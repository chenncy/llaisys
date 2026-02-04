"""Qwen2 model inference via LLAISYS backend (no PyTorch in inference path)."""
from typing import Sequence
from ctypes import byref, cast, c_int, c_int64, c_size_t, POINTER

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights, LlaisysQwen2Model_t
from ..libllaisys.llaisys_types import DataType

from pathlib import Path
import json
import numpy as np
import safetensors

try:
    import torch
except ImportError:
    torch = None


# Safetensors key -> tensor_handle (support multiple key names for same weight)
def _weight_key_to_handle(weights_ptr, nlayer: int):
    """Yield (safetensors_key, tensor_handle) for each weight."""
    w = weights_ptr.contents
    yield "model.embed_tokens.weight", w.in_embed
    yield "model.norm.weight", w.out_norm_w
    yield "model.norm.w", w.out_norm_w  # ModelScope / some checkpoints
    yield "lm_head.weight", w.out_embed
    for i in range(nlayer):
        yield f"model.layers.{i}.input_layernorm.weight", w.attn_norm_w[i]
        yield f"model.layers.{i}.self_attn.q_proj.weight", w.attn_q_w[i]
        yield f"model.layers.{i}.self_attn.q_proj.bias", w.attn_q_b[i]
        yield f"model.layers.{i}.self_attn.k_proj.weight", w.attn_k_w[i]
        yield f"model.layers.{i}.self_attn.k_proj.bias", w.attn_k_b[i]
        yield f"model.layers.{i}.self_attn.v_proj.weight", w.attn_v_w[i]
        yield f"model.layers.{i}.self_attn.v_proj.bias", w.attn_v_b[i]
        yield f"model.layers.{i}.self_attn.o_proj.weight", w.attn_o_w[i]
        yield f"model.layers.{i}.post_attention_layernorm.weight", w.mlp_norm_w[i]
        yield f"model.layers.{i}.mlp.gate_proj.weight", w.mlp_gate_w[i]
        yield f"model.layers.{i}.mlp.up_proj.weight", w.mlp_up_w[i]
        yield f"model.layers.{i}.mlp.down_proj.weight", w.mlp_down_w[i]


def _numpy_to_backend(arr: np.ndarray, tensor_handle) -> None:
    """Copy numpy array into LLAISYS tensor (host to device/host)."""
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.float32:
        # Safetensors often returns float32 for bfloat16; convert to bf16 (high 16 bits)
        arr_bf16 = (arr.view(np.uint32) >> 16).astype(np.uint16)
        LIB_LLAISYS.tensorLoad(tensor_handle, arr_bf16.ctypes.data)
    elif arr.dtype == np.uint16 or arr.dtype == np.float16:
        LIB_LLAISYS.tensorLoad(tensor_handle, arr.ctypes.data)
    else:
        LIB_LLAISYS.tensorLoad(tensor_handle, arr.ctypes.data)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        hidden_size = config["hidden_size"]
        num_hidden_layers = config["num_hidden_layers"]
        num_attention_heads = config["num_attention_heads"]
        num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
        intermediate_size = config["intermediate_size"]
        vocab_size = config["vocab_size"]
        rms_norm_eps = float(config.get("rms_norm_eps", 1e-6))
        rope_theta = float(config.get("rope_theta", 10000.0))
        eos_id = config.get("eos_token_id", config.get("bos_token_id", 151643))
        max_position = config.get("max_position_embeddings", 131072)
        maxseq = min(4096, max_position)

        # Prefer "dtype" (new), fallback to "torch_dtype"
        cfg_dtype = config.get("dtype", config.get("torch_dtype", "bfloat16"))
        if isinstance(cfg_dtype, str) and "bfloat" in cfg_dtype.lower():
            dtype = DataType.BF16
        elif isinstance(cfg_dtype, str) and "float16" in cfg_dtype.lower():
            dtype = DataType.F16
        else:
            dtype = DataType.BF16

        dh = hidden_size // num_attention_heads

        meta = LlaisysQwen2Meta(
            dtype=dtype,
            nlayer=num_hidden_layers,
            hs=hidden_size,
            nh=num_attention_heads,
            nkvh=num_key_value_heads,
            dh=dh,
            di=intermediate_size,
            maxseq=maxseq,
            voc=vocab_size,
            epsilon=rms_norm_eps,
            theta=rope_theta,
            end_token=eos_id,
        )

        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device,
            None,
            0,
        )
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate failed")

        self._end_token = eos_id
        self._nlayer = num_hidden_layers

        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        key_to_handle = dict(_weight_key_to_handle(weights_ptr, num_hidden_layers))
        loaded_keys = set()

        # NumPy 不原生支持 bfloat16，safetensors 用 numpy 加载 bf16 会报错，必须用 PyTorch 打开后转成 numpy
        use_pt = (dtype == DataType.BF16) or (torch is not None)
        if dtype == DataType.BF16 and torch is None:
            raise RuntimeError("Loading bfloat16 weights requires PyTorch (pip install torch)")
        for fpath in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(
                fpath, framework="pt" if use_pt else "numpy", device="cpu"
            ) as data:
                for key in data.keys():
                    if key not in key_to_handle:
                        continue
                    handle = key_to_handle[key]
                    t = data.get_tensor(key)
                    if use_pt:
                        if t.dtype == torch.bfloat16:
                            arr = t.float().numpy()
                        else:
                            arr = t.numpy()
                    else:
                        arr = np.ascontiguousarray(t)
                    arr = np.ascontiguousarray(arr)
                    _numpy_to_backend(arr, handle)
                    loaded_keys.add(key)

        # Require at least embed to be loaded
        embed_loaded = "model.embed_tokens.weight" in loaded_keys
        if not embed_loaded:
            # List keys from file (metadata only) to help debug
            sample_keys = []
            for fpath in sorted(model_path.glob("*.safetensors")):
                with safetensors.safe_open(
                    fpath, framework="pt" if use_pt else "numpy", device="cpu"
                ) as data:
                    sample_keys.extend(list(data.keys())[:40])
                break
            raise RuntimeError(
                "No embedding weights loaded. Loaded %d keys; sample keys from file: %s"
                % (len(loaded_keys), sample_keys[:30])
            )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # Argmax sampling only (top_k=1, temperature=1, top_p=1 for test)
        if not inputs:
            raise ValueError("generate() called with empty input token list")
        tokens = list(inputs)
        import os
        if os.environ.get("LLAISYS_DEBUG"):
            print(f"[LLAISYS] Qwen2.generate() n_inputs={len(inputs)} max_new_tokens={max_new_tokens}")
        for _ in range(max_new_tokens):
            n = len(tokens)
            token_arr = (c_int64 * n)(*tokens)
            next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                cast(token_arr, POINTER(c_int64)),
                n,
            )
            if next_tok == -1:
                raise RuntimeError("llaisysQwen2ModelInfer failed (returned -1)")
            tokens.append(next_tok)
            if next_tok == self._end_token:
                break
        return tokens

    def __del__(self):
        if getattr(self, "_model", None) is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
