import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Ensure we use the project's llaisys (when running from repo root or with PYTHONPATH)
import llaisys
print(f"[test_infer] llaisys loaded from: {os.path.abspath(os.path.dirname(llaisys.__file__))}")


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    if not inputs:
        raise ValueError(
            "tokenizer.encode returned empty list for prompt %r" % (prompt[:50],)
        )
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    if not outputs:
        raise RuntimeError(
            "LLAISYS generate() returned no tokens. "
            "Rebuild the shared library: run 'xmake build llaisys' in the project root, "
            "then ensure python/llaisys/libllaisys/ contains the updated llaisys.dll (or .so)."
        )
    if len(outputs) < len(inputs):
        raise RuntimeError(
            "LLAISYS generate() returned %d tokens but input had %d (expected >= input length). "
            "Check that the project llaisys is used (see path printed at start)."
            % (len(outputs), len(inputs))
        )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, model, model_path = load_hf_model(args.model, args.device)

    # Example prompt
    start_time = time.time()
    tokens, output = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    del model
    gc.collect()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    try:
        model = load_llaisys_model(model_path, args.device)
    except Exception as e:
        print("\n\033[91mLLAISYS model load failed:\033[0m")
        import traceback
        traceback.print_exc()
        raise

    start_time = time.time()
    try:
        llaisys_tokens, llaisys_output = llaisys_infer(
            args.prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
    except Exception as e:
        print("\n\033[91mLLAISYS inference failed:\033[0m")
        import traceback
        traceback.print_exc()
        raise

    end_time = time.time()

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    if args.test:
        if llaisys_tokens != tokens:
            raise AssertionError(
                "LLAISYS token sequence did not match PyTorch. "
                "If LLAISYS returned [], ensure 'xmake build llaisys' was run and the DLL in "
                "python/llaisys/libllaisys/ is up to date."
            )
        print("\033[92mTest passed!\033[0m\n")
