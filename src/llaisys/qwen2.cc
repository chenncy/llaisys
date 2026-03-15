/**
 * Qwen2 模型推理的 C++ 实现：模型创建、权重分配、单步前向与 KV Cache 管理。
 *
 * 对外暴露 C 接口（见 include/llaisys/models/qwen2.h）：
 * - llaisysQwen2ModelCreate / Destroy / Weights：创建、销毁、获取权重句柄；
 * - llaisysQwen2ModelInfer：给定当前 token 序列，执行一次前向，返回下一个 token id。
 *
 * 前向流程：embedding -> 逐层 Transformer Block（attention + MLP）-> 最后一层 norm -> 输出层 linear -> argmax。
 */
#include "llaisys/llaisys_tensor.hpp"
#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/sample/op.hpp"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <cstring>
#include <cmath>
#include <vector>

namespace {

using namespace llaisys;
using tensor_t = llaisys::tensor_t;

/// 从 C 接口的 LlaisysTensor 包装中取出内部 tensor_t 指针
inline tensor_t get_t(llaisysTensor_t t) { return t->tensor; }

/// 按当前设备做同步内存拷贝（H2H 或 D2D），供写 KV cache 和写回 hidden 用
void copy_sync(void *dst, const void *src, size_t bytes, llaisysDeviceType_t dev) {
    llaisys::core::context().setDevice(dev, 0);
    llaisysMemcpyKind_t kind = (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
    llaisys::core::context().runtime().api()->memcpy_sync(dst, src, bytes, kind);
}

/// 设备 -> 主机拷贝（Export 用）
void copy_device_to_host(void *host_dst, const void *dev_src, size_t bytes, llaisysDeviceType_t dev) {
    if (dev == LLAISYS_DEVICE_CPU) {
        std::memcpy(host_dst, dev_src, bytes);
        return;
    }
    llaisys::core::context().setDevice(dev, 0);
    llaisys::core::context().runtime().api()->memcpy_sync(host_dst, dev_src, bytes, LLAISYS_MEMCPY_D2H);
}

/// 主机 -> 设备拷贝（Import 用）
void copy_host_to_device(void *dev_dst, const void *host_src, size_t bytes, llaisysDeviceType_t dev) {
    if (dev == LLAISYS_DEVICE_CPU) {
        std::memcpy(dev_dst, host_src, bytes);
        return;
    }
    llaisys::core::context().setDevice(dev, 0);
    llaisys::core::context().runtime().api()->memcpy_sync(dev_dst, host_src, bytes, LLAISYS_MEMCPY_H2D);
}

} // namespace

/// Qwen2 模型内部表示：元信息、权重张量、每层 K/V cache 及当前已填充长度
/// 当 meta.max_batch_size > 1 时：k/v_caches 每层形状 [max_batch_size, maxseq, nkvh, dh]，cache_lens 长度为 max_batch_size
/// 当 max_batch_size == 1：k/v_caches 每层形状 [maxseq, nkvh, dh]，cache_lens.size()==1
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    std::vector<tensor_t> k_caches;   /// 每层 key cache；单 slot [maxseq,nkvh,dh]，多 slot [max_batch,maxseq,nkvh,dh]
    std::vector<tensor_t> v_caches;   /// 每层 value cache，同上
    std::vector<size_t> cache_lens;   /// 每 slot 已写入长度；单 slot 时 size()==1
    llaisysDeviceType_t device_type;
    int device_id;
};

LLAISYS_EXTERN_C {

/// 根据 meta 为模型分配所有权重张量（embed、输出 norm、每层 attention/MLP 的 weight/bias），不填数据，由 Python 侧 tensorLoad 灌入
static void create_weight_tensors(LlaisysQwen2Model *m) {
    const LlaisysQwen2Meta *meta = &m->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs;
    const size_t nh = meta->nh;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t di = meta->di;
    const size_t voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const llaisysDeviceType_t dev = m->device_type;
    const int dev_id = m->device_id;

    auto mk = [&](const std::vector<size_t> &shape) {
        return LlaisysTensor{llaisys::Tensor::create(shape, dtype, dev, dev_id)};
    };

    // 词嵌入 [voc, hs]，输出层嵌入 [voc, hs]，最后一层 RMSNorm 权重 [hs]
    m->weights.in_embed = new LlaisysTensor(mk({voc, hs}));
    m->weights.out_embed = new LlaisysTensor(mk({voc, hs}));
    m->weights.out_norm_w = new LlaisysTensor(mk({hs}));

    // 每层指针数组，下面按层填充
    m->weights.attn_norm_w = new llaisysTensor_t[nlayer];
    m->weights.attn_q_w = new llaisysTensor_t[nlayer];
    m->weights.attn_q_b = new llaisysTensor_t[nlayer];
    m->weights.attn_k_w = new llaisysTensor_t[nlayer];
    m->weights.attn_k_b = new llaisysTensor_t[nlayer];
    m->weights.attn_v_w = new llaisysTensor_t[nlayer];
    m->weights.attn_v_b = new llaisysTensor_t[nlayer];
    m->weights.attn_o_w = new llaisysTensor_t[nlayer];
    m->weights.mlp_norm_w = new llaisysTensor_t[nlayer];
    m->weights.mlp_gate_w = new llaisysTensor_t[nlayer];
    m->weights.mlp_up_w = new llaisysTensor_t[nlayer];
    m->weights.mlp_down_w = new llaisysTensor_t[nlayer];

    for (size_t i = 0; i < nlayer; i++) {
        m->weights.attn_norm_w[i] = new LlaisysTensor(mk({hs}));
        m->weights.attn_q_w[i] = new LlaisysTensor(mk({nh * dh, hs}));
        m->weights.attn_q_b[i] = new LlaisysTensor(mk({nh * dh}));
        m->weights.attn_k_w[i] = new LlaisysTensor(mk({nkvh * dh, hs}));
        m->weights.attn_k_b[i] = new LlaisysTensor(mk({nkvh * dh}));
        m->weights.attn_v_w[i] = new LlaisysTensor(mk({nkvh * dh, hs}));
        m->weights.attn_v_b[i] = new LlaisysTensor(mk({nkvh * dh}));
        m->weights.attn_o_w[i] = new LlaisysTensor(mk({hs, nh * dh}));
        m->weights.mlp_norm_w[i] = new LlaisysTensor(mk({hs}));
        m->weights.mlp_gate_w[i] = new LlaisysTensor(mk({di, hs}));
        m->weights.mlp_up_w[i] = new LlaisysTensor(mk({di, hs}));
        m->weights.mlp_down_w[i] = new LlaisysTensor(mk({hs, di}));
    }
}

/// 创建模型：分配 meta、权重张量、每层 KV cache，返回模型指针；权重数据由调用方通过 llaisysQwen2ModelWeights + tensorLoad 写入
struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                  llaisysDeviceType_t device,
                                                  int *device_ids,
                                                  int ndevice) {
    (void)device_ids;
    (void)ndevice;
    LlaisysQwen2Model *m = new LlaisysQwen2Model();
    m->meta = *meta;
    if (m->meta.max_batch_size == 0)
        m->meta.max_batch_size = 1;
    const size_t max_batch = m->meta.max_batch_size;
    m->cache_lens.assign(max_batch, 0);
    m->device_type = device;
    m->device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;

    create_weight_tensors(m);

    const size_t nlayer = meta->nlayer;
    const size_t maxseq = meta->maxseq;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;

    m->k_caches.resize(nlayer);
    m->v_caches.resize(nlayer);
    if (max_batch > 1) {
        for (size_t i = 0; i < nlayer; i++) {
            m->k_caches[i] = llaisys::Tensor::create(
                {max_batch, maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
            m->v_caches[i] = llaisys::Tensor::create(
                {max_batch, maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
        }
    } else {
        for (size_t i = 0; i < nlayer; i++) {
            m->k_caches[i] = llaisys::Tensor::create(
                {maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
            m->v_caches[i] = llaisys::Tensor::create(
                {maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
        }
    }
    return m;
}

/// 释放模型：先销毁所有权重张量和指针数组，再 delete 模型本体
void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) return;
    const size_t nlayer = model->meta.nlayer;

    tensorDestroy(model->weights.in_embed);
    tensorDestroy(model->weights.out_embed);
    tensorDestroy(model->weights.out_norm_w);

    for (size_t i = 0; i < nlayer; i++) {
        tensorDestroy(model->weights.attn_norm_w[i]);
        tensorDestroy(model->weights.attn_q_w[i]);
        tensorDestroy(model->weights.attn_q_b[i]);
        tensorDestroy(model->weights.attn_k_w[i]);
        tensorDestroy(model->weights.attn_k_b[i]);
        tensorDestroy(model->weights.attn_v_w[i]);
        tensorDestroy(model->weights.attn_v_b[i]);
        tensorDestroy(model->weights.attn_o_w[i]);
        tensorDestroy(model->weights.mlp_norm_w[i]);
        tensorDestroy(model->weights.mlp_gate_w[i]);
        tensorDestroy(model->weights.mlp_up_w[i]);
        tensorDestroy(model->weights.mlp_down_w[i]);
    }
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;

    delete model;
}

/// 返回模型权重结构体指针，供 Python 侧根据 safetensors key 找到对应句柄并 tensorLoad
struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return model ? &model->weights : nullptr;
}

} // extern "C"

namespace {

/**
 * 单层 Transformer Block 前向：Attention（norm -> q/k/v -> RoPE -> 写 KV cache -> attention -> o_proj -> 残差）+ MLP（norm -> gate/up -> SwiGLU -> down -> 残差）。
 * 输入输出通过 hidden 传入并在本函数内原地更新；slot_id 指定使用哪一槽的 KV cache（多 slot 时）。
 */
void forward_layer(LlaisysQwen2Model *m, size_t layer_idx, size_t slot_id,
                   tensor_t hidden,      // [seq, hs]
                   tensor_t normed,      // [seq, hs]
                   tensor_t q_buf,       // [seq, nh*dh]
                   tensor_t k_buf,       // [seq, nkvh*dh]
                   tensor_t v_buf,       // [seq, nkvh*dh]
                   tensor_t q_rope,      // [seq, nh, dh]
                   tensor_t k_rope,      // [seq, nkvh, dh]
                   tensor_t attn_val,    // [seq, nh, dh]
                   tensor_t o_proj_out,  // [seq, hs]
                   tensor_t res_buf,     // [seq, hs]
                   tensor_t gate_buf,    // [seq, di]
                   tensor_t up_buf,      // [seq, di]
                   tensor_t mlp_buf,     // [seq, di]
                   tensor_t down_buf,    // [seq, hs]
                   size_t seq_len, size_t cache_start,
                   tensor_t pos_ids_t) {
    const LlaisysQwen2Meta *meta = &m->meta;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh;
    const size_t maxseq = meta->maxseq;
    const float eps = meta->epsilon, theta = meta->theta;
    LlaisysQwen2Weights *w = &m->weights;
    tensor_t wt = get_t(w->attn_norm_w[layer_idx]);
    const float scale = 1.f / std::sqrt(static_cast<float>(dh));

    tensor_t k_cache_raw = m->k_caches[layer_idx];
    tensor_t v_cache_raw = m->v_caches[layer_idx];
    tensor_t k_cache, v_cache;
    if (meta->max_batch_size > 1) {
        k_cache = k_cache_raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
        v_cache = v_cache_raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
    } else {
        k_cache = k_cache_raw;
        v_cache = v_cache_raw;
    }

    // ---------- Attention 分支：norm -> Q/K/V 投影 -> RoPE -> 写 KV cache -> causal attention -> o_proj -> 残差 ----------
    llaisys::ops::rms_norm(normed, hidden, wt, eps);
    llaisys::ops::linear(q_buf, normed, get_t(w->attn_q_w[layer_idx]), get_t(w->attn_q_b[layer_idx]));
    llaisys::ops::linear(k_buf, normed, get_t(w->attn_k_w[layer_idx]), get_t(w->attn_k_b[layer_idx]));
    llaisys::ops::linear(v_buf, normed, get_t(w->attn_v_w[layer_idx]), get_t(w->attn_v_b[layer_idx]));

    // 展平为 [seq, nh, dh] / [seq, nkvh, dh] 以做 RoPE
    std::vector<size_t> shape_q = {seq_len, nh, dh};
    std::vector<size_t> shape_kv = {seq_len, nkvh, dh};
    tensor_t q_view = q_buf->view(shape_q);
    tensor_t k_view = k_buf->view(shape_kv);
    tensor_t v_view = v_buf->view(shape_kv);

    llaisys::ops::rope(q_rope, q_view, pos_ids_t, theta);
    llaisys::ops::rope(k_rope, k_view, pos_ids_t, theta);

    // 将本步的 K/V 写入 cache 的 [cache_start, cache_start+seq_len) 位置，decode 时复用历史
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const size_t kv_row_bytes = nkvh * dh * elem_size;
    for (size_t s = 0; s < seq_len; s++) {
        size_t cache_pos = cache_start + s;
        copy_sync(
            reinterpret_cast<std::byte *>(k_cache->data()) + cache_pos * kv_row_bytes,
            reinterpret_cast<const std::byte *>(k_rope->data()) + s * kv_row_bytes,
            kv_row_bytes, m->device_type);
        copy_sync(
            reinterpret_cast<std::byte *>(v_cache->data()) + cache_pos * kv_row_bytes,
            reinterpret_cast<const std::byte *>(v_buf->data()) + s * kv_row_bytes,
            kv_row_bytes, m->device_type);
    }

    // 当前有效长度为 cache_start + seq_len；取 cache 的该前缀做 attention
    size_t kv_len = cache_start + seq_len;
    tensor_t k_slice = k_cache->slice(0, 0, kv_len);
    tensor_t v_slice = v_cache->slice(0, 0, kv_len);

    llaisys::ops::self_attention(attn_val, q_rope, k_slice, v_slice, scale);

    std::vector<size_t> shape_attn_flat = {seq_len, nh * dh};
    tensor_t attn_flat = attn_val->view(shape_attn_flat);
    llaisys::ops::linear(o_proj_out, attn_flat, get_t(w->attn_o_w[layer_idx]), nullptr);

    llaisys::ops::add(res_buf, hidden, o_proj_out);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, m->device_type);

    // ---------- MLP 分支：norm -> gate/up -> SwiGLU -> down -> 残差 ----------
    llaisys::ops::rms_norm(normed, hidden, get_t(w->mlp_norm_w[layer_idx]), eps);
    llaisys::ops::linear(gate_buf, normed, get_t(w->mlp_gate_w[layer_idx]), nullptr);
    llaisys::ops::linear(up_buf, normed, get_t(w->mlp_up_w[layer_idx]), nullptr);
    llaisys::ops::swiglu(mlp_buf, gate_buf, up_buf);
    llaisys::ops::linear(down_buf, mlp_buf, get_t(w->mlp_down_w[layer_idx]), nullptr);
    llaisys::ops::add(res_buf, hidden, down_buf);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, m->device_type);
}

} // namespace

LLAISYS_EXTERN_C {

size_t llaisysQwen2ModelGetCacheLen(struct LlaisysQwen2Model *model) {
    return model && !model->cache_lens.empty() ? model->cache_lens[0] : 0;
}

size_t llaisysQwen2ModelGetCacheLenSlot(struct LlaisysQwen2Model *model, size_t slot_id) {
    if (!model || model->cache_lens.empty()) return 0;
    if (slot_id >= model->cache_lens.size()) return 0;
    return model->cache_lens[slot_id];
}

size_t llaisysQwen2ModelGetKVCacheBytes(struct LlaisysQwen2Model *model, size_t prefix_len) {
    if (!model || prefix_len == 0) return 0;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    return nlayer * 2 * prefix_len * nkvh * dh * elem_size;
}

static tensor_t get_slot_k_cache(LlaisysQwen2Model *model, size_t layer_idx, size_t slot_id) {
    const size_t maxseq = model->meta.maxseq, nkvh = model->meta.nkvh, dh = model->meta.dh;
    tensor_t raw = model->k_caches[layer_idx];
    if (model->meta.max_batch_size > 1)
        return raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
    return raw;
}
static tensor_t get_slot_v_cache(LlaisysQwen2Model *model, size_t layer_idx, size_t slot_id) {
    const size_t maxseq = model->meta.maxseq, nkvh = model->meta.nkvh, dh = model->meta.dh;
    tensor_t raw = model->v_caches[layer_idx];
    if (model->meta.max_batch_size > 1)
        return raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
    return raw;
}

void llaisysQwen2ModelExportKVCache(struct LlaisysQwen2Model *model, void *ptr_out) {
    if (!model || !ptr_out || model->cache_lens.empty()) return;
    const size_t cache_len = model->cache_lens[0];
    if (cache_len == 0) return;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const size_t row_bytes = nkvh * dh * elem_size;
    const size_t layer_bytes = cache_len * row_bytes;
    std::byte *out = static_cast<std::byte *>(ptr_out);
    llaisys::core::context().setDevice(model->device_type, model->device_id);
    for (size_t i = 0; i < nlayer; i++) {
        tensor_t k_slot = get_slot_k_cache(model, i, 0);
        tensor_t v_slot = get_slot_v_cache(model, i, 0);
        copy_device_to_host(out, k_slot->data(), layer_bytes, model->device_type);
        out += layer_bytes;
        copy_device_to_host(out, v_slot->data(), layer_bytes, model->device_type);
        out += layer_bytes;
    }
}

void llaisysQwen2ModelImportKVCache(struct LlaisysQwen2Model *model, const void *ptr_in, size_t prefix_len) {
    if (!model || !ptr_in || prefix_len == 0) return;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const size_t row_bytes = nkvh * dh * elem_size;
    const size_t layer_bytes = prefix_len * row_bytes;
    const std::byte *in = static_cast<const std::byte *>(ptr_in);
    llaisys::core::context().setDevice(model->device_type, model->device_id);
    for (size_t i = 0; i < nlayer; i++) {
        tensor_t k_slot = get_slot_k_cache(model, i, 0);
        tensor_t v_slot = get_slot_v_cache(model, i, 0);
        copy_host_to_device(k_slot->data(), in, layer_bytes, model->device_type);
        in += layer_bytes;
        copy_host_to_device(v_slot->data(), in, layer_bytes, model->device_type);
        in += layer_bytes;
    }
    model->cache_lens[0] = prefix_len;
}

void llaisysQwen2ModelResetKVCache(struct LlaisysQwen2Model *model) {
    if (model) {
        for (size_t i = 0; i < model->cache_lens.size(); i++)
            model->cache_lens[i] = 0;
    }
}

void llaisysQwen2ModelResetKVCacheSlot(struct LlaisysQwen2Model *model, size_t slot_id) {
    if (!model || model->cache_lens.empty()) return;
    if (slot_id < model->cache_lens.size())
        model->cache_lens[slot_id] = 0;
}

/**
 * 单步推理：根据当前 token 序列做一次前向，返回下一个 token 的 id。
 * - 若 cache_len==0（prefill）：传入整段 token_ids，seq_len=ntoken，并填充 KV cache；
 * - 若 cache_len>0 且 ntoken>1（suffix prefill）：传入后缀 token_ids，seq_len=ntoken，只对后缀做 prefill；
 * - 否则（decode）：只传入最后一个 token，seq_len=1，用已有 cache 做 attention。
 * - temperature<=0 或极小且 top_k<=1、top_p>=1 时使用 argmax；否则使用随机采样（Temperature/Top-K/Top-P）。
 */
int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model,
                                int64_t *token_ids,
                                size_t ntoken,
                                float temperature,
                                int top_k,
                                float top_p,
                                unsigned long long seed) {
    if (!model || ntoken == 0) return static_cast<int64_t>(-1);

    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di;
    const size_t voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const llaisysDeviceType_t dev = model->device_type;
    const int dev_id = model->device_id;

    const size_t slot0_len = model->cache_lens[0];
    const bool is_prefill = (slot0_len == 0);
    const bool is_suffix_prefill = (slot0_len > 0 && ntoken > 1);
    const size_t seq_len = is_suffix_prefill ? ntoken : (is_prefill ? ntoken : 1);
    const size_t cache_start = slot0_len;

    llaisys::core::context().setDevice(dev, dev_id);

    // 本步输入的 token id：prefill 为整段，suffix prefill 为后缀整段，decode 为最后一个
    tensor_t token_tensor = llaisys::Tensor::create(
        {seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    if (is_suffix_prefill) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids, ntoken * sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    } else if (is_prefill) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids, ntoken * sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    } else {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids + ntoken - 1, sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    }

    // 前向临时张量：hidden 每层更新，其余为每层复用缓冲区
    //去 llaisys 命名空间下的 Tensor 类里，调用 create 函数来创建一个张量。
    tensor_t hidden = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t normed = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t q_buf = llaisys::Tensor::create({seq_len, nh * dh}, dtype, dev, dev_id);
    tensor_t k_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, dev, dev_id);
    tensor_t v_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, dev, dev_id);
    tensor_t q_rope = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t k_rope = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, dev, dev_id);
    tensor_t attn_val = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t o_proj_out = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t res_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t gate_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t up_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t mlp_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t down_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);

    // 位置 id：当前步在全局序列中的位置，用于 RoPE
    tensor_t pos_ids_t = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    std::vector<int64_t> pos_ids_host(seq_len);
    for (size_t s = 0; s < seq_len; s++)
        pos_ids_host[s] = static_cast<int64_t>(cache_start + s);
    llaisys::core::context().runtime().api()->memcpy_sync(
        pos_ids_t->data(), pos_ids_host.data(), seq_len * sizeof(int64_t),
        (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);

    llaisys::ops::embedding(hidden, token_tensor, get_t(model->weights.in_embed));

    for (size_t i = 0; i < nlayer; i++) {
        forward_layer(model, i, 0, hidden, normed, q_buf, k_buf, v_buf,
                     q_rope, k_rope, attn_val, o_proj_out, res_buf,
                     gate_buf, up_buf, mlp_buf, down_buf,
                     seq_len, cache_start, pos_ids_t);
    }

    model->cache_lens[0] = cache_start + seq_len;

    // 最后一层 norm + 输出层 linear：hidden -> logits [seq_len, voc]
    // 步骤 1：对模型脑子里最后一层的抽象特征（hidden）做最后一次梳理（RMSNorm 归一化）
    llaisys::ops::rms_norm(normed, hidden, get_t(model->weights.out_norm_w), meta->epsilon);
    
    // 步骤 2：准备一张长长的“打分表”（张量 logits_t），长度是整本字典的大小（voc，通常是十几万）
    tensor_t logits_t = llaisys::Tensor::create({seq_len, voc}, dtype, dev, dev_id);
    
    // 步骤 3：把抽象特征（normed）映射到字典上，给字典里的每一个词打分
    // 分数（logits）越高，代表模型觉得这个词接在后面的概率越大
    llaisys::ops::linear(logits_t, normed, get_t(model->weights.out_embed), nullptr);

    // 取最后一个位置的 logits，argmax 得到下一个 token id（当前实现为贪心）
    // 步骤 4：因为我们输入了一大段话（seq_len），但我们只关心“接下来这一个字”该填什么
    // 所以用 slice 操作，把打分表里“最后那个字”的所有候选词得分切出来
    tensor_t last_logit = logits_t->slice(0, seq_len - 1, seq_len);
    
    // 步骤 5：把切出来的多维数组拍扁成一维的列表
    tensor_t last_logit_1d = last_logit->view(std::vector<size_t>{voc});

    // 随机采样：temperature 有效且（top_k>1 或 top_p 在 (0,1)）；否则使用 argmax 贪心
    const bool use_sampling = (temperature > 1e-6f && (top_k > 1 || (top_p > 0.f && top_p < 1.f)));
    int64_t next_token = meta->end_token;

    if (use_sampling) {
        // 随机采样：支持 Temperature、Top-K、Top-P（sample 算子目前仅实现 CPU）
        tensor_t logits_for_sample = last_logit_1d;
        if (dev != LLAISYS_DEVICE_CPU) {
            tensor_t logits_cpu = llaisys::Tensor::create({voc}, dtype, LLAISYS_DEVICE_CPU, 0);
            size_t logits_bytes = voc * llaisys::utils::dsize(dtype);
            llaisys::core::context().runtime().api()->memcpy_sync(
                logits_cpu->data(), last_logit_1d->data(), logits_bytes, LLAISYS_MEMCPY_D2H);
            logits_for_sample = logits_cpu;
        }
        tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::sample(sampled_idx, logits_for_sample, temperature, top_k, top_p, static_cast<uint64_t>(seed));
        llaisys::core::context().runtime().api()->memcpy_sync(
            &next_token, sampled_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
    } else {
        // 贪心：argmax
        tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
        tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, dev, dev_id);
        llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
        llaisys::core::context().runtime().api()->memcpy_sync(
            &next_token, max_idx_t->data(), sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2H);
    }

    return next_token;
}

int64_t llaisysQwen2ModelInferWithSlot(struct LlaisysQwen2Model *model,
                                        size_t slot_id,
                                        int64_t *token_ids,
                                        size_t ntoken,
                                        float temperature,
                                        int top_k,
                                        float top_p,
                                        unsigned long long seed) {
    if (!model || ntoken == 0) return static_cast<int64_t>(-1);
    if (slot_id >= model->cache_lens.size()) return static_cast<int64_t>(-1);

    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di;
    const size_t voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const llaisysDeviceType_t dev = model->device_type;
    const int dev_id = model->device_id;

    size_t *p_cache_len = &model->cache_lens[slot_id];
    const bool is_prefill = (*p_cache_len == 0);
    const bool is_suffix_prefill = (*p_cache_len > 0 && ntoken > 1);
    const size_t seq_len = is_suffix_prefill ? ntoken : (is_prefill ? ntoken : 1);
    const size_t cache_start = *p_cache_len;

    llaisys::core::context().setDevice(dev, dev_id);

    tensor_t token_tensor = llaisys::Tensor::create(
        {seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    if (is_suffix_prefill) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids, ntoken * sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    } else if (is_prefill) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids, ntoken * sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    } else {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids + ntoken - 1, sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    }

    tensor_t hidden = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t normed = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t q_buf = llaisys::Tensor::create({seq_len, nh * dh}, dtype, dev, dev_id);
    tensor_t k_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, dev, dev_id);
    tensor_t v_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, dev, dev_id);
    tensor_t q_rope = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t k_rope = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, dev, dev_id);
    tensor_t attn_val = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t o_proj_out = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t res_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t gate_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t up_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t mlp_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t down_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);

    tensor_t pos_ids_t = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    std::vector<int64_t> pos_ids_host(seq_len);
    for (size_t s = 0; s < seq_len; s++)
        pos_ids_host[s] = static_cast<int64_t>(cache_start + s);
    llaisys::core::context().runtime().api()->memcpy_sync(
        pos_ids_t->data(), pos_ids_host.data(), seq_len * sizeof(int64_t),
        (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);

    llaisys::ops::embedding(hidden, token_tensor, get_t(model->weights.in_embed));

    for (size_t i = 0; i < nlayer; i++) {
        forward_layer(model, i, slot_id, hidden, normed, q_buf, k_buf, v_buf,
                     q_rope, k_rope, attn_val, o_proj_out, res_buf,
                     gate_buf, up_buf, mlp_buf, down_buf,
                     seq_len, cache_start, pos_ids_t);
    }

    *p_cache_len = cache_start + seq_len;

    llaisys::ops::rms_norm(normed, hidden, get_t(model->weights.out_norm_w), meta->epsilon);
    tensor_t logits_t = llaisys::Tensor::create({seq_len, voc}, dtype, dev, dev_id);
    llaisys::ops::linear(logits_t, normed, get_t(model->weights.out_embed), nullptr);

    tensor_t last_logit = logits_t->slice(0, seq_len - 1, seq_len);
    tensor_t last_logit_1d = last_logit->view(std::vector<size_t>{voc});

    const bool use_sampling = (temperature > 1e-6f && (top_k > 1 || (top_p > 0.f && top_p < 1.f)));
    int64_t next_token = meta->end_token;

    if (use_sampling) {
        tensor_t logits_for_sample = last_logit_1d;
        if (dev != LLAISYS_DEVICE_CPU) {
            tensor_t logits_cpu = llaisys::Tensor::create({voc}, dtype, LLAISYS_DEVICE_CPU, 0);
            size_t logits_bytes = voc * llaisys::utils::dsize(dtype);
            llaisys::core::context().runtime().api()->memcpy_sync(
                logits_cpu->data(), last_logit_1d->data(), logits_bytes, LLAISYS_MEMCPY_D2H);
            logits_for_sample = logits_cpu;
        }
        tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::sample(sampled_idx, logits_for_sample, temperature, top_k, top_p, static_cast<uint64_t>(seed));
        llaisys::core::context().runtime().api()->memcpy_sync(
            &next_token, sampled_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
    } else {
        tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
        tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, dev, dev_id);
        llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
        llaisys::core::context().runtime().api()->memcpy_sync(
            &next_token, max_idx_t->data(), sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2H);
    }

    return next_token;
}

void llaisysQwen2ModelBatchedDecode(struct LlaisysQwen2Model *model,
                                     const size_t *slot_ids,
                                     const int64_t *token_ids,
                                     size_t n_batch,
                                     int64_t *out_next_tokens,
                                     float temperature,
                                     int top_k,
                                     float top_p,
                                     unsigned long long seed) {
    if (!model || !slot_ids || !token_ids || !out_next_tokens || n_batch == 0)
        return;
    const size_t max_batch = model->meta.max_batch_size;
    if (n_batch > max_batch)
        n_batch = max_batch;
    for (size_t i = 0; i < n_batch; i++) {
        int64_t one_token = token_ids[i];
        out_next_tokens[i] = llaisysQwen2ModelInferWithSlot(
            model, slot_ids[i], &one_token, 1,
            temperature, top_k, top_p, seed);
    }
}

} // extern "C"
