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
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <cstring>
#include <cmath>
#include <vector>

namespace {

using namespace llaisys;
using tensor_t = llaisys::tensor_t;

inline tensor_t get_t(llaisysTensor_t t) { return t->tensor; }

void copy_sync(void *dst, const void *src, size_t bytes, llaisysDeviceType_t dev) {
    llaisys::core::context().setDevice(dev, 0);
    llaisysMemcpyKind_t kind = (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
    llaisys::core::context().runtime().api()->memcpy_sync(dst, src, bytes, kind);
}

} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    std::vector<tensor_t> k_caches;
    std::vector<tensor_t> v_caches;
    size_t cache_len;
    llaisysDeviceType_t device_type;
    int device_id;
};

__C {

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

    m->weights.in_embed = new LlaisysTensor(mk({voc, hs}));
    m->weights.out_embed = new LlaisysTensor(mk({voc, hs}));
    m->weights.out_norm_w = new LlaisysTensor(mk({hs}));

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

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                  llaisysDeviceType_t device,
                                                  int *device_ids,
                                                  int ndevice) {
    (void)device_ids;
    (void)ndevice;
    LlaisysQwen2Model *m = new LlaisysQwen2Model();
    m->meta = *meta;
    m->cache_len = 0;
    m->device_type = device;
    m->device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;

    create_weight_tensors(m);

    const size_t nlayer = meta->nlayer;
    const size_t maxseq = meta->maxseq;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;

    m->k_caches.resize(nlayer);
    m->v_caches.resize(nlayer);
    for (size_t i = 0; i < nlayer; i++) {
        m->k_caches[i] = llaisys::Tensor::create(
            {maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
        m->v_caches[i] = llaisys::Tensor::create(
            {maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
    }

    return m;
}

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

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return model ? &model->weights : nullptr;
}

} // extern "C"

namespace {

void forward_layer(LlaisysQwen2Model *m, size_t layer_idx,
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
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di;
    const float eps = meta->epsilon, theta = meta->theta;
    LlaisysQwen2Weights *w = &m->weights;
    tensor_t wt = get_t(w->attn_norm_w[layer_idx]);
    const float scale = 1.f / std::sqrt(static_cast<float>(dh));

    tensor_t k_cache = m->k_caches[layer_idx];
    tensor_t v_cache = m->v_caches[layer_idx];

    // Attention: norm -> q,k,v proj -> rope -> cache update -> attention -> o_proj -> residual
    llaisys::ops::rms_norm(normed, hidden, wt, eps);
    llaisys::ops::linear(q_buf, normed, get_t(w->attn_q_w[layer_idx]), get_t(w->attn_q_b[layer_idx]));
    llaisys::ops::linear(k_buf, normed, get_t(w->attn_k_w[layer_idx]), get_t(w->attn_k_b[layer_idx]));
    llaisys::ops::linear(v_buf, normed, get_t(w->attn_v_w[layer_idx]), get_t(w->attn_v_b[layer_idx]));

    std::vector<size_t> shape_q = {seq_len, nh, dh};
    std::vector<size_t> shape_kv = {seq_len, nkvh, dh};
    tensor_t q_view = q_buf->view(shape_q);
    tensor_t k_view = k_buf->view(shape_kv);
    tensor_t v_view = v_buf->view(shape_kv);

    llaisys::ops::rope(q_rope, q_view, pos_ids_t, theta);
    llaisys::ops::rope(k_rope, k_view, pos_ids_t, theta);

    // Append to KV cache
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

    size_t kv_len = cache_start + seq_len;
    tensor_t k_slice = k_cache->slice(0, 0, kv_len);
    tensor_t v_slice = v_cache->slice(0, 0, kv_len);

    llaisys::ops::self_attention(attn_val, q_rope, k_slice, v_slice, scale);

    std::vector<size_t> shape_attn_flat = {seq_len, nh * dh};
    tensor_t attn_flat = attn_val->view(shape_attn_flat);
    llaisys::ops::linear(o_proj_out, attn_flat, get_t(w->attn_o_w[layer_idx]), nullptr);

    llaisys::ops::add(res_buf, hidden, o_proj_out);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, m->device_type);

    // MLP: norm -> gate, up -> swiglu -> down -> residual
    llaisys::ops::rms_norm(normed, hidden, get_t(w->mlp_norm_w[layer_idx]), eps);
    llaisys::ops::linear(gate_buf, normed, get_t(w->mlp_gate_w[layer_idx]), nullptr);
    llaisys::ops::linear(up_buf, normed, get_t(w->mlp_up_w[layer_idx]), nullptr);
    llaisys::ops::swiglu(mlp_buf, gate_buf, up_buf);
    llaisys::ops::linear(down_buf, mlp_buf, get_t(w->mlp_down_w[layer_idx]), nullptr);
    llaisys::ops::add(res_buf, hidden, down_buf);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, m->device_type);
}

} // namespace

__C {

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model,
                                int64_t *token_ids,
                                size_t ntoken) {
    if (!model || ntoken == 0) return static_cast<int64_t>(-1);

    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di;
    const size_t voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const llaisysDeviceType_t dev = model->device_type;
    const int dev_id = model->device_id;

    const bool is_prefill = (model->cache_len == 0);
    const size_t seq_len = is_prefill ? ntoken : 1;
    const size_t cache_start = model->cache_len;

    llaisys::core::context().setDevice(dev, dev_id);

    tensor_t token_tensor = llaisys::Tensor::create(
        {seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    if (is_prefill) {
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
        forward_layer(model, i, hidden, normed, q_buf, k_buf, v_buf,
                     q_rope, k_rope, attn_val, o_proj_out, res_buf,
                     gate_buf, up_buf, mlp_buf, down_buf,
                     seq_len, cache_start, pos_ids_t);
    }

    model->cache_len = cache_start + seq_len;

    llaisys::ops::rms_norm(normed, hidden, get_t(model->weights.out_norm_w), meta->epsilon);
    tensor_t logits_t = llaisys::Tensor::create({seq_len, voc}, dtype, dev, dev_id);
    llaisys::ops::linear(logits_t, normed, get_t(model->weights.out_embed), nullptr);

    tensor_t last_logit = logits_t->slice(0, seq_len - 1, seq_len);
    tensor_t last_logit_1d = last_logit->view(std::vector<size_t>{voc});

    tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
    tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, dev, dev_id);
    llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);

    int64_t next_token = meta->end_token;
    llaisys::core::context().runtime().api()->memcpy_sync(
        &next_token, max_idx_t->data(), sizeof(int64_t),
        (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2H);

    return next_token;
}

} // extern "C"
