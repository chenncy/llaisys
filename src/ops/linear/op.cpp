#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace {

// Y = X W^T + b; out (B,M), in (B,K), weight (M,K), bias (M,) or null
template <typename T>
void linear_impl(T *out, const T *in, const T *weight, const T *bias, size_t B, size_t M, size_t K) {
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < M; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float sum_f = 0;
                for (size_t k = 0; k < K; k++)
                    sum_f += llaisys::utils::cast<float>(in[i * K + k]) * llaisys::utils::cast<float>(weight[j * K + k]);
                if (bias) sum_f += llaisys::utils::cast<float>(bias[j]);
                out[i * M + j] = llaisys::utils::cast<T>(sum_f);
            } else {
                T sum = T(0);
                for (size_t k = 0; k < K; k++)
                    sum += in[i * K + k] * weight[j * K + k];
                if (bias) sum += bias[j];
                out[i * M + j] = sum;
            }
        }
    }
}

void linear_cpu(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
               llaisysDataType_t dtype, size_t B, size_t M, size_t K) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_impl(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                    reinterpret_cast<const float *>(weight), bias ? reinterpret_cast<const float *>(bias) : nullptr, B, M, K);
        return;
    case LLAISYS_DTYPE_F16:
        linear_impl(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr, B, M, K);
        return;
    case LLAISYS_DTYPE_BF16:
        linear_impl(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr, B, M, K);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);
    ASSERT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(), "linear: out, in, weight must have same dtype");
    if (bias) ASSERT(out->dtype() == bias->dtype(), "linear: bias dtype must match");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "linear: out, in, weight must be contiguous");
    if (bias) ASSERT(bias->isContiguous() && bias->ndim() == 1, "linear: bias must be 1D contiguous");
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "linear: out, in, weight must be 2D");
    size_t B = in->shape()[0], K = in->shape()[1];
    size_t M = weight->shape()[0];
    ASSERT(weight->shape()[1] == K, "linear: weight second dim must match in second dim");
    ASSERT(out->shape()[0] == B && out->shape()[1] == M, "linear: out shape must be (B, M)");
    if (bias) ASSERT(bias->numel() == M, "linear: bias size must equal M");

    std::byte *bias_ptr = bias ? bias->data() : nullptr;

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return linear_cpu(out->data(), in->data(), weight->data(), bias_ptr, out->dtype(), B, M, K);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
