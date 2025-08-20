#include <torch/python.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <utility>
#include <vector>
#include "cuda_utils.cuh"
#include "flash_attention.cuh"
#include "flash_kernels.cuh"

using namespace flash;

FlashForwardKernelConfig py_to_cpp_kernel_config(const py::object &py_cfg) {
    return FlashForwardKernelConfig{
        py::cast<torch::ScalarType>(
            py_cfg.attr("dtype").attr("to_torch_dtype")()),
        py::cast<int>(py_cfg.attr("d_head")),
        py::cast<int>(py_cfg.attr("B_r")),
        py::cast<int>(py_cfg.attr("B_c")),
        py::cast<int>(py_cfg.attr("n_warps")),
        py::cast<bool>(py_cfg.attr("async_copy")),
        py::cast<bool>(py_cfg.attr("eager_load_blocks")),
        py::cast<bool>(py_cfg.attr("swizzled")),
        py::cast<int>(py_cfg.attr("Q_mma_load_K_tiles")),
        py::cast<int>(py_cfg.attr("K_mma_load_K_tiles")),
        py::cast<int>(py_cfg.attr("V_mma_load_K_tiles")),
        py::cast<bool>(py_cfg.attr("mma_double_buffer_loads")),
        py::cast<bool>(py_cfg.attr("optimized_softmax"))};
}

decltype(auto)
flash_attention_forward(const py::object &py_cfg, const torch::Tensor &TQ,
                        const torch::Tensor &TK, const torch::Tensor &TV,
                        std::optional<at::Tensor> &out_, bool benchmark) {
    CHECK_INPUT(TQ);
    CHECK_INPUT(TK);
    CHECK_INPUT(TV);

    at::cuda::CUDAGuard device_guard{TQ.device()};
    const int compute_capability =
        cuda_device_compute_capability(TQ.device().index());
    TORCH_CHECK(compute_capability >= 80,
                "Flash Attention requires SM_80 or higher (current: SM_",
                compute_capability / 10, ".", compute_capability % 10, ")");

    // Check data types
    const auto Q_dtype = TQ.dtype();
    TORCH_CHECK(Q_dtype == torch::kFloat16 || Q_dtype == torch::kBFloat16,
                "Only fp16 and bf16 are supported");
    TORCH_CHECK(TK.dtype() == Q_dtype,
                "Input tensors must have the same data type");
    TORCH_CHECK(TV.dtype() == Q_dtype,
                "Input tensors must have the same data type");

    const auto d_head = TQ.size(3);
    const FlashForwardKernelConfig cfg{py_to_cpp_kernel_config(py_cfg)};
    TORCH_CHECK(forward_kernels.contains(cfg),
                "Kernel configuration was not found in flash_kernels.cuh");
    const auto kernel = forward_kernels[cfg];

    TORCH_CHECK(cfg.dtype == Q_dtype,
                "Kernel configuration dtype does not match input dtype");

    const auto batch_size = TQ.size(0);
    const auto seq_len = TQ.size(1);
    const auto n_heads = TQ.size(2);

    // Only supported configuration currently.
    TORCH_CHECK(TQ.sizes() == TK.sizes(),
                "Query and key tensors have same shape");
    TORCH_CHECK(TQ.sizes() == TV.sizes(),
                "Query and value tensors have same shape");

    const int B_r = cfg.B_r;
    const int B_c = cfg.B_c;
    TORCH_CHECK(seq_len % B_r == 0,
                "Only multiples of B_r are supported for seq_len Q currently");
    TORCH_CHECK(seq_len % B_c == 0,
                "Only multiples of B_c are supported for seq_len K currently");

    const auto batch_stride = TQ.stride(0);
    const auto seq_stride = TQ.stride(1);
    const auto head_stride = TQ.stride(2);

    torch::Tensor TO;
    if (out_.has_value()) {
        TO = out_.value();
        TORCH_CHECK(TO.dtype() == Q_dtype,
                    "Output tensor must have the same dtype as inputs");

        TORCH_CHECK(TQ.sizes() == TV.sizes(),
                    "Query and output tensors have same shape");
    } else {
        TO = torch::empty_like(TQ);
    }

    const int n_Q_blocks = CEIL_DIV(seq_len, B_r);
    const int n_KV_blocks = CEIL_DIV(seq_len, B_c);
    const int n_threads = cfg.n_warps * WARP_SIZE;

    ForwardKernelArgs args{TQ.data_ptr(), TK.data_ptr(), TV.data_ptr(),
                           TO.data_ptr(), batch_stride,  seq_stride,
                           head_stride,   seq_len,       n_heads,
                           n_Q_blocks,    n_KV_blocks};

    dim3 blockDim(n_threads);
    dim3 gridDim{static_cast<uint>(n_Q_blocks), static_cast<uint>(n_heads),
                 static_cast<uint>(batch_size)};

    float runtime;
    cudaEvent_t start, stop;

    const int smem_bytes = cfg.smem_bytes();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
    }

    kernel<<<gridDim, blockDim, smem_bytes, stream>>>(args);
    if (benchmark) {
        cudaEventRecord(stop, stream);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runtime, start, stop);
    }

    return std::make_tuple(TO, runtime);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, py::arg("kernel_cfg"),
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"),
          py::arg("benchmark") = false, "Flash Attention forward (CUDA)");

    // Set kernel max dynamic smem on module initialization.
    for (const auto &[cfg, kernel] : forward_kernels) {
        int smem_used = cfg.smem_bytes();
        if (smem_used > 48 * 1024) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_used);
        }
    }
}