#pragma once

namespace flash {

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                             \
    {                                                                          \
        auto status = static_cast<cudaError_t>(error);                         \
        if (status != cudaSuccess) {                                           \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":"  \
                      << __LINE__ << std::endl;                                \
            std::exit(status);                                                 \
        }                                                                      \
    }
#endif

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__device__ __forceinline__ bool is_cta_leader() { return threadIdx.x == 0; }

inline int cuda_device_num_sms(int device) {
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    return sms;
}

inline int cuda_device_max_smem_bytes(int device) {
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    return max_smem;
}

inline int cuda_device_compute_capability(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major * 10 + prop.minor;
}

} // namespace flash
