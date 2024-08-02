#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void wrapper(void *A, void *B, void *C, const int m, const int n, const int k, cudaStream_t stream);

void gemm(torch::Tensor a, torch::Tensor b, torch::Tensor c, int m, int n, int k)
{

    // TORCH_CHECK(a.device() == b.device() && a.device() == c.device(), "All tensors must be on the same device");
    // // TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "A and B tensors must be of dtype int8");
    // TORCH_CHECK(c.dtype() == torch::kInt32, "C tensor must be of dtype int32");
    // TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "All tensors must be 2D");
    // TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "All tensors must be contiguous");

    wrapper(a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            m,
            n,
            k,
            at::cuda::getCurrentCUDAStream(a.get_device()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm", &gemm, "Int4xInt4 Matrix Multiplication Kernel");
}