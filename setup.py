from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

torch.utils.cpp_extension.COMMON_NVCC_FLAGS = ["--expt-relaxed-constexpr", "-lcublas"]

setup(
    name="quadmul",
    ext_modules=[
        CUDAExtension(
            "quadmul",
            ["extension.cpp", "kernel.cu", "configs.cu", "helpers.cu"],
        ),
    ],
    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lcublas"]},
    cmdclass={"build_ext": BuildExtension},
)
