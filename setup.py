import glob
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))
CSRC = "csrc"

# every kernel file is picked up automatically; bindings.cpp holds the Python-facing wrappers
sources = [os.path.join(CSRC, "bindings.cpp")] + sorted(glob.glob(os.path.join(CSRC, "*.cu")))

# no -std flag: torch's build helper picks the C++ standard its headers need
if sys.platform == "win32":
    # CUDA 13's headers refuse MSVC's legacy preprocessor; nvcc forwards host flags with -Xcompiler
    cxx_flags = ["/O2", "/Zc:preprocessor"]
    nvcc_flags = ["-O3", "-Xcompiler", "/Zc:preprocessor"]
else:
    cxx_flags = ["-O3"]
    nvcc_flags = ["-O3"]

setup(
    name="cuda_cnn",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="cuda_cnn",
            sources=sources,
            include_dirs=[os.path.join(ROOT, CSRC, "include")],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
