import os
import re
import shutil
import tarfile
import tempfile

from setuptools import find_namespace_packages, setup
from urllib.request import urlretrieve

try:
    import torch
    import torch.version

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    torch_version = torch.version.__version__.split(".")[:2]
    cuda_version = torch.version.cuda

    # This will be e.g. "+pt23cu121"
    assert cuda_version is not None, "Pytorch CUDA is required for this installation."
    version_suffix = f"+pt{torch_version[0]}{torch_version[1]}cu{cuda_version.replace('.', '')}"

except ImportError:
    raise ValueError("Pytorch not found, please install it first.")

PACKAGE_NAME = "vipe"
DEFAULT_CUDA_ARCHES = ["7.5", "8.0", "8.6", "8.7", "9.0+PTX"]

# Avoid directly importing the package
with open(f"{PACKAGE_NAME}/__init__.py", "r") as fh:
    __version__ = re.findall(r"__version__ = \"(.*?)\"", fh.read())[0]
__version__ += version_suffix

coder_finder_path = f"{PACKAGE_NAME}/ext/specs.py"
code_finder_namespace = {"__file__": coder_finder_path}
with open(coder_finder_path, "r") as fh:
    exec(fh.read(), code_finder_namespace)
get_sources = code_finder_namespace["get_sources"]
get_cpp_flags = code_finder_namespace["get_cpp_flags"]
get_cuda_flags = code_finder_namespace["get_cuda_flags"]


def _normalize_arch_token(token: str) -> str:
    return token.strip().replace(" ", "")


def _parse_arch_list(raw_arch_list: str | None) -> dict[str, bool]:
    arch_map: dict[str, bool] = {}
    if not raw_arch_list:
        return arch_map

    for raw_token in raw_arch_list.replace(",", ";").split(";"):
        token = _normalize_arch_token(raw_token)
        if not token:
            continue
        has_ptx = token.endswith("+PTX")
        arch = token.removesuffix("+PTX")
        arch_map[arch] = arch_map.get(arch, False) or has_ptx
    return arch_map


def _detect_local_cuda_arches() -> dict[str, bool]:
    arch_map: dict[str, bool] = {}
    if not torch.cuda.is_available():
        return arch_map

    for device_idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_idx)
        arch = f"{props.major}.{props.minor}"
        arch_map[arch] = arch_map.get(arch, False)
    return arch_map


def _format_arch_list(arch_map: dict[str, bool]) -> str:
    def arch_sort_key(item: tuple[str, bool]) -> tuple[int, int]:
        major, minor = item[0].split(".")
        return int(major), int(minor)

    tokens = []
    for arch, has_ptx in sorted(arch_map.items(), key=arch_sort_key):
        tokens.append(f"{arch}+PTX" if has_ptx else arch)
    return ";".join(tokens)


def _resolve_cuda_arch_list() -> str:
    arch_map = _parse_arch_list(";".join(DEFAULT_CUDA_ARCHES))

    for source_arches in (
        _parse_arch_list(os.environ.get("TORCH_CUDA_ARCH_LIST")),
        _detect_local_cuda_arches(),
    ):
        for arch, has_ptx in source_arches.items():
            arch_map[arch] = arch_map.get(arch, False) or has_ptx

    return _format_arch_list(arch_map)

# Setup CUDA_HOME for conda environment for consistency
if "CONDA_PREFIX" in os.environ:
    conda_nvcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc")
    if os.path.exists(conda_nvcc_path):
        os.environ["PYTORCH_NVCC"] = conda_nvcc_path

resolved_cuda_arch_list = _resolve_cuda_arch_list()
os.environ["TORCH_CUDA_ARCH_LIST"] = resolved_cuda_arch_list
print(f"Building {PACKAGE_NAME} CUDA extension for archs: {resolved_cuda_arch_list}")

# Download the put Eigen 3.4 in a correct place
cpp_flags = get_cpp_flags()
cuda_flags = get_cuda_flags()
if os.environ.get("USE_SYSTEM_EIGEN", "0") == "0":
    eigen_include_dir = "csrc/include/eigen3"
    eigen_url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"

    if not os.path.exists(eigen_include_dir):
        os.makedirs(eigen_include_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_tar_path = os.path.join(temp_dir, "eigen.gz")
            extracted_dir = os.path.join(temp_dir, "eigen-extracted")
            urlretrieve(eigen_url, tmp_tar_path)
            with tarfile.open(tmp_tar_path, "r:gz") as tar:
                tar.extractall(path=extracted_dir)

            shutil.move(os.path.join(extracted_dir, "eigen-3.4.0", "Eigen"), eigen_include_dir)

    # Use full path
    additional_include_path = os.path.join(os.path.dirname(__file__), "csrc/include")
    cpp_flags += ["-isystem", additional_include_path]
    cuda_flags += ["-isystem", additional_include_path]

packages = find_namespace_packages(include=["vipe", "vipe.*"])
setup(
    packages=packages,
    version=__version__,
    ext_modules=[
        CUDAExtension(
            f"{PACKAGE_NAME}_ext",
            sources=get_sources(),  # type: ignore
            extra_compile_args={"cxx": cpp_flags, "nvcc": cuda_flags},  # type: ignore
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
