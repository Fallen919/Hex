from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys


def platform_compile_args():
    if sys.platform.startswith("win"):
        return ["/O2"]
    return ["-O3"]


ext_modules = [
    Pybind11Extension(
        "hex_cpp",
        [
            "cpp_module/bindings.cpp",
            "cpp_module/gamestate.cpp",
            "cpp_module/unionfind.cpp",
        ],
        include_dirs=["cpp_module"],
        cxx_std=17,
        extra_compile_args=platform_compile_args(),
    )
]


setup(
    name="hex_cpp",
    version="0.1.0",
    description="Hex C++ acceleration module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
