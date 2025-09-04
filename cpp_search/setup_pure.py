import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# ONNX Runtime paths
onnx_include_path = "/opt/homebrew/Cellar/onnxruntime/1.22.2_2/include/onnxruntime"
onnx_lib_path = "/opt/homebrew/Cellar/onnxruntime/1.22.2_2/lib"

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "pure_cpp_chess_engine",
        [
            "pure_cpp_chess_engine.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            onnx_include_path,
        ],
        libraries=["onnxruntime"],
        library_dirs=[onnx_lib_path],
        language="c++",
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
        # Add runtime library path
        extra_link_args=[f"-Wl,-rpath,{onnx_lib_path}"],
    ),
]

setup(
    name="pure_cpp_chess_engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
