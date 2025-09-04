import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "hybrid_chess_engine",
        [
            "hybrid_chess_engine.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="hybrid_chess_engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)