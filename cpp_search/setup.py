import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "cpp_search",
        [
            "search_engine.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="cpp_search",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
