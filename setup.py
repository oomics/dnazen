import sys
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import tomllib

from os import path

if "--dev" in sys.argv:
    dev_build = True
    sys.argv.remove("--dev")
else:
    dev_build = False

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
    __version__ = pyproject["project"]["version"]
    name = pyproject["project"]["name"]
    python_requires = pyproject["project"]["requires-python"]


# --- cpp extension ---
CSRC_SEARCH_PATH = path.join("src", "_ngram", "pybind.cpp")

CPP_MACROS = [
    ("VERSION_INFO", __version__),
]

if dev_build:
    CPP_MACROS.append(("__DEBUG__", 1))

ext = Pybind11Extension(
    "_ngram",
    sorted(glob(CSRC_SEARCH_PATH)),
    define_macros=CPP_MACROS,
)
if dev_build:
    ext._add_cflags(["-std=c++17", "-O0", "-g"])
else:
    ext._add_cflags(["-std=c++17", "-O3"])

ext_modules = [
    ext,
]

# --- setup ---
setup(
    name=name,
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=python_requires,
)
