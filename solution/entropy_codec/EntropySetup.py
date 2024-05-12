from pathlib import Path

from setuptools import setup, Extension
from sysconfig import get_paths
import os
import pybind11

python_paths = get_paths()
current_dir = Path(__file__).parent
source_file = str(current_dir / 'wrapper.cpp')

functions_module = Extension(
    name='EntropyCodec',
    sources=[source_file],
    include_dirs=[
        python_paths.get("include", None),
        os.path.join(pybind11.__path__[0], 'include')]
)

setup(ext_modules=[functions_module], options={"build_ext": {"build_lib": ".."}}, dist_dir=current_dir)
