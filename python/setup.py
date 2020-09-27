from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
import glob
import torch


this_dir = os.path.dirname(os.path.abspath(__file__))
extension_dir = os.path.join(this_dir, 'src')
sources = glob.glob(os.path.join(extension_dir, "*.cpp"))

setup(
    name='boa',
    version='0.1',
    author='zhuxiaolong',
    ext_modules=[
        CppExtension(
            'boa._C',
            sources,
            include_dirs=['..'],
            library_dirs=['../build'],
            libraries=['boa']
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)