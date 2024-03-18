from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='chamfer',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
