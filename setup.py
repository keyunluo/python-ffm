from setuptools import setup, Extension
import sys

sys.dont_write_bytecode = True
os_type = '_WIN32' if sys.platform.startswith('win32') else 'LINUX'

libffm = Extension('ffm.libffm',
                   extra_compile_args = ["-Wall", "-O3",  "-std=c++11", "-march=native", "-DUSESSE", "-DDEBUG=0", "-D%s" % os_type], 
                   include_dirs = ['libffm'], 
                   sources = ['ffm/ffm-wrapper.cpp', 'libffm/timer.cpp'],
                   language='c++',)


# Please use setup_pip.py for generating and deploying pip installation
# detailed instruction in setup_pip.py
setup(name='ffm',
      version='1.0',
      description="LibFFM Python Package",
      long_description="LibFFM Python Package",
      install_requires=[
          'numpy',
          'scikit-learn',
          'tqdm',
          'pandas'
      ],
      maintainer='keyunluo',
      maintainer_email='streamer.ky@foxmail.com',
      zip_safe=False,
      packages=['ffm'],
      ext_modules = [libffm],
      include_package_data=True,
      exclude_package_data = { '': ['*.pyc', '*.cpp'] },
      license='BSD 3-clause',
      classifiers=['License :: OSI Approved :: BSD 3-clause'],
      url='https://github.com/keyunluo/python-libffm'
)
