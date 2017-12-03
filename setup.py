from setuptools import setup, Extension
import os

os_type = '_WIN32' if os.name == 'nt' else 'LINUX'

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
      ],
      maintainer='',
      maintainer_email='',
      zip_safe=False,
      packages=['ffm'],
      ext_modules = [libffm],
      include_package_data=True,
      license='BSD 3-clause',
      classifiers=['License :: OSI Approved :: BSD 3-clause'],
      url='https://github.com/keyunluo/python-libffm'
)
