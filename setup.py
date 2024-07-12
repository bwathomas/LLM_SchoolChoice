from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Function to get environment variable or default to None
def get_env_var(var_name):
    return os.environ.get(var_name, None)

# Include directories
include_dirs = [
    get_env_var('CURL_INCLUDE_DIR'),
    get_env_var('MSVC_INCLUDE_DIR'),
    get_env_var('UCRT_INCLUDE_DIR'),
    get_env_var('UM_INCLUDE_DIR'),
    get_env_var('SHARED_INCLUDE_DIR'),
    get_env_var('PYTHON_INCLUDE_DIR'),
    get_env_var('PYTHON_INCLUDE_DIR2')
]

# Library directories
library_dirs = [
    get_env_var('CURL_LIB_DIR'),
    get_env_var('MSVC_LIB_DIR'),
    get_env_var('UCRT_LIB_DIR'),
    get_env_var('UM_LIB_DIR'),
    get_env_var('PYTHON_LIB_DIR'),
    get_env_var('PYTHON_LIB_DIR2')
]

# Filter out None values
include_dirs = [dir for dir in include_dirs if dir is not None]
library_dirs = [dir for dir in library_dirs if dir is not None]

extensions = [
    Extension(
        "chatgpt_wrapper",
        sources=["chatgpt_wrapper.pyx", "chatgpt_query.c"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["curl"],  # Use 'curl' for linking
    )
]

setup(
    name="chatgpt_batch_query",
    version="1.0.0",
    description="A batch query processor for ChatGPT using Cython and C for performance optimization.",
    author="Your Name",
    author_email="your.email@example.com",
    ext_modules=cythonize(extensions),
    install_requires=[
        "pandas==1.3.3",
        "xlsxwriter==3.0.2",
        "cython==0.29.24",
        "numpy==1.21.2",
        "openpyxl==3.0.9",
        "configparser==5.0.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
