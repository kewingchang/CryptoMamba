# 完整的包版本检测代码
packages = [
    'ipython', 'jedi', 'google_colab', 'requests', 'datasets', 'fsspec',
    'opencv_python', 'numpy', 'opencv_contrib_python', 'tensorflow',
    'gcsfs', 'cupy_cuda12x', 'numba', 'opencv_python_headless'
]

for pkg in packages:
    try:
        module = __import__(pkg.replace('-', '_').replace('.', '_'))  # 处理包名中的-和.，转换为import友好格式
        version = module.__version__ if hasattr(module, '__version__') else "Version attribute not found"
        print(f"{pkg}: {version}")
    except ImportError:
        print(f"{pkg}: Not installed")
    except Exception as e:
        print(f"{pkg}: Error checking version - {str(e)}")