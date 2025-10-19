import importlib

# 定义需要检查的包和预期版本
packages = {
    'mamba_ssm': '2.2.5',
    'causal_conv1d': '1.5.2',
    'requests': '2.32.4',
    'datasets': '4.0.0',
    'fsspec': '2025.3.0',
    'numpy': '2.0.2',
    'gcsfs': '2025.3.0',
    'numba': '0.60.0'
}

# 检查每个包的版本
for pkg, expected_version in packages.items():
    try:
        # 动态导入包，处理带-的包名
        module = importlib.import_module(pkg.replace('-', '_'))
        # 获取实际版本
        actual_version = getattr(module, '__version__', 'Version attribute not found')
        # 断言版本匹配
        assert actual_version == expected_version, f"{pkg} version mismatch: expected {expected_version}, got {actual_version}"
        print(f"{pkg}: {actual_version} (matches expected version)")
    except ImportError:
        raise AssertionError(f"{pkg}: Not installed")
    except AssertionError as e:
        raise AssertionError(str(e))
    except Exception as e:
        raise AssertionError(f"{pkg}: Error checking version - {str(e)}")

print("All package versions match expected values!")
