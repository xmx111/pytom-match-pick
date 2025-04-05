# This file is named testzz_* as it should run last,
# because it permanently destroys the imports
# 此文件命名为 testzz_* 是因为它应该最后运行，
# 因为它会永久破坏导入操作
# No imports of pytom_tm outside of the methods
# 除了方法内部，不导入 pytom_tm

# 导入 unittest 模块，用于编写单元测试
import unittest
# 从 importlib 模块导入 reload 函数，用于重新加载模块
from importlib import reload

# Mock out installed dependencies
# 模拟已安装的依赖项
# 保存原始的 __import__ 函数
orig_import = __import__

# skip tests if optional stuff is not installed
# 如果可选的组件未安装，则跳过测试
# 初始化 SKIP_PLOT 标志为 False
SKIP_PLOT = False
try:
    # 尝试导入 pytom_tm.plotting 模块
    import pytom_tm.plotting  # noqa: F401
except RuntimeError:
    # 如果导入时出现 RuntimeError 异常，将 SKIP_PLOT 标志设置为 True
    SKIP_PLOT = True

# 定义一个函数，用于模拟模块未找到的情况
def module_not_found_mock(missing_name):
    """
    创建一个模拟导入函数，当尝试导入指定名称的模块时，抛出 ModuleNotFoundError。

    参数:
    missing_name (str): 要模拟找不到的模块名称。

    返回:
    function: 模拟导入函数。
    """
    def import_mock(name, *args):
        """
        模拟导入函数，当尝试导入指定名称的模块时，抛出 ModuleNotFoundError。

        参数:
        name (str): 要导入的模块名称。
        *args: 其他参数。

        返回:
        module: 如果不是指定的模块名称，调用原始的 __import__ 函数进行导入。
        """
        if name == missing_name:
            # 如果要导入的模块名称与指定的缺失模块名称相同，抛出 ModuleNotFoundError 异常
            raise ModuleNotFoundError(f"No module named '{name}'")
        # 否则，调用原始的 __import__ 函数进行导入
        return orig_import(name, *args)

    # 返回模拟导入函数
    return import_mock

# 定义一个函数，用于模拟 cupy 导入错误的情况
def cupy_import_error_mock(name, *args):
    """
    创建一个模拟导入函数，当尝试导入 'cupy' 模块时，抛出 ImportError。

    参数:
    name (str): 要导入的模块名称。
    *args: 其他参数。

    返回:
    module: 如果不是 'cupy' 模块，调用原始的 __import__ 函数进行导入。
    """
    if name == "cupy":
        # 如果要导入的模块名称是 'cupy'，抛出 ImportError 异常
        raise ImportError("Failed to import cupy")
    # 否则，调用原始的 __import__ 函数进行导入
    return orig_import(name, *args)

# 定义一个测试类，继承自 unittest.TestCase
class TestMissingDependencies(unittest.TestCase):
    """
    测试在缺少依赖项时 pytom_tm 模块的导入情况。
    """
    def test_missing_cupy(self):
        """
        测试在缺少 cupy 模块时 pytom_tm 的导入情况。
        验证当 cupy 模块不可用时，是否会记录警告信息。
        """
        # assert working import
        # 断言导入操作正常，且不产生警告日志
        with self.assertNoLogs(level="WARNING"):
            # 导入 pytom_tm 模块
            import pytom_tm
        # 创建一个模拟导入函数，用于模拟 cupy 模块未找到的情况
        cupy_not_found = module_not_found_mock("cupy")
        # Test missing cupy
        # 测试缺少 cupy 模块的情况
        with unittest.mock.patch("builtins.__import__", side_effect=cupy_not_found):
            # 捕获警告日志
            with self.assertLogs(level="WARNING") as cm:
                # 重新加载 pytom_tm 模块
                reload(pytom_tm)
            # 断言警告日志的数量为 1
            self.assertEqual(len(cm.output), 1)
            # 断言警告日志中包含指定的信息
            self.assertIn("cupy installation not found or not functional", cm.output[0])

    def test_broken_cupy(self):
        """
        测试在 cupy 模块导入失败时 pytom_tm 的导入情况。
        验证当 cupy 模块导入失败时，是否会记录警告信息。
        """
        # assert working import
        # 断言导入操作正常，且不产生警告日志
        with self.assertNoLogs(level="WARNING"):
            # 导入 pytom_tm 模块
            import pytom_tm
        # Test cupy ImportError
        # 测试 cupy 模块导入错误的情况
        with unittest.mock.patch(
            "builtins.__import__", side_effect=cupy_import_error_mock
        ):
            # 捕获警告日志
            with self.assertLogs(level="WARNING") as cm:
                # 重新加载 pytom_tm 模块
                reload(pytom_tm)
            # 断言警告日志的数量为 1
            self.assertEqual(len(cm.output), 1)
            # 断言警告日志中包含指定的信息
            self.assertIn("cupy installation not found or not functional", cm.output[0])

    # 如果 SKIP_PLOT 为 True，跳过此测试
    @unittest.skipIf(SKIP_PLOT, "plotting module not installed")
    def test_missing_matplotlib(self):
        """
        测试在缺少 matplotlib 模块时 pytom_tm 的导入情况。
        验证当 matplotlib 模块不可用时，是否会抛出 ModuleNotFoundError，
        并且 pytom_tm 的绘图功能是否被禁用。
        """
        # assert working import
        # 导入 pytom_tm 模块
        import pytom_tm

        # 创建一个模拟导入函数，用于模拟 matplotlib.pyplot 模块未找到的情况
        matplotlib_not_found = module_not_found_mock("matplotlib.pyplot")
        with unittest.mock.patch(
            "builtins.__import__", side_effect=matplotlib_not_found
        ):
            # 断言会抛出 ModuleNotFoundError 异常，且异常信息中包含 'matplotlib'
            with self.assertRaisesRegex(ModuleNotFoundError, "matplotlib"):
                # 仅 pyplot 是直接导入的，所以应该测试这个
                import matplotlib.pyplot as plt  # noqa: F401
            # force reload
            # 强制重新加载 pytom_tm 模块
            # check if we can still import pytom_tm
            # 检查是否仍然可以导入 pytom_tm 模块
            reload(pytom_tm)

            # check if plotting is indeed disabled after reload
            # 检查重新加载后绘图功能是否确实被禁用
            # (reload is needed to prevent python import caching)
            # （需要重新加载以防止 Python 导入缓存）
            self.assertFalse(reload(pytom_tm.extract).plotting_available)
            # assert that importing the plotting module fails completely
            # 断言导入绘图模块会完全失败
            with self.assertRaisesRegex(RuntimeError, "matplotlib and seaborn"):
                # 重新加载 pytom_tm.plotting 模块
                reload(pytom_tm.plotting)

    # 如果 SKIP_PLOT 为 True，跳过此测试
    @unittest.skipIf(SKIP_PLOT, "plotting module not installed")
    def test_missing_seaborn(self):
        """
        测试在缺少 seaborn 模块时 pytom_tm 的导入情况。
        验证当 seaborn 模块不可用时，是否会抛出 ModuleNotFoundError，
        并且 pytom_tm 的绘图功能是否被禁用。
        """
        # assert working import
        # 导入 pytom_tm 模块
        import pytom_tm

        # 创建一个模拟导入函数，用于模拟 seaborn 模块未找到的情况
        seaborn_not_found = module_not_found_mock("seaborn")
        with unittest.mock.patch("builtins.__import__", side_effect=seaborn_not_found):
            # 断言会抛出 ModuleNotFoundError 异常，且异常信息中包含 'seaborn'
            with self.assertRaisesRegex(ModuleNotFoundError, "seaborn"):
                # 导入 seaborn 模块
                import seaborn  # noqa: F401
            # check if we can still import pytom_tm
            # 检查是否仍然可以导入 pytom_tm 模块
            reload(pytom_tm)
            # check if plotting is indeed disabled
            # 检查绘图功能是否确实被禁用
            # (reload is needed to prevent python import caching)
            # （需要重新加载以防止 Python 导入缓存）
            self.assertFalse(reload(pytom_tm.extract).plotting_available)
            # assert that importing the plotting module fails completely
            # 断言导入绘图模块会完全失败
            with self.assertRaisesRegex(RuntimeError, "matplotlib and seaborn"):
                # 重新加载 pytom_tm.plotting 模块
                reload(pytom_tm.plotting)
