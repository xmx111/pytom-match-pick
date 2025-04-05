# 导入unittest模块，用于编写单元测试
import unittest
# 导入pathlib模块，用于处理文件路径
import pathlib
# 导入numpy库，用于数值计算
import numpy as np
# 导入cupy库，用于GPU加速计算
import cupy as cp
# 导入logging模块，用于记录日志
import logging
# 从shutil模块导入which函数，用于查找可执行文件的路径
from shutil import which
# 从contextlib模块导入redirect_stdout和redirect_stderr，用于重定向标准输出和标准错误输出
from contextlib import redirect_stdout, redirect_stderr
# 从io模块导入StringIO，用于在内存中操作字符串
from io import StringIO
# 从pytom_tm模块导入entry_points，包含要测试的入口点函数
from pytom_tm import entry_points
# 从pytom_tm模块导入io，用于文件输入输出操作
from pytom_tm import io

# (命令行函数, entry_points文件中的函数)
# 定义要测试的入口点列表，包含命令行函数名和对应的Python函数名
ENTRY_POINTS_TO_TEST = [
    ("pytom_create_mask.py", "pytom_create_mask"),
    ("pytom_create_template.py", "pytom_create_template"),
    ("pytom_match_template.py", "match_template"),
    ("pytom_extract_candidates.py", "extract_candidates"),
    ("pytom_merge_stars.py", "merge_stars"),
]
# 测试可选依赖是否安装
try:
    # 尝试导入pytom_tm中的plotting模块
    from pytom_tm import plotting  # noqa: F401
except RuntimeError:
    # 若导入失败，忽略该异常
    pass
else:
    # 若导入成功，将pytom_estimate_roc.py及其对应的函数添加到测试列表中
    ENTRY_POINTS_TO_TEST.append(("pytom_estimate_roc.py", "estimate_roc"))

# 命令行match_template的输入文件
# 获取当前文件所在目录，并拼接test_data目录路径
TEST_DATA = pathlib.Path(__file__).parent.joinpath("test_data")
# 拼接模板文件路径
TEMPLATE = TEST_DATA.joinpath("template.mrc")
# 拼接掩码文件路径
MASK = TEST_DATA.joinpath("mask.mrc")
# 拼接断层图像文件路径
TOMOGRAM = TEST_DATA.joinpath("tomogram.mrc")
# 拼接输出目录路径
DESTINATION = TEST_DATA.joinpath("output")
# 拼接倾斜角度文件路径
TILT_ANGLES = TEST_DATA.joinpath("angles.rawtlt")
# 拼接剂量文件路径
DOSE = TEST_DATA.joinpath("test_dose.txt")
# 拼接散焦文件路径
DEFOCUS = TEST_DATA.joinpath("defocus.txt")
# 拼接IMOD散焦文件路径
DEFOCUS_IMOD = (
    pathlib.Path(__file__).parent.joinpath("Data").joinpath("test_imod.defocus")
)
# 拼接RELION5断层图像星文件路径
RELION5_TOMOGRAMS_STAR = pathlib.Path(__file__).parent.joinpath(
    "Data/relion5_project_example/Tomograms/job009/tomograms.star"
)
# 拼接RELION5断层图像文件路径
RELION5_TOMOGRAM = TEST_DATA.joinpath("rec_tomo200528_107.mrc")

# 初始日志级别
# 获取当前日志记录器的日志级别
LOG_LEVEL = logging.getLogger().level

# 此函数用于将参数字典转换为命令行参数列表
def prep_argv(arg_dict):
    """
    将参数字典转换为命令行参数列表。

    参数:
    arg_dict (dict): 参数字典，键为参数名，值为参数值。

    返回:
    list: 命令行参数列表。
    """
    argv = []
    # 遍历参数字典
    [
        # 如果参数值不为空，将参数名和参数值拆分为多个元素添加到argv列表中
        argv.extend([k] + v.split()) if v != "" else argv.append(k)
        for k, v in arg_dict.items()
    ]
    return argv

# 定义测试入口点的测试类，继承自unittest.TestCase
class TestEntryPoints(unittest.TestCase):
    # 类方法，在测试类的所有测试方法执行前运行
    @classmethod
    def setUpClass(cls) -> None:
        """
        在所有测试方法执行前设置测试环境。
        创建必要的文件和目录。
        """
        # 创建输出目录，若父目录不存在则一并创建
        DESTINATION.mkdir(parents=True)
        # 写入模板文件，内容为全零的5x5x5数组
        io.write_mrc(TEMPLATE, np.zeros((5, 5, 5), dtype=np.float32), 1)
        # 写入掩码文件，内容为全零的5x5x5数组
        io.write_mrc(MASK, np.zeros((5, 5, 5), dtype=np.float32), 1)
        # 写入断层图像文件，内容为全零的10x10x10数组
        io.write_mrc(TOMOGRAM, np.zeros((10, 10, 10), dtype=np.float32), 1)
        # 写入RELION5断层图像文件，内容为全零的10x10x10数组
        io.write_mrc(RELION5_TOMOGRAM, np.zeros((10, 10, 10), dtype=np.float32), 1)
        # 写入倾斜角度文件，内容为从-50到50的35个均匀分布的值
        np.savetxt(TILT_ANGLES, np.linspace(-50, 50, 35))
        # 写入剂量文件，内容为从0到100的35个均匀分布的值
        np.savetxt(DOSE, np.linspace(0, 100, 35))
        # 写入散焦文件，内容为35个值为3000的数组
        np.savetxt(DEFOCUS, np.ones(35) * 3000)

    # 类方法，在测试类的所有测试方法执行后运行
    @classmethod
    def tearDownClass(cls) -> None:
        """
        在所有测试方法执行后清理测试环境。
        删除创建的文件和目录。
        """
        # 删除模板文件
        TEMPLATE.unlink()
        # 删除掩码文件
        MASK.unlink()
        # 删除断层图像文件
        TOMOGRAM.unlink()
        # 删除RELION5断层图像文件
        RELION5_TOMOGRAM.unlink()
        # 删除倾斜角度文件
        TILT_ANGLES.unlink()
        # 删除剂量文件
        DOSE.unlink()
        # 删除散焦文件
        DEFOCUS.unlink()
        # 遍历输出目录下的所有文件并删除
        for f in DESTINATION.iterdir():
            f.unlink()  # 应该测试特定输出吗？
        # 删除输出目录
        DESTINATION.rmdir()
        # 删除测试数据目录
        TEST_DATA.rmdir()

    # 测试入口点是否存在
    def test_entry_points_exist(self):
        """
        测试命令行入口点是否存在，并且可以使用 -h 选项正常退出。
        """
        # 遍历要测试的入口点列表
        for cli, fname in ENTRY_POINTS_TO_TEST:
            # 测试命令行函数是否可以找到
            self.assertIsNotNone(which(cli))
            # 断言入口点可以使用 -h 选项调用并干净退出
            # 捕获标准输出以防止污染shell
            # 从entry_points模块中获取对应的函数
            func = getattr(entry_points, fname)
            # 创建一个StringIO对象，用于捕获标准输出
            dump = StringIO()
            # 使用with语句捕获SystemExit异常
            with self.assertRaises(SystemExit) as ex, redirect_stdout(dump):
                # 调用函数并传入命令行参数
                func([cli, "-h"])
            # 关闭StringIO对象
            dump.close()
            # 检查系统返回码是否为0（成功）
            self.assertEqual(ex.exception.code, 0)

    # 测试模板匹配功能
    def test_match_template(self):
        """
        测试模板匹配功能，包括不同参数组合和错误处理。
        """
        # 定义默认参数
        defaults = {
            "-t": str(TEMPLATE),
            "-m": str(MASK),
            "-v": str(TOMOGRAM),
            "-d": str(DESTINATION),
            "--angular-search": "35",
            "--tilt-angles": str(TILT_ANGLES),
            "--per-tilt-weighting": "",
            "--dose-accumulation": str(DOSE),
            "--defocus": str(DEFOCUS_IMOD),
            "--amplitude-contrast": "0.08",
            "--spherical-aberration": "2.7",
            "--voltage": "300",
            "--tomogram-ctf-model": "phase-flip",
            "-g": "0",
        }

        # 定义一个辅助函数，用于简化运行过程
        def start(arg_dict):
            """
            简化模板匹配函数的调用。

            参数:
            arg_dict (dict): 参数字典。
            """
            # 调用entry_points模块中的match_template函数
            entry_points.match_template(prep_argv(arg_dict))

        # 测试有效的散焦参数
        for z in [str(DEFOCUS_IMOD), str(DEFOCUS), "3000"]:
            # 复制默认参数
            arguments = defaults.copy()
            # 更新散焦参数
            arguments["--defocus"] = z
            # 启动模板匹配
            start(arguments)

        # 测试错误参数
        for z in ["asdf.txt", "asdf"]:
            # 创建一个StringIO对象，用于捕获标准输出和标准错误输出
            dump = StringIO()
            # 使用with语句捕获SystemExit异常
            with (
                self.assertRaises(SystemExit) as ex,
                redirect_stdout(dump),
                redirect_stderr(dump),
            ):
                # 复制默认参数
                arguments = defaults.copy()
                # 更新散焦参数
                arguments["--defocus"] = z
                # 启动模板匹配
                start(arguments)
            # 关闭StringIO对象
            dump.close()
            # 检查系统返回码是否为2（错误）
            self.assertEqual(ex.exception.code, 2)

        # 移除每倾斜加权参数
        arguments = defaults.copy()
        # 从参数字典中移除每倾斜加权参数
        arguments.pop("--per-tilt-weighting")
        # 启动模板匹配
        start(arguments)

        # 测试缺少CTF参数的情况
        with self.assertRaises(
            ValueError, msg="Missing CTF params should produce error"
        ):
            # 复制默认参数
            arguments = defaults.copy()
            # 从参数字典中移除电压参数
            arguments.pop("--voltage")
            # 启动模板匹配
            start(arguments)

        # 测试角度搜索和粒子直径选项
        with self.assertRaises(
            ValueError, msg="Missing angular search should raise an error."
        ):
            # 复制默认参数
            arguments = defaults.copy()
            # 从参数字典中移除角度搜索参数
            arguments.pop("--angular-search")
            # 启动模板匹配
            start(arguments)

        # 复制默认参数
        arguments = defaults.copy()
        # 从参数字典中移除角度搜索参数
        arguments.pop("--angular-search")
        # 添加粒子直径参数
        arguments["--particle-diameter"] = "50"
        # 设置低通滤波器参数
        arguments["--low-pass"] = "50"
        # 启动模板匹配
        start(arguments)

        # 相位随机化测试
        # 复制默认参数
        arguments = defaults.copy()
        # 添加相位随机化参数
        arguments["-r"] = ""
        # 启动模板匹配
        start(arguments)
        # 测试是否可以设置随机数种子，见问题 #194
        # 添加随机数种子参数
        arguments["--rng-seed"] = "42"
        # 启动模板匹配
        start(arguments)

        # 测试调试文件
        # 复制默认参数
        arguments = defaults.copy()
        # 添加日志级别为调试的参数
        arguments["--log"] = "debug"
        # 启动模板匹配
        start(arguments)
        # 这些文件只有在测试成功设置日志记录时才会存在
        # 检查模板PSF文件是否存在
        self.assertTrue(
            DESTINATION.joinpath("template_psf.mrc").exists(),
            msg="File should exist in debug mode",
        )
        # 检查模板卷积文件是否存在
        self.assertTrue(
            DESTINATION.joinpath("template_convolved.mrc").exists(),
            msg="File should exist in debug mode",
        )

        # 在入口点修改日志级别后重置日志级别
        # 重置日志级别为初始日志级别
        logging.basicConfig(level=LOG_LEVEL, force=True)

        # 测试提供无效的GPU索引
        # 获取可用的GPU设备数量
        n_devices = cp.cuda.runtime.getDeviceCount()
        # 遍历无效的GPU索引
        for indices in ["-1", f"0 {n_devices}"]:
            # 创建一个StringIO对象，用于捕获标准输出和标准错误输出
            dump = StringIO()
            # 使用with语句捕获SystemExit异常
            with (
                self.assertRaises(SystemExit) as ex,
                redirect_stdout(dump),
                redirect_stderr(dump),
            ):
                # 复制默认参数
                arguments = defaults.copy()
                # 更新GPU索引参数
                arguments["-g"] = indices
                # 启动模板匹配
                start(arguments)
            # 检查输出中是否包含gpu indices
            self.assertIn("gpu indices", dump.getvalue())
            # 关闭StringIO对象
            dump.close()

        # 测试RELION5元数据读取
        # 复制默认参数
        arguments = defaults.copy()
        # 移除多个参数
        [
            arguments.pop(x)
            for x in [
                "--tilt-angles",
                "--per-tilt-weighting",
                "--dose-accumulation",
                "--defocus",
                "--amplitude-contrast",
                "--spherical-aberration",
                "--voltage",
            ]
        ]
        # 更新断层图像文件路径
        arguments["-v"] = str(RELION5_TOMOGRAM)
        # 添加RELION5断层图像星文件路径
        arguments["--relion5-tomograms-star"] = str(RELION5_TOMOGRAMS_STAR)
        # 启动模板匹配
        start(arguments)

        # 测试缺少倾斜角度的情况
        with self.assertRaises(
            ValueError, msg="Missing tilt angles should raise an error."
        ):
            # 从参数字典中移除RELION5断层图像星文件路径
            arguments.pop("--relion5-tomograms-star")
            # 启动模板匹配
            start(arguments)