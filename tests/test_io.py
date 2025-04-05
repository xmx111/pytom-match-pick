# 导入unittest模块，用于编写单元测试
import unittest
# 导入pathlib模块，用于处理文件路径
import pathlib
# 导入warnings模块，用于处理警告信息
import warnings
# 导入contextlib模块，用于创建上下文管理器
import contextlib
# 从tempfile模块导入TemporaryDirectory类，用于创建临时目录
from tempfile import TemporaryDirectory
# 导入numpy库，用于数值计算
import numpy as np
# 导入mrcfile库，用于处理MRC格式的文件
import mrcfile

# 从pytom_tm.io模块导入需要测试的函数
from pytom_tm.io import read_mrc, read_mrc_meta_data, write_mrc, parse_relion5_star_data

# 定义一个失败的MRC文件路径
FAILING_MRC = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/human_ribo_mask_32_8_5.mrc")
)
# 注释说明下面的文件是通过截取部分数据得到的，用于模拟损坏的MRC文件
# The below file was made with head -c 1024 human_ribo_mask_32_8_5.mrc > header_only.mrc
# 定义一个损坏的MRC文件路径
CORRUPT_MRC = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/header_only.mrc")
)
# 定义一个Relion 5格式的tomograms.star文件路径
RELION5_TOMOGRAMS_STAR = pathlib.Path(__file__).parent.joinpath(
    "Data/relion5_project_example/Tomograms/job009/tomograms.star"
)


class TestBrokenMRC(unittest.TestCase):
    def setUp(self):
        """
        测试用例执行前的初始化操作。
        主要功能包括屏蔽其他代码库产生的RuntimeWarnings，以及创建临时目录。
        """
        # Mute the RuntimeWarnings comming from other code-base inside these tests
        # following this SO answer: https://stackoverflow.com/a/45809502
        # 创建一个上下文管理器栈
        stack = contextlib.ExitStack()
        # 进入一个捕获警告的上下文
        _ = stack.enter_context(warnings.catch_warnings())
        # 简单过滤所有警告
        warnings.simplefilter("ignore")
        # The follwing line is better, but only works in python >= 3.11
        # _ = stack.enter_context(warnings.catch_warnings(action="ignore"))

        # 注册清理函数，在测试结束后关闭上下文管理器栈
        self.addCleanup(stack.close)

        # prep temporary directory
        # 创建一个临时目录
        tempdir = TemporaryDirectory()
        # 保存临时目录的名称
        self.tempdirname = tempdir.name
        # 注册清理函数，在测试结束后清理临时目录
        self.addCleanup(tempdir.cleanup)

    def test_read_mrc_minor_broken(self):
        """
        测试读取轻微损坏的MRC文件，检查是否能读取文件并打印相应的日志信息。
        """
        # Test if this mrc can be read and if the approriate logs are printed
        # 使用assertLogs上下文管理器捕获日志信息
        with self.assertLogs(level="WARNING") as cm:
            # 读取轻微损坏的MRC文件
            mrc = read_mrc(FAILING_MRC)
        # 断言读取的MRC文件不为None
        self.assertIsNotNone(mrc)
        # 断言日志输出的数量为1
        self.assertEqual(len(cm.output), 1)
        # 断言日志输出中包含文件名
        self.assertIn(FAILING_MRC.name, cm.output[0])
        # 断言日志输出中包含提示信息
        self.assertIn("make sure this is correct", cm.output[0])

    def test_read_mrc_too_broken(self):
        """
        测试读取严重损坏的MRC文件，检查是否会抛出预期的错误。
        """
        # Test if this mrc raises an error as expected
        # 使用assertRaises上下文管理器捕获ValueError异常
        with self.assertRaises(ValueError) as err:
            # 尝试读取严重损坏的MRC文件
            _ = read_mrc(CORRUPT_MRC)
        # 断言异常信息中包含文件名
        self.assertIn(CORRUPT_MRC.name, str(err.exception))
        # 断言异常信息中包含提示信息
        self.assertIn("too corrupt", str(err.exception))

    def test_read_mrc_meta_data(self):
        """
        测试读取轻微损坏的MRC文件的元数据，检查是否能读取并打印相应的日志信息。
        """
        # Test if this mrc can be read and if the approriate logs are printed
        # 使用assertLogs上下文管理器捕获日志信息
        with self.assertLogs(level="WARNING") as cm:
            # 读取轻微损坏的MRC文件的元数据
            mrc = read_mrc_meta_data(FAILING_MRC)
        # 断言读取的MRC元数据不为None
        self.assertIsNotNone(mrc)
        # 断言日志输出的数量为1
        self.assertEqual(len(cm.output), 1)
        # 断言日志输出中包含文件名
        self.assertIn(FAILING_MRC.name, cm.output[0])
        # 断言日志输出中包含提示信息
        self.assertIn("make sure this is correct", cm.output[0])

    def test_half_precision_read_write_cycle(self):
        """
        测试半精度（float16）数据的读写循环，检查写入的文件是否能正确读取，
        并且数据类型和内容是否与原始数据一致。
        """
        # 生成一个随机的3x3x3数组，并转换为float16类型
        array = np.random.rand(27).reshape((3, 3, 3)).astype(np.float16)
        # 定义临时文件的路径
        fname = pathlib.Path(self.tempdirname) / "test_half.mrc"
        # Make sure no warnings are raised
        # 使用assertNoLogs上下文管理器确保没有警告信息
        with self.assertNoLogs(level="WARNING"):
            # 将数组写入MRC文件
            write_mrc(fname, array, 1.0)
        # Make sure the file can be read back
        # make sure mode is as expected for float16
        # https://mrcfile.readthedocs.io/en/stable/source/mrcfile.html#mrcfile.utils.dtype_from_mode
        # 打开MRC文件
        mrc = mrcfile.open(fname)
        # 断言文件头的mode属性为12（对应float16类型）
        self.assertEqual(mrc.header.mode, 12)
        # 关闭MRC文件
        mrc.close()
        # make sure dtype is expected
        # 读取MRC文件
        mrc = read_mrc(fname)
        # 断言读取的数据类型为float16
        self.assertEqual(mrc.dtype, np.float16)
        # make sure data is identical
        # 断言读取的数据与原始数据一致
        np.testing.assert_equal(mrc, array)

    def test_cast_warning(self):
        """
        测试写入整数类型数组时是否会抛出警告信息。
        """
        # make sure a warning is raised when writing an integer based array
        # 生成一个随机的3x3x3数组，并转换为int32类型
        array = np.random.rand(27).reshape((3, 3, 3)).astype(np.int32)
        # 定义临时文件的路径
        fname = pathlib.Path(self.tempdirname) / "test_cast.mrc"
        # 使用assertLogs上下文管理器捕获日志信息
        with self.assertLogs(level="WARNING") as cm:
            # 将数组写入MRC文件
            write_mrc(fname, array, 1.0)
        # 断言日志输出的数量为1
        self.assertEqual(len(cm.output), 1)
        # 断言日志输出中包含提示信息
        self.assertIn("np.float32", cm.output[0])

    def test_parse_relion5_star_data(self):
        """
        测试解析Relion 5格式的tomograms.star文件，检查解析结果的长度和数据类型是否符合预期，
        以及处理不匹配文件名时是否会抛出预期的错误。
        """
        # 定义一个tomogram文件路径
        tomogram = pathlib.Path("rec_tomo200528_107.mrc")
        # 解析Relion 5格式的tomograms.star文件
        meta_data = parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)
        # 断言解析结果的长度为5
        self.assertEqual(len(meta_data), 5)
        # 断言解析结果的第一个元素为float类型
        self.assertIsInstance(meta_data[0], float)
        # 断言解析结果的第二个元素为list类型
        self.assertIsInstance(meta_data[1], list)
        # 断言解析结果的第三个元素为list类型
        self.assertIsInstance(meta_data[2], list)
        # 断言解析结果的第四个元素为list类型
        self.assertIsInstance(meta_data[3], list)
        # 断言解析结果的第四个元素的第一个元素为dict类型
        self.assertIsInstance(meta_data[3][0], dict)
        # 断言解析结果的第五个元素为int类型
        self.assertIsInstance(meta_data[4], int)

        # 定义一个不匹配的tomogram文件路径
        tomogram = pathlib.Path("tomogram.mrc")
        # 使用assertRaises上下文管理器捕获ValueError异常
        with self.assertRaises(
            ValueError, msg="Unmatching tomograms name should raise an error."
        ):
            # 尝试解析不匹配的tomogram文件
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)

        # 定义一个部分匹配的tomogram文件路径
        tomogram = pathlib.Path("rec_tomogram200528_1077.mrc")
        # 使用assertRaises上下文管理器捕获ValueError异常
        with self.assertRaises(
            ValueError, msg="Partially matching tomogram name should raise an error."
        ):
            # 尝试解析部分匹配的tomogram文件
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)

        # 定义一个部分匹配的tomogram文件路径
        tomogram = pathlib.Path("rec_tomogram200528_10.mrc")
        # 使用assertRaises上下文管理器捕获ValueError异常
        with self.assertRaises(
            ValueError, msg="Partially matching tomogram name should raise an error."
        ):
            # 尝试解析部分匹配的tomogram文件
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)
