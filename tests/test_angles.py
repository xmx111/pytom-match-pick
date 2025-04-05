# 导入 unittest 模块，用于编写和运行测试用例
import unittest
# 导入 pathlib 模块，用于处理文件路径
import pathlib
# 从 pytom_tm.angles 模块导入 load_angle_list 和 angle_to_angle_list 函数
from pytom_tm.angles import load_angle_list, angle_to_angle_list
# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 itertools 库，用于迭代器操作
import itertools as itt
# 导入 re 模块，用于正则表达式操作
import re
# 导入 logging 模块，用于记录日志
import logging

# 定义测试数据目录，使用当前文件所在目录的父目录下的 test_data 文件夹
TEST_DATA_DIR = pathlib.Path(__file__).parent.joinpath("test_data")
# 定义包含错误角度的文件路径
ERRONEOUS_ANGLE_FILE = TEST_DATA_DIR.joinpath("error_angles.txt")
# 定义包含未排序角度的文件路径
UNORDERED_ANGLE_FILE = TEST_DATA_DIR.joinpath("unordered_angles.txt")


class TestAngles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        在测试类开始执行前运行的方法，用于创建测试所需的文件。
        """
        # 创建测试数据目录
        TEST_DATA_DIR.mkdir()
        # 创建一个包含错误角度的文件
        with open(ERRONEOUS_ANGLE_FILE, "w") as fstream:
            # 写入一行包含 4 个 1.0 的数据
            fstream.write(" ".join(map(str, [1.0] * 4)) + "\n")
            # 写入一行包含 3 个 1.0 的数据
            fstream.write(" ".join(map(str, [1.0] * 3)) + "\n")
        # 创建一个包含未排序角度的文件
        with open(UNORDERED_ANGLE_FILE, "w") as fstream:
            # 写入不同的角度数据
            fstream.write(" ".join(["3.", "3.", "1."]) + "\n")
            fstream.write(" ".join(["3", "2.", "1."]) + "\n")
            fstream.write(" ".join(["2.", "3.", "1."]) + "\n")
            fstream.write(" ".join(["3.", "2.", "2."]) + "\n")

    @classmethod
    def tearDownClass(cls) -> None:
        """
        在测试类执行结束后运行的方法，用于清理测试创建的文件和目录。
        """
        # 删除错误角度文件和未排序角度文件
        for f in [ERRONEOUS_ANGLE_FILE, UNORDERED_ANGLE_FILE]:
            f.unlink()
        # 删除测试数据目录
        TEST_DATA_DIR.rmdir()

    def test_load_list(self):
        """
        测试 load_angle_list 函数在处理包含错误角度的文件时是否抛出预期的异常。
        """
        with self.assertRaisesRegex(
            ValueError,
            "each line should have 3",
            msg="Invalid angle file should raise an error",
        ):
            # 尝试加载包含错误角度的文件，期望抛出 ValueError 异常
            load_angle_list(ERRONEOUS_ANGLE_FILE)

    def test_load_sort(self):
        """
        测试 load_angle_list 函数在处理未排序角度文件时是否能正确排序。
        """
        # 加载未排序角度文件，并设置排序参数为 True
        angles = load_angle_list(UNORDERED_ANGLE_FILE, sort_angles=True)
        # 定义预期的排序结果
        expected = [(2.0, 3.0, 1.0), (3.0, 2.0, 1.0), (3.0, 2.0, 2.0), (3.0, 3.0, 1.0)]
        # 断言加载的角度列表与预期结果相等
        self.assertEqual(angles, expected)

    def test_angle_to_angle_list(self):
        """
        测试 angle_to_angle_list 函数的功能，包括日志记录、角度范围检查和排序检查。
        """
        # 生成一个 1 到 90 之间的随机角度
        angle = 1 + np.random.random() * 89
        # 捕获日志输出
        with self.assertLogs(level="INFO") as cm:
            # 调用 angle_to_angle_list 函数生成角度列表
            angles = angle_to_angle_list(angle, log_level=logging.INFO)

        # 检查日志输出的数量是否为 2
        self.assertEqual(len(cm.output), 2)
        for out in cm.output:
            # 使用正则表达式查找日志中的角度值
            possible_match = re.findall(r"\s\d*[.]\d+\s", out)
            # 检查找到的角度值数量是否为 1
            self.assertEqual(len(possible_match), 1)
            # 检查找到的角度值是否小于等于输入的角度
            self.assertLessEqual(float(possible_match[0]), angle)

        # 遍历角度列表，检查相邻元素是否按升序排列
        for a, b in itt.pairwise(angles):
            # 检查相邻元素是否按升序排列
            self.assertLess(a, b)
            # 检查每个元素的第二个值是否不为 0
            self.assertNotEqual(a[1], 0)
        # 检查最后一个元素的第二个值是否不为 0
        self.assertNotEqual(b[1], 0)
