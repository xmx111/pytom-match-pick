# 导入unittest模块，用于编写和运行测试用例
import unittest
# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 从scipy.ndimage模块导入center_of_mass函数，用于计算数组的质心
from scipy.ndimage import center_of_mass
# 从pytom_tm.template模块导入generate_template_from_map和phase_randomize_template函数
from pytom_tm.template import generate_template_from_map, phase_randomize_template


class TestTemplate(unittest.TestCase):
    def setUp(self):
        """
        测试用例执行前的初始化操作。
        创建一个三维的零数组作为模板，并设置部分元素的值，同时记录模板的中心位置。
        """
        # 创建一个形状为(13, 13, 13)的零数组，数据类型为float32
        self.template = np.zeros((13, 13, 13), dtype=np.float32)
        # 将模板中指定区域的元素值设为 -1
        self.template[2:5, 2:5, 7:9] = -1
        # 将模板中更小的指定区域的元素值设为 2
        self.template[3:5, 3:5, 7:9] = 2
        # 记录模板的中心位置
        self.template_center = (6, 6, 6)

    def test_template_padding(self):
        """
        测试模板填充功能。
        验证不同情况下模板的形状是否符合预期，以及在不合法输出盒子大小下是否发出警告。
        """
        # 创建一个形状为(13, 13, 7)的零数组
        uneven_box = np.zeros((13, 13, 7))
        # 调用generate_template_from_map函数生成新模板，不指定输出盒子大小
        new_template = generate_template_from_map(uneven_box, 1, 1)
        # 断言新模板的形状与原模板形状相同
        self.assertEqual(
            new_template.shape, self.template.shape, msg="Box should be made square"
        )
        # 调用generate_template_from_map函数生成新模板，指定输出盒子大小为 20
        new_template = generate_template_from_map(uneven_box, 1, 2, output_box_size=20)
        # 断言新模板的形状为(20, 20, 20)
        self.assertEqual(
            new_template.shape,
            (20,) * 3,
            msg="Template was not padded to output box size",
        )

        # 捕获日志信息，检查在输出盒子大小为 3 时是否发出警告
        with self.assertLogs(level="WARNING") as cm:
            new_template = generate_template_from_map(
                uneven_box, 1, 2, output_box_size=3
            )
        # 断言日志输出的数量为 1
        self.assertEqual(len(cm.output), 1)
        # 断言日志输出中包含指定的警告信息
        self.assertIn("Could not set specified box size", cm.output[0])

    def test_template_centering(self):
        """
        测试模板居中功能。
        验证在不进行重新居中以及进行重新居中时模板的变化情况，以及绝对值模板的中心是否一致。
        """
        # 调用generate_template_from_map函数生成新模板，不进行重新居中
        new_template = generate_template_from_map(
            self.template,
            1,
            1,
            center=False,
        )
        # 计算新模板与原模板差值的平方和
        square_sum = np.square(new_template - self.template).sum()
        # 断言差值的平方和小于 10，即模板没有显著变化
        self.assertTrue(
            square_sum < 10,
            msg="Template should not change strongly without recentering.",
        )
        # 调用generate_template_from_map函数生成新模板，进行重新居中
        new_template = generate_template_from_map(
            self.template,
            1,
            1,
            center=True,
        )
        # 计算新模板与原模板差值的平方和
        square_sum = np.square(new_template - self.template).sum()
        # 断言差值的平方和大于 10，即模板发生了变化
        self.assertTrue(square_sum > 10, msg="Template didnt change after shift")
        # 计算新模板质心与原模板中心的差值
        diff = np.array(center_of_mass(new_template**2)) - np.array(
            self.template_center
        )
        # 计算差值的绝对值之和
        diff = np.abs(diff).sum()
        # 断言差值的绝对值之和小于 1，即总偏移差值较小
        self.assertTrue(diff < 1, msg="Total shift difference should be small")
        # 调用generate_template_from_map函数生成绝对值模板，进行重新居中
        abs_template = generate_template_from_map(
            np.abs(self.template),
            1,
            1,
            center=True,
        )
        # 计算新模板质心与绝对值模板质心的差值
        diff = np.array(center_of_mass(new_template**2)) - np.array(
            center_of_mass(abs_template**2)
        )
        # 计算差值的绝对值之和
        diff = np.abs(diff).sum()
        # 断言差值的绝对值之和小于 1，即绝对值模板应提供相同的中心
        self.assertTrue(diff < 1, msg="Absolute should provide exactly same center")

    def test_lowpass_resolution(self):
        """
        测试低通滤波分辨率功能。
        验证在不同滤波分辨率下是否发出警告，以及是否能正常工作。
        """
        # 捕获日志信息，检查在滤波分辨率为 1.5 时是否发出警告
        with self.assertLogs(level="WARNING") as cm:
            _ = generate_template_from_map(
                self.template, 1.0, 1.0, filter_to_resolution=1.5
            )
        # 断言日志输出的数量为 1
        self.assertEqual(len(cm.output), 1)
        # 断言日志输出中包含指定的警告信息
        self.assertIn("Filter resolution", cm.output[0])
        self.assertIn("too low", cm.output[0])

        # 检查在滤波分辨率为 2.5 时是否不发出警告
        with self.assertNoLogs(level="WARNING"):
            _ = generate_template_from_map(
                self.template, 1.0, 1.0, filter_to_resolution=2.5
            )

    def test_phase_randomize_template(self):
        """
        测试模板相位随机化功能。
        验证相位随机化后模板是否与原模板不同，以及不同种子是否产生不同的随机化结果。
        """
        # 调用phase_randomize_template函数对模板进行相位随机化，使用默认种子
        randomized = phase_randomize_template(
            self.template,  # use default seed
        )
        # 断言随机化后的模板形状与原模板形状相同
        self.assertEqual(self.template.shape, randomized.shape)
        # 断言随机化后的模板与原模板不相等的元素数量大于 0
        self.assertGreater(
            (self.template != randomized).sum(),
            0,
            msg="After phase randomization the template should "
            "no longer be equal to the input.",
        )

        # 调用phase_randomize_template函数对模板进行相位随机化，使用种子 11
        randomized_seeded = phase_randomize_template(
            self.template,
            11,  # use default seed
        )
        # 计算两次随机化结果的差值之和
        diff = np.abs(randomized_seeded - randomized).sum()
        # 断言两次随机化结果的差值之和不为 0，即不同种子应返回不同的随机化结果
        self.assertNotEqual(
            diff, 0, msg="Different seed should return different randomization"
        )
