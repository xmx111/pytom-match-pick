# 导入 unittest 模块，用于编写和运行测试用例
import unittest
# 导入 voltools 库，用于处理 3D 体积数据
import voltools as vt
# 导入 numpy 库，用于数值计算和数组操作
import numpy as np
# 从 pytom_tm.matching 模块导入 TemplateMatchingGPU 类，用于模板匹配
from pytom_tm.matching import TemplateMatchingGPU
# 从 pytom_tm.mask 模块导入 spherical_mask 函数，用于创建球形掩码
from pytom_tm.mask import spherical_mask
# 从 pytom_tm.angles 模块导入 angle_to_angle_list 函数，用于生成角度列表
from pytom_tm.angles import angle_to_angle_list


class TestTM(unittest.TestCase):
    """
    该类用于测试 TemplateMatchingGPU 类的功能。
    继承自 unittest.TestCase，包含了多个测试方法。
    """
    def setUp(self):
        """
        测试用例执行前的初始化方法。
        初始化了测试所需的体积数据、模板、掩码、GPU ID 和角度列表。
        """
        # 定义模板的大小
        self.t_size = 12
        # 创建一个大小为 (100, 100, 100) 的零数组作为搜索体积
        self.volume = np.zeros((100,) * 3, dtype=float)
        # 创建一个大小为 (t_size, t_size, t_size) 的零数组作为模板
        self.template = np.zeros((self.t_size,) * 3, dtype=float)
        # 在模板的特定区域赋值为 1
        self.template[3:8, 4:8, 3:7] = 1.0
        self.template[7, 8, 5:7] = 1.0
        # 调用 spherical_mask 函数创建一个球形掩码
        self.mask = spherical_mask(self.t_size, 5, 0.5)
        # 定义使用的 GPU ID
        self.gpu_id = "gpu:0"
        # 调用 angle_to_angle_list 函数生成角度列表
        self.angles = angle_to_angle_list(38.53)

    def test_search_spherical_mask(self):
        """
        测试使用球形掩码进行模板匹配的功能。
        验证在给定旋转角度和位置的情况下，模板匹配的结果是否符合预期。
        """
        # 选择一个角度 ID
        angle_id = 100
        # 根据角度 ID 获取对应的旋转角度
        rotation = self.angles[angle_id]
        # 定义模板在搜索体积中的位置
        loc = (77, 26, 40)
        # 将旋转后的模板放置到搜索体积的指定位置
        self.volume[
            loc[0] - self.t_size // 2 : loc[0] + self.t_size // 2,
            loc[1] - self.t_size // 2 : loc[1] + self.t_size // 2,
            loc[2] - self.t_size // 2 : loc[2] + self.t_size // 2,
        ] = vt.transform(
            self.template,
            rotation=rotation,
            rotation_units="rad",
            rotation_order="rzxz",
            device="cpu",
        )

        # 创建 TemplateMatchingGPU 类的实例
        tm = TemplateMatchingGPU(
            0,
            0,
            self.volume,
            self.template,
            self.mask,
            self.angles,
            list(range(len(self.angles))),
        )
        # 运行模板匹配
        score_volume, angle_volume, stats = tm.run()

        # 获取分数体积中最大值的索引
        ind = np.unravel_index(score_volume.argmax(), self.volume.shape)
        # 验证分数体积的最大值是否大于 0.99
        self.assertTrue(
            score_volume.max() > 0.99, msg="lcc max value lower than expected"
        )
        # 验证角度体积中最大值索引处的角度 ID 是否与预期一致
        self.assertEqual(angle_id, angle_volume[ind])
        # 验证最大值索引是否与预期位置一致
        self.assertSequenceEqual(loc, ind)
        # 计算预期的搜索空间大小
        expected_search_space = len(self.angles) * self.volume.size
        # 验证实际搜索空间大小是否与预期一致
        self.assertEqual(
            stats["search_space"],
            expected_search_space,
            msg="Search space should exactly equal this value",
        )
        # 验证搜索的标准差是否与预期值近似相等
        self.assertAlmostEqual(
            stats["std"],
            0.005163,
            places=5,
            msg="Standard deviation of the search should be almost equal",
        )

    def test_search_non_spherical_mask(self):
        """
        测试使用非球形掩码进行模板匹配的功能。
        验证在给定旋转角度和位置的情况下，模板匹配的结果是否符合预期。
        """
        # 选择一个角度 ID
        angle_id = 100
        # 根据角度 ID 获取对应的旋转角度
        rotation = self.angles[angle_id]
        # 定义模板在搜索体积中的位置
        loc = (77, 26, 40)
        # 将旋转后的模板放置到搜索体积的指定位置
        self.volume[
            loc[0] - self.t_size // 2 : loc[0] + self.t_size // 2,
            loc[1] - self.t_size // 2 : loc[1] + self.t_size // 2,
            loc[2] - self.t_size // 2 : loc[2] + self.t_size // 2,
        ] = vt.transform(
            self.template,
            rotation=rotation,
            rotation_units="rad",
            rotation_order="rzxz",
            device="cpu",
        )

        # 创建 TemplateMatchingGPU 类的实例，指定掩码不是球形
        tm = TemplateMatchingGPU(
            0,
            0,
            self.volume,
            self.template,
            self.mask,
            self.angles,
            list(range(len(self.angles))),
            mask_is_spherical=False,
        )
        # 运行模板匹配
        score_volume, angle_volume, stats = tm.run()

        # 获取分数体积中最大值的索引
        ind = np.unravel_index(score_volume.argmax(), self.volume.shape)
        # 验证分数体积的最大值是否大于 0.99
        self.assertTrue(
            score_volume.max() > 0.99, msg="lcc max value lower than expected"
        )
        # 验证角度体积中最大值索引处的角度 ID 是否与预期一致
        self.assertEqual(angle_id, angle_volume[ind])
        # 验证最大值索引是否与预期位置一致
        self.assertSequenceEqual(loc, ind)
        # 计算预期的搜索空间大小
        expected_search_space = len(self.angles) * self.volume.size
        # 验证实际搜索空间大小是否与预期一致
        self.assertEqual(
            stats["search_space"],
            expected_search_space,
            msg="Search space should exactly equal this value",
        )
        # 验证搜索的标准差是否与预期值近似相等
        self.assertAlmostEqual(
            stats["std"],
            0.005187,
            places=4,
            msg="Standard deviation of the search should be almost equal",
        )

    def test_search_noise_correction(self):
        """
        测试使用噪声校正进行模板匹配的功能。
        验证在给定旋转角度和位置的情况下，模板匹配的结果是否符合预期。
        """
        # 选择一个角度 ID
        angle_id = 100
        # 根据角度 ID 获取对应的旋转角度
        rotation = self.angles[angle_id]
        # 定义模板在搜索体积中的位置
        loc = (77, 26, 40)
        # 将旋转后的模板放置到搜索体积的指定位置
        self.volume[
            loc[0] - self.t_size // 2 : loc[0] + self.t_size // 2,
            loc[1] - self.t_size // 2 : loc[1] + self.t_size // 2,
            loc[2] - self.t_size // 2 : loc[2] + self.t_size // 2,
        ] = vt.transform(
            self.template,
            rotation=rotation,
            rotation_units="rad",
            rotation_order="rzxz",
            device="cpu",
        )

        # 创建 TemplateMatchingGPU 类的实例，启用噪声校正
        tm = TemplateMatchingGPU(
            0,
            0,
            self.volume,
            self.template,
            self.mask,
            self.angles,
            list(range(len(self.angles))),
            noise_correction=True,
        )
        # 运行模板匹配
        score_volume, angle_volume, stats = tm.run()

        # 获取分数体积中最大值的索引
        ind = np.unravel_index(score_volume.argmax(), self.volume.shape)
        # 验证分数体积的最大值是否大于 0.99
        self.assertTrue(
            score_volume.max() > 0.99, msg="lcc max value lower than expected"
        )
        # 验证角度体积中最大值索引处的角度 ID 是否与预期一致
        self.assertEqual(angle_id, angle_volume[ind])
        # 验证最大值索引是否与预期位置一致
        self.assertSequenceEqual(loc, ind)
        # 计算预期的搜索空间大小
        expected_search_space = len(self.angles) * self.volume.size
        # 验证实际搜索空间大小是否与预期一致
        self.assertEqual(
            stats["search_space"],
            expected_search_space,
            msg="Search space should exactly equal this value",
        )
        # 验证搜索的标准差是否与预期值近似相等
        self.assertAlmostEqual(
            stats["std"],
            0.005187,
            places=4,
            msg="Standard deviation of the search should be almost equal",
        )
