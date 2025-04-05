# 导入 unittest 模块，用于编写单元测试
import unittest
# 导入 numpy 库，用于数值计算
import numpy as np
# 从 pytom_tm.extract 模块导入 predict_tophat_mask 函数
from pytom_tm.extract import predict_tophat_mask


class TestExtract(unittest.TestCase):
    """
    定义一个测试类，继承自 unittest.TestCase，用于测试 extract 模块的功能
    """
    def test_predict_tophat_mask(self):
        """
        测试 predict_tophat_mask 函数的功能
        """
        # 创建一个随机数生成器，种子为 0，保证结果可复现
        rng = np.random.default_rng(0)
        # 生成一个形状为 (50, 50, 50) 的三维数组，元素服从均值为 0，标准差为 0.1 的正态分布，模拟随机峰值
        volume = rng.normal(loc=0, scale=0.1, size=(50,) * 3)
        # 将数组中索引为 (20, 20, 20) 的元素设置为 1
        volume[20, 20, 20] = 1
        # 调用 predict_tophat_mask 函数，对 volume 数组进行处理，得到 tophat 掩码
        tophat_mask = predict_tophat_mask(volume)
        # 断言 tophat 掩码的形状与输入数组的形状相同
        self.assertEqual(
            tophat_mask.shape,
            volume.shape,
            msg="tophat mask should have same size as input",
        )
        # 断言 tophat 掩码的数据类型为布尔型
        self.assertEqual(
            tophat_mask.dtype, bool, msg="predicted tophat mask should be boolean"
        )
        # 断言 tophat 掩码不全为 0，即掩码不为空
        self.assertNotEqual(tophat_mask.sum(), 0, msg="tophat mask should not be empty")

        # 测试使用 float16 类型的分数，函数内部会将其转换为 float32 类型
        tophat_mask = predict_tophat_mask(volume.astype(np.float16))
        # 断言处理 float16 类型分数时，tophat 掩码不全为 0
        self.assertNotEqual(
            tophat_mask.sum(), 0, msg="float16 scores failing for tophat mask"
        )

    # 此测试函数是 test_tmjob.py 中提取测试的一部分，暂时不实现具体逻辑
    # def test_extract_job_with_tomogram_mask(self):
    #    pass
