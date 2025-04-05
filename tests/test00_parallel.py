"""
此文件名称显式以 test00_ 开头，以确保在测试期间首先运行。
其他测试会在 GPU 上运行作业，这会使主单元测试进程在 GPU 上持续占用资源。
当在独占进程模式下的 GPU 上启动并行管理器测试时，
由于其他单元测试的占用，并行管理器生成的进程会失败。
如果先测试并行管理器，生成的进程会完全关闭，从而允许其余测试使用 GPU。
"""

# 导入 unittest 模块，用于编写和运行单元测试
import unittest
# 导入 pathlib 模块，用于处理文件路径
import pathlib
# 导入 time 模块，用于处理时间相关操作
import time
# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 voltools 库，用于处理 3D 体积数据
import voltools as vt
# 导入 multiprocessing 模块，用于实现多进程编程
import multiprocessing
# 从 pytom_tm.mask 模块导入 spherical_mask 函数，用于生成球形掩码
from pytom_tm.mask import spherical_mask
# 从 pytom_tm.angles 模块导入 angle_to_angle_list 函数，用于生成角度列表
from pytom_tm.angles import angle_to_angle_list
# 从 pytom_tm.parallel 模块导入 run_job_parallel 函数，用于并行运行作业
from pytom_tm.parallel import run_job_parallel
# 从 pytom_tm.tmjob 模块导入 TMJob 类，用于定义模板匹配作业
from pytom_tm.tmjob import TMJob
# 从 pytom_tm.io 模块导入 write_mrc 函数，用于将数据写入 MRC 文件
from pytom_tm.io import write_mrc

# 定义断层图像的形状
TOMO_SHAPE = (100, 107, 59)
# 定义模板的大小
TEMPLATE_SIZE = 13
# 定义模板在断层图像中的位置
LOCATION = (77, 26, 40)
# 定义角度 ID
ANGLE_ID = 100
# 定义角度搜索范围
ANGULAR_SEARCH = 38.53
# 定义测试数据目录的路径
TEST_DATA_DIR = pathlib.Path(__file__).parent.joinpath("test_data")
# 定义测试断层图像文件的路径
TEST_TOMOGRAM = TEST_DATA_DIR.joinpath("tomogram.mrc")
# 定义测试模板文件的路径
TEST_TEMPLATE = TEST_DATA_DIR.joinpath("template.mrc")
# 定义测试掩码文件的路径
TEST_MASK = TEST_DATA_DIR.joinpath("mask.mrc")

class TestTMJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        类方法，在类的所有测试用例执行之前执行一次。
        用于创建测试所需的模板、掩码和断层图像数据，并将其保存到文件中。
        """
        # 创建一个形状为 TOMO_SHAPE 的零数组，用于表示断层图像
        volume = np.zeros(TOMO_SHAPE, dtype=np.float32)
        # 创建一个形状为 (TEMPLATE_SIZE, TEMPLATE_SIZE, TEMPLATE_SIZE) 的零数组，用于表示模板
        template = np.zeros((TEMPLATE_SIZE,) * 3, dtype=np.float32)
        # 在模板的特定区域赋值为 1.0
        template[3:8, 4:8, 3:7] = 1.0
        template[7, 8, 5:7] = 1.0
        # 生成一个球形掩码
        mask = spherical_mask(TEMPLATE_SIZE, 5, 0.5)
        # 根据角度搜索范围和角度 ID 获取旋转角度
        rotation = angle_to_angle_list(ANGULAR_SEARCH)[ANGLE_ID]

        # 将旋转后的模板放置到断层图像的指定位置
        volume[
            LOCATION[0] - TEMPLATE_SIZE // 2 : LOCATION[0]
            + TEMPLATE_SIZE // 2
            + TEMPLATE_SIZE % 2,
            LOCATION[1] - TEMPLATE_SIZE // 2 : LOCATION[1]
            + TEMPLATE_SIZE // 2
            + TEMPLATE_SIZE % 2,
            LOCATION[2] - TEMPLATE_SIZE // 2 : LOCATION[2]
            + TEMPLATE_SIZE // 2
            + TEMPLATE_SIZE % 2,
        ] = vt.transform(
            template,
            rotation=rotation,
            rotation_units="rad",
            rotation_order="rzxz",
            device="cpu",
        )

        # 为断层图像添加一些噪声
        rng = np.random.default_rng(0)
        volume += rng.normal(loc=0, scale=0.1, size=volume.shape)

        # 创建测试数据目录，如果目录已存在则不会报错
        TEST_DATA_DIR.mkdir(exist_ok=True)
        # 将掩码数据写入 MRC 文件
        write_mrc(TEST_MASK, mask, 1.0)
        # 将模板数据写入 MRC 文件
        write_mrc(TEST_TEMPLATE, template, 1.0)
        # 将断层图像数据写入 MRC 文件
        write_mrc(TEST_TOMOGRAM, volume, 1.0)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        类方法，在类的所有测试用例执行之后执行一次。
        用于删除测试过程中创建的文件和目录。
        """
        # 删除测试掩码文件
        TEST_MASK.unlink()
        # 删除测试模板文件
        TEST_TEMPLATE.unlink()
        # 删除测试断层图像文件
        TEST_TOMOGRAM.unlink()
        # 删除测试数据目录
        TEST_DATA_DIR.rmdir()

    def setUp(self):
        """
        实例方法，在每个测试用例执行之前执行。
        用于创建一个 TMJob 实例，用于后续的测试。
        """
        self.job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=38.53,
            voxel_size=1.0,
        )

    def test_parallel_breaking(self):
        """
        测试用例，用于测试并行作业在部分资源无效时的异常处理。
        当并行作业启动时，如果使用了部分无效的资源，应该抛出 RuntimeError 异常。
        """
        try:
            # 尝试并行运行作业，使用部分无效的资源
            _ = run_job_parallel(
                self.job, volume_splits=(1, 2, 1), gpu_ids=[0, -1], unittest_mute=True
            )
        except RuntimeError:
            # 若捕获到 RuntimeError 异常，等待 2 秒以确保所有子进程都已清理
            time.sleep(2)
            # 检查是否还有活动的子进程
            self.assertEqual(
                len(multiprocessing.active_children()),
                0,
                msg="在启动部分资源无效的并行作业后，仍有进程在运行",
            )
        else:  # pragma: no cover
            # 若未抛出异常，则测试失败
            self.fail("此操作应该抛出 RuntimeError 异常")

    def test_parallel_manager(self):
        """
        测试用例，用于测试并行管理器的功能。
        并行运行作业，检查得分、角度和位置是否符合预期。
        """
        # 并行运行作业，获取得分和角度
        score, angle = run_job_parallel(self.job, volume_splits=(1, 3, 1), gpu_ids=[0])
        # 获取得分最大值的索引
        ind = np.unravel_index(score.argmax(), score.shape)

        # 检查得分的最大值是否大于预期值
        self.assertTrue(score.max() > 0.931, msg="LCC 最大值低于预期")
        # 检查角度是否与预期的角度 ID 相等
        self.assertEqual(ANGLE_ID, angle[ind])
        # 检查位置是否与预期的位置相等
        self.assertSequenceEqual(LOCATION, ind)
