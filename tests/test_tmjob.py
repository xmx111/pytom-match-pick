import unittest
import pathlib
import numpy as np
import voltools as vt
import mrcfile
from tempfile import TemporaryDirectory
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import angle_to_angle_list
from pytom_tm.tmjob import TMJob, TMJobError, load_json_to_tmjob, get_defocus_offsets
from pytom_tm.io import read_mrc, write_mrc, UnequalSpacingError
from pytom_tm.extract import extract_particles
from testing_utils import CTF_PARAMS, ACCUMULATED_DOSE, TILT_ANGLES

# 定义断层图像的形状
TOMO_SHAPE = (100, 107, 59)
# 定义模板的大小
TEMPLATE_SIZE = 13
# 定义模板在断层图像中的位置
LOCATION = (77, 26, 40)
# 定义角度ID
ANGLE_ID = 100
# 定义角度搜索的增量
ANGULAR_SEARCH = "38.53"
# 创建一个临时目录
TEMP_DIR = TemporaryDirectory()
# 获取临时目录的路径
TEST_DATA_DIR = pathlib.Path(TEMP_DIR.name)
# 定义测试用的断层图像文件路径
TEST_TOMOGRAM = TEST_DATA_DIR.joinpath("tomogram.mrc")
# 定义测试用的损坏的断层图像掩码文件路径
TEST_BROKEN_TOMOGRAM_MASK = TEST_DATA_DIR.joinpath("broken_tomogram_mask.mrc")
# 定义测试用的大小错误的断层图像掩码文件路径
TEST_WRONG_SIZE_TOMO_MASK = TEST_DATA_DIR.joinpath("wrong_size_tomogram_mask.mrc")
# 定义测试用的提取掩码（外部）文件路径
TEST_EXTRACTION_MASK_OUTSIDE = TEST_DATA_DIR.joinpath("extraction_mask_outside.mrc")
# 定义测试用的提取掩码（内部）文件路径
TEST_EXTRACTION_MASK_INSIDE = TEST_DATA_DIR.joinpath("extraction_mask_inside.mrc")
# 定义测试用的模板文件路径
TEST_TEMPLATE = TEST_DATA_DIR.joinpath("template.mrc")
# 定义测试用的间距不等的模板文件路径
TEST_TEMPLATE_UNEQUAL_SPACING = TEST_DATA_DIR.joinpath("template_unequal_spacing.mrc")
# 定义测试用的体素大小错误的模板文件路径
TEST_TEMPLATE_WRONG_VOXEL_SIZE = TEST_DATA_DIR.joinpath("template_voxel_error_test.mrc")
# 定义测试用的掩码文件路径
TEST_MASK = TEST_DATA_DIR.joinpath("mask.mrc")
# 定义测试用的分数文件路径
TEST_SCORES = TEST_DATA_DIR.joinpath("tomogram_scores.mrc")
# 定义测试用的角度文件路径
TEST_ANGLES = TEST_DATA_DIR.joinpath("tomogram_angles.mrc")
# 定义测试用的自定义角度搜索文件路径
TEST_CUSTOM_ANGULAR_SEARCH = TEST_DATA_DIR.joinpath("custom_angular_search.txt")
# 定义测试用的白化滤波器文件路径
TEST_WHITENING_FILTER = TEST_DATA_DIR.joinpath("tomogram_whitening_filter.npy")
# 定义测试用的作业JSON文件路径
TEST_JOB_JSON = TEST_DATA_DIR.joinpath("tomogram_job.json")
# 定义测试用的带白化的作业JSON文件路径
TEST_JOB_JSON_WHITENING = TEST_DATA_DIR.joinpath("tomogram_job_whitening.json")
# 定义测试用的旧版本作业JSON文件路径
TEST_JOB_OLD_VERSION = TEST_DATA_DIR.joinpath("tomogram_job_old_version.json")


class TestTMJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        类级别的设置方法，在所有测试方法执行前运行。
        此方法用于创建模板、掩码和断层图像，并将它们保存到临时文件中。
        """
        # 创建一个全零的断层图像数组
        volume = np.zeros(TOMO_SHAPE, dtype=np.float32)
        # 创建一个全零的模板数组
        template = np.zeros((TEMPLATE_SIZE,) * 3, dtype=np.float32)
        # 在模板中设置部分值为1，模拟模板的特征
        template[3:8, 4:8, 3:7] = 1.0
        template[7, 8, 5:7] = 1.0
        # 创建一个球形掩码
        mask = spherical_mask(TEMPLATE_SIZE, 5, 0.5)
        # 根据角度搜索增量生成角度列表，并选择指定ID的旋转角度
        rotation = angle_to_angle_list(float(ANGULAR_SEARCH))[ANGLE_ID]

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

        # 创建提取掩码（外部）
        extraction_mask_outside = np.zeros(TOMO_SHAPE, dtype=np.float32)
        extraction_mask_outside[20:40, 60:80, 10:30] = 1
        # 创建提取掩码（内部）
        extraction_mask_inside = np.zeros(TOMO_SHAPE, dtype=np.float32)
        extraction_mask_inside[70:90, 15:35, 30:50] = 1

        # 创建临时数据目录
        TEST_DATA_DIR.mkdir(exist_ok=True)
        # 将提取掩码（外部）保存为MRC文件
        write_mrc(TEST_EXTRACTION_MASK_OUTSIDE, extraction_mask_outside, 1.0)
        # 将提取掩码（内部）保存为MRC文件
        write_mrc(TEST_EXTRACTION_MASK_INSIDE, extraction_mask_inside, 1.0)
        # 将掩码保存为MRC文件
        write_mrc(TEST_MASK, mask, 1.0)
        # 将模板保存为MRC文件
        write_mrc(TEST_TEMPLATE, template, 1.0)
        # 将体素大小错误的模板保存为MRC文件
        write_mrc(TEST_TEMPLATE_WRONG_VOXEL_SIZE, template, 1.5)
        # 将间距不等的模板保存为MRC文件
        mrcfile.write(
            TEST_TEMPLATE_UNEQUAL_SPACING,
            template,
            voxel_size=(1.5, 1.0, 2.0),
            overwrite=True,
        )
        # 将断层图像保存为MRC文件
        write_mrc(TEST_TOMOGRAM, volume, 1.0)

        # 运行一次不分割的模板匹配作业，用于后续比较
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=ANGULAR_SEARCH,
            voxel_size=1.0,
        )
        # 启动作业并获取分数和角度
        score, angle = job.start_job(0, return_volumes=True)
        # 将分数保存为MRC文件
        write_mrc(TEST_SCORES, score, job.voxel_size)
        # 将角度保存为MRC文件
        write_mrc(TEST_ANGLES, angle, job.voxel_size)
        # 将作业信息保存为JSON文件
        job.write_to_json(TEST_JOB_JSON)

        # 保存一个包含随机角度的文本文件，用于自定义角度搜索
        np.savetxt(TEST_CUSTOM_ANGULAR_SEARCH, np.random.rand(100, 3))

        # 创建一个带频谱白化的作业
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            whiten_spectrum=True,
        )
        # 将带频谱白化的作业信息保存为JSON文件
        job.write_to_json(TEST_JOB_JSON_WHITENING)

        # 创建一个损坏的断层图像掩码
        broken_tomogram_mask = np.zeros(TOMO_SHAPE, dtype=np.float32)
        # 将损坏的断层图像掩码保存为MRC文件
        write_mrc(TEST_BROKEN_TOMOGRAM_MASK, broken_tomogram_mask, 1.0)

        # 创建一个大小错误的断层图像掩码
        size = list(TOMO_SHAPE)
        size[0] += 1
        wrong_size_tomogram_mask = np.ones(tuple(size), dtype=np.float32)
        # 将大小错误的断层图像掩码保存为MRC文件
        write_mrc(TEST_WRONG_SIZE_TOMO_MASK, wrong_size_tomogram_mask, 1.0)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        类级别的清理方法，在所有测试方法执行后运行。
        此方法用于清理临时目录。
        """
        TEMP_DIR.cleanup()

    def setUp(self):
        """
        每个测试方法执行前的设置方法。
        此方法用于创建一个TMJob实例，供后续测试使用。
        """
        self.job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=ANGULAR_SEARCH,
            voxel_size=1.0,
        )

    def test_tm_job_errors(self):
        """
        测试TMJob类在不同错误情况下的异常处理。
        包括体素大小不匹配、间距不等、搜索索引无效、角度输入错误、模板掩码损坏或大小错误等情况。
        """
        with self.assertRaises(
            ValueError,
            msg="不同的体素大小在断层图像和模板中，且未提供体素大小，应引发ValueError",
        ):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE_WRONG_VOXEL_SIZE,
                TEST_MASK,
                TEST_DATA_DIR,
            )

        with self.assertRaises(
            UnequalSpacingError, msg="间距不等应引发特定错误"
        ):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE_UNEQUAL_SPACING,
                TEST_MASK,
                TEST_DATA_DIR,
            )

        # 测试搜索参数的错误处理
        for param in ["search_x", "search_y", "search_z"]:
            with self.assertRaises(
                ValueError, msg="搜索中的无效起始索引应引发ValueError"
            ):
                TMJob(
                    "0",
                    10,
                    TEST_TOMOGRAM,
                    TEST_TEMPLATE,
                    TEST_MASK,
                    TEST_DATA_DIR,
                    voxel_size=1.0,
                    **{param: [-10, 100]},
                )
            with self.assertRaises(
                ValueError, msg="搜索中的无效起始索引应引发ValueError"
            ):
                TMJob(
                    "0",
                    10,
                    TEST_TOMOGRAM,
                    TEST_TEMPLATE,
                    TEST_MASK,
                    TEST_DATA_DIR,
                    voxel_size=1.0,
                    **{param: [110, 130]},
                )
            with self.assertRaises(
                ValueError, msg="搜索中的无效结束索引应引发ValueError"
            ):
                TMJob(
                    "0",
                    10,
                    TEST_TOMOGRAM,
                    TEST_TEMPLATE,
                    TEST_MASK,
                    TEST_DATA_DIR,
                    voxel_size=1.0,
                    **{param: [0, 120]},
                )
        # 测试错误的角度输入
        with self.assertRaisesRegex(TMJobError, "无效的角度搜索"):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE,
                TEST_MASK,
                TEST_DATA_DIR,
                angle_increment="1.2.3",
                voxel_size=1.0,
            )

        # 测试损坏的模板掩码
        with self.assertRaisesRegex(ValueError, str(TEST_BROKEN_TOMOGRAM_MASK).replace("\\", "/")):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE,
                TEST_MASK,
                TEST_DATA_DIR,
                angle_increment=ANGULAR_SEARCH,
                voxel_size=1.0,
                tomogram_mask=TEST_BROKEN_TOMOGRAM_MASK,
            )
        # 测试大小错误的模板掩码
        with self.assertRaisesRegex(ValueError, str(TOMO_SHAPE)):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE,
                TEST_MASK,
                TEST_DATA_DIR,
                angle_increment=ANGULAR_SEARCH,
                voxel_size=1.0,
                tomogram_mask=TEST_WRONG_SIZE_TOMO_MASK,
            )

    def test_tm_job_copy(self):
        """
        测试TMJob类的复制功能。
        验证复制后的对象是否为新对象，以及断层图像形状是否正确。
        """
        # 复制TMJob实例
        copy = self.job.copy()
        # 验证复制后的对象与原对象不是同一个对象
        self.assertIsNot(
            self.job, copy, msg="复制作业应创建一个新对象。"
        )
        # 验证复制后的对象的断层图像形状是否与原对象相同
        self.assertEqual(
            TOMO_SHAPE,
            copy.tomo_shape,
            msg="作业中的断层图像形状不正确，可能是转置问题？",
        )

    def test_tm_job_weighting_options(self):
        """
        测试TMJob类的各种加权选项。
        包括低通滤波、高通滤波、剂量累积、CTF校正、倾斜加权、频谱白化等选项的组合测试。
        """
        # 运行包含所有加权选项的作业
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            low_pass=10,
            high_pass=100,
            dose_accumulation=ACCUMULATED_DOSE,
            ctf_data=CTF_PARAMS,
            tilt_angles=TILT_ANGLES,
            whiten_spectrum=True,
            tilt_weighting=True,
            defocus_handedness=1,
        )
        # 启动作业并获取分数和角度
        score, angle = job.start_job(0, return_volumes=True)
        # 验证作业结果的形状是否与断层图像形状相同
        self.assertEqual(
            score.shape, job.tomo_shape, msg="包含所有选项的TMJob失败"
        )

        # 运行仅包含倾斜加权的作业
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            dose_accumulation=ACCUMULATED_DOSE,
            ctf_data=CTF_PARAMS,
            tilt_angles=TILT_ANGLES,
            tilt_weighting=True,
        )
        score, angle = job.start_job(0, return_volumes=True)
        self.assertEqual(
            score.shape, job.tomo_shape, msg="仅创建楔形的TMJob失败"
        )

        # 运行仅包含带通滤波的作业
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            low_pass=10,
            high_pass=100,
        )
        score, angle = job.start_job(0, return_volumes=True)
        self.assertEqual(
            score.shape, job.tomo_shape, msg="仅带通滤波的TMJob失败"
        )

        # 运行仅包含频谱白化的作业
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            whiten_spectrum=True,
        )
        score, angle = job.start_job(0, return_volumes=True)
        self.assertEqual(
            score.shape, job.tomo_shape, msg="仅白化滤波器的TMJob失败"
        )

        # 加载之前作业的白化滤波器
        whitening_filter = np.load(TEST_WHITENING_FILTER)
        # 运行一个修改搜索区域的作业
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            whiten_spectrum=True,
            search_y=[10, 90],
        )
        # 加载新的白化滤波器
        new_whitening_filter = np.load(TEST_WHITENING_FILTER)
        # 验证修改搜索区域后白化滤波器的形状是否发生变化
        self.assertNotEqual(
            whitening_filter.shape,
            new_whitening_filter.shape,
            msg="在沿最大维度缩小搜索区域后，白化滤波器的采样点数应减少",
        )
        # 验证新的白化滤波器的形状是否符合预期
        self.assertEqual(
            new_whitening_filter.shape,
            (max(job.search_size) // 2 + 1,),
            msg="白化滤波器的大小不符合预期，应为 (x // 2) + 1，其中x是搜索框的最大维度。",
        )

        # 本文件中的其他运行测试了不包含这些加权选项的TMJob

    def test_load_json_to_tmjob(self):
        """
        测试从JSON文件加载TMJob实例的功能。
        验证基本加载、防止白化滤波器重新计算、向后兼容性等功能。
        """
        # 从JSON文件加载TMJob实例
        job = load_json_to_tmjob(TEST_JOB_JSON)
        # 验证加载的对象是否为TMJob实例
        self.assertIsInstance(
            job, TMJob, msg="TMJob无法从磁盘正确加载。"
        )

        # 测试加载作业并防止白化滤波器重新计算
        with self.assertNoLogs(level="INFO"):
            _ = load_json_to_tmjob(TEST_JOB_JSON_WHITENING, load_for_extraction=True)
        with self.assertLogs(level="INFO") as cm:
            _ = load_json_to_tmjob(TEST_JOB_JSON_WHITENING, load_for_extraction=False)
        # 验证日志中是否包含白化滤波器估计的信息
        self.assertIn("Estimating whitening filter...", "".join(cm.output))

        # 将当前作业转换为旧版本（0.6.0）的作业，并添加CTF参数
        job.pytom_tm_version_number = "0.6.0"
        job.ctf_data = []
        for ctf in CTF_PARAMS:
            job.ctf_data.append(ctf.copy())
            del job.ctf_data[-1]["phase_shift_deg"]
        # 将旧版本作业信息保存为JSON文件
        job.write_to_json(TEST_JOB_OLD_VERSION)

        # 测试旧版本作业的向后兼容性
        job = load_json_to_tmjob(TEST_JOB_OLD_VERSION)
        # 验证CTF参数中的相位偏移是否正确恢复
        self.assertEqual(job.ctf_data[0]["phase_shift_deg"], 0.0)

    def test_custom_angular_search(self):
        """
        测试自定义角度搜索的功能。
        验证传递自定义角度搜索文件到TMJob实例的功能，以及使用自定义角度文件进行提取的功能。
        """
        with TemporaryDirectory() as data_dir:
            data_dir = pathlib.Path(data_dir)
            # 创建一个使用自定义角度搜索文件的TMJob实例
            job = TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE,
                TEST_MASK,
                data_dir,
                angle_increment=TEST_CUSTOM_ANGULAR_SEARCH,
                voxel_size=1.0,
            )
            # 验证自定义角度搜索文件是否正确传递
            self.assertEqual(
                job.rotation_file,
                TEST_CUSTOM_ANGULAR_SEARCH,
                msg="将自定义角度搜索文件传递给TMJob失败。",
            )

            # 测试使用自定义角度文件进行提取的功能
            scores, angles = job.start_job(0, return_volumes=True)
            # 将分数保存为MRC文件
            write_mrc(data_dir / "tomogram_scores.mrc", scores, job.voxel_size)
            # 将角度保存为MRC文件
            write_mrc(data_dir / "tomogram_angles.mrc", angles, job.voxel_size)
            # 提取粒子
            df, scores = extract_particles(
                job, 100, particle_diameter=10, create_plot=False
            )
            # 验证提取的粒子数量是否不为零
            self.assertNotEqual(
                len(scores), 0, msg="这里期望得到一些注释。"
            )

    def test_tm_job_split_volume(self):
        """
        测试TMJob类的体积分割功能。
        验证将体积分割为小于模板的小盒子不会引发错误，以及分割后作业的结果和统计信息是否正确。
        """
        # 分割体积搜索，验证分割为小于模板的小盒子不会引发错误
        _ = self.job.split_volume_search((10, 3, 2))
        # 重置子作业列表
        self.job.sub_jobs = []
        # 验证请求的分割数超过像素数时，作业数量是否等于像素数
        with self.assertWarnsRegex(RuntimeWarning, "More splits than pixels"):
            self.job.split_volume_search((TOMO_SHAPE[0] + 42, 1, 1))
        self.assertEqual(len(self.job.sub_jobs), TOMO_SHAPE[0])
        # 重置子作业列表
        self.job.sub_jobs = []
        # 验证负分割数是否会引发错误
        with self.assertRaisesRegex(RuntimeError, "splits=-42"):
            self.job.split_volume_search((-42, 1, 1))
        # 分割体积搜索
        sub_jobs = self.job.split_volume_search((2, 3, 2))
        stats = []
        for x in sub_jobs:
            # 启动子作业
            stats.append(x.start_job(0))
            # 获取子作业的分数文件路径
            job_scores = TEST_DATA_DIR.joinpath(f"tomogram_scores_{x.job_key}.mrc")
            # 获取子作业的角度文件路径
            job_angles = TEST_DATA_DIR.joinpath(f"tomogram_angles_{x.job_key}.mrc")
            # 验证子作业的输出文件是否存在
            self.assertTrue(
                job_scores.exists(), msg="作业的预期输出不存在。"
            )
            self.assertTrue(
                job_angles.exists(), msg="作业的预期输出不存在。"
            )
        # 合并子作业的结果
        score, angle = self.job.merge_sub_jobs(stats)
        # 获取分数图中最大值的索引
        ind = np.unravel_index(score.argmax(), score.shape)

        # 验证分数图的最大值是否符合预期
        self.assertTrue(score.max() > 0.931, msg="lcc最大值低于预期")
        # 验证角度图中最大值对应的角度ID是否正确
        self.assertEqual(ANGLE_ID, angle[ind])
        # 验证分数图中最大值的索引是否与模板位置一致
        self.assertSequenceEqual(LOCATION, ind)

        # 由于交叉相关函数在边界区域的定义不明确，分割维度的边缘区域可能存在小差异
        ok_region = slice(TEMPLATE_SIZE // 2, -TEMPLATE_SIZE // 2)
        # 计算分割后分数图与参考分数图的差异
        score_diff = np.abs(
            score[ok_region, ok_region, ok_region]
            - read_mrc(TEST_SCORES)[ok_region, ok_region, ok_region]
        ).sum()

        # 验证分数图的差异是否在允许范围内
        self.assertAlmostEqual(
            score_diff, 0, places=1, msg="分数差异不应大于0.01"
        )
        # 由于FFT填充可能存在一些竞争条件，角度图可能存在差异
        # angle_diff = np.abs(
        #    angle[ok_region, ok_region, ok_region] -
        #    read_mrc(TEST_ANGLES)[ok_region, ok_region, ok_region]
        #    ).sum()

        # self.assertAlmostEqual(angle_diff, 0, places=1,
        #    msg='角度差异不应改变')

        # 获取分割前后的搜索统计信息
        split_stats = self.job.job_stats
        reference_stats = load_json_to_tmjob(TEST_JOB_JSON).job_stats
        # 验证分割后搜索空间是否保持不变
        self.assertEqual(
            split_stats["search_space"],
            reference_stats["search_space"],
            msg="子体积分割后搜索空间应保持不变。",
        )
        # 验证分割后标准差是否几乎相同
        self.assertAlmostEqual(
            split_stats["std"],
            reference_stats["std"],
            places=3,
            msg="子体积分割后的模板匹配标准差应几乎相同。",
        )

    def test_splitting_with_tomogram_mask(self):
        """
        测试使用断层图像掩码进行分割的功能。
        验证使用掩码分割后子作业的数量是否减少。
        """
        # 复制当前作业
        job = self.job.copy()
        # 设置断层图像掩码
        job.tomogram_mask = TEST_EXTRACTION_MASK_INSIDE
        # 使用掩码进行体积分割
        job.split_volume_search((10, 10, 10))
        # 验证分割后子作业的数量是否减少
        self.assertLess(len(job.sub_jobs), 10 * 10 * 10)

    def test_splitting_with_offsets(self):
        """
        测试带偏移量的分割功能。
        验证子作业的偏移量是否正确，以及最后一个子作业的起始位置和大小是否与主作业的搜索大小一致。
        """
        # 创建一个带搜索范围的TMJob实例
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=ANGULAR_SEARCH,
            voxel_size=1.0,
            search_x=[9, 90],
            search_y=[25, 102],
            search_z=[19, 54],
        )
        # 沿每个维度分割作业，并获取最后一个子作业
        last_sub_job = job.split_volume_search((2, 3, 2))[-1]
        # 计算最后一个子作业的最终大小
        final_size = [
            i + j for i, j in zip(last_sub_job.whole_start, last_sub_job.sub_step)
        ]
        # 验证最后一个子作业的起始位置和大小是否与主作业的搜索大小一致
        self.assertEqual(
            final_size,
            job.search_size,
            msg="最后一个子作业的起始位置加上其大小应等于主作业的搜索大小",
        )

    def test_tm_job_split_angles(self):
        """
        测试TMJob类的角度分割功能。
        验证分割角度搜索后作业的结果和统计信息是否正确。
        """
        # 分割旋转搜索
        sub_jobs = self.job.split_rotation_search(3)
        stats = []
        for x in sub_jobs:
            # 启动子作业
            stats.append(x.start_job(0))
            # 获取子作业的分数文件路径
            job_scores = TEST_DATA_DIR.joinpath(f"tomogram_scores_{x.job_key}.mrc")
            # 获取子作业的角度文件路径
            job_angles = TEST_DATA_DIR.joinpath(f"tomogram_angles_{x.job_key}.mrc")
            # 验证子作业的输出文件是否存在
            self.assertTrue(
                job_scores.exists(), msg="作业的预期输出不存在。"
            )
            self.assertTrue(
                job_angles.exists(), msg="作业的预期输出不存在。"
            )
        # 合并子作业的结果
        score, angle = self.job.merge_sub_jobs(stats)
        # 获取分数图中最大值的索引
        ind = np.unravel_index(score.argmax(), score.shape)

        # 验证分数图的最大值是否符合预期
        self.assertTrue(score.max() > 0.931, msg="lcc最大值低于预期")
        # 验证角度图中最大值对应的角度ID是否正确
        self.assertEqual(ANGLE_ID, angle[ind])
        # 验证分数图中最大值的索引是否与模板位置一致
        self.assertSequenceEqual(LOCATION, ind)

        # 验证分割旋转搜索后的分数图是否与参考分数图相同
        self.assertTrue(
            np.abs(score - read_mrc(TEST_SCORES)).sum() == 0,
            msg="分割旋转搜索结果应相同",
        )
        # 验证分割旋转搜索后的角度图是否与参考角度图相同
        self.assertTrue(
            np.abs(angle - read_mrc(TEST_ANGLES)).sum() == 0,
            msg="分割旋转搜索结果应相同",
        )

        # 获取分割前后的搜索统计信息
        split_stats = self.job.job_stats
        reference_stats = load_json_to_tmjob(TEST_JOB_JSON).job_stats
        # 验证分割后搜索空间是否保持不变
        self.assertEqual(
            split_stats["search_space"],
            reference_stats["search_space"],
            msg="角度搜索分割后搜索空间应保持不变。",
        )
        # 验证分割后标准差是否几乎相同
        self.assertAlmostEqual(
            split_stats["std"],
            reference_stats["std"],
            places=6,
            msg="角度搜索分割后的模板匹配标准差应几乎相同。",
        )

    def test_tm_job_half_precision(self):
        """
        测试TMJob类的半精度输出功能。
        验证作业输出的分数和角度的数据类型是否符合预期。
        """
        # 创建一个使用半精度输出的TMJob实例
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=ANGULAR_SEARCH,
            voxel_size=1.0,
            output_dtype=np.float16,
        )
        # 启动作业并获取分数和角度
        s, a = job.start_job(0, return_volumes=True)
        # 验证分数的数据类型是否为半精度浮点数
        self.assertEqual(s.dtype, np.float16)
        # 验证角度的数据类型是否为单精度浮点数
        self.assertEqual(a.dtype, np.float32)

    def test_extractions(self):
        """
        测试粒子提取功能。
        验证不同条件下的粒子提取结果，包括提取掩码的使用、错误处理、Relion 5 兼容模式等。
        """
        # 修改作业的断层图像ID
        self.job.tomo_id = "rec_" + self.job.tomo_id
        # 启动作业并获取分数和角度
        scores, angles = self.job.start_job(0, return_volumes=True)
        # 将分数保存为MRC文件
        write_mrc(
            TEST_DATA_DIR.joinpath(f"{self.job.tomo_id}_scores.mrc"),
            scores,
            self.job.voxel_size,
        )
        # 将角度保存为MRC文件
        write_mrc(
            TEST_DATA_DIR.joinpath(f"{self.job.tomo_id}_angles.mrc"),
            angles,
            self.job.voxel_size,
        )

        # 提取粒子
        df, scores = extract_particles(
            self.job, 100, particle_diameter=10, create_plot=False
        )
        # 验证提取的粒子数量是否不为零
        self.assertNotEqual(
            len(scores), 0, msg="这里期望得到一些注释。"
        )

        # 测试粒子直径参数错误的情况
        with self.assertRaisesRegex(ValueError, "particle diameter"):
            _ = extract_particles(self.job, 100, create_plot=False)
        # 复制当前作业
        job = self.job.copy()
        # 设置粒子直径
        job.particle_diameter = 10
        # 测试未提供粒子直径时的日志记录
        with self.assertLogs(level="INFO") as cm:
            _ = extract_particles(job, 100, create_plot=False)
        # 验证日志中是否包含未提供粒子直径的信息
        self.assertIn("No particle diameter was provided,", "".join(cm.output))

        # 测试Relion 5兼容模式下的粒子提取
        df_rel5, scores = extract_particles(
            self.job, 100, particle_diameter=10, create_plot=False, relion5_compat=True
        )
        # 验证Relion 5兼容模式下的DataFrame是否包含预期的列
        for column in (
            "rlnCenteredCoordinateXAngst",
            "rlnCenteredCoordinateYAngst",
            "rlnCenteredCoordinateZAngst",
            "rlnTomoName",
            "rlnTomoTiltSeriesPixelSize",
        ):
            self.assertTrue(
                column in df_rel5.columns,
                msg=f"预期在Relion 5 DataFrame中包含 {column}。",
            )
        # 计算中心位置
        centered_location = (
            np.array(LOCATION) - (np.array(TOMO_SHAPE) / 2)
        ) * self.job.voxel_size
        # 计算Relion 5兼容模式下提取的位置与中心位置的差异
        diff = np.abs(np.array(df_rel5.iloc[0, 0:3]) - centered_location).sum()
        # 验证Relion 5兼容模式下提取的位置是否为中心位置
        self.assertEqual(
            diff,
            0,
            msg="Relion 5兼容模式应返回对象的中心位置",
        )
        # 验证Relion 5兼容模式下的断层图像名称是否不包含前缀
        self.assertNotIn("rec_", df_rel5["rlnTomoName"][0])

        # 测试使用不覆盖粒子的提取掩码的情况
        df, scores = extract_particles(
            self.job,
            5,
            100,
            tomogram_mask_path=TEST_EXTRACTION_MASK_OUTSIDE,
            create_plot=False,
        )
        # 验证使用不覆盖粒子的提取掩码后提取的粒子数量是否为零
        self.assertEqual(
            len(scores),
            0,
            msg="应用不覆盖对象的掩码后，返回列表的长度应为0。",
        )
        # 测试从作业中获取提取掩码的情况
        job = self.job.copy()
        job.tomogram_mask = TEST_EXTRACTION_MASK_OUTSIDE
        df, scores = extract_particles(
            job,
            100,
            particle_diameter=10,
            create_plot=False,
        )
        # 验证使用不覆盖粒子的提取掩码后提取的粒子数量是否为零
        self.assertEqual(
            len(scores),
            0,
            msg="应用不覆盖对象的掩码后，返回列表的长度应为0。",
        )
        # 测试忽略提取掩码的情况
        with self.assertLogs(level="WARNING") as cm:
            df, scores = extract_particles(
                job,
                100,
                particle_diameter=10,
                tomogram_mask_path=TEST_EXTRACTION_MASK_OUTSIDE,
                create_plot=False,
                ignore_tomogram_mask=True,
            )
        # 验证日志中是否包含忽略提取掩码的信息
        for o in cm.output:
            if "Ignoring tomogram mask" in o:
                break
        else:
            # 如果未找到忽略提取掩码的信息，测试失败
            self.fail("预期的警告未记录")
        # 验证忽略提取掩码后提取的粒子数量是否不为零
        self.assertNotEqual(
            len(scores),
            0,
            msg="如果忽略所有断层图像掩码，期望得到一些注释",
        )

        # 测试使用覆盖粒子的提取掩码的情况
        df, scores = extract_particles(
            job,
            100,
            particle_diameter=5,
            tomogram_mask_path=TEST_EXTRACTION_MASK_INSIDE,
            create_plot=False,
        )
        # 验证使用覆盖粒子的提取掩码后提取的粒子数量是否不为零
        self.assertNotEqual(
            len(scores),
            0,
            msg="使用覆盖对象的提取掩码时，期望检测到粒子。",
        )

        # 测试使用大小错误的提取掩码的情况
        with self.assertRaisesRegex(ValueError, str(TOMO_SHAPE)):
            _, _ = extract_particles(
                job,
                100,
                particle_diameter=5,
                tomogram_mask_path=TEST_WRONG_SIZE_TOMO_MASK,
                create_plot=False,
            )

        # 测试作业中附带大小错误的提取掩码的情况
        job = self.job.copy()
        job.tomogram_mask = TEST_WRONG_SIZE_TOMO_MASK
        with self.assertRaisesRegex(ValueError, str(TOMO_SHAPE)):
            _, _ = extract_particles(
                job,
                5,
                100,
                create_plot=False,
            )

        # 测试使用Tophat滤波器和绘图功能的粒子提取
        df, scores = extract_particles(
            job,
            100,
            particle_diameter=5,
            tomogram_mask_path=TEST_EXTRACTION_MASK_INSIDE,
            create_plot=True,
            tophat_filter=True,
        )
        # 验证使用Tophat滤波器和绘图功能后提取的粒子数量是否不为零
        self.assertNotEqual(
            len(scores),
            0,
            msg="使用覆盖对象的提取掩码时，期望检测到粒子。",
        )
        # 不检查绘图文件，因为可能由于没有绘图功能而跳过

    def test_get_defocus_offsets(self):
        """
        测试获取散焦偏移的功能。
        验证计算得到的散焦偏移列表的长度是否与倾斜角度列表的长度一致，以及反转手性后的偏移是否符合预期。
        """
        # 定义倾斜角度列表
        tilt_angles = list(range(-51, 54, 3))
        # 计算X方向的偏移量（微米）
        x_offset_um = 200 * 13.79 * 1e-4
        # 计算Z方向的偏移量（微米）
        z_offset_um = 100 * 13.79 * 1e-4
        # 计算散焦偏移
        defocus_offsets = get_defocus_offsets(x_offset_um, z_offset_um, tilt_angles)
        # 验证散焦偏移列表的长度是否与倾斜角度列表的长度一致
        self.assertEqual(
            len(defocus_offsets),
            len(tilt_angles),
            msg="get_defocus_offsets返回的列表长度应与倾斜角度数量相同",
        )
        # 计算反转手性后的散焦偏移
        defocus_offsets_inverted = get_defocus_offsets(
            x_offset_um, z_offset_um, tilt_angles, invert_handedness=True
        )
        # 验证反转手性后只有一个偏移量相同
        self.assertTrue(
            (defocus_offsets == defocus_offsets_inverted).sum() == 1,
            msg="反转手性后应有一个相同的偏移量",
        )
