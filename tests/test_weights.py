# 导入NumPy库，用于数值计算
import numpy as np
# 导入unittest模块，用于编写单元测试
import unittest
# 从pytom_tm.weights模块导入多个函数，这些函数用于创建不同类型的权重和处理数据
from pytom_tm.weights import (
    create_wedge,
    create_ctf,
    create_gaussian_band_pass,
    radial_reduced_grid,
    radial_average,
    power_spectrum_profile,
    profile_to_weighting,
)
# 从testing_utils模块导入测试所需的常量
from testing_utils import TILT_ANGLES, ACCUMULATED_DOSE, CTF_PARAMS

# 定义一个测试类，继承自unittest.TestCase
class TestWeights(unittest.TestCase):
    def setUp(self):
        """
        测试前的初始化操作，设置一些常用的参数和形状
        """
        # 定义偶数形状的体积
        self.volume_shape_even = (10, 10, 10)
        # 定义奇数形状的体积
        self.volume_shape_uneven = (11, 11, 11)
        # 定义不规则形状的体积
        self.volume_shape_irregular = (7, 12, 6)
        # 定义体素大小
        self.voxel_size = 3.34
        # 定义低通滤波器的截止频率
        self.low_pass = 10
        # 定义高通滤波器的截止频率
        self.high_pass = 50

        # 定义偶数形状体积的3D缩减后的形状
        self.reduced_even_shape_3d = (10, 10, 6)
        # 定义偶数形状体积的2D缩减后的形状
        self.reduced_even_shape_2d = (10, 6)
        # 定义奇数形状体积的3D缩减后的形状
        self.reduced_uneven_shape_3d = (11, 11, 6)
        # 定义奇数形状体积的2D缩减后的形状
        self.reduced_uneven_shape_2d = (11, 6)
        # 定义不规则形状体积的3D缩减后的形状
        self.reduced_irregular_shape_3d = (7, 12, 6 // 2 + 1)
        # 定义不规则形状体积的2D缩减后的形状
        self.reduced_irregular_shape_2d = (7, 12 // 2 + 1)

    def test_radial_reduced_grid(self):
        """
        测试radial_reduced_grid函数，验证其在不同输入下的行为
        """
        # 测试当输入形状不是2D或3D时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Radial reduced grid should raise ValueError if the shape is "
            "not 2- or 3-dimensional.",
        ):
            radial_reduced_grid((5,))
        # 测试当输入形状不是2D或3D时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Radial reduced grid should raise ValueError if the shape is "
            "not 2- or 3-dimensional.",
        ):
            radial_reduced_grid((5,) * 4)

        # 验证3D径向缩减网格的形状是否正确
        self.assertEqual(
            radial_reduced_grid(self.volume_shape_even).shape,
            self.reduced_even_shape_3d,
            msg="3D radial reduced grid does not have the correct shape",
        )
        # 验证2D径向缩减网格的形状是否正确
        self.assertEqual(
            radial_reduced_grid(self.volume_shape_even[:2]).shape,
            self.reduced_even_shape_2d,
            msg="2D radial reduced grid does not have the correct shape",
        )

    def test_band_pass(self):
        """
        测试create_gaussian_band_pass函数，验证其在不同输入下的行为
        """
        # 测试当低通和高通截止频率都为None时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Bandpass should raise ValueError if both low and high pass are None",
        ):
            create_gaussian_band_pass(
                self.volume_shape_even, self.voxel_size, None, None
            )
        # 测试当低通截止频率大于高通截止频率时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Bandpass should raise ValueError if low pass resolution > high pass "
            "resolution",
        ):
            create_gaussian_band_pass(self.volume_shape_even, self.voxel_size, 50, 10)

        # 创建带通滤波器
        band_pass = create_gaussian_band_pass(
            self.volume_shape_even, self.voxel_size, self.low_pass, self.high_pass
        )
        # 创建低通滤波器
        low_pass = create_gaussian_band_pass(
            self.volume_shape_even, self.voxel_size, low_pass=self.low_pass
        )
        # 创建高通滤波器
        high_pass = create_gaussian_band_pass(
            self.volume_shape_even, self.voxel_size, high_pass=self.high_pass
        )

        # 验证带通滤波器的形状是否正确
        self.assertEqual(
            band_pass.shape,
            self.reduced_even_shape_3d,
            msg="Bandpass filter does not have expected output shape",
        )
        # 验证带通滤波器的数据类型是否正确
        self.assertEqual(
            band_pass.dtype,
            np.float64,
            msg="Bandpass filter does not have expected dtype",
        )
        # 验证低通滤波器的形状是否正确
        self.assertEqual(
            low_pass.shape,
            self.reduced_even_shape_3d,
            msg="Low-pass filter does not have expected output shape",
        )
        # 验证低通滤波器的数据类型是否正确
        self.assertEqual(
            low_pass.dtype,
            np.float64,
            msg="Low-pass filter does not have expected dtype",
        )
        # 验证高通滤波器的形状是否正确
        self.assertEqual(
            high_pass.shape,
            self.reduced_even_shape_3d,
            msg="High-pass filter does not have expected output shape",
        )
        # 验证高通滤波器的数据类型是否正确
        self.assertEqual(
            high_pass.dtype,
            np.float64,
            msg="High-pass filter does not have expected dtype",
        )

        # 验证带通滤波器和低通滤波器是否不同
        self.assertTrue(
            np.sum((band_pass != low_pass) * 1) != 0,
            msg="Band-pass and low-pass should be different",
        )
        # 验证带通滤波器和高通滤波器是否不同
        self.assertTrue(
            np.sum((band_pass != high_pass) * 1) != 0,
            msg="Band-pass and low-pass filter should be different",
        )
        # 验证低通滤波器和高通滤波器是否不同
        self.assertTrue(
            np.sum((low_pass != high_pass) * 1) != 0,
            msg="Low-pass and high-pass filter should be different",
        )

    def test_create_wedge(self):
        """
        测试create_wedge函数，验证其在不同输入下的行为
        """
        # 测试当倾斜角度列表元素少于两个时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Create wedge should raise ValueError if tilt_angles list does not "
            "contain at least two values",
        ):
            create_wedge(self.volume_shape_even, [1.0], 1.0)
        # 测试当倾斜角度输入不是列表时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Create wedge should raise ValueError if tilt_angles input is not a "
            "list",
        ):
            create_wedge(self.volume_shape_even, 1.0, 1.0)
        # 测试当体素大小小于等于0时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Create wedge should raise ValueError if voxel_size is smaller or "
            "equal to 0",
        ):
            create_wedge(self.volume_shape_even, TILT_ANGLES, 0.0)
        # 测试当截止半径小于等于0时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Create wedge should raise ValueError if cut_off_radius is smaller or "
            "equal to 0",
        ):
            create_wedge(self.volume_shape_even, TILT_ANGLES, 1.0, cut_off_radius=0.0)

        # 创建结构化楔形滤波器
        structured_wedge = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            1.0,
            tilt_weighting=True,
            ctf_params_per_tilt=CTF_PARAMS,
        )
        # 创建对称楔形滤波器
        symmetric_wedge = create_wedge(
            self.volume_shape_even,
            [TILT_ANGLES[0], TILT_ANGLES[-1]],
            1.0,
            tilt_weighting=False,
            ctf_params_per_tilt=CTF_PARAMS,
        )
        # 创建非对称楔形滤波器
        asymmetric_wedge = create_wedge(
            self.volume_shape_even,
            [TILT_ANGLES[0], TILT_ANGLES[-2]],
            1.0,
            tilt_weighting=False,
            ctf_params_per_tilt=CTF_PARAMS,
        )

        # 验证结构化楔形滤波器的形状是否正确
        self.assertEqual(
            structured_wedge.shape,
            self.reduced_even_shape_3d,
            msg="Structured wedge does not have expected output shape",
        )
        # 验证结构化楔形滤波器的数据类型是否正确
        self.assertEqual(
            structured_wedge.dtype,
            np.float32,
            msg="Structured wedge does not have expected dtype",
        )

        # 验证对称楔形滤波器的形状是否正确
        self.assertEqual(
            symmetric_wedge.shape,
            self.reduced_even_shape_3d,
            msg="Symmetric wedge does not have expected output shape",
        )
        # 验证对称楔形滤波器的数据类型是否正确
        self.assertEqual(
            symmetric_wedge.dtype,
            np.float32,
            msg="Symmetric wedge does not have expected dtype",
        )

        # 验证非对称楔形滤波器的形状是否正确
        self.assertEqual(
            asymmetric_wedge.shape,
            self.reduced_even_shape_3d,
            msg="Asymmetric wedge does not have expected output shape",
        )
        # 验证非对称楔形滤波器的数据类型是否正确
        self.assertEqual(
            asymmetric_wedge.dtype,
            np.float32,
            msg="Asymmetric wedge does not have expected dtype",
        )

        # 验证对称楔形滤波器和非对称楔形滤波器是否不同
        self.assertTrue(
            np.sum((symmetric_wedge != asymmetric_wedge) * 1) != 0,
            msg="Symmetric and asymmetric wedge should be different!",
        )

        # 创建带有带通滤波器的结构化楔形滤波器
        structured_wedge = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            self.voxel_size,
            tilt_weighting=True,
            cut_off_radius=1.0,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
        )
        # 验证带有带通滤波器的结构化楔形滤波器的形状是否正确
        self.assertEqual(
            structured_wedge.shape,
            self.reduced_even_shape_3d,
            msg="Wedge with band-pass does not have expected output shape",
        )
        # 验证带有带通滤波器的结构化楔形滤波器的数据类型是否正确
        self.assertEqual(
            structured_wedge.dtype,
            np.float32,
            msg="Wedge with band-pass does not have expected dtype",
        )

        # 测试不同形状体积的楔形滤波器
        weights = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            self.voxel_size * 3,
            tilt_weighting=True,
            low_pass=40,
            accumulated_dose_per_tilt=ACCUMULATED_DOSE,
            ctf_params_per_tilt=CTF_PARAMS,
        )
        # 验证3D楔形滤波器的形状是否正确
        self.assertEqual(
            weights.shape,
            self.reduced_even_shape_3d,
            msg="3D CTF does not have the correct reduced fourier shape.",
        )
        weights = create_wedge(
            self.volume_shape_uneven,
            TILT_ANGLES,
            self.voxel_size * 3,
            tilt_weighting=True,
            low_pass=40,
            accumulated_dose_per_tilt=ACCUMULATED_DOSE,
            ctf_params_per_tilt=CTF_PARAMS,
        )
        # 验证3D楔形滤波器的形状是否正确
        self.assertEqual(
            weights.shape,
            self.reduced_uneven_shape_3d,
            msg="3D CTF does not have the correct reduced fourier shape.",
        )

        # 测试倾斜加权楔形滤波器的参数灵活性
        weights = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            self.voxel_size * 3,
            tilt_weighting=True,
            low_pass=self.low_pass,
            accumulated_dose_per_tilt=None,
            ctf_params_per_tilt=None,
        )
        # 验证倾斜加权楔形滤波器在没有散焦和剂量信息时是否正常工作
        self.assertEqual(
            weights.shape,
            self.reduced_even_shape_3d,
            msg="Tilt weighted wedge should also work without defocus and dose info.",
        )
        weights = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            self.voxel_size * 3,
            tilt_weighting=True,
            low_pass=self.low_pass,
            accumulated_dose_per_tilt=None,
            ctf_params_per_tilt=CTF_PARAMS[:1],
        )
        # 验证倾斜加权楔形滤波器在使用单个散焦信息时是否正常工作
        self.assertEqual(
            weights.shape,
            self.reduced_even_shape_3d,
            msg="Tilt weighted wedge should work with single defocus.",
        )

    def test_ctf(self):
        """
        测试create_ctf函数，验证其在不同输入下的行为
        """
        # 创建原始CTF滤波器
        ctf_raw = create_ctf(
            self.volume_shape_even, self.voxel_size * 1e-10, 3e-6, 0.08, 300e3, 2.7e-3
        )
        # 创建截断后的CTF滤波器
        ctf_cut = create_ctf(
            self.volume_shape_even,
            self.voxel_size * 1e-10,
            3e-6,
            0.08,
            300e3,
            2.7e-3,
            cut_after_first_zero=True,
        )
        # 验证原始CTF滤波器的形状是否正确
        self.assertEqual(
            ctf_raw.shape,
            self.reduced_even_shape_3d,
            msg="CTF does not have expected output shape",
        )
        # 验证原始CTF滤波器和截断后的CTF滤波器是否不同
        self.assertTrue(
            np.sum((ctf_raw != ctf_cut) * 1) != 0,
            msg="CTF should be different when cutting it off after the first zero "
            "crossing",
        )

    def test_radial_average(self):
        """
        测试radial_average函数，验证其在不同输入下的行为
        """
        # 定义二维数组的尺寸
        x, y = 100, 50
        # 测试当输入不是2D或3D数组时，函数是否抛出异常
        with self.assertRaises(
            ValueError,
            msg="Radial average should raise error if something other than 2d/3d "
            "array is provided.",
        ):
            radial_average(np.zeros(x))
        # 计算二维数组的径向平均值
        q, m = radial_average(np.zeros((x, y)))
        # 验证径向平均值的形状是否正确
        self.assertEqual(
            m.shape[0],
            x // 2 + 1,
            msg="Radial average shape should equal largest sampling dimension.",
        )
        # 计算不同二维数组的径向平均值
        q, m = radial_average(np.zeros((30, y)))
        # 验证径向平均值的形状是否正确
        self.assertEqual(
            m.shape[0],
            y,
            msg="Radial average shape should equal largest sampling dimension, "
            "considering Fourier reduced form.",
        )
        # 计算三维数组的径向平均值
        q, m = radial_average(np.zeros((20, x, y)))
        # 验证径向平均值的形状是否正确
        self.assertEqual(
            m.shape[0],
            x // 2 + 1,
            msg="Radial average shape should equal largest sampling dimension.",
        )
        # 计算不同三维数组的径向平均值
        q, m = radial_average(np.zeros((20, 30, y)))
        # 验证径向平均值的形状是否正确
        self.assertEqual(
            m.shape[0],
            y,
            msg="Radial average shape should equal largest sampling dimension, "
            "considering Fourier reduced form.",
        )

    def test_power_spectrum_profile(self):
        """
        测试power_spectrum_profile函数，验证其在不同输入下的行为
        """
        # 测试当输入图像不是2D或3D时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Power spectrum profile should raise ValueError if input image is "
            "not 2- or 3-dimensional.",
        ):
            power_spectrum_profile(np.zeros(5))
        # 测试当输入图像不是2D或3D时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Power spectrum profile should raise ValueError if input image is "
            "not 2- or 3-dimensional.",
        ):
            power_spectrum_profile(np.zeros((5,) * 4))
        # 计算不规则形状体积的功率谱轮廓
        profile = power_spectrum_profile(np.zeros(self.volume_shape_irregular))
        # 验证功率谱轮廓的形状是否正确
        self.assertEqual(
            profile.shape,
            (max(self.volume_shape_irregular) // 2 + 1,),
            msg="Power spectrum profile output shape should be a 1-dimensional array "
            "with length equal to max(input_shape) // 2 + 1, corresponding to largest "
            "sampling component in Fourier space.",
        )

    def test_profile_to_weighting(self):
        """
        测试profile_to_weighting函数，验证其在不同输入下的行为
        """
        # 测试当输入的轮廓不是一维数组时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Profile to weighting should raise a ValueError if the profile is not "
            "1-dimensional.",
        ):
            profile_to_weighting(np.zeros((5, 5)), (5, 5))
        # 测试当输出权重的形状不是2D或3D时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Profile to weighting should raise a ValueError if the output shape "
            "for the weighting is not 2- or 3-dimensional.",
        ):
            profile_to_weighting(np.zeros(5), (5,))
        # 测试当输出权重的形状不是2D或3D时，函数是否抛出ValueError异常
        with self.assertRaises(
            ValueError,
            msg="Profile to weighting should raise a ValueError if the output shape "
            "for the weighting is not 2- or 3-dimensional.",
        ):
            profile_to_weighting(np.zeros(5), (5,) * 4)

        # 计算不规则形状体积的功率谱轮廓
        profile = power_spectrum_profile(np.zeros(self.volume_shape_irregular))
        # 验证将轮廓转换为权重后的3D数组形状是否正确
        self.assertEqual(
            profile_to_weighting(profile, self.volume_shape_irregular).shape,
            self.reduced_irregular_shape_3d,
            msg="Profile to weighting should return 3D Fourier reduced array.",
        )
        # 验证将轮廓转换为权重后的2D数组形状是否正确
        self.assertEqual(
            profile_to_weighting(profile, self.volume_shape_irregular[:2]).shape,
            self.reduced_irregular_shape_2d,
            msg="Profile to weighting should return 2D Fourier reduced array.",
        )
