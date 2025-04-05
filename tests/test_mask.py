# 导入 unittest 模块，用于编写单元测试
import unittest
# 导入 voltools 库，用于处理 3D 体积数据
import voltools as vt
# 导入 cupy 库，用于 GPU 加速计算
import cupy as cp
# 从 pytom_tm.mask 模块导入 spherical_mask 函数，用于生成球形掩码
from pytom_tm.mask import spherical_mask
# 从 pytom_tm.angles 模块导入 angle_to_angle_list 函数，用于生成角度列表
from pytom_tm.angles import angle_to_angle_list
# 从 pytom_tm.correlation 模块导入 normalised_cross_correlation 函数，用于计算归一化互相关
from pytom_tm.correlation import normalised_cross_correlation


class TestMask(unittest.TestCase):
    def setUp(self):
        """
        测试用例执行前的初始化操作。
        生成一个角度列表，用于后续的旋转测试。
        """
        # 调用 angle_to_angle_list 函数生成角度列表，角度差为 50.00
        self.angles = angle_to_angle_list(50.00)

    def test_rotational_invariance_even(self):
        """
        测试偶数尺寸掩码的旋转不变性。
        比较以中心旋转和非中心旋转的掩码与原始掩码的归一化互相关之和。
        """
        # 提示信息：测试偶数尺寸的掩码
        print("# TEST EVEN MASK")
        # 初始化两个列表，用于存储非中心旋转和中心旋转的归一化互相关值
        nxcc_offcenter, nxcc_centered = [], []

        # 生成一个偶数尺寸（12）的球形掩码，半径为 4，阈值为 0.5，并转换为 cupy 数组
        mask = cp.asarray(spherical_mask(12, 4, 0.5))
        # 创建一个与掩码形状相同的零数组，用于存储旋转后的掩码
        mask_rotated = cp.zeros_like(mask)

        # 创建一个静态体积对象，用于对掩码进行插值和旋转操作
        mask_texture = vt.StaticVolume(mask, interpolation="filt_bspline", device="gpu")

        # 遍历角度列表，进行非中心旋转操作
        for i in range(len(self.angles)):
            # 对掩码进行旋转操作，指定旋转角度、旋转单位、旋转顺序和旋转中心
            mask_texture.transform(
                rotation=self.angles[i],
                rotation_units="rad",
                rotation_order="rzxz",
                center=tuple([x // 2 for x in mask.shape]),
                output=mask_rotated,
            )
            # 计算旋转后的掩码与原始掩码的归一化互相关，并添加到非中心旋转列表中
            nxcc_offcenter.append(
                normalised_cross_correlation(mask, mask_rotated).get()
            )

        # 遍历角度列表，进行中心旋转操作
        for i in range(len(self.angles)):
            # 对掩码进行旋转操作，指定旋转角度、旋转单位和旋转顺序
            mask_texture.transform(
                rotation=self.angles[i],
                rotation_units="rad",
                rotation_order="rzxz",
                output=mask_rotated,
                # center=np.divide(np.subtract(mask.shape, 1), 2, dtype=np.float32),
            )
            # 计算旋转后的掩码与原始掩码的归一化互相关，并添加到中心旋转列表中
            nxcc_centered.append(normalised_cross_correlation(mask, mask_rotated).get())

        # 断言中心旋转的归一化互相关之和大于非中心旋转的归一化互相关之和
        self.assertTrue(
            sum(nxcc_centered) > sum(nxcc_offcenter),
            msg="Center of rotation for mask is incorrect.",
        )
        # 断言中心旋转的归一化互相关之和大于 99.27
        self.assertTrue(
            sum(nxcc_centered) > 99.27, msg="Precision of mask rotation is too low."
        )

    def test_rotational_invariance_uneven(self):
        """
        测试奇数尺寸掩码的旋转不变性。
        比较以中心旋转和非中心旋转的掩码与原始掩码的归一化互相关之和。
        """
        # 提示信息：测试奇数尺寸的掩码
        print("# TEST UNEVEN MASK")
        # 初始化两个列表，用于存储非中心旋转和中心旋转的归一化互相关值
        nxcc_offcenter, nxcc_centered = [], []

        # 生成一个奇数尺寸（13）的球形掩码，半径为 4，阈值为 0.5，并转换为 cupy 数组
        mask = cp.asarray(spherical_mask(13, 4, 0.5))
        # 创建一个与掩码形状相同的零数组，用于存储旋转后的掩码
        mask_rotated = cp.zeros_like(mask)

        # 创建一个静态体积对象，用于对掩码进行插值和旋转操作
        mask_texture = vt.StaticVolume(mask, interpolation="filt_bspline", device="gpu")

        # 遍历角度列表，进行非中心旋转操作
        for i in range(len(self.angles)):
            # 对掩码进行旋转操作，指定旋转角度、旋转单位、旋转顺序和旋转中心
            mask_texture.transform(
                rotation=self.angles[i],
                rotation_units="rad",
                rotation_order="rzxz",
                center=tuple([x // 2 for x in mask.shape]),
                output=mask_rotated,
            )
            # 计算旋转后的掩码与原始掩码的归一化互相关，并添加到非中心旋转列表中
            nxcc_offcenter.append(
                normalised_cross_correlation(mask, mask_rotated).get()
            )

        # 遍历角度列表，进行中心旋转操作
        for i in range(len(self.angles)):
            # 对掩码进行旋转操作，指定旋转角度、旋转单位和旋转顺序
            mask_texture.transform(
                rotation=self.angles[i],
                rotation_units="rad",
                rotation_order="rzxz",
                output=mask_rotated,
                # center=np.divide(np.subtract(mask.shape, 1), 2, dtype=np.float32),
            )
            # 计算旋转后的掩码与原始掩码的归一化互相关，并添加到中心旋转列表中
            nxcc_centered.append(normalised_cross_correlation(mask, mask_rotated).get())

        # 断言中心旋转和非中心旋转的归一化互相关之和在小数点后 4 位内近似相等
        self.assertAlmostEqual(
            sum(nxcc_centered),
            sum(nxcc_offcenter),
            places=4,
            msg="Center of rotation for mask is incorrect.",
        )
        # 断言中心旋转的归一化互相关之和大于 99.09
        self.assertTrue(
            sum(nxcc_centered) > 99.09, msg="Precision of mask rotation is too low."
        )
