# 导入numpy的类型注解模块，用于类型提示
import numpy.typing as npt
# 导入numpy库，用于数值计算
import numpy as np
# 导入voltools库，用于处理3D体积数据
import voltools as vt
# 导入logging模块，用于记录日志
import logging
# 从scipy.ndimage模块导入center_of_mass和zoom函数
from scipy.ndimage import center_of_mass, zoom
# 从scipy.fft模块导入rfftn和irfftn函数，用于快速傅里叶变换
from scipy.fft import rfftn, irfftn
# 从pytom_tm.weights模块导入create_gaussian_low_pass和radial_reduced_grid函数
from pytom_tm.weights import (
    create_gaussian_low_pass,
    radial_reduced_grid,
)


def generate_template_from_map(
    input_map: npt.NDArray[float],
    input_spacing: float,
    output_spacing: float,
    center: bool = False,
    filter_to_resolution: float | None = None,
    output_box_size: int | None = None,
) -> npt.NDArray[float]:
    """
    从密度图生成模板。

    参数
    ----------
    input_map: npt.NDArray[float]
        用于生成模板的3D密度图，如果盒子不是正方形，将被填充为正方形
    input_spacing: float
        输入图的体素大小（以埃为单位）
    output_spacing: float
        输出图的体素大小（以埃为单位），输入与输出的比例将用于下采样
    center: bool, 默认值为 False
        设置为 True 以通过计算质心将模板居中于盒子中
    filter_to_resolution: Optional[float], 默认值为 None
        应用于模板的低通滤波器分辨率，如果未提供，将设置为 2 * 输出体素大小
    output_box_size:  Optional[int], 默认值为 None
        模板的最终盒子大小
    display_filter: bool, 默认值为 False
        标志，用于显示应用于模板的滤波器的绘图

    返回
    -------
    template: npt.NDArray[float]
        处理后的模板，具有指定的输出盒子大小，盒子将为正方形
    """
    # 确保输入图是一个具有相等维度的盒子
    if len(set(input_map.shape)) != 1:
        # 计算每个维度需要填充的差值
        diff = [max(input_map.shape) - s for s in input_map.shape]
        # 使用零填充输入图，使其成为正方形
        input_map = np.pad(
            input_map,
            tuple([(d // 2, d // 2 + d % 2) for d in diff]),
            mode="constant",
            constant_values=0,
        )

    # 如果未提供滤波器分辨率，则设置为奈奎斯特分辨率
    if filter_to_resolution is None:
        # 设置为奈奎斯特分辨率
        filter_to_resolution = 2 * output_spacing
    # 如果滤波器分辨率低于 2 * 输出体素大小，发出警告并调整分辨率
    elif filter_to_resolution < (2 * output_spacing):
        warning_text = (
            f"滤波器分辨率过低，"
            f" 设置为 {2 * output_spacing} 埃 (2 * 输出体素大小)"
        )
        logging.warning(warning_text)
        filter_to_resolution = 2 * output_spacing

    # 如果需要将模板居中
    if center:
        # 计算体积的中心坐标
        volume_center = np.divide(np.subtract(input_map.shape, 1), 2, dtype=np.float32)
        # 对输入图进行平方，确保质心计算中的值为正
        input_center_of_mass = center_of_mass(input_map**2)
        # 计算需要平移的偏移量
        shift = np.subtract(volume_center, input_center_of_mass)
        # 对输入图进行平移操作
        input_map = vt.transform(input_map, translation=shift, device="cpu")

        # 记录质心的变化
        logging.debug(
            f"质心，之前是 "
            f"{np.round(input_center_of_mass, 2)} "
            f"之后是 {np.round(center_of_mass(input_map**2), 2)}"
        )

    # 在应用卷积之前，将体积扩展到所需的输出大小
    if output_box_size is not None:
        # 记录大小检查信息
        logging.debug(
            f"大小检查 {output_box_size} > "
            f"{(input_map.shape[0] * input_spacing) // output_spacing}"
        )
        # 如果输出盒子大小大于计算得到的大小，进行填充
        if output_box_size > (input_map.shape[0] * input_spacing) // output_spacing:
            # 计算需要填充的零的数量
            pad = (
                int(output_box_size * (output_spacing / input_spacing))
                - input_map.shape[0]
            )
            # 记录填充的零的数量
            logging.debug(f"用以下数量的零填充: {pad}")
            # 对输入图进行填充操作
            input_map = np.pad(
                input_map,
                (pad // 2, pad // 2 + pad % 2),
                mode="constant",
                constant_values=0,
            )
        # 如果输出盒子大小小于计算得到的大小，发出警告
        elif output_box_size < (input_map.shape[0] * input_spacing) // output_spacing:
            logging.warning(
                "无法设置指定的盒子大小，因为图需要被裁剪，"
                " 这可能会导致结构信息的丢失。请手动减小图的盒子大小（例如使用chimera）"
            )

    # 创建低通滤波器
    lpf = create_gaussian_low_pass(
        input_map.shape, input_spacing, filter_to_resolution
    ).astype(np.float32)

    # 记录卷积和下采样的信息
    logging.info("将体积与滤波器卷积，然后进行下采样。")
    # 对输入图进行傅里叶变换，乘以滤波器，再进行逆傅里叶变换，最后进行下采样
    return zoom(
        irfftn(rfftn(input_map) * lpf, s=input_map.shape),
        input_spacing / output_spacing,
    )


def phase_randomize_template(
    template: npt.NDArray[float],
    seed: int = 321,
):
    """
    创建一个在傅里叶空间中相位随机排列的模板版本。

    参数
    ----------
    template: npt.NDArray[float]
        输入结构
    seed: int, 默认值为 321
        用于相位排列的随机数生成器的种子

    返回
    -------
    result: npt.NDArray[float]
        相位随机化的模板版本
    """
    # 对模板进行实值快速傅里叶变换
    ft = rfftn(template)
    # 计算傅里叶变换结果的幅度
    amplitude = np.abs(ft)

    # 在数组的扁平化版本中对相位进行排列
    # 获取傅里叶变换结果的相位，并将其扁平化
    phase = np.angle(ft).flatten()
    # 计算傅里叶空间的径向缩减网格，并进行逆傅里叶变换的移轴操作，然后扁平化
    grid = np.fft.ifftshift(radial_reduced_grid(template.shape), axes=(0, 1)).flatten()
    # 确定相关频率，仅对直到奈奎斯特频率的部分进行排列
    relevant_freqs = grid <= 1
    # 创建一个与相位数组相同形状的零数组
    noise = np.zeros_like(phase)
    # 创建一个随机数生成器，使用指定的种子
    rng = np.random.default_rng(seed)
    # 对相关频率的相位进行随机排列
    noise[relevant_freqs] = rng.permutation(phase[relevant_freqs])

    # 构建新的模板
    # 将噪声数组重新调整为幅度数组的形状
    noise = np.reshape(noise, amplitude.shape)
    # 计算新的模板，通过幅度乘以相位的指数形式，再进行逆傅里叶变换
    result = irfftn(amplitude * np.exp(1j * noise), s=template.shape)
    return result
