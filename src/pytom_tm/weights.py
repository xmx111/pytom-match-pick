import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndimage
import voltools as vt
from pytom_tm.io import UnequalSpacingError
from itertools import pairwise

# 定义一个字典，包含计算所需的物理常数
constants = {
    # 字典中存储了计算所需的物理常数，每个常数都有对应的注释说明其含义和单位
    "c": 299792458,  # 光速，单位：m/s
    "el": 1.60217646e-19,  # 电子电荷，单位：C
    "h": 6.62606896e-34,  # 普朗克常数，单位：J*S
    "h_ev": 4.13566733e-15,  # 普朗克常数（以eV为单位），单位：eV*s
    "h_bar": 1.054571628e-34,  # 约化普朗克常数，单位：J*s
    "h_bar_ev": 6.58211899e-16,  # 约化普朗克常数（以eV为单位），单位：eV*s
    "na": 6.02214179e23,  # 阿伏伽德罗常数，单位：mol-1
    "re": 2.817940289458e-15,  # 经典电子半径，单位：m
    "rw": 2.976e-10,  # 未知物理量，单位：m
    "me": 9.10938215e-31,  # 电子质量，单位：kg
    "me_ev": 0.510998910e6,  # 电子质量（以eV为单位），单位：ev/c^2
    "kb": 1.3806503e-23,  # 玻尔兹曼常数，单位：m^2 kgs^-2 K^-1
    "eps0": 8.854187817620e-12,  # 真空介电常数，单位：F/m
}


def hwhm_to_sigma(hwhm: float) -> float:
    """
    将高斯函数的半高宽（HWHM）转换为标准差（sigma）。通过将半高宽除以sqrt(2 * ln(2))来实现。

    参数
    ----------
    hwhm: float
        高斯函数的半高宽

    返回
    -------
    sigma: float
        高斯函数的标准差
    """
    return hwhm / (np.sqrt(2 * np.log(2)))


def sigma_to_hwhm(sigma: float) -> float:
    """
    将高斯函数的标准差（sigma）转换为半高宽（HWHM）。通过将标准差乘以sqrt(2 * ln(2))来实现。

    参数
    ----------
    sigma: float
        高斯函数的标准差

    返回
    -------
    hwhm: float
        高斯函数的半高宽
    """
    return sigma * (np.sqrt(2 * np.log(2)))


def wavelength_ev2m(voltage: float) -> float:
    """
    根据电压计算电子的波长。

    参数
    ----------
    voltage: float
        电子波的电压，单位：eV

    返回
    -------
    lambda: float
        电子的波长，单位：m
    """
    # 从常数字典中获取所需的物理常数
    h = constants["h"]
    e = constants["el"]
    m = constants["me"]
    c = constants["c"]

    # 根据相对论公式计算电子的波长
    _lambda = h / np.sqrt(e * voltage * m * (e / m * voltage / c**2 + 2))

    return _lambda


def radial_reduced_grid(
    shape: tuple[int, int, int] | tuple[int, int], shape_is_reduced: bool = False
) -> npt.NDArray[float]:
    """
    计算给定输入形状的傅里叶空间径向缩减网格，其中零频率位于输出图像的中心。
    网格值从中心的0到奈奎斯特频率的1变化。

    默认情况下，假设形状属于实空间数组，这会导致函数返回一个最后一维缩减的网格，
    即 shape[-1] // 2 + 1（适用于创建频率相关滤波器）。
    但是，如果设置 radial_reduced_grid(..., shape_is_reduced=True)，则假设形状已经是缩减形式。

    参数
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int]]
        2D或3D输入形状，通常是numpy数组的.shape属性
    shape_is_reduced: bool, default False
        形状是否已经是缩减的傅里叶格式，默认为False

    返回
    ----------
    radial_reduced_grid: npt.NDArray[float]
        傅里叶空间频率网格，中心为0，奈奎斯特频率为1
    """
    # 检查输入形状是否为2D或3D
    if len(shape) not in [2, 3]:
        raise ValueError("radial_reduced_grid() 仅适用于2D或3D形状")
    # 根据形状是否已经缩减，确定最后一维的大小
    reduced_dim = shape[-1] if shape_is_reduced else shape[-1] // 2 + 1
    if len(shape) == 3:
        # 3D情况下，计算x、y、z方向的坐标
        x = (
            np.abs(
                np.arange(
                    -shape[0] // 2 + shape[0] % 2, shape[0] // 2 + shape[0] % 2, 1.0
                )
            )
            / (shape[0] // 2)
        )[:, np.newaxis, np.newaxis]
        y = (
            np.abs(
                np.arange(
                    -shape[1] // 2 + shape[1] % 2, shape[1] // 2 + shape[1] % 2, 1.0
                )
            )
            / (shape[1] // 2)
        )[:, np.newaxis]
        z = np.arange(0, reduced_dim, 1.0) / (reduced_dim - 1)
        # 计算径向距离
        return np.sqrt(x**2 + y**2 + z**2)
    elif len(shape) == 2:
        # 2D情况下，计算x、y方向的坐标
        x = (
            np.abs(
                np.arange(
                    -shape[0] // 2 + shape[0] % 2, shape[0] // 2 + shape[0] % 2, 1.0
                )
            )
            / (shape[0] // 2)
        )[:, np.newaxis]
        y = np.arange(0, reduced_dim, 1.0) / (reduced_dim - 1)
        # 计算径向距离
        return np.sqrt(x**2 + y**2)


def create_gaussian_low_pass(
    shape: tuple[int, int, int] | tuple[int, int],
    spacing: float,
    resolution: float,
) -> npt.NDArray[float]:
    """
    创建一个在傅里叶空间缩减的3D高斯低通滤波器，具有截止频率（或半高宽）。

    参数
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int]]
        包含x、y或x、y、z维度的形状元组
    spacing: float
        实空间中的体素大小
    resolution: float
        要滤波到的实空间分辨率

    返回
    ----------
    output: npt.NDArray[float]
        包含滤波器的数组
    """
    # 计算傅里叶空间的径向缩减网格
    q = radial_reduced_grid(shape)

    # 2 * spacing / resolution 是傅里叶空间的截止频率
    # 然后将截止频率（半高宽）转换为高斯函数的标准差
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    # 计算高斯低通滤波器，并进行逆傅里叶移轴操作
    return np.fft.ifftshift(np.exp(-(q**2) / (2 * sigma_fourier**2)), axes=(0, 1))


def create_gaussian_high_pass(
    shape: tuple[int, int, int] | tuple[int, int],
    spacing: float,
    resolution: float,
) -> npt.NDArray[float]:
    """
    创建一个在傅里叶空间缩减的3D高斯高通滤波器，具有截止频率（或半高宽）。

    参数
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int]]
        包含x、y或x、y、z维度的形状元组
    spacing: float
        实空间中的体素大小
    resolution: float
        要滤波到的实空间分辨率

    返回
    ----------
    output: npt.NDArray[float]
        包含滤波器的数组
    """
    # 计算傅里叶空间的径向缩减网格
    q = radial_reduced_grid(shape)

    # 2 * spacing / resolution 是傅里叶空间的截止频率
    # 然后将截止频率（半高宽）转换为高斯函数的标准差
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    # 计算高斯高通滤波器，并进行逆傅里叶移轴操作
    return np.fft.ifftshift(1 - np.exp(-(q**2) / (2 * sigma_fourier**2)), axes=(0, 1))


def create_gaussian_band_pass(
    shape: tuple[int, int, int] | tuple[int, int],
    spacing: float,
    low_pass: float | None = None,
    high_pass: float | None = None,
) -> npt.NDArray[float]:
    """
    分辨率带表示需要保留信息的分辨率壳层。例如，带可能是 (150A, 40A)。
    对于间距为15A（奈奎斯特分辨率为30A），这是一个温和的低通滤波器。
    然而，相当多的低空间频率将被其截断。

    参数
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int]]
        包含x、y或x、y、z维度的形状元组
    spacing: float
        实空间中的体素大小
    low_pass: Optional[float], default None
        低通滤波器的分辨率
    high_pass: Optional[float], default None
        高通滤波器的分辨率

    返回
    ----------
    output: npt.NDArray[float]
        包含带通滤波器的数组
    """
    # 检查是否至少设置了低通或高通滤波器
    if high_pass is None and low_pass is None:
        raise ValueError("带通滤波器需要至少设置低通或高通滤波器")

    if high_pass is None:
        # 如果只设置了低通滤波器，调用 create_gaussian_low_pass 函数
        return create_gaussian_low_pass(shape, spacing, low_pass)
    elif low_pass is None:
        # 如果只设置了高通滤波器，调用 create_gaussian_high_pass 函数
        return create_gaussian_high_pass(shape, spacing, high_pass)
    elif low_pass >= high_pass:
        # 检查低通分辨率是否小于高通分辨率
        raise ValueError("带通滤波器的第二个值应该是高分辨率壳层")
    else:
        # 计算傅里叶空间的径向缩减网格
        q = radial_reduced_grid(shape)

        # 2 * spacing / resolution 是傅里叶空间的截止频率
        # 然后将截止频率（半高宽）转换为高斯函数的标准差
        sigma_high_pass = hwhm_to_sigma(2 * spacing / high_pass)
        sigma_low_pass = hwhm_to_sigma(2 * spacing / low_pass)

        # 计算高斯带通滤波器，并进行逆傅里叶移轴操作
        return np.fft.ifftshift(
            (1 - np.exp(-(q**2) / (2 * sigma_high_pass**2)))
            * np.exp(-(q**2) / (2 * sigma_low_pass**2)),
            axes=(0, 1),
        )


def create_wedge(
    shape: tuple[int, int, int],
    tilt_angles: list[float, ...],
    voxel_size: float,
    cut_off_radius: float = 1.0,
    angles_in_degrees: bool = True,
    low_pass: float | None = None,
    high_pass: float | None = None,
    tilt_weighting: bool = False,
    accumulated_dose_per_tilt: list[float, ...] | None = None,
    ctf_params_per_tilt: list[dict] | None = None,
) -> npt.NDArray[float]:
    """
    此函数根据输入的楔形角度返回一个楔形体积，该体积可以是对称的或不对称的。

    参数
    ----------
    shape: tuple[int, int, int]
        需要应用楔形的实空间体积形状
    tilt_angles: list[float, ...]
        用于重建断层图像的倾斜角度列表
    voxel_size: float
        体素大小，用于各种滤波器的计算
    cut_off_radius: float, default 1.
        截止半径，以奈奎斯特频率的分数表示，即1.0表示一直到奈奎斯特频率
    angles_in_degrees: bool, default True
        倾斜角度是否以度为单位
    low_pass: Optional[float], default None
        低通滤波器的分辨率，单位：A
    high_pass: Optional[float], default None
        高通滤波器的分辨率，单位：A
    tilt_weighting: bool, default False
        是否应用倾斜加权
    accumulated_dose_per_tilt: Optional[list[float, ...]], default None
        每个倾斜角度的累积剂量，用于剂量加权
    ctf_params_per_tilt: Optional[list[dict]], default None
        每个倾斜角度的CTF参数（请参阅 _create_tilt_weighted_wedge() 了解字典规范）

    返回
    -------
    wedge: npt.NDArray[float]
        楔形体积，是z方向上缩减的傅里叶空间对象，即 shape[2] // 2 + 1
    """
    # 检查倾斜角度列表是否有效
    if not isinstance(tilt_angles, list) or len(tilt_angles) < 2:
        raise ValueError("楔形生成需要至少两个倾斜角度的列表")

    # 检查体素大小是否有效
    if voxel_size <= 0.0:
        raise ValueError(
            "创建楔形时体素大小小于或等于0，这是无效的体素间距"
        )

    # 检查截止半径是否有效
    if cut_off_radius > 1:
        print(
            "警告：楔形截止半径需要定义为奈奎斯特频率的分数，0 < c <= 1。将值设置为1.0。"
        )
        cut_off_radius = 1.0
    elif cut_off_radius <= 0:
        raise ValueError("无效的楔形截止半径：需要大于0")

    # 将倾斜角度转换为弧度
    if angles_in_degrees:
        tilt_angles_rad = [np.deg2rad(w) for w in tilt_angles]
    else:
        tilt_angles_rad = tilt_angles

    if tilt_weighting:
        # 如果应用倾斜加权
        if ctf_params_per_tilt is not None and len(ctf_params_per_tilt) == 1:
            # 如果只有一个CTF参数，将其复制到每个倾斜角度
            ctf_params_per_tilt = ctf_params_per_tilt * len(tilt_angles_rad)
        # 调用 _create_tilt_weighted_wedge 函数创建倾斜加权楔形
        wedge = _create_tilt_weighted_wedge(
            shape,
            tilt_angles_rad,
            cut_off_radius,
            voxel_size,
            accumulated_dose_per_tilt=accumulated_dose_per_tilt,
            ctf_params_per_tilt=ctf_params_per_tilt,
        ).astype(np.float32)
    else:
        # 计算楔形角度
        wedge_angles = (
            np.pi / 2 - np.abs(min(tilt_angles_rad)),
            np.pi / 2 - np.abs(max(tilt_angles_rad)),
        )
        if np.round(wedge_angles[0], 2) == np.round(wedge_angles[1], 2):
            # 如果楔形角度对称，调用 _create_symmetric_wedge 函数
            wedge = _create_symmetric_wedge(
                shape, wedge_angles[0], cut_off_radius
            ).astype(np.float32)
        else:
            # 如果楔形角度不对称，调用 _create_asymmetric_wedge 函数
            wedge = _create_asymmetric_wedge(
                shape, (wedge_angles[0], wedge_angles[1]), cut_off_radius
            ).astype(np.float32)
        if ctf_params_per_tilt is not None:
            # 如果提供了CTF参数，应用CTF滤波器
            ctf_params = ctf_params_per_tilt[len(ctf_params_per_tilt) // 2]
            wedge *= create_ctf(
                shape,
                voxel_size * 1e-10,
                **ctf_params,
            )

    if not (low_pass is None and high_pass is None):
        # 如果设置了低通或高通滤波器，应用带通滤波器
        return wedge * create_gaussian_band_pass(
            shape, voxel_size, low_pass, high_pass
        ).astype(np.float32)
    else:
        return wedge


def _create_symmetric_wedge(
    shape: tuple[int, int, int], wedge_angle: float, cut_off_radius: float
) -> npt.NDArray[float]:
    """
    此函数返回一个对称的楔形对象。
    该函数不应被导入，用户应调用 create_wedge()。

    参数
    ----------
    shape: tuple[int, int, int]
        需要应用楔形的实空间体积形状
    wedge_angle: float
        描述对称楔形的角度，单位：弧度
    cut_off_radius: float
        截止半径，以奈奎斯特频率的分数表示，即1.0表示一直到奈奎斯特频率

    返回
    ----------
    wedge: npt.NDArray[float]
        楔形体积，是z方向上缩减的傅里叶空间对象，即 shape[2] // 2 + 1
    """
    # 计算x方向的坐标
    x = (
        np.abs(
            np.arange(-shape[0] // 2 + shape[0] % 2, shape[0] // 2 + shape[0] % 2, 1.0)
        )
        / (shape[0] // 2)
    )[:, np.newaxis]
    # 计算z方向的坐标
    z = np.arange(0, shape[2] // 2 + 1, 1.0) / (shape[2] // 2)

    # 计算具有平滑边缘的楔形掩码
    wedge_2d = x - np.tan(wedge_angle) * z
    limit = (wedge_2d.max() - wedge_2d.min()) / (2 * min(shape[0], shape[2]) // 2)
    wedge_2d[wedge_2d > limit] = limit
    wedge_2d[wedge_2d < -limit] = -limit
    wedge_2d = (wedge_2d - wedge_2d.min()) / (wedge_2d.max() - wedge_2d.min())
    # 确保零频率点等于1
    wedge_2d[shape[0] // 2 + 1, 0] = 1

    # 在x方向上复制楔形掩码
    wedge = np.tile(wedge_2d[:, np.newaxis, :], (1, shape[1], 1))

    # 应用截止半径
    wedge[radial_reduced_grid(shape) > cut_off_radius] = 0

    # 进行逆傅里叶移轴操作
    return np.fft.ifftshift(wedge, axes=(0, 1))


def _create_asymmetric_wedge(
    shape: tuple[int, int, int],
    wedge_angles: tuple[float, float],
    cut_off_radius: float,
) -> npt.NDArray[float]:
    """
    此函数返回一个不对称的楔形对象。
    该函数不应被导入，用户应调用 create_wedge()。

    参数
    ----------
    shape: tuple[int, int, int]
        需要应用楔形的实空间体积形状
    wedge_angles: tuple[float, float]
        描述不对称缺失楔形的两个角度，单位：弧度
    cut_off_radius: float
        截止半径，以奈奎斯特频率的分数表示，即1.0表示一直到奈奎斯特频率

    返回
    ----------
    wedge: npt.NDArray[float]
        楔形体积，是z方向上缩减的傅里叶空间对象，即 shape[2] // 2 + 1
    """
    # 计算x方向的坐标
    x = (
        np.abs(
            np.arange(-shape[0] // 2 + shape[0] % 2, shape[0] // 2 + shape[0] % 2, 1.0)
        )
        / (shape[0] // 2)
    )[:, np.newaxis]
    # 计算z方向的坐标
    z = np.arange(0, shape[2] // 2 + 1, 1.0) / (shape[2] // 2)

    # 计算第一个角度的楔形
    wedge_section = x - np.tan(wedge_angles[0]) * z
    limit = (wedge_section.max() - wedge_section.min()) / (
        2 * min(shape[0], shape[2]) // 2
    )
    wedge_section[wedge_section > limit] = limit
    wedge_section[wedge_section < -limit] = -limit
    wedge_section = (wedge_section - wedge_section.min()) / (
        wedge_section.max() - wedge_section.min()
    )

    # 设置楔形的顶部
    wedge_2d = wedge_section.copy()

    # 计算第二个角度的楔形
    wedge_section = x - np.tan(wedge_angles[1]) * z
    limit = (wedge_section.max() - wedge_section.min()) / (
        2 * min(shape[0], shape[2]) // 2
    )
    wedge_section[wedge_section > limit] = limit
    wedge_section[wedge_section < -limit] = -limit
    wedge_section = (wedge_section - wedge_section.min()) / (
        wedge_section.max() - wedge_section.min()
    )

    # 设置楔形的底部，并将零频率点设置为1
    wedge_2d[shape[0] // 2 + 1 :] = wedge_section[shape[0] // 2 + 1 :]
    wedge_2d[shape[0] // 2 + 1, 0] = 1

    # 在x方向上复制楔形掩码
    wedge = np.tile(wedge_2d[:, np.newaxis, :], (1, shape[1], 1))

    # 应用截止半径
    wedge[radial_reduced_grid(shape) > cut_off_radius] = 0

    # 进行逆傅里叶移轴操作
    return np.fft.ifftshift(wedge, axes=(0, 1))


def _create_tilt_weighted_wedge(
    shape: tuple[int, int, int],
    tilt_angles: list[float, ...],
    cut_off_radius: float,
    pixel_size_angstrom: float,
    accumulated_dose_per_tilt: list[float, ...] | None = None,
    ctf_params_per_tilt: list[dict] | None = None,
) -> npt.NDArray[float]:
    """
    使用以下B因子启发式方法（如M论文中所述，并在RELION 1.4中引入）：
        "B因子每暴露1e− Å−2增加4Å2，每个倾斜角度的权重为cos θ。"

    B因子与高斯函数的标准差之间的关系：

        B = 8 * pi ** 2 * sigma_motion ** 2

    即 sigma_motion = sqrt( B / (8 * pi ** 2))。属于高斯模糊：

        exp( -2 * pi ** 2 * sigma_motion ** 2 * q ** 2 )

    参数
    ----------
    shape: tuple[int, int, int]
        用于建模楔形的体积形状
    tilt_angles: list[float, ...]
        倾斜角度列表，单位：弧度
    cut_off_radius: float
        掩码的截止半径，以奈奎斯特频率的分数表示，值在0和1之间
    pixel_size_angstrom: float
        像素大小，单位：Å
    accumulated_dose_per_tilt: list[float, ...], default None
        每个倾斜角度的累积剂量，单位：e− Å−2
    ctf_params_per_tilt: list[dict, ...], default None
        每个倾斜角度的CTF参数列表，每个字典包含以下键：
        - 'defocus': 散焦，单位：um
        - 'amplitude': 振幅对比度分数，在0和1之间
        - 'voltage': 电压，单位：keV
        - 'cs': 球差，单位：mm
        - 'phase_shift_deg': 相位板的相移，单位：deg

    返回
    -------
    wedge: npt.NDArray[float]
        结构化楔形掩码，以缩减的傅里叶形式表示，即输出形状为 (shape[0], shape[1], shape[2] // 2 + 1)
    """
    # 检查累积剂量列表的长度是否与倾斜角度列表的长度一致
    if accumulated_dose_per_tilt is not None and len(accumulated_dose_per_tilt) != len(
        tilt_angles
    ):
        raise ValueError(
            "在 _create_tilt_weighted_wedge 中，累积剂量列表的长度与倾斜角度列表的长度不一致！"
        )
    # 检查CTF参数列表的长度是否与倾斜角度列表的长度一致
    if ctf_params_per_tilt is not None and len(ctf_params_per_tilt) != len(tilt_angles):
        raise ValueError(
            "在 _create_tilt_weighted_wedge 中，CTF参数列表的长度与倾斜角度列表的长度不一致！"
        )
    # 检查输入形状是否为正方形盒子
    if not all([shape[0] == s for s in shape[1:]]):
        raise UnequalSpacingError(
            "结构化楔形的输入形状需要是正方形盒子。"
            " 否则，傅里叶空间中的频率在各个维度上不相等。"
        )

    # 由于所有维度大小相等，将形状的第一个维度赋值给 image_size
    image_size = shape[0]
    # 初始化倾斜角度数组
    tilt = np.zeros(shape)
    # 计算傅里叶空间的径向缩减网格
    q_grid = radial_reduced_grid(shape)
    # 初始化倾斜加权楔形数组
    tilt_weighted_wedge = np.zeros((image_size, image_size, image_size // 2 + 1))

    # 创建斜坡权重以校正倾斜求和的重叠
    # 计算相邻倾斜角度的最小增量
    tilt_increment = min([abs(x - y) for x, y in pairwise(tilt_angles)])
    # 计算克劳瑟频率，确定相邻倾斜在傅里叶空间中的重叠点
    overlap_frequency = 1 / (tilt_increment * image_size)
    # 计算一维频率数组
    freq_1d = (
        np.abs(
            np.arange(
                -image_size // 2 + image_size % 2, image_size // 2 + image_size % 2, 1.0
            )
        )
        / (image_size // 2)
        * 0.5
    )  # 乘以0.5以得到奈奎斯特频率
    # 计算斜坡滤波器
    ramp_filter = freq_1d / overlap_frequency
    # 将斜坡滤波器中大于1的值设置为1
    ramp_filter[ramp_filter > 1] = 1  # 线性增加到重叠频率

    # 生成沿倾斜轴的二维权重
    # 在y方向上复制斜坡滤波器
    ramp_weighting = np.tile(ramp_filter[:, np.newaxis], (1, image_size))

    # 遍历每个倾斜角度
    for i, alpha in enumerate(tilt_angles):
        if ctf_params_per_tilt is not None:
            # 如果提供了CTF参数，计算CTF
            ctf = np.fft.fftshift(
                create_ctf(
                    (image_size,) * 2,
                    pixel_size_angstrom * 1e-10,
                    **ctf_params_per_tilt[i],
                ),
                axes=0,
            )
            # 将CTF复制并翻转，然后与斜坡权重相乘，赋值给倾斜数组的中间平面
            tilt[:, :, image_size // 2] = (
                np.concatenate(
                    (  # 复制并翻转CTF围绕零频率；
                        # 然后连接以使其非缩减
                        np.flip(ctf[:, 1 : 1 + image_size - ctf.shape[1]], axis=1),
                        ctf,
                    ),
                    axis=1,
                )
                * ramp_weighting
            )
        else:
            # 如果未提供CTF参数，将斜坡权重赋值给倾斜数组的中间平面
            tilt[:, :, image_size // 2] = ramp_weighting

        # 将倾斜数组旋转到倾斜角度
        rotated = np.flip(
            vt.transform(
                tilt,
                rotation=(0, alpha, 0),
                rotation_units="rad",
                rotation_order="rxyz",
                center=(image_size // 2,) * 3,
                interpolation="filt_bspline",
                device="cpu",
            )[:, :, : image_size // 2 + 1],  # 将z轴裁剪回缩减的傅里叶形式
            axis=2,
        )

        # 用曝光和倾斜衰减进行加权
        if accumulated_dose_per_tilt is not None:
            # 计算q的平方
            q_squared = (q_grid / (2 * pixel_size_angstrom)) ** 2
            # 计算运动标准差
            sigma_motion = np.sqrt(accumulated_dose_per_tilt[i] * 4 / (8 * np.pi**2))
            # 计算加权后的倾斜数组
            weighted_tilt = (
                rotated
                * np.cos(alpha)  # 应用倾斜相关的加权
                * np.exp(
                    -2 * np.pi**2 * sigma_motion**2 * q_squared
                )  # 应用剂量加权
            )
        else:
            # 如果未提供累积剂量，只应用倾斜相关的加权
            weighted_tilt = (
                rotated * np.cos(alpha)  # 应用倾斜相关的加权
            )

        # 将加权后的倾斜数组累加到倾斜加权楔形数组中
        tilt_weighted_wedge += weighted_tilt

    # 应用截止半径
    tilt_weighted_wedge[q_grid > cut_off_radius] = 0

    # 进行逆傅里叶移轴操作
    return np.fft.ifftshift(tilt_weighted_wedge, axes=(0, 1))


def create_ctf(
    shape: tuple[int, int, int] | tuple[int, int],
    pixel_size: float,
    defocus: float,
    amplitude_contrast: float,
    voltage: float,
    spherical_aberration: float,
    cut_after_first_zero: bool = False,
    flip_phase: bool = False,
    phase_shift_deg: float = 0.0,
) -> npt.NDArray[float]:
    """
    在3D体积中以缩减格式创建一个CTF（对比度传递函数）。

    参数
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int]]
        用于创建CTF的体积维度
    pixel_size: float
        CTF的像素大小，单位：m
    defocus: float
        CTF的散焦，单位：m
    amplitude_contrast: float
        CTF中的振幅对比度分数
    voltage: float
        显微镜的加速电压，单位：eV
    spherical_aberration: float
        球差，单位：m
    cut_after_first_zero: bool, default False
        是否在第一个零交叉点后截断CTF
    flip_phase: bool, default False
        使CTF完全为正/负，以模拟通过相位翻转进行的CTF校正
    phase_shift_deg: float, default .0
        额外的相移，用于模拟相位板，类似于 `https://github.com/dtegunov/tom_deconv`
        除了tom中的CTF定义产生与我们这里相反的曲线

    返回
    -------
    ctf: npt.NDArray[float]
        3D中的CTF
    """
    # 计算傅里叶空间的频率
    k = radial_reduced_grid(shape) / (2 * pixel_size)

    # 计算电子的波长
    _lambda = wavelength_ev2m(voltage)

    # 计算相位对比度传递函数
    chi = (
        np.pi * _lambda * defocus * k**2
        - 0.5 * np.pi * spherical_aberration * _lambda**3 * k**4
    )
    # 计算振幅对比度项
    tan_term = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast**2))

    # 确定CTF
    ctf = -np.sin(chi + tan_term + np.deg2rad(phase_shift_deg))

    if cut_after_first_zero:  # 找到第一个零交叉点的频率进行截断
        def chi_1d(q):
            return (
                np.pi * _lambda * defocus * q**2
                - 0.5 * np.pi * spherical_aberration * _lambda**3 * q**4
            )

        def ctf_1d(q):
            return -np.sin(chi_1d(q) + tan_term)

        # 采样一维CTF并获取零交叉点的索引
        k_range = np.arange(max(k.shape)) / max(k.shape) / (2 * pixel_size)
        values = ctf_1d(k_range)
        zero_crossings = np.where(np.diff(np.sign(values)))[0]

        # 对于过焦情况，跳过第一个交叉点
        # 例如，参见：Yonekura et al. 2006 JSB
        k_cutoff = (
            k_range[zero_crossings[0]] if defocus > 0 else k_range[zero_crossings[1]]
        )

        # 用截止频率过滤CTF
        ctf[k > k_cutoff] = 0

    if flip_phase:  # 取绝对值，确保对比度匹配
        ctf = np.abs(ctf)
    else:  # 如果是过焦情况，将CTF乘以 -1，这允许用户始终匹配输入模板的对比度与断层图像的对比度
        # 如果断层图像是黑色的，参考应该是黑色的。
        ctf *= -1 if defocus > 0 else 1

    # 进行逆傅里叶移轴操作
    return np.fft.ifftshift(ctf, axes=(0, 1) if len(shape) == 3 else 0)


def radial_average(
    weights: npt.NDArray[float],
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    计算缩减傅里叶空间函数的径向平均值。

    参数
    ----------
    weights: npt.NDArray[float]
        要进行径向平均的3D数组：以缩减的傅里叶形式表示，且原点在角落。

    返回
    -------
    (q, mean): tuple[npt.NDArray[float], npt.NDArray[float]]
        两个一维numpy数组的元组。它们的长度等于最大输入维度的一半。
    """
    # 检查输入数组是否为2D或3D
    if len(weights.shape) not in [2, 3]:
        raise ValueError("径向平均计算仅适用于2D/3D数组")

    # 获取采样点的数量，从最大的傅里叶维度获取，除非缩减维度已经是最大的
    sampling_points = max(max(weights.shape[:-1]) // 2 + 1, weights.shape[-1])

    # 生成采样点的索引数组
    q = np.arange(sampling_points)
    # 计算傅里叶功率谱中的径向索引
    q_grid = np.floor(
        # 转换为傅里叶功率谱中的径向索引，
        # 加上0.5以获得正确的环
        radial_reduced_grid(weights.shape, shape_is_reduced=True)
        * (sampling_points - 1)
        + 0.5
    ).astype(int)
    # 计算每个径向索引对应的平均值
    mean = ndimage.mean(
        np.fft.fftshift(weights, axes=(0, 1) if len(weights.shape) == 3 else 0),
        labels=q_grid,
        index=q,
    )

    return q, mean


def power_spectrum_profile(image: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    计算实空间数组的功率谱，然后找到其轮廓（径向平均值）。

    参数
    ----------
    image: npt.NDArray[float]
        要计算功率谱轮廓的2D/3D实空间数组

    返回
    -------
    profile: npt.NDArray[float]
        一维numpy数组
    """
    # 检查输入数组是否为2D或3D
    if len(image.shape) not in [2, 3]:
        raise ValueError(
            "功率谱轮廓计算仅适用于2D/3D数组。"
        )

    # 计算图像的实值快速傅里叶变换的绝对值的平方，然后进行径向平均
    _, power_profile = radial_average(np.abs(np.fft.rfftn(image)) ** 2)

    return power_profile


def profile_to_weighting(
    profile: npt.NDArray[float], shape: tuple[int, int] | tuple[int, int, int]
) -> npt.NDArray[float]:
    """
    根据频谱轮廓计算径向加权（滤波器）。

    参数
    ----------
    profile: npt.NDArray[float]
        功率谱轮廓（或其他一维轮廓），用于转换为傅里叶空间滤波器
    shape: Union[tuple[int, int], tuple[int, int, int]]
        实空间中的2D/3D数组形状，用于计算缩减的傅里叶空间权重

    返回
    -------
    weighting: npt.NDArray[float]
        针对给定形状的缩减傅里叶空间权重
    """
    # 检查输入轮廓是否为一维数组
    if len(profile.shape) != 1:
        raise ValueError("传递给 profile_to_weighting 的轮廓不是一维的。")
    # 检查输入形状是否为2D或3D
    if len(shape) not in [2, 3]:
        raise ValueError("传递给 profile_to_weighting 的形状需要是2D/3D。")

    # 计算傅里叶空间的径向缩减网格
    q_grid = radial_reduced_grid(shape)

    # 使用 ndimage.map_coordinates 函数根据网格坐标从轮廓中插值得到权重
    weights = ndimage.map_coordinates(
        profile, q_grid.flatten()[np.newaxis, :] * profile.shape[0], order=1
    ).reshape(q_grid.shape)

    # 将网格中大于1的部分对应的权重设置为0
    weights[q_grid > 1] = 0

    # 进行逆傅里叶移轴操作
    return np.fft.ifftshift(weights, axes=(0, 1) if len(shape) == 3 else 0)
