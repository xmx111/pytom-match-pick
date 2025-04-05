# 导入numpy库，用于数值计算
import numpy as np
# 导入numpy的类型注解模块，用于类型提示
import numpy.typing as npt


def spherical_mask(
    box_size: int,
    radius: float,
    smooth: float | None = None,
    cutoff_sd: int = 3,
    center: float | None = None,
) -> npt.NDArray[float]:
    """
    围绕ellipsoidal_mask()函数的包装器，用于创建仅具有单个半径的球形掩码。

    参数
    ----------
    box_size: int
        掩码的盒子大小，各维度相等
    radius: float
        球体的半径
    smooth: Optional[float], default None
        掩码周围高斯衰减的标准差（相对于像素数量的浮点数）
    cutoff_sd: int, default 3
        包含高斯衰减的标准差数量，默认值3是一个不错的选择
    center: Optional[float], default None
        掩码的可选中心，默认值为 (size - 1) / 2

    返回
    -------
    mask: npt.NDArray[float]
        中心带有掩码的3D numpy数组
    """
    # 调用ellipsoidal_mask函数，传入相同的半径以创建球形掩码
    return ellipsoidal_mask(
        box_size, radius, radius, radius, smooth, cutoff_sd=cutoff_sd, center=center
    )


def ellipsoidal_mask(
    box_size: int,
    major: float,
    minor1: float,
    minor2: float,
    smooth: float | None = None,
    cutoff_sd: int = 3,
    center: float | None = None,
) -> npt.NDArray[float]:
    """
    在指定的方形盒子中创建一个椭球形掩码。椭球体由x、y和z轴上的3个半径定义。

    参数
    ----------
    box_size: int
        掩码的盒子大小，各维度相等
    major: float
        椭球体在x轴上的半径
    minor1: float
        椭球体在y轴上的半径
    minor2: float
        椭球体在z轴上的半径
    smooth: Optional[float], default None
        掩码周围高斯衰减的标准差（相对于像素数量的浮点数）
    cutoff_sd: int, default 3
        包含高斯衰减的标准差数量，默认值3是一个不错的选择
    center: Optional[float], default None
        掩码的可选中心，默认值为 (size - 1) / 2

    返回
    -------
    mask: npt.NDArray[float]
        中心带有掩码的3D numpy数组
    """
    # 检查输入是否有效，若盒子大小或半径小于等于0，则抛出异常
    if not all([box_size > 0, major > 0, minor1 > 0, minor2 > 0]):
        raise ValueError("Invalid input for mask creation: box_size or radii are <= 0")

    # 若未指定中心，则使用默认中心
    center = (box_size - 1) / 2 if center is None else center
    # 生成x、y、z轴上的坐标数组，并减去中心坐标
    x, y, z = (
        np.arange(box_size) - center,
        np.arange(box_size) - center,
        np.arange(box_size) - center,
    )

    # 使用广播机制计算每个点到中心的相对距离
    r = np.sqrt(
        ((x / major) ** 2)[:, np.newaxis, np.newaxis]
        + ((y / minor1) ** 2)[:, np.newaxis]
        + (z / minor2) ** 2
    ).astype(np.float32)

    if smooth is not None:
        # 若指定了平滑参数，检查平滑参数和标准差截断值是否有效
        if not all([smooth >= 0, cutoff_sd >= 0]):
            raise ValueError(
                "Invalid input for mask smoothing: smooth or sd cutoff are <= 0"
            )
        # 将距离小于等于1的点的距离设为1
        r[r <= 1] = 1
        # 计算高斯衰减的标准差
        sigma = smooth / ((major + minor1 + minor2) / 3)
        # 计算高斯衰减的掩码
        mask = np.exp(-1 * ((r - 1) / sigma) ** 2)
        # 将小于等于截断值的掩码元素设为0
        mask[mask <= np.exp(-(cutoff_sd**2) / 2.0)] = 0
    else:
        # 若未指定平滑参数，创建一个全零的掩码
        mask = np.zeros_like(r)
        # 将距离小于等于1的点的掩码设为1
        mask[r <= 1] = 1.0

    return mask
