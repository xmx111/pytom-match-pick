"""
此文件中的函数在 CPU 和 GPU 上均可使用。
"""

# 导入 numpy 的类型注解模块，用于类型提示
import numpy.typing as npt
# 导入 cupy 的类型注解模块，用于类型提示
import cupy.typing as cpt


def mean_under_mask(
    data: npt.NDArray[float] | cpt.NDArray[float],
    mask: npt.NDArray[float] | cpt.NDArray[float],
    mask_weight: float | None = None,
) -> float | cpt.NDArray[float]:
    """计算数组在掩码区域内的均值。

    data 和 mask 可以是 cupy 或 numpy 数组。

    参数
    ----------
    data: Union[npt.NDArray[float], cpt.NDArray[float]]
        输入数组
    mask: Union[npt.NDArray[float], cpt.NDArray[float]]
        输入掩码，与 data 具有相同的维度
    mask_weight: Optional[float], 默认值为 None
        可选的掩码权重，如果未提供，则使用 mask.sum() 来确定权重

    返回
    -------
    output: Union[float, cpt.NDArray[float]]
        数据在掩码区域内的均值
    """
    # 计算数据在掩码区域内的元素和，然后除以掩码权重
    output = (data * mask).sum() / (
        mask_weight if mask_weight is not None else mask.sum()
    )
    return output


def std_under_mask(
    data: npt.NDArray[float] | cpt.NDArray[float],
    mask: npt.NDArray[float] | cpt.NDArray[float],
    mean: float,
    mask_weight: float | None = None,
) -> float | cpt.NDArray[float]:
    """计算数组在掩码区域内的标准差。使用 mean_under_mask() 函数
    计算数据平方在掩码内的均值。

    data 和 mask 可以是 cupy 或 numpy 数组。

    参数
    ----------
    data: Union[npt.NDArray[float], cpt.NDArray[float]]
        输入数组
    mask: Union[npt.NDArray[float], cpt.NDArray[float]]
        输入掩码，与 data 具有相同的维度
    mean: float
        数组在掩码区域内的均值
    mask_weight: Optional[float], 默认值为 None
        可选的掩码权重，如果未提供，则使用 mask.sum() 来确定权重

    返回
    -------
    output: Union[float, cpt.NDArray[float]]
        数据在掩码区域内的标准差
    """
    # 先计算数据平方在掩码区域内的均值，减去均值的平方，再开方得到标准差
    output = (mean_under_mask(data**2, mask, mask_weight=mask_weight) - mean**2) ** 0.5
    return output


def normalise(
    data: npt.NDArray[float] | cpt.NDArray[float],
    mask: npt.NDArray[float] | cpt.NDArray[float] | None = None,
    mask_weight: float | None = None,
) -> npt.NDArray[float] | cpt.NDArray[float]:
    """通过减去均值并除以标准差来归一化数组。如果提供了掩码，则使用在掩码内计算的均值和标准差来归一化数组。

    data 和 mask 可以是 cupy 或 numpy 数组。

    参数
    ----------
    data: Union[npt.NDArray[float], cpt.NDArray[float]]
        要归一化的输入数组
    mask: Optional[Union[npt.NDArray[float], cpt.NDArray[float]]], 默认值为 None
        可选的掩码，用于在掩码区域内计算均值和标准差进行归一化
    mask_weight: Optional[float], 默认值为 None
        可选的浮点数，指定掩码权重，如果未提供，则使用 mask.sum()

    返回
    -------
    output: Union[npt.NDArray[float], cpt.NDArray[float]]
        归一化后的数组
    """
    if mask is None:
        # 若未提供掩码，直接计算数据的均值和标准差
        mean, std = data.mean(), data.std()
    else:
        # 若提供了掩码，计算掩码区域内的均值和标准差
        mean = mean_under_mask(data, mask, mask_weight=mask_weight)
        std = std_under_mask(data, mask, mean, mask_weight=mask_weight)
    # 对数据进行归一化操作
    output = (data - mean) / std
    return output


def normalised_cross_correlation(
    data1: npt.NDArray[float] | cpt.NDArray[float],
    data2: npt.NDArray[float] | cpt.NDArray[float],
    mask: npt.NDArray[float] | cpt.NDArray[float] | None = None,
) -> float | cpt.NDArray[float]:
    """计算两个数组之间的归一化互相关。可选择仅在掩码区域内计算。

    data1、data2 和 mask 可以是 cupy 或 numpy 数组。

    参数
    ----------
    data1: Union[npt.NDArray[float], cpt.NDArray[float]]
        用于相关计算的第一个数组
    data2: Union[npt.NDArray[float], cpt.NDArray[float]]
        用于相关计算的第二个数组
    mask: Optional[Union[npt.NDArray[float], cpt.NDArray[float]]], 默认值为 None
        可选的掩码，用于在掩码区域内计算相关性

    返回
    -------
    output: Union[float, cpt.NDArray[float]]
        数组之间的归一化互相关
    """
    if mask is None:
        # 若未提供掩码，对两个数组分别归一化后计算乘积和，再除以数据元素个数
        output = (normalise(data1) * normalise(data2)).sum() / data1.size
    else:
        # 若提供了掩码，对两个数组在掩码区域内归一化后计算乘积和，再除以掩码元素和
        output = (
            normalise(data1, mask) * mask * normalise(data2, mask)
        ).sum() / mask.sum()
    return output
