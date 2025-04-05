import pathlib
import os
# 从scipy库中导入Rotation模块，用于处理欧拉角的转换
from scipy.spatial.transform import Rotation
import numpy as np
# 导入healpix库，用于处理球面像素化
import healpix as hp
# 导入logging模块，用于记录日志
import logging


def angle_to_angle_list(
    angle_diff: float, sort_angles: bool = True, log_level: int = logging.DEBUG
) -> list[tuple[float, float, float]]:
    """自动生成一个给定最大角度差的角度列表。

    代码使用healpix来确定Z1和X，并线性分割Z2。

    参数
    ----------
    angle_diff: float
        角度列表的最大差值（以度为单位）
    sort_angles: bool, default True
        对列表进行排序，使用Python默认的angle_list.sort()，先按Z1排序，然后是X，最后是Z2
    log_level: int, default logging.DEBUG
        生成日志时使用的日志级别

    返回
    -------
    angle_list: list[tuple[float, float, float]]
        一个列表，其中每个元素是一个包含3个浮点数的元组，表示逆时针ZXZ欧拉旋转（以弧度为单位）
    """
    # 我们使用面积的平方根的近似值作为中值角度差
    # 这在一定程度上是合理的，基于以下公式：
    # angle_diff = (4*np.pi/npix)**0.5 * 360/(2*np.pi)
    npix = 4 * np.pi / (angle_diff * np.pi / 180) ** 2
    nside = 0
    # 找到合适的nside值，使得npix足够大
    while hp.nside2npix(nside) < npix:
        nside += 1
    used_npix = hp.nside2npix(nside)
    used_angle_diff = (4 * np.pi / used_npix) ** 0.5 * (180 / np.pi)
    # 记录使用的Z1和X的角度差
    logging.log(
        log_level, f"Using an angle difference of {used_angle_diff:.4f} for Z1 and X"
    )
    # 获取theta和phi角度
    theta, phi = hp.pix2ang(nside, np.arange(used_npix))
    # 计算psi角度的数量
    n_psi_angles = int(np.ceil(360 / angle_diff))
    # 线性分割psi角度
    psi, used_psi_diff = np.linspace(
        0, 2 * np.pi, n_psi_angles, endpoint=False, retstep=True
    )
    # 记录使用的Z2的角度差
    logging.log(
        log_level,
        f"Using an angle difference of {np.rad2deg(used_psi_diff):.4f} for Z2",
    )
    # 生成角度列表
    angle_list = [(ph, th, ps) for ph, th in zip(phi, theta) for ps in psi]
    if sort_angles:
        # 对角度列表进行排序
        angle_list.sort()
    return angle_list


def load_angle_list(
    file_name: pathlib.Path, sort_angles: bool = True
) -> list[tuple[float, float, float]]:
    """从磁盘加载一个角度搜索列表。

    参数
    ----------
    file_name: pathlib.Path
        包含角度搜索的文本文件的路径，每行应包含3个逆时针ZXZ的浮点数
    sort_angles: bool, default True
        对列表进行排序，使用Python默认的angle_list.sort()，先按Z1排序，然后是X，最后是Z2

    返回
    -------
    angle_list: list[tuple[float, float, float]]
        一个列表，其中每个元素是一个包含3个浮点数的元组，表示逆时针ZXZ欧拉旋转（以弧度为单位）
    """
    # 打开文件并读取所有行
    with open(str(file_name)) as fstream:
        lines = fstream.readlines()
    # 将每行转换为浮点数元组
    angle_list = [tuple(map(float, x.strip().split(" "))) for x in lines]
    # 检查每行是否包含3个角度
    if not all([len(a) == 3 for a in angle_list]):
        raise ValueError(
            "Invalid angle file provided, each line should have 3 ZXZ Euler angles!"
        )
    if sort_angles:
        # 对角度列表进行排序，否则无法使用对称缩减
        angle_list.sort()
    return angle_list


def get_angle_list(
    angle: pathlib.Path | float,
    sort_angles: bool = True,
    symmetry: int = 1,
    log_level: int = logging.DEBUG,
):
    """从磁盘获取一个角度搜索文件，或者从一个浮点数生成一个角度搜索文件。

    参数
    ----------
    angle: Union[pathlib.Path, float]
        可以是包含角度搜索的文本文件的路径，每行应包含3个逆时针ZXZ的浮点数
        或者如果是一个浮点数：
          角度列表的最大差值（以度为单位）
    sort_angles: bool, default True
        对列表进行排序，使用Python默认的angle_list.sort()，先按Z1排序，然后是X，最后是Z2
    symmetry: int, default 1
        返回的列表将只包含Z2角度在[0, (2*pi/symmetry))范围内的角度
    log_level: int, default logging.DEBUG
        生成日志时使用的日志级别

    返回
    -------
    angle_list: list[tuple[float, float, float]]
        一个列表，其中每个元素是一个包含3个浮点数的元组，表示逆时针ZXZ欧拉旋转（以弧度为单位）
    """
    out = None
    # 计算Z1的最大角度
    max_z1 = 2 * np.pi / symmetry
    try:
        # 尝试将angle转换为浮点数
        angle = float(angle)
        angle_is_float = True
    except (ValueError, TypeError):
        angle_is_float = False
    if angle_is_float:
        # 记录将生成角度列表
        logging.log(
            log_level,
            f"Will generate an angle list with a maximum increment of {angle}",
        )
        # 生成角度列表
        out = angle_to_angle_list(angle, sort_angles, log_level)
    elif isinstance(angle, str | os.PathLike):
        # 将angle转换为Path对象
        possible_file_path = pathlib.Path(angle)
        if possible_file_path.exists() and possible_file_path.suffix == ".txt":
            # 记录将检查自定义文件
            logging.log(
                log_level,
                "Custom file provided for the angular search. "
                "Checking if it can be read...",
            )
            # 从文件中加载角度列表
            out = load_angle_list(angle, sort_angles)

    if out is None:  # 如果此时没有角度列表，抛出错误
        raise ValueError("Invalid angle input provided")
    # 过滤角度列表，只保留Z1小于max_z1的角度
    return [i for i in out if i[0] < max_z1]


def convert_euler(
    angles: tuple[float, float, float],
    order_in: str = "ZXZ",
    order_out: str = "ZXZ",
    degrees_in: bool = True,
    degrees_out: bool = True,
) -> tuple[float, float, float]:
    """将一组欧拉角从一种欧拉表示法转换为另一种。此函数使用scipy.spatial.transform.Rotation，
    大写字母（如ZXZ）表示内在旋转（常用于冷冻电镜），小写字母（如zxz）表示外在旋转。

    参数
    ----------
    angles: tuple[float, float, float]
        三个角度的元组
    order_in: str, default 'ZXZ'
        输入角度的欧拉旋转轴
    order_out: str, default 'ZXZ'
        输出角度的欧拉旋转轴
    degrees_in: bool, default True
        输入角度是否以度为单位
    degrees_out: bool, default True
        输出角度是否应以度为单位

    返回
    -------
    output: tuple[float, float, float]
        三个角度的元组
    """
    # 根据输入的欧拉角和旋转顺序创建Rotation对象
    r = Rotation.from_euler(order_in, angles, degrees=degrees_in)
    # 将Rotation对象转换为指定顺序的欧拉角
    return tuple(r.as_euler(order_out, degrees=degrees_out))
