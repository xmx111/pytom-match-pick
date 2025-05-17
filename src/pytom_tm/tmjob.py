# 从 __future__ 导入 annotations，允许在类型注解中使用前向引用
from __future__ import annotations
# 导入 version 模块，用于版本比较
from packaging import version
# 导入 pathlib 模块，用于处理文件路径
import pathlib
# 导入 warnings 模块，用于发出警告信息
import warnings
# 导入 copy 模块，用于对象的深拷贝
import copy
# 导入 itertools 模块并简称为 itt，用于迭代器操作
import itertools as itt
# 导入 numpy 库并简称为 np，用于数值计算
import numpy as np
# 导入 numpy 的类型注解模块
import numpy.typing as npt
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 logging 模块，用于记录日志
import logging
# 从 scipy.fft 模块导入 next_fast_len、rfftn 和 irfftn 函数，用于快速傅里叶变换
from scipy.fft import next_fast_len, rfftn, irfftn
# 从 pytom_tm.angles 模块导入 get_angle_list 函数，用于获取角度列表
from pytom_tm.angles import get_angle_list
# 从 pytom_tm.matching 模块导入 TemplateMatchingGPU 类，用于模板匹配
from pytom_tm.matching import TemplateMatchingGPU
# 从 pytom_tm.weights 模块导入多个函数，用于创建权重
from pytom_tm.weights import (
    create_wedge,
    power_spectrum_profile,
    profile_to_weighting,
    create_gaussian_band_pass,
)
# 从 pytom_tm.io 模块导入多个函数和异常类，用于文件输入输出
from pytom_tm.io import read_mrc_meta_data, read_mrc, write_mrc, UnequalSpacingError
# 从 pytom_tm 模块导入版本号
from pytom_tm import __version__ as PYTOM_TM_VERSION

import os
from datetime import datetime

def load_json_to_tmjob(
    file_name: pathlib.Path, load_for_extraction: bool = True
) -> 'TMJob':
    """
    从 JSON 文件中加载之前保存的 TMJob 任务。

    参数
    ----------
    file_name: pathlib.Path
        指向 TMJob JSON 文件的路径
    load_for_extraction: bool, 默认值为 True
        指示是否为提取目的加载已完成的任务，默认值为 True，因为该函数当前仅用于
        pytom_extract_candidates 和 pytom_estimate_roc，这两个函数处理已完成的任务

    返回
    -------
    job: TMJob
        初始化后的 TMJob 实例
    """
    # 打开 JSON 文件并加载数据
    with open(file_name) as fstream:
        data = json.load(fstream)

    # 处理数据类型
    # 获取输出数据类型，默认为 float32
    output_dtype = data.get("output_dtype", "float32")
    # 将数据类型转换为 numpy 的数据类型
    output_dtype = np.dtype(output_dtype)

    # 创建 TMJob 实例
    job = TMJob(
        data["job_key"],
        data["log_level"],
        pathlib.Path(data["tomogram"]),
        pathlib.Path(data["template"]),
        pathlib.Path(data["mask"]),
        pathlib.Path(data["output_dir"]),
        angle_increment=data.get("angle_increment", data["rotation_file"]),
        mask_is_spherical=data["mask_is_spherical"],
        tilt_angles=data["tilt_angles"],
        tilt_weighting=data["tilt_weighting"],
        search_x=data["search_x"],
        search_y=data["search_y"],
        search_z=data["search_z"],
        # 使用 get 方法以实现向后兼容
        tomogram_mask=data.get("tomogram_mask", None),
        voxel_size=data["voxel_size"],
        low_pass=data["low_pass"],
        # 使用 get 方法以实现向后兼容
        high_pass=data.get("high_pass", None),
        dose_accumulation=data.get("dose_accumulation", None),
        ctf_data=data.get("ctf_data", None),
        whiten_spectrum=data.get("whiten_spectrum", False),
        rotational_symmetry=data.get("rotational_symmetry", 1),
        # 如果 JSON 文件中未包含版本号，则默认为 0.3.0 或更早版本
        pytom_tm_version_number=data.get("pytom_tm_version_number", "0.3.0"),
        job_loaded_for_extraction=load_for_extraction,
        particle_diameter=data.get("particle_diameter", None),
        random_phase_correction=data.get("random_phase_correction", False),
        rng_seed=data.get("rng_seed", 321),
        defocus_handedness=data.get("defocus_handedness", 0),
        output_dtype=output_dtype,
    )
    # 如果文件来自旧版本，为兼容性设置相移
    if (
        version.parse(job.pytom_tm_version_number) < version.parse("0.6.1")
        and job.ctf_data is not None
    ):
        for tilt in job.ctf_data:
            tilt["phase_shift_deg"] = 0.0
    # 从数据中获取并设置相关属性
    job.whole_start = data["whole_start"]
    job.sub_start = data["sub_start"]
    job.sub_step = data["sub_step"]
    job.n_rotations = data["n_rotations"]
    job.start_slice = data["start_slice"]
    job.steps_slice = data["steps_slice"]
    job.job_stats = data["job_stats"]
    return job


def get_defocus_offsets(
    patch_center_x: float,
    patch_center_z: float,
    tilt_angles: list[float, ...],
    angles_in_degrees: bool = True,
    invert_handedness: bool = False,
) -> npt.NDArray[float]:
    """
    基于倾斜几何计算子体积的散焦偏移量。

    对于散焦左右手性的默认设置，我使用了 Pyle & Zianetti (https://doi.org/10.1042/BCJ20200715) 中的定义。
    它假设在样品右侧（相对于中心的正 X 坐标），正倾斜角度的散焦会增加。

    偏移量的计算如下：
        z_offset = z_center * np.cos(tilt_angle) + x_center * np.sin(tilt_angle)

    参数
    ----------
    patch_center_x: float
        子体积相对于断层图像中心的 X 坐标
    patch_center_z: float
        子体积相对于断层图像中心的 Z 坐标
    tilt_angles: list[float, ...]
        倾斜角度列表
    angles_in_degrees: bool, 默认值为 True
        指示倾斜角度是否以度为单位
    invert_handedness: bool, 默认值为 False
        反转散焦左右手性几何

    返回
    -------
    z_offsets: npt.NDArray[float]
        每个倾斜角度的散焦偏移量数组
    """
    # 获取倾斜角度的数量
    n_tilts = len(tilt_angles)
    # 创建一个长度为 n_tilts 的数组，每个元素都为 patch_center_x
    x_centers = np.full(n_tilts, patch_center_x)
    # 创建一个长度为 n_tilts 的数组，每个元素都为 patch_center_z
    z_centers = np.full(n_tilts, patch_center_z)
    # 将倾斜角度列表转换为 numpy 数组
    ta_array = np.array(tilt_angles)
    # 如果角度是以度为单位，将其转换为弧度
    if angles_in_degrees:
        ta_array = np.deg2rad(ta_array)
    # 如果需要反转左右手性，将角度取负
    if invert_handedness:
        ta_array *= -1
    # 计算散焦偏移量
    z_offsets = z_centers * np.cos(ta_array) + x_centers * np.sin(ta_array)
    return z_offsets


def _determine_1D_fft_splits(
    length: int, splits: int, overhang: int = 0
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    将一维长度分割为 FFT 最优大小，并考虑重叠部分。

    参数
    ----------
    length: int
        要分割的一维总长度
    splits: int
        分割的次数
    overhang: int, 默认值为 0
        分割之间的最小重叠/覆盖量

    返回
    -------
    output: list[tuple[tuple[int, int], tuple[int, int]]]
        分割列表，每个分割包含两个元组：
          [start, end) 表示此分割中断层图像数据的范围
          [start, end) 表示此分割中唯一数据点的范围
        如果一个数据点存在于两个分割中，将其添加到数据量最大的分割中，
        如果两个分割的数据量相同，则添加到左侧的分割中
    """
    # 此代码假设默认切片为 [x, y)，即包含 x 但不包含 y
    data_slices = []
    valid_data_slices = []
    sub_len = []
    # 如果只进行一次分割，直接返回
    if splits == 1:
        return [((0, length), (0, length))]
    # 如果分割次数大于长度，发出警告并将分割次数设置为长度
    if splits > length:
        warnings.warn(
            "请求的分割次数超过了像素数量，将默认设置为每个像素一次分割",
            RuntimeWarning,
        )
        splits = length
    # 向上取整，确保有足够的缓冲区覆盖整个长度
    min_len = int(np.ceil(length / splits)) + overhang
    min_unique_len = min_len - overhang
    no_overhang_left = 0
    while True:
        if no_overhang_left == 0:
            # 特殊处理第一个分割，只考虑右侧重叠
            split_length = next_fast_len(min_len)
            data_slices.append((0, split_length))
            valid_data_slices.append((0, split_length - overhang))
            no_overhang_left = split_length - overhang
            sub_len.append(split_length)
        elif no_overhang_left + min_unique_len >= length:
            # 最后一个切片，只考虑左侧重叠
            split_length = next_fast_len(min_len)
            data_slices.append((length - split_length, length))
            valid_data_slices.append((length - split_length + overhang, length))
            sub_len.append(split_length)
            break
        else:
            # 其他切片
            split_length = next_fast_len(min_len + overhang)
            left_overhang = (split_length - min_unique_len) // 2
            temp_left = no_overhang_left - left_overhang
            temp_right = temp_left + split_length
            data_slices.append((temp_left, temp_right))
            valid_data_slices.append((temp_left + overhang, temp_right - overhang))
            sub_len.append(split_length)
            no_overhang_left = temp_right - overhang
        # 如果分割长度或剩余无重叠长度小于等于 0，抛出运行时错误
        if split_length <= 0 or no_overhang_left <= 0:
            raise RuntimeError(
                f"无法为 {length=}, {splits=}, {overhang=} 生成合法的分割"
            )
    # 现在生成最佳的唯一数据点，
    # 我们总是选择数据量最大的子集或左侧的子集
    unique_data = []
    unique_left = 0
    # 遍历相邻的子长度对
    for i, (len1, len2) in enumerate(itt.pairwise(sub_len)):
        if len1 >= len2:
            right = valid_data_slices[i][1]
        else:
            right = valid_data_slices[i + 1][0]
        unique_data.append((unique_left, right))
        unique_left = right
    # 添加最后一部分
    if unique_left != length:
        unique_data.append((unique_left, length))
    # 确保唯一切片是唯一的，并且在有效数据范围内
    last_right = 0
    for (vd_left, vd_right), (ud_left, ud_right) in zip(valid_data_slices, unique_data):
        if (
            ud_left < vd_left
            or ud_right > vd_right
            or ud_right > length
            or ud_left != last_right
        ):  # pragma: no cover
            raise RuntimeError(
                f"我们为 {length=}, {splits=}, {overhang=} 生成了不一致的切片"
            )
        last_right = ud_right
    return list(zip(data_slices, unique_data))


class TMJobError(Exception):
    """
    带有指定消息的 TMJob 异常。
    """

    def __init__(self, message):
        # 调用基类构造函数并传入所需参数
        super().__init__(message)


class TMJob:
    def __init__(
        self,
        job_key: str,
        log_level: int,
        tomogram: pathlib.Path,
        template: pathlib.Path,
        mask: pathlib.Path,
        output_dir: pathlib.Path,
        angle_increment: str | float | None = None,
        mask_is_spherical: bool = True,
        tilt_angles: list[float, ...] | None = None,
        tilt_weighting: bool = False,
        search_x: list[int, int] | None = None,
        search_y: list[int, int] | None = None,
        search_z: list[int, int] | None = None,
        tomogram_mask: pathlib.Path | None = None,
        sphere_file: pathlib.Path | None = None,
        voxel_size: float | None = None,
        low_pass: float | None = None,
        high_pass: float | None = None,
        dose_accumulation: list[float, ...] | None = None,
        ctf_data: list[dict, ...] | None = None,
        whiten_spectrum: bool = False,
        rotational_symmetry: int = 1,
        pytom_tm_version_number: str = PYTOM_TM_VERSION,
        job_loaded_for_extraction: bool = False,
        particle_diameter: float | None = None,
        random_phase_correction: bool = False,
        rng_seed: int = 321,
        defocus_handedness: int = 0,
        output_dtype: np.dtype = np.float32,
    ):
        """
        初始化 TMJob 实例。

        参数
        ----------
        job_key: str
            任务标识符
        log_level: int
            日志记录模块的日志级别
        tomogram: pathlib.Path
            断层图像 MRC 文件的路径
        template: pathlib.Path
            模板 MRC 文件的路径
        mask: pathlib.Path
            掩码 MRC 文件的路径
        output_dir: pathlib.Path
            输出目录的路径
        angle_increment: Union[str, float]; 默认值为 7.00
            模板搜索的角度增量
        mask_is_spherical: bool, 默认值为 True
            指示模板掩码是否为球形，可降低计算复杂度
        tilt_angles: Optional[list[float, ...]], 默认值为 None
            用于重建断层图像的倾斜系列的倾斜角度，如果只有两个浮点数，
            将用于生成连续楔形模型
        tilt_weighting: bool, 默认值为 False
            使用高级倾斜加权选项，可与 CTF 参数和累积剂量一起使用
        search_x: Optional[list[int, int]], 默认值为 None
            限制断层图像在 x 轴上的搜索区域
        search_y: Optional[list[int, int]], 默认值为 None
            限制断层图像在 y 轴上的搜索区域
        search_z: Optional[list[int, int]], 默认值为 None
            限制断层图像在 z 轴上的搜索区域
        tomogram_mask: Optional[pathlib.Path], 默认值为 None
            当分割断层图像体积时，仅生成掩码中存在大于 0 的值的子任务
        sphere_file: Optional[pathlib.Path], 默认值为 None
            标志的球心文件
        voxel_size: Optional[float], 默认值为 None
            断层图像和模板的体素大小（以埃为单位），如果未提供，将从模板/断层图像 MRC 文件中读取
        low_pass: Optional[float], 默认值为 None
            可选的低通滤波器（分辨率以埃为单位），应用于断层图像和模板
        high_pass: Optional[float], 默认值为 None
            可选的高通滤波器（分辨率以埃为单位），应用于断层图像和模板
        dose_accumulation: Optional[list[float, ...]], 默认值为 None
            每个倾斜图像的累积剂量列表
        ctf_data: Optional[list[dict, ...]], 默认值为 None
            每个倾斜图像的 CTF 参数列表，参数定义见 pytom_tm.weight.create_ctf()
        whiten_spectrum: bool, 默认值为 False
            指示是否应用频谱白化
        rotational_symmetry: int, 默认值为 1
            指定围绕 z 轴的旋转对称性，仅当模板的对称轴与 z 轴对齐时有效
        pytom_tm_version_number: str, 默认值为当前版本
            用于向后兼容的 pytom_tm 版本号字符串
        job_loaded_for_extraction: bool, 默认值为 False
            为已完成的模板匹配任务设置的标志，用于加载回来进行提取，可防止重新计算白化滤波器
        particle_diameter: Optional[float], 默认值为 None
            用于计算角度搜索的粒子直径（以埃为单位）
        random_phase_correction: bool, 默认值为 False
            使用模板的相位随机化版本进行匹配，以校正噪声分数
        rng_seed: int, 默认值为 321
            用于相位随机化的随机数生成器种子
        defocus_handedness: int, 默认值为 0
            指定散焦左右手性：
            -1 = 反转
             0 = 不校正偏移（如果未知，建议使用此值）
             1 = 常规（如 Pyle & Zianetti (2021) 中所述）
        output_dtype: np.dtype, 默认值为 np.float32
            输出分数体积的数据类型，可选值为 np.float32 和 np.float16
        """
        # 初始化基本属性
        self.mask = mask
        self.mask_is_spherical = mask_is_spherical
        self.output_dir = output_dir

        self.tomogram = tomogram
        self.template = template
        # 从断层图像路径中提取文件名（不包含扩展名）
        self.tomo_id = self.tomogram.stem

        try:
            # 读取断层图像的元数据
            meta_data_tomo = read_mrc_meta_data(self.tomogram)
        except UnequalSpacingError:  # 添加信息，表明问题出在断层图像上
            raise UnequalSpacingError(
                "输入断层图像的体素间距在每个维度上不相等！"
            )

        try:
            # 读取模板的元数据
            meta_data_template = read_mrc_meta_data(self.template)
        except UnequalSpacingError:  # 添加信息，表明问题出在模板上
            raise UnequalSpacingError(
                "输入模板的体素间距在每个维度上不相等！"
            )

        # 获取断层图像和模板的形状
        self.tomo_shape = meta_data_tomo["shape"]
        self.template_shape = meta_data_template["shape"]

        if voxel_size is not None:
            if voxel_size <= 0:
                raise ValueError(
                    "提供的体素大小无效，小于或等于零。"
                )
            self.voxel_size = voxel_size
            if (  # 允许微小的数值差异，这些差异对模板匹配无关紧要
                round(self.voxel_size, 3) != round(meta_data_tomo["voxel_size"], 3)
                or round(self.voxel_size, 3)
                != round(meta_data_template["voxel_size"], 3)
            ):
                logging.debug(
                    f"提供的体素大小为 {self.voxel_size}，断层图像的体素大小为 "
                    f"{meta_data_tomo['voxel_size']}，模板的体素大小为 "
                    f"{meta_data_template['voxel_size']}"
                )
                print(
                    "警告：提供的体素大小与断层图像/模板 MRC 文件中注释的体素大小不匹配。"
                )
        elif (
            round(meta_data_tomo["voxel_size"], 3)
            == round(meta_data_template["voxel_size"], 3)
            and meta_data_tomo["voxel_size"] > 0
        ):
            self.voxel_size = round(meta_data_tomo["voxel_size"], 3)
        else:
            raise ValueError(
                "无法分配体素大小，可能是断层图像和模板之间不匹配，或者注释为 0。"
            )

        # 确定搜索区域的起始位置
        search_origin = [
            x[0] if x is not None else 0 for x in (search_x, search_y, search_z)
        ]
        # 检查断层图像的起始位置是否有效
        if all([0 <= x < y for x, y in zip(search_origin, self.tomo_shape)]):
            self.search_origin = search_origin
        else:
            raise ValueError("为断层图像的搜索起始位置提供了无效的输入。")

        # 如果结束位置无效，抛出错误
        search_end = []
        for x, s in zip([search_x, search_y, search_z], self.tomo_shape):
            if x is not None:
                if not x[1] <= s:
                    raise ValueError(
                        "其中一个搜索结束索引大于断层图像的维度。"
                    )
                search_end.append(x[1])
            else:
                search_end.append(s)
        # 计算搜索区域的大小
        self.search_size = [
            end - start for end, start in zip(search_end, self.search_origin)
        ]

        logging.debug(f"起始位置，大小 = {self.search_origin}, {self.search_size}")

        self.sphere_list = None
        if sphere_file is not None:
            # 传入球心文件并读取
            self.sphere_list = []
            with open(sphere_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z, rmax = map(float, parts[:4])
                        rmin = rmax * 0.5
                        self.sphere_list.append((x, y, z, rmin, rmax))
        
        self.tomogram_mask = tomogram_mask
        self.has_tomogram_mask = tomogram_mask is not None
        if self.has_tomogram_mask:
            # 读取断层图像掩码
            temp = read_mrc(tomogram_mask)
            if temp.shape != self.tomo_shape:
                raise ValueError(
                    "断层图像掩码的像素数量与断层图像不同。\n"
                    f"断层图像掩码的形状: {temp.shape}，"
                    f"断层图像的形状: {self.tomo_shape}"
                )
            if np.all(temp <= 0):
                raise ValueError(
                    "在断层图像掩码中未找到大于 0 的值："
                    f"{tomogram_mask}"
                )

        # 初始化相关属性
        self.whole_start = None
        # 对于主任务，这些值始终为 [0, 0, 0] 和 self.search_size，对于子任务，这些值将与 self.search_origin 和 self.search_size 不同。
        # 主任务仅使用它们来计算搜索体积的感兴趣区域以进行统计。子任务还使用这些值来提取和放回主任务中的相关区域。
        self.sub_start, self.sub_step = [0, 0, 0], self.search_size.copy()

        # 旋转参数
        self.start_slice = 0
        self.steps_slice = 1
        self.rotational_symmetry = rotational_symmetry
        self.particle_diameter = particle_diameter
        # 根据粒子直径计算角度增量
        # angle_increment 7.00
        if angle_increment is None:
            # particle_diameter "LH1RC": 140, "LH2": 80
            if particle_diameter is not None:
                # 计算最大分辨率
                max_res = max(
                    2 * self.voxel_size, low_pass if low_pass is not None else 0
                )
                # 计算角度增量
                angle_increment = np.rad2deg(max_res / particle_diameter)
            else:
                angle_increment = 7.0
        self.rotation_file = angle_increment
        try:
            # 获取角度列表
            angle_list = get_angle_list(
                angle_increment,
                sort_angles=False,
                symmetry=rotational_symmetry,
                # 此 log_level 与稍后分配的 self.log_level 不同。
                # TMJob.log_level 指的是用户提供的日志记录设置，而这里的 log_level 用于控制候选提取/模板匹配期间的作业输出。
                log_level=logging.DEBUG if job_loaded_for_extraction else logging.INFO,
            )
        except ValueError:
            raise TMJobError("提供的角度搜索无效。")

        # 获取旋转的数量
        self.n_rotations = len(angle_list)

        # 缺失楔形相关
        self.tilt_angles = tilt_angles
        self.tilt_weighting = tilt_weighting
        # 设置带通分辨率壳
        self.low_pass = low_pass
        self.high_pass = high_pass

        # 设置剂量和 CTF
        self.dose_accumulation = dose_accumulation
        self.ctf_data = ctf_data
        self.defocus_handedness = defocus_handedness
        self.whiten_spectrum = whiten_spectrum
        # 定义白化滤波器的文件路径
        self.whitening_filter = self.output_dir.joinpath(
            f"{self.tomo_id}_whitening_filter.npy"
        )
        if self.whiten_spectrum and not job_loaded_for_extraction:
            logging.info("正在估计白化滤波器...")
            # 计算功率谱轮廓
            weights = 1 / np.sqrt(
                power_spectrum_profile(
                    read_mrc(self.tomogram)[
                        self.search_origin[0] : self.search_origin[0]
                        + self.search_size[0],
                        self.search_origin[1] : self.search_origin[1]
                        + self.search_size[1],
                        self.search_origin[2] : self.search_origin[2]
                        + self.search_size[2],
                    ]
                )
            )
            # 对权重进行缩放
            weights /= weights.max()  # 缩放到 1
            # 将权重保存到文件
            np.save(self.whitening_filter, weights)

        # 相位随机化选项
        self.random_phase_correction = random_phase_correction
        self.rng_seed = rng_seed

        # 任务详细信息
        self.job_key = job_key
        # 生成此任务的上级任务
        self.leader = None
        # 子任务列表
        self.sub_jobs = []  # 如果此任务没有子任务，则应执行该任务

        # 用于跟踪任务统计信息的字典
        self.job_stats = None

        # 日志级别
        self.log_level = log_level

        # 任务的版本号
        self.pytom_tm_version_number = pytom_tm_version_number

        # 输出数据类型
        self.output_dtype = output_dtype

    def copy(self) -> 'TMJob':
        """
        创建 TMJob 实例的副本。

        返回
        -------
        job: TMJob
            复制后的 TMJob 实例
        """
        return copy.deepcopy(self)

    def write_to_json(self, file_name: pathlib.Path) -> None:
        """
        将任务信息写入 JSON 文件。

        参数
        ----------
        file_name: pathlib.Path
            输出文件的路径
        """
        # 复制实例的属性字典
        d = self.__dict__.copy()
        # 移除不需要保存的属性
        d.pop("sub_jobs")
        d.pop("search_origin")
        d.pop("search_size")
        # 重新构建搜索区域的表示
        d["search_x"] = [
            self.search_origin[0],
            self.search_origin[0] + self.search_size[0],
        ]
        d["search_y"] = [
            self.search_origin[1],
            self.search_origin[1] + self.search_size[1],
        ]
        d["search_z"] = [
            self.search_origin[2],
            self.search_origin[2] + self.search_size[2],
        ]
        # 将路径对象转换为字符串
        for key, value in d.items():
            if isinstance(value, pathlib.Path):
                d[key] = str(value)
        # 处理数据类型转换
        d["output_dtype"] = str(np.dtype(d["output_dtype"]))
        # 将数据写入 JSON 文件
        with open(file_name, "w") as fstream:
            json.dump(d, fstream, indent=4)

    def split_rotation_search(self, n: int) -> list['TMJob']:
        """
        通过分割旋转将搜索任务拆分为子任务。子任务的键将是 self.job_key + str(i)，其中 i 是循环范围(n) 内的索引。

        参数
        ----------
        n: int
            角度搜索的分割次数

        返回
        -------
        sub_jobs: list[TMJob]
            从当前任务拆分出来的 TMJob 列表，这些任务也将被分配给 TMJob.sub_jobs 属性
        """
        if len(self.sub_jobs) > 0:
            raise TMJobError(
                "无法进一步拆分此任务，因为它已经分配了子任务！"
            )

        sub_jobs = []
        for i in range(n):
            # 复制当前任务
            new_job = self.copy()
            # 设置起始切片
            new_job.start_slice = i
            # 设置切片步长
            new_job.steps_slice = n
            # 设置上级任务的键
            new_job.leader = self.job_key
            # 设置新任务的键
            new_job.job_key = self.job_key + str(i)
            sub_jobs.append(new_job)

        # 将子任务列表分配给当前任务的属性
        self.sub_jobs = sub_jobs

        return self.sub_jobs

    def split_volume_search(self, split: tuple[int, int, int]) -> list['TMJob']:
        """
        通过将搜索区域分割为子体积，将搜索任务拆分为子任务。最终的子体积数量通过将所有分割数相乘得到，
        例如 (2, 2, 1) 将产生 4 个子体积。子任务的键将是 self.job_key + str(i)，其中 i 是循环范围(n) 内的索引。

        子任务在整个断层图像中的搜索区域由以下属性定义：
        new_job.search_origin 和 new_job.search_size。
        这些属性用于从整个断层图像中加载搜索体积。

        属性 new_job.whole_start 定义了该体积如何映射回父任务的分数体积（当搜索在 x、y 或 z 方向上受到限制时，
        分数体积的大小可能与断层图像不同）。

        最后，new_job.sub_start 和 new_job.sub_step 用于从子体积中提取不包含模板重叠部分的分数和角度图。

        如果设置了 self.tomogram_mask，将跳过掩码中所有值都小于等于 0 的子任务。

        参数
        ----------
        split: tuple[int, int, int]
            一个元组，定义了搜索体积在每个轴上应分割为子体积的次数

        返回
        -------
        sub_jobs: list[TMJob]
            从当前任务拆分出来的 TMJob 列表，这些任务也将被分配给 TMJob.sub_jobs 属性
        """
        if len(self.sub_jobs) > 0:
            raise TMJobError(
                "无法进一步拆分此任务，因为它已经分配了子任务！"
            )

        # 获取搜索区域的大小
        search_size = self.search_size
        if self.tomogram_mask is not None:
            # 读取断层图像掩码
            tomogram_mask = read_mrc(self.tomogram_mask)
        else:
            tomogram_mask = None
        # 模板的形状，用于计算重叠部分
        overhang = self.template_shape
        # 使用 overhang//2 (+1 用于奇数大小)
        overhang = tuple(sum(divmod(o, 2)) for o in overhang)

        # 对每个轴进行一维 FFT 分割
        x_splits = _determine_1D_fft_splits(search_size[0], split[0], overhang[0])
        y_splits = _determine_1D_fft_splits(search_size[1], split[1], overhang[1])
        z_splits = _determine_1D_fft_splits(search_size[2], split[2], overhang[2])

        sub_jobs = []
        # 遍历所有可能的分割组合
        for i, data_3D in enumerate(itt.product(x_splits, y_splits, z_splits)):
            # 每个维度的数据点是搜索空间的切片(left, right)
            # 和搜索空间中唯一数据点的切片(left, right)
            # 查看 new_job.attribute 中的注释，了解每个属性的含义

            # 计算搜索区域的起始位置
            search_origin = tuple(
                data_3D[d][0][0] + self.search_origin[d] for d in range(3)
            )
            # 计算搜索区域的大小
            search_size = tuple(dim_data[0][1] - dim_data[0][0] for dim_data in data_3D)
            # 计算唯一数据在整个搜索数组中的起始位置
            whole_start = tuple(dim_data[1][0] for dim_data in data_3D)
            # 计算唯一数据在分割数组中的起始位置
            sub_start = tuple(dim_data[1][0] - dim_data[0][0] for dim_data in data_3D)
            # 计算唯一数据在分割数组中的步长
            sub_step = tuple(dim_data[1][1] - dim_data[1][0] for dim_data in data_3D)

            # 检查是否包含任何掩码值大于 0 的唯一数据点
            if tomogram_mask is not None:
                # 创建切片对象
                slices = [slice(origin, origin + step) for origin, step in zip(whole_start, sub_step)]
                if np.all(tomogram_mask[slices[0], slices[1], slices[2]] <= 0):
                    # 没有未掩码的唯一数据点，跳过此子任务
                    continue
            # 复制当前任务
            new_job = self.copy()
            # 设置上级任务的键
            new_job.leader = self.job_key
            # 设置新任务的键
            new_job.job_key = self.job_key + str(i)

            # 搜索区域相对于整个断层图像的起始位置
            new_job.search_origin = search_origin
            # 搜索区域的大小 TODO: 应与起始位置合并为切片
            new_job.search_size = search_size

            # whole_start 是唯一数据在整个搜索数组中的起始位置
            new_job.whole_start = whole_start
            # sub_start 是唯一数据在分割数组中的起始位置
            new_job.sub_start = sub_start
            # sub_step 是唯一数据在分割数组中的步长。
            # TODO: 应改为切片
            new_job.sub_step = sub_step
            sub_jobs.append(new_job)

        # 将子任务列表分配给当前任务的属性
        self.sub_jobs = sub_jobs

        return self.sub_jobs

    def merge_sub_jobs(
        self, stats: list[dict, ...] | None = None
    ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        合并当前任务的子任务，生成最终的输出分数和角度图。

        参数
        ----------
        stats: Optional[list[dict, ...]], 默认值为 None
            可选的子任务统计信息列表，用于合并

        返回
        -------
        output: tuple[npt.NDArray[float], npt.NDArray[float]]
            合并后的子任务的分数和角度图
        """
        if len(self.sub_jobs) == 0:
            # 读取体积文件，删除文件并返回结果
            score_file, angle_file = (
                self.output_dir.joinpath(f"{self.tomo_id}_scores_{self.job_key}.mrc"),
                self.output_dir.joinpath(f"{self.tomo_id}_angles_{self.job_key}.mrc"),
            )
            result = (read_mrc(score_file), read_mrc(angle_file))
            # 删除文件
            (score_file.unlink(), angle_file.unlink())
            return result

        if stats is not None:
            # 计算搜索空间的总和
            search_space = sum([s["search_space"] for s in stats])
            # 计算方差的平均值
            variance = sum([s["variance"] for s in stats]) / len(stats)
            # 更新任务统计信息
            self.job_stats = {
                "search_space": search_space,
                "variance": variance,
                "std": np.sqrt(variance),
            }

        # 检查是否为子体积分割
        is_subvolume_split = np.all(
            np.array([x.start_slice for x in self.sub_jobs]) == 0
        )

        score_volumes, angle_volumes = [], []
        for x in self.sub_jobs:
            # 递归合并子任务
            result = x.merge_sub_jobs()
            score_volumes.append(result[0])
            angle_volumes.append(result[1])

        if not is_subvolume_split:
            # 初始化分数和角度数组
            scores, angles = (
                np.zeros_like(score_volumes[0]) - 1.0,
                np.zeros_like(angle_volumes[0]) - 1.0,
            )
            for s, a in zip(score_volumes, angle_volumes):
                # 更新角度数组
                angles = np.where(s > scores, a, angles)
                # 防止切片引起的竞争条件
                angles = np.where(s == scores, np.minimum(a, angles), angles)
                # 更新分数数组
                scores = np.where(s > scores, s, scores)
        else:
            # 初始化分数和角度数组
            scores, angles = (
                np.zeros(self.search_size, dtype=np.float32),
                np.zeros(self.search_size, dtype=np.float32),
            )
            for job, s, a in zip(self.sub_jobs, score_volumes, angle_volumes):
                # 提取子任务的分数和角度
                sub_scores = s[
                    job.sub_start[0] : job.sub_start[0] + job.sub_step[0],
                    job.sub_start[1] : job.sub_start[1] + job.sub_step[1],
                    job.sub_start[2] : job.sub_start[2] + job.sub_step[2],
                ]
                sub_angles = a[
                    job.sub_start[0] : job.sub_start[0] + job.sub_step[0],
                    job.sub_start[1] : job.sub_start[1] + job.sub_step[1],
                    job.sub_start[2] : job.sub_start[2] + job.sub_step[2],
                ]
                # 将提取的子部分放回完整体积中
                scores[
                    job.whole_start[0] : job.whole_start[0] + sub_scores.shape[0],
                    job.whole_start[1] : job.whole_start[1] + sub_scores.shape[1],
                    job.whole_start[2] : job.whole_start[2] + sub_scores.shape[2],
                ] = sub_scores
                angles[
                    job.whole_start[0] : job.whole_start[0] + sub_scores.shape[0],
                    job.whole_start[1] : job.whole_start[1] + sub_scores.shape[1],
                    job.whole_start[2] : job.whole_start[2] + sub_scores.shape[2],
                ] = sub_angles
        # 将分数数组转换为指定的数据类型
        return scores.astype(self.output_dtype), angles

    def start_job(
        self, gpu_id: int, return_volumes: bool = False
    ) -> tuple[npt.NDArray[float], npt.NDArray[float]] | dict:
        """
        在指定的 GPU 上运行此模板匹配任务。任务的搜索统计信息将始终分配给 self.job_stats。

        参数
        ----------
        gpu_id: int
            运行任务的 GPU 索引
        return_volumes: bool, 默认值为 False
            False（默认）不返回体积，而是将其写入磁盘；设置为 True 则直接返回分数和角度体积

        返回
        -------
        output: Union[tuple[npt.NDArray[float], npt.NDArray[float]], dict]
            当返回体积时，输出由两个 numpy 数组（分数和角度图）组成；
            当不返回体积时，输出由包含搜索统计信息的字典组成
        """
        # 记录下一个快速 FFT 形状的信息
        logging.debug(
            "下一个快速 FFT 形状: "
            f"{tuple([next_fast_len(s, real=True) for s in self.search_size])}"
        )
        # 创建一个零数组，用于存储搜索体积
        search_volume = np.zeros(
            tuple([next_fast_len(s, real=True) for s in self.search_size]),
            dtype=np.float32,
        )

        # 加载（子）体积
        volume_mrc = read_mrc(self.tomogram)
        volume_mrc_part = volume_mrc[
                self.search_origin[0] : self.search_origin[0] + self.search_size[0],
                self.search_origin[1] : self.search_origin[1] + self.search_size[1],
                self.search_origin[2] : self.search_origin[2] + self.search_size[2],
            ]
        search_volume[
            : self.search_size[0], : self.search_size[1], : self.search_size[2]
        ] = np.ascontiguousarray(volume_mrc_part)

        # 加载模板和掩码
        template, mask = (read_mrc(self.template), read_mrc(self.mask))

        # 初始化断层图像和模板的加权
        tomo_filter, template_wedge = 1, 1
        # 首先生成带通滤波器
        if not (self.low_pass is None and self.high_pass is None):
            tomo_filter *= create_gaussian_band_pass(
                search_volume.shape, self.voxel_size, self.low_pass, self.high_pass
            ).astype(np.float32)
            template_wedge *= create_gaussian_band_pass(
                self.template_shape, self.voxel_size, self.low_pass, self.high_pass
            ).astype(np.float32)

        # 然后乘以可选的白化滤波器
        if self.whiten_spectrum:
            tomo_filter *= profile_to_weighting(
                np.load(self.whitening_filter), search_volume.shape
            ).astype(np.float32)
            template_wedge *= profile_to_weighting(
                np.load(self.whitening_filter), self.template_shape
            ).astype(np.float32)

        # 如果提供了倾斜角度，可以创建楔形滤波器
        if self.tilt_angles is not None:
            if self.tilt_weighting and self.defocus_handedness != 0:
                # 调整此特定断层图像块的 CTF 参数
                full_tomo_center = np.array(self.tomo_shape) / 2
                patch_center = (
                    np.array(self.search_origin) + np.array(self.search_size) / 2
                )
                relative_patch_center_angstrom = (
                    patch_center - full_tomo_center
                ) * self.voxel_size
                # 计算散焦偏移量
                defocus_offsets = get_defocus_offsets(
                    relative_patch_center_angstrom[0],  # x 坐标
                    relative_patch_center_angstrom[2],  # z 坐标
                    self.tilt_angles,
                    angles_in_degrees=True,
                    invert_handedness=self.defocus_handedness < 0,
                )
                for ctf, defocus_shift in zip(self.ctf_data, defocus_offsets):
                    ctf["defocus"] = ctf["defocus"] + defocus_shift * 1e-10
                logging.debug(
                    "块中心（体素数量）: "
                    f"{np.array_str(relative_patch_center_angstrom, precision=2)}"
                )
                logging.debug(
                    "散焦值（微米）: "
                    f"{[round(ctf['defocus'] * 1e6, 2) for ctf in self.ctf_data]}",
                )

            # 对于断层图像，生成一个二进制楔形滤波器，将缺失楔形区域明确设置为 0
            tomo_filter *= create_wedge(
                search_volume.shape,
                self.tilt_angles,
                self.voxel_size,
                cut_off_radius=1.0,
                angles_in_degrees=True,
                tilt_weighting=False,
            ).astype(np.float32)
            # 对于模板，根据选项生成二进制或按倾斜加权的楔形滤波器
            template_wedge *= create_wedge(
                self.template_shape,
                self.tilt_angles,
                self.voxel_size,
                cut_off_radius=1.0,
                angles_in_degrees=True,
                tilt_weighting=self.tilt_weighting,
                accumulated_dose_per_tilt=self.dose_accumulation,
                ctf_params_per_tilt=self.ctf_data,
            ).astype(np.float32)

            if logging.DEBUG >= logging.root.level:
                # 将模板的点扩散函数写入文件
                write_mrc(
                    self.output_dir.joinpath("template_psf.mrc"),
                    template_wedge,
                    self.voxel_size,
                )
                # 将卷积后的模板写入文件
                write_mrc(
                    self.output_dir.joinpath("template_convolved.mrc"),
                    irfftn(rfftn(template) * template_wedge, s=template.shape),
                    self.voxel_size,
                )

        # 将可选的带通和白化滤波器应用于搜索区域
        search_volume = np.real(
            irfftn(rfftn(search_volume) * tomo_filter, s=search_volume.shape)
        )

        # 加载旋转搜索
        angle_ids = list(range(self.start_slice, self.n_rotations, self.steps_slice))
        angle_list = get_angle_list(
            self.rotation_file,
            sort_angles=version.parse(self.pytom_tm_version_number)
            > version.parse("0.3.0"),
            symmetry=self.rotational_symmetry,
        )

        angle_list = angle_list[
            slice(self.start_slice, self.n_rotations, self.steps_slice)
        ]

        mask_volume = None
        # 加载标记了区域的（子）体积
        if self.tomogram_mask is not None:
            # 创建一个零数组，用于存储搜索体积
            mask_volume = np.zeros(
                tuple([next_fast_len(s, real=True) for s in self.search_size]),
                dtype=np.float32,
            )
            # 加载标记的（子）体积
            mask_volume[
                : self.search_size[0], : self.search_size[1], : self.search_size[2]
            ] = np.ascontiguousarray(
                read_mrc(self.tomogram_mask)[
                    self.search_origin[0] : self.search_origin[0] + self.search_size[0],
                    self.search_origin[1] : self.search_origin[1] + self.search_size[1],
                    self.search_origin[2] : self.search_origin[2] + self.search_size[2],
                ]
            )
            search_volume = search_volume * mask_volume
            search_volume_roi = mask_volume
        else:
            # 用于任务统计的相关部分的切片
            search_volume_roi = (
                slice(self.sub_start[0], self.sub_start[0] + self.sub_step[0]),
                slice(self.sub_start[1], self.sub_start[1] + self.sub_step[1]),
                slice(self.sub_start[2], self.sub_start[2] + self.sub_step[2]),
            )

        # 创建 TemplateMatchingGPU 实例
        tm = TemplateMatchingGPU(
            job_id=self.job_key,
            device_id=gpu_id,
            volume=search_volume,
            template=template,
            mask=mask,
            angle_list=angle_list,
            angle_ids=angle_ids,
            mask_is_spherical=self.mask_is_spherical,
            wedge=template_wedge,
            stats_roi=search_volume_roi,
            noise_correction=self.random_phase_correction,
            rng_seed=self.rng_seed,
            sphere_list=self.sphere_list,
            search_origin=self.search_origin,
            search_size=self.search_size,
            volume_origin_shape=volume_mrc.shape,
            mask_volume=mask_volume
        )
        # 运行模板匹配任务
        results = tm.run()
        # 提取分数体积
        score_volume = results[0][
            : self.search_size[0], : self.search_size[1], : self.search_size[2]
        ]
        # 提取角度体积
        angle_volume = results[1][
            : self.search_size[0], : self.search_size[1], : self.search_size[2]
        ]
        # 记录任务统计信息
        self.job_stats = results[2]

        # 删除模板匹配计划
        del tm

        # 将分数体积转换为指定的数据类型
        score_volume = score_volume.astype(self.output_dtype)
        angle_volume = angle_volume

        if return_volumes:
            # 如果需要返回体积，直接返回分数和角度体积
            return score_volume, angle_volume
        else:  # 否则将它们写入磁盘并返回任务统计信息
            write_mrc(
                self.output_dir.joinpath(f"{self.tomo_id}_scores_{self.job_key}.mrc"),
                score_volume,
                self.voxel_size,
            )
            write_mrc(
                self.output_dir.joinpath(f"{self.tomo_id}_angles_{self.job_key}.mrc"),
                angle_volume,
                self.voxel_size,
            )
            return self.job_stats
