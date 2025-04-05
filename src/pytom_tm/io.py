# 导入 pathlib 模块，用于处理文件路径
import pathlib
# 导入 mrcfile 模块，用于处理 MRC 文件
import mrcfile
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 logging 模块，用于记录日志
import logging
# 导入 numpy 的类型注解模块，用于类型提示
import numpy.typing as npt
# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 starfile 模块，用于处理 STAR 文件
import starfile
# 导入 contextmanager 用于创建上下文管理器
from contextlib import contextmanager
# 导入 attrgetter 用于获取对象属性
from operator import attrgetter


class ParseLogging(argparse.Action):
    """
    argparse.Action 子类，用于从输入脚本中解析日志记录参数。用户可以将其设置为 info/debug。
    """

    def __call__(
        self, parser, namespace, values: str, option_string: str | None = None
    ):
        # 检查输入的日志级别是否为 INFO 或 DEBUG
        if values.upper() not in ["INFO", "DEBUG"]:
            # 若不是，抛出错误提示用户设置正确的日志级别
            parser.error(
                f"{option_string} log got an invalid option, "
                "set either to `info` or `debug` "
            )
        else:
            # 获取对应的日志级别数值
            numeric_level = getattr(logging, values.upper(), None)
            # 将日志级别数值设置到命名空间中
            setattr(namespace, self.dest, numeric_level)


class CheckDirExists(argparse.Action):
    """
    argparse.Action 子类，用于检查预期的输入目录是否存在。
    """

    def __call__(
        self,
        parser,
        namespace,
        values: pathlib.Path,
        option_string: str | None = None,
    ):
        # 检查输入的路径是否为有效的目录
        if not values.is_dir():
            # 若不是，抛出错误提示用户路径不存在
            parser.error(f"{option_string} got a file path that does not exist ")
        # 将路径设置到命名空间中
        setattr(namespace, self.dest, values)


class CheckFileExists(argparse.Action):
    """
    argparse.Action 子类，用于检查预期的输入文件是否存在。
    """

    def __call__(
        self,
        parser,
        namespace,
        values: pathlib.Path,
        option_string: str | None = None,
    ):
        # 检查输入的文件是否存在
        if not values.exists():
            # 若不存在，抛出错误提示用户路径不存在
            parser.error(f"{option_string} got a file path that does not exist ")
        # 将文件路径设置到命名空间中
        setattr(namespace, self.dest, values)


class LargerThanZero(argparse.Action):
    """
    argparse.Action 子类，用于限制输入值必须大于零。
    """

    def __call__(
        self,
        parser,
        namespace,
        values: int | float,
        option_string: str | None = None,
    ):
        # 检查输入值是否小于等于零
        if values <= 0.0:
            # 若小于等于零，抛出错误提示用户输入值必须大于零
            parser.error(f"{option_string} must be larger than 0")
        # 将输入值设置到命名空间中
        setattr(namespace, self.dest, values)


class BetweenZeroAndOne(argparse.Action):
    """
    argparse.Action 子类，用于限制输入值为一个分数，即介于 0 和 1 之间。
    """

    def __call__(
        self, parser, namespace, values: float, option_string: str | None = None
    ):
        # 检查输入值是否不在 0 到 1 之间
        if 1.0 <= values <= 0.0:
            # 若不在范围内，抛出错误提示用户输入值必须在 0 到 1 之间
            parser.error(
                f"{option_string} is a fraction and can only range between 0 and 1"
            )
        # 将输入值设置到命名空间中
        setattr(namespace, self.dest, values)


class ParseSearch(argparse.Action):
    """
    argparse.Action 子类，用于将断层图像的搜索区域限制在沿某个轴的这些索引范围内。
    检查这些值是否大于零，并且第二个值是否大于第一个值。
    """

    def __call__(
        self,
        parser,
        namespace,
        values: list[int, int],
        option_string: str | None = None,
    ):
        # 检查输入的起始和结束索引是否满足条件
        if not (0 <= values[0] < values[1]):
            # 若不满足条件，抛出错误提示用户起始和结束索引的要求
            parser.error(
                f"{option_string} start and end indices must be larger than 0 and end "
                "must be larger than start"
            )
        # 将输入的索引列表设置到命名空间中
        setattr(namespace, self.dest, values)


class ParseTiltAngles(argparse.Action):
    """
    argparse.Action 子类，用于解析倾斜角度信息。输入可以是两个浮点数，用于指定连续楔形模型的倾斜范围。
    或者可以是一个 .tlt/.rawtlt 文件，用于指定倾斜系列的所有倾斜角度，以用于更精细的楔形模型。
    """

    def __call__(
        self,
        parser,
        namespace,
        values: list[str, str] | str,
        option_string: str | None = None,
    ):
        if len(values) == 2:  # 提供了两个楔形角度，分别是最小值和最大值
            try:
                # 将输入值转换为浮点数并排序
                values = sorted(list(map(float, values)))
                # 将排序后的角度列表设置到命名空间中
                setattr(namespace, self.dest, values)
            except ValueError:
                # 若转换失败，抛出错误提示用户输入的参数无法解析为浮点数
                parser.error(
                    f"{option_string} the two arguments provided could not be parsed "
                    "to floats"
                )
        elif len(values) == 1:
            # 将输入值转换为路径对象
            values = pathlib.Path(values[0])
            # 检查文件是否存在以及文件后缀是否为 .tlt 或 .rawtlt
            if not values.exists() or values.suffix not in [".tlt", ".rawtlt"]:
                # 若文件不存在或格式不正确，抛出错误提示用户
                parser.error(
                    f"{option_string} provided tilt angle file does not exist or does "
                    "not have the right format"
                )
            # 读取倾斜角度文件并将结果设置到命名空间中
            setattr(namespace, self.dest, read_tlt_file(values))
        else:
            # 若输入参数数量不符合要求，抛出错误提示用户只能输入一个或两个参数
            parser.error(f"{option_string} can only take one or two arguments")


class ParseGPUIndices(argparse.Action):
    """
    argparse.Action 子类，用于解析 GPU 索引。输入可以是一个整数或一个整数列表，用于指定要使用的 GPU 索引。
    """

    def __call__(
        self,
        parser,
        namespace,
        values: list[int, ...],
        option_string: str | None = None,
    ):
        # 导入 cupy 库，用于处理 GPU 相关操作
        import cupy
        # 获取可用的 GPU 设备数量
        max_value = cupy.cuda.runtime.getDeviceCount()
        # 遍历输入的 GPU 索引列表
        for val in values:
            # 检查索引是否在有效范围内
            if val < 0 or val >= max_value:
                # 若索引无效，抛出错误提示用户索引必须在有效范围内
                parser.error(
                    f"{option_string} all gpu indices should be between 0 "
                    f"and {max_value - 1}"
                )
        # 将有效的 GPU 索引列表设置到命名空间中
        setattr(namespace, self.dest, values)


class ParseDoseFile(argparse.Action):
    """
    argparse.Action 子类，用于解析包含每个倾斜累积剂量信息的文本文件。
    """

    def __call__(
        self, parser, namespace, values: str, option_string: str | None = None
    ):
        # 将输入值转换为路径对象
        file_path = pathlib.Path(values)
        # 检查文件是否存在
        if not file_path.exists():
            # 若文件不存在，抛出错误提示用户文件不存在
            parser.error(
                f"{option_string} provided dose accumulation file does not exist"
            )
        # 定义允许的文件后缀
        allowed_suffixes = [".txt"]
        # 检查文件后缀是否在允许的范围内
        if file_path.suffix not in allowed_suffixes:
            # 若后缀不符合要求，抛出错误提示用户文件后缀不正确
            parser.error(
                f"{option_string}  provided dose accumulation file does not have the "
                f"right suffix, allowed are: {', '.join(allowed_suffixes)}"
            )
        # 读取剂量文件并将结果设置到命名空间中
        setattr(namespace, self.dest, read_dose_file(file_path))


class ParseDefocus(argparse.Action):
    """
    argparse.Action 子类，用于读取散焦文件，该文件可以是遵循 IMOD 文件格式的文件，
    也可以是每行包含每个倾斜散焦值的文本文件。
    """

    def __call__(
        self, parser, namespace, values: str, option_string: str | None = None
    ):
        # 检查输入值的文件后缀是否为 .defocus 或 .txt
        if values.endswith((".defocus", ".txt")):
            # 将输入值转换为路径对象
            file_path = pathlib.Path(values)
            # 检查文件是否存在
            if not file_path.exists():
                # 若文件不存在，抛出错误提示用户文件不存在
                parser.error(f"{option_string} provided defocus file does not exist")
            # 读取散焦文件并将结果设置到命名空间中
            setattr(namespace, self.dest, read_defocus_file(file_path))
        else:
            try:
                # 尝试将输入值转换为浮点数
                defocus = float(values)
            except ValueError:
                # 若转换失败，抛出错误提示用户无法将输入值读取为浮点数
                parser.error(f"{option_string} not possible to read defocus as float")
            # 将散焦值作为列表设置到命名空间中
            setattr(namespace, self.dest, [defocus])


class UnequalSpacingError(Exception):
    """
    当 MRC 文件的 xyz 维度上的体素间距在其元数据中注释不相等时抛出的异常。
    """

    pass


def write_angle_list(
    data: npt.NDArray[float],
    file_name: pathlib.Path,
    order: tuple[int, int, int] = (0, 2, 1),
):
    """
    辅助函数，用于将旧 PyTom 中的角度搜索列表写入当前模块。
    由于旧 PyTom 总是将其存储为 Z1, Z2, X，而这里是 Z1, X, Z2，因此需要更改顺序。

    @todo remove function
    """
    # 以写入模式打开文件
    with open(file_name, "w") as fstream:
        # 遍历数据的列
        for i in range(data.shape[1]):
            # 按照指定顺序提取数据并写入文件，每个值用空格分隔，每行末尾添加换行符
            fstream.write(
                " ".join([str(x) for x in [data[j, i] for j in order]]) + "\n"
            )


@contextmanager
def _wrap_mrcfile_readers(func, *args, **kwargs):
    """
    尝试恢复损坏的 MRC 文件，假设 'permissive' 是一个关键字参数而不是位置参数。
    """
    try:
        # 尝试调用传入的函数打开 MRC 文件
        mrc = func(*args, **kwargs)
    except ValueError as err:
        # 若打开失败，记录错误信息并尝试以宽松模式打开
        logging.debug(f"mrcfile raised the following error: {err}, will try to recover")
        kwargs["permissive"] = True
        mrc = func(*args, **kwargs)
        if mrc.data is not None:
            # 若宽松模式下成功获取到数据，记录警告信息提醒用户检查数据的正确性
            logging.warning(
                f"Loading {args[0]} in strict mode gave an error. "
                "However, loading with 'permissive=True' did generate data, make sure "
                "this is correct!"
            )
        else:
            # 若宽松模式下仍无法获取到有效数据，记录调试信息并抛出异常
            logging.debug("Could not reasonably recover")
            raise ValueError(
                f"{args[0]} header or data is too corrupt to recover, please fix the "
                "header or data"
            ) from err
    # 生成 MRC 文件对象供上下文使用
    yield mrc
    # 上下文结束后关闭 MRC 文件
    mrc.close()


def read_mrc_meta_data(file_name: pathlib.Path) -> dict:
    """
    读取提供的 MRC 文件路径的元数据（使用 mrcfile）并作为字典返回。

    如果 x、y 和 z 维度上的体素大小差异很大（小数点后三位不一致），
    函数将引发 UnequalSpacingError，因为这可能意味着在这些体积上进行模板匹配可能不一致。

    参数
    ----------
    file_name: pathlib.Path
        MRC 文件的路径

    返回
    -------
    metadata: dict
        一个包含 MRC 元数据的字典，键 'shape' 包含文件的 x、y、z 维度，
        键 'voxel_size' 包含 x、y 和 z 维度上的体素大小，单位为 Å
    """
    # 初始化元数据字典
    meta_data = {}
    # 使用上下文管理器打开 MRC 文件
    with _wrap_mrcfile_readers(mrcfile.mmap, file_name) as mrc:
        # 获取 MRC 文件的形状信息并存储到元数据字典中
        meta_data["shape"] = tuple(map(int, attrgetter("nx", "ny", "nz")(mrc.header)))
        # 允许 MRC 头文件中的体素大小存在小的数值不一致，有时在 Warp 中会看到这种情况
        if not all(
            [
                np.round(mrc.voxel_size.x, 3) == np.round(s, 3)
                for s in attrgetter("y", "z")(mrc.voxel_size)
            ]
        ):
            # 若体素大小不一致，抛出异常提示用户输入体积的体素间距在各维度上不相同
            raise UnequalSpacingError(
                "Input volume voxel spacing is not identical in each dimension!"
            )
        else:
            if not all(
                [mrc.voxel_size.x == s for s in attrgetter("y", "z")(mrc.voxel_size)]
            ):
                # 若体素大小存在微小差异，记录警告信息提醒用户检查是否会有问题
                logging.warning(
                    "Voxel size annotation in MRC is slightly different between "
                    f"dimensions, namely {mrc.voxel_size}. It might be a tiny "
                    "numerical inaccuracy, but please ensure this is not problematic."
                )
            # 将体素大小存储到元数据字典中
            meta_data["voxel_size"] = float(mrc.voxel_size.x)
    return meta_data


def write_mrc(
    file_name: pathlib.Path,
    data: npt.NDArray[float],
    voxel_size: float,
    overwrite: bool = True,
    transpose: bool = True,
) -> None:
    """
    将数据写入 MRC 文件。在写入之前，数据会进行转置，因为 pytom 内部使用 xyz 顺序，而 MRC 使用 zyx 顺序。

    参数
    ----------
    file_name: pathlib.Path
        要写入文件的磁盘路径
    data: npt.NDArray[float]
        要作为 MRC 写入的 numpy 数组
    voxel_size: float
        要在 MRC 头文件中注释的数组体素大小
    overwrite: bool, default True
        True（默认）将覆盖路径上的当前 MRC 文件，设置为 False 时，写入现有文件将出错
    transpose: bool, default True
        True（默认）在写入之前转置数组，设置为 False 可防止转置，但在使用此模块的函数时可能不是一个好主意

    返回
    -------
    """
    # 检查数据的数据类型是否为 np.float32 或 np.float16
    if data.dtype not in [np.float32, np.float16]:
        # 若不是，记录警告信息并将数据类型转换为 np.float32
        logging.warning(
            "data for mrc writing is not np.float32 or np.float16, will convert to "
            "np.float32"
        )
        data = data.astype(np.float32)
    # 使用 mrcfile 写入文件，根据 transpose 参数决定是否转置数据
    mrcfile.write(
        file_name,
        data.T if transpose else data,
        voxel_size=voxel_size,
        overwrite=overwrite,
    )


def read_mrc(file_name: pathlib.Path, transpose: bool = True) -> npt.NDArray[float]:
    """
    从磁盘读取 MRC 文件。读取后，数据会进行转置，因为 pytom 内部使用 xyz 顺序，而 MRC 使用 zyx 顺序。

    参数
    ----------
    file_name: pathlib.Path
        磁盘上文件的路径
    transpose: bool, default True
        True（默认）在读取后转置体积，设置为 False 可防止转置，但在使用此模块的函数时可能不是一个好主意

    返回
    -------
    data: npt.NDArray[float]
        以 numpy 数组形式返回 MRC 数据
    """
    # 使用上下文管理器打开 MRC 文件
    with _wrap_mrcfile_readers(mrcfile.open, file_name) as mrc:
        # 根据 transpose 参数决定是否转置数据并将其转换为连续数组
        data = np.ascontiguousarray(mrc.data.T) if transpose else mrc.data
    return data


def read_txt_file(file_name: pathlib.Path) -> list[float, ...]:
    """
    从磁盘读取一个文本文件，每行包含一个浮点数。

    参数
    ----------
    file_name: pathlib.Path
        要读取的磁盘文件

    返回
    -------
    output: list[float, ...]
        浮点数列表
    """
    # 以只读模式打开文件
    with open(file_name) as fstream:
        # 读取文件的所有行
        lines = fstream.readlines()
    # 去除每行的空白字符，过滤掉空白行，将每行转换为浮点数并存储到列表中
    return list(map(float, [x.strip() for x in lines if not x.isspace()]))


def read_tlt_file(file_name: pathlib.Path) -> list[float, ...]:
    """
    使用 read_txt_file() 从磁盘读取一个文本文件。文件预计包含倾斜角度，单位为度。

    参数
    ----------
    file_name: pathlib.Path
        要读取的磁盘文件

    返回
    -------
    output: list[float, ...]
        包含倾斜角度的浮点数列表
    """
    # 调用 read_txt_file 函数读取文件内容
    return read_txt_file(file_name)


def read_dose_file(file_name: pathlib.Path) -> list[float, ...]:
    """
    使用 read_txt_file() 从磁盘读取一个文本文件。文件预计包含累积剂量，单位为 e-/(Å^2)。

    参数
    ----------
    file_name: pathlib.Path
        要读取的磁盘文件

    返回
    -------
    output: list[float, ...]
        包含累积剂量的浮点数列表
    """
    # 调用 read_txt_file 函数读取文件内容
    return read_txt_file(file_name)


def read_imod_defocus_file(file_name: pathlib.Path) -> list[float, ...]:
    """
    读取 IMOD 风格的散焦文件。此函数可以读取版本 2 和 3 的散焦文件。
    有关格式规范，请参阅：https://bio3d.colorado.edu/imod/doc/man/ctfphaseflip.html（部分：散焦文件格式）。

    参数
    ----------
    file_name: pathlib.Path
        要读取的磁盘文件

    返回
    -------
    output: list[float, ...]
        包含散焦值（单位为 μm）的浮点数列表
    """
    # 以只读模式打开文件
    with open(file_name) as fstream:
        # 读取文件的所有行
        lines = fstream.readlines()
    # 获取 IMOD 散焦文件的版本号
    imod_defocus_version = float(lines[0].strip().split()[5])
    # IMOD 散焦文件中的值以 nm 为单位
    if imod_defocus_version == 2:  # 包含一个散焦值的文件；数据从第 0 行开始
        # 提取每行的散焦值并转换为 μm 单位
        return [float(x.strip().split()[4]) * 1e-3 for x in lines]
    elif (
        imod_defocus_version == 3
    ):  # 包含像散的文件；第 0 行包含我们不需要的元数据
        # 提取除第一行外每行的散焦值并转换为 μm 单位
        return [
            (float(x.strip().split()[4]) + float(x.strip().split()[5])) / 2 * 1e-3
            for x in lines[1:]
        ]
    else:
        # 若版本号不是 2 或 3，抛出异常提示用户文件版本无效
        raise ValueError("Invalid IMOD defocus file inversion, can only be 2 or 3.")


def read_defocus_file(file_name: pathlib.Path) -> list[float, ...]:
    """
    读取散焦文件，文件中的值以 nm 为单位。输出的散焦值以 μm 为单位。

    根据文件后缀，函数会调用：
     - read_imod_defocus_file() 处理 .defocus 后缀的文件
     - read_txt_file 处理 .txt 后缀的文件

    参数
    ----------
    file_name: pathlib.Path
        要读取的磁盘文件

    返回
    -------
    output: list[float, ...]
        包含散焦值（单位为 μm）的浮点数列表
    """
    # 检查文件后缀是否为 .defocus
    if file_name.suffix == ".defocus":
        # 若为 .defocus 后缀，调用 read_imod_defocus_file 函数读取文件
        return read_imod_defocus_file(file_name)
    # 检查文件后缀是否为 .txt
    elif file_name.suffix == ".txt":
        # 若为 .txt 后缀，调用 read_txt_file 函数读取文件
        return read_txt_file(file_name)
    else:
        # 若文件后缀不符合要求，抛出异常提示用户文件格式必须为 .defocus 或 .txt
        raise ValueError("Defocus file needs to have format .defocus or .txt")


def parse_relion5_star_data(
    tomograms_star_path: pathlib.Path,
    tomogram_path: pathlib.Path,
    phase_flip_correction: bool = False,
    phase_shift: float = 0.0,
) -> tuple[float, list[float, ...], list[float, ...], list[dict, ...], int]:
    """
    从项目目录中读取 RELION5 元数据。

    参数
    ----------
    tomograms_star_path: pathlib.Path
        RELION5 重建作业中的 tomograms.star 文件包含不变的元数据，
        并指向一个包含拟合值的倾斜系列 STAR 文件
    tomogram_path: pathlib.Path
        用于模板匹配的断层图像的路径；我们使用名称在 RELION5 STAR 文件中进行模式匹配
    phase_flip_correction: bool, default False
    phase_shift: float, default 0.0

    返回
    -------
    tomogram_voxel_size, tilt_angles, dose_accumulation, ctf_params, defocus_handedness:
        tuple[float, list[float, ...], list[float, ...], list[dict, ...], int]
    """
    # 获取断层图像的文件名（不包含扩展名）
    tomogram_id = tomogram_path.stem
    # 读取 tomograms.star 文件
    tomograms_star_data = starfile.read(tomograms_star_path)

    # 匹配断层图像 ID 并检查是否可行
    matches = [
        i
        for i, x in enumerate(tomograms_star_data["rlnTomoName"])
        if tomogram_id.endswith(x)
    ]
    if len(matches) == 1:
        # 若匹配到一个结果，获取对应的断层图像元数据
        tomogram_meta_data = tomograms_star_data.loc[matches[0]]
    else:
        # 若匹配结果数量不为 1，抛出异常提示用户匹配结果的情况
        raise ValueError(
            f"{'Multiple' if len(matches) > 1 else 'Zero'} matches "
            f"of tomogram id: {tomogram_id}, "
            f"in RELION5 STAR file: {tomograms_star_path}. "
            "Aborting..."
        )

    # 获取倾斜系列 STAR 文件的路径
    tilt_series_star_path = pathlib.Path(
        tomogram_meta_data["rlnTomoTiltSeriesStarFile"]
    )
    # 更新路径为实际可找到的位置
    tilt_series_star_path = tomograms_star_path.parent.joinpath("tilt_series").joinpath(
        tilt_series_star_path.name
    )
    # 读取倾斜系列 STAR 文件
    tilt_series_star_data = starfile.read(tilt_series_star_path)

    # 提取倾斜角度、剂量累积和 CTF 参数
    # TODO 我们需要为倾斜系列元数据建立一个内部结构
    tilt_angles = list(tilt_series_star_data["rlnTomoNominalStageTiltAngle"])
    dose_accumulation = list(tilt_series_star_data["rlnMicrographPreExposure"])

    # 计算断层图像的体素大小
    tomogram_voxel_size = float(
        tomogram_meta_data["rlnTomoTiltSeriesPixelSize"]
        * tomogram_meta_data["rlnTomoTomogramBinning"]
    )
    # 获取散焦手性
    defocus_handedness = int(tomogram_meta_data["rlnTomoHand"])

    # 构建 CTF 参数列表
    ctf_params = [
        {
            "defocus": defocus * 1e-10,
            "amplitude_contrast": tomogram_meta_data["rlnAmplitudeContrast"],
            "voltage": tomogram_meta_data["rlnVoltage"] * 1e3,
            "spherical_aberration": tomogram_meta_data["rlnSphericalAberration"] * 1e-3,
            "flip_phase": phase_flip_correction,
            "phase_shift_deg": phase_shift,  # RELION5 似乎不存储此信息
        }
        for defocus in (
            tilt_series_star_data.rlnDefocusV + tilt_series_star_data.rlnDefocusU
        )
        / 2
    ]

    return (
        tomogram_voxel_size,
        tilt_angles,
        dose_accumulation,
        ctf_params,
        defocus_handedness,
    )
