# 导入packaging库的version模块，用于版本比较
from packaging import version
# 导入pandas库，用于数据处理和分析
import pandas as pd
# 导入numpy库，用于数值计算
import numpy as np
# 导入numpy的类型注解模块，用于类型提示
import numpy.typing as npt
# 导入logging模块，用于记录日志
import logging
# 导入scipy.ndimage模块，用于图像处理和滤波操作
import scipy.ndimage as ndimage
# 导入pathlib模块，用于处理文件路径
import pathlib
# 从pytom_tm.tmjob模块导入TMJob类
from pytom_tm.tmjob import TMJob
# 从pytom_tm.mask模块导入spherical_mask函数
from pytom_tm.mask import spherical_mask
# 从pytom_tm.angles模块导入get_angle_list和convert_euler函数
from pytom_tm.angles import get_angle_list, convert_euler
# 从pytom_tm.io模块导入read_mrc函数，用于读取MRC格式的文件
from pytom_tm.io import read_mrc
# 从scipy.special模块导入erfcinv函数，用于误差函数的反函数计算
from scipy.special import erfcinv
# 从scipy.optimize模块导入curve_fit函数，用于曲线拟合
from scipy.optimize import curve_fit
# 导入tqdm模块，用于显示进度条
from tqdm import tqdm

# 初始化绘图可用性标志
plotting_available = False
try:
    # 尝试导入matplotlib.pyplot和seaborn库，用于绘图
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 设置seaborn的绘图风格
    sns.set(context="talk", style="ticks")
    # 若导入成功，设置绘图可用性标志为True
    plotting_available = True
except ModuleNotFoundError:
    # 若导入失败，不做处理
    pass


def predict_tophat_mask(
    score_volume: npt.NDArray[float],
    output_path: pathlib.Path | None = None,
    n_false_positives: float = 1.0,
    create_plot: bool = True,
    tophat_connectivity: int = 1,
    bins: int = 50,
) -> npt.NDArray[bool]:
    """
    此函数接收一个分数图作为输入，并返回一个通过顶帽变换确定的峰值掩码。

    它执行以下操作：
    - 使用scipy.ndimage.white_tophat()和一个内核（ndimage.generate_binary_structure(rank=3, connectivity=1)）计算顶帽变换。
    - 计算变换后分数图的直方图，并取其对数以更关注小值。
    - 对对数直方图取二阶导数，以找到拟合高斯分布的区域，当二阶导数从负变为正时，背景噪声可能开始变化。
    - 使用Rickgauer等人（2017年，eLife）的优秀工作中的公式，该公式使用误差函数来确定背景高斯分布上的假阳性可能性：N**(-1) = erfc( theta / ( sigma * sqrt(2) ) ) / 2

    参数
    ----------
    score_volume: npt.NDArray[float]
        模板匹配分数图
    output_path: Optional[pathlib.Path], default None
        如果提供（且绘图可用），将拟合图写入输出文件夹
    n_false_positives: float, default 1.0
        用于误差函数截止计算的假阳性数量
    create_plot: bool, default True
        是否绘制高斯拟合和截止估计图
    tophat_connectivity: int, default 1
        二进制结构的连通性
    bins: int, default 50
        用于估计和绘图的直方图的箱数

    返回
    -------
    peak_mask: npt.NDArray[bool]
        包含顶帽滤波后峰值位置的布尔掩码
    """
    # 检查分数图的数据类型，如果是float16，将其转换为float32，避免ndimage.white_tophat()失败和score_volume.std()产生np.inf
    if score_volume.dtype == np.float16:
        score_volume = score_volume.astype(np.float32)
    # 计算顶帽变换
    tophat = ndimage.white_tophat(
        score_volume,
        structure=ndimage.generate_binary_structure(
            rank=3, connectivity=tophat_connectivity
        ),
    )
    # 计算顶帽变换结果的直方图
    y, bins = np.histogram(tophat.flatten(), bins=bins)
    # 计算直方图箱的中心位置
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # 丢弃前两个点，因为零值可能过度代表
    x_raw, y_raw = (
        bin_centers[2:],
        y[2:],
    )
    # 对对数直方图取二阶导数，并丢弃不准确的边界值
    with np.errstate(divide="ignore"):
        y_log = np.log(y_raw)
        y_log[np.isinf(y_log)] = 0
    second_derivative = np.gradient(np.gradient(y_log))[2:]
    # 找到二阶导数为负的位置
    m1 = second_derivative[:-1] < 0
    # 计算二阶导数符号变化的位置
    sign = np.sign(second_derivative[1:] * second_derivative[:-1])
    # 处理二阶导数为零的情况
    sign = np.where(sign == 0, np.roll(sign, -1), sign)
    sign = np.where(sign == 0, np.roll(sign, 1), sign)
    # 找到二阶导数从负变为正的第一个位置
    m2 = sign == -1
    idx = (
        int(np.argmax(m1 & m2))
        + 2
        + 1
    )
    # 提取用于拟合的区域
    x_fit, y_fit = x_raw[:idx], y_raw[:idx]

    def gauss(x, amp, mu, sigma):
        """高斯函数，用于拟合"""
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def log_gauss(x, amp, mu, sigma):
        """高斯函数的对数，用于拟合"""
        return np.log(gauss(x, amp, mu, sigma))

    # 对高斯函数的参数进行初始猜测
    guess = np.array(
        [y.max(), 0, score_volume.std()]
    )
    # 首先对常规高斯函数进行拟合，以获得更好的初始猜测
    coeff = curve_fit(gauss, x_fit, y_fit, p0=guess)[
        0
    ]
    # 对对数高斯函数进行精确拟合
    coeff_log = curve_fit(log_gauss, x_fit, np.log(y_fit), p0=coeff)[
        0
    ]
    # 计算搜索空间
    search_space = coeff_log[0] / (coeff_log[2] * np.sqrt(2 * np.pi))
    # 根据Rickgauer等人（2017年，eLife）的公式计算截止值
    cut_off = (
        erfcinv((2 * n_false_positives) / search_space) * np.sqrt(2) * coeff_log[2]
        + coeff_log[1]
    )

    # 如果绘图可用且提供了输出路径，并且需要创建绘图
    if plotting_available and output_path is not None and create_plot:
        # 创建绘图对象
        fig, ax = plt.subplots()
        # 绘制顶帽变换结果的散点图
        ax.scatter(x_raw, y_raw, label="tophat", marker="o")
        # 绘制拟合的高斯曲线
        ax.plot(x_raw, gauss(x_raw, *coeff_log), label="pred", color="tab:orange")
        # 绘制截止线
        ax.axvline(cut_off, color="gray", linestyle="dashed", label="cut-off")
        # 绘制拟合区域的阴影
        ax.axvspan(x_fit[0], x_fit[-1], alpha=0.25, color="gray", label="fitted data")
        # 设置y轴为对数刻度
        ax.set_yscale("log")
        # 设置y轴下限
        ax.set_ylim(bottom=0.1)
        # 设置y轴标签
        ax.set_ylabel("Occurence")
        # 设置x轴标签
        ax.set_xlabel("Tophat scores")
        # 显示图例
        ax.legend()
        # 调整布局
        plt.tight_layout()
        # 保存绘图
        plt.savefig(output_path, dpi=600, transparent=False, bbox_inches="tight")

    # 创建峰值掩码
    peak_mask = tophat > cut_off

    return peak_mask


def extract_particles(
    job: TMJob,
    n_particles: int,
    particle_diameter: float | None = None,
    cut_off: float | None = None,
    n_false_positives: float = 1.0,
    tomogram_mask_path: pathlib.Path | None = None,
    tophat_filter: bool = False,
    create_plot: bool = True,
    tophat_connectivity: int = 1,
    relion5_compat: bool = False,
    ignore_tomogram_mask: bool = False,
    tophat_bins: int = 50,
    plot_bins: int = 20,
) -> tuple[pd.DataFrame, list[float, ...]]:
    """
    从模板匹配作业中提取粒子。

    参数
    ----------
    job: pytom_tm.tmjob.TMJob
        用于注释粒子的模板匹配作业
    n_particles: int
        要提取的最大粒子数
    particle_diameter: Optional[float]
        粒子直径，用于在注释分数后移除峰值。提取后最小峰间距离将为直径/2。
    cut_off: Optional[float]
        手动覆盖自动分数截止估计，值应在0和1之间
    n_false_positives: float, default 1.0
        调整自动误差函数截止估计中包含的假阳性数量：应为大于0的浮点数
    tomogram_mask_path: Optional[pathlib.Path]
        用于提取的断层图像二进制掩码的路径，将覆盖job.tomogram_mask
    tophat_filter: bool
        尝试使用顶帽滤波器仅选择尖锐的峰值
    create_plot: bool, default True
        创建提取图的标志
    tophat_connectivity: int, default 1
        顶帽变换内核的连通性
    relion5_compat: bool, default False
        Relion5兼容性，将坐标相对于中心并以埃为单位写入
        中心定义应为：tomo_shape / 2 - 1
    ignore_tomogram_mask: bool, default False
        调试选项，强制代码忽略job.tomogram_mask和输入掩码。
        允许在不重新运行TM作业的情况下重新提取（假设分数体积看起来合理）
    tophat_bins: int, default 50
        顶帽直方图中使用的箱数
    plot_bins: int, default 20
        用于绘制出现次数直方图的箱数

    返回
    -------
    dataframe, scores: tuple[pd.DataFrame, list[float, ...]]
        包含可作为STAR文件写出的注释的数据框和所选分数的列表
    """
    # 读取模板匹配的分数体积文件
    score_volume = read_mrc(job.output_dir.joinpath(f"{job.tomo_id}_scores.mrc"))
    # 读取模板匹配的角度体积文件
    angle_volume = read_mrc(job.output_dir.joinpath(f"{job.tomo_id}_angles.mrc"))
    # 获取角度列表
    angle_list = get_angle_list(
        job.rotation_file,
        sort_angles=version.parse(job.pytom_tm_version_number) > version.parse("0.3.0"),
        symmetry=job.rotational_symmetry,
    )

    # 如果使用顶帽滤波器
    if tophat_filter:
        # 预测峰值掩码
        predicted_peaks = predict_tophat_mask(
            score_volume,
            output_path=job.output_dir.joinpath(f"{job.tomo_id}_tophat_filter.svg"),
            n_false_positives=n_false_positives,
            create_plot=create_plot,
            tophat_connectivity=tophat_connectivity,
            bins=tophat_bins,
        )
        # 将分数体积与预测的峰值掩码相乘，只保留峰值位置
        score_volume *= predicted_peaks

    # 初始化断层图像掩码
    tomogram_mask = None
    # 如果忽略断层图像掩码
    if ignore_tomogram_mask:
        logging.warning("Ignoring tomogram mask")
    # 如果提供了断层图像掩码路径
    elif tomogram_mask_path is not None:
        tomogram_mask = read_mrc(tomogram_mask_path)
    # 如果作业中提供了断层图像掩码
    elif job.tomogram_mask is not None:
        tomogram_mask = read_mrc(job.tomogram_mask)

    # 如果存在断层图像掩码
    if tomogram_mask is not None:
        # 检查断层图像掩码和断层图像的形状是否一致
        if tomogram_mask.shape != job.tomo_shape:
            raise ValueError(
                "Tomogram mask does not have the same number of pixels as the "
                f"tomogram.\n Tomogram mask shape: {tomogram_mask.shape}, "
                f"tomogram shape: {job.tomo_shape}"
            )
        # 计算搜索区域的切片
        slices = [slice(origin, origin + size) for origin, size in zip(job.search_origin, job.search_size)]
        # 提取搜索区域的断层图像掩码
        tomogram_mask = tomogram_mask[slices[0], slices[1], slices[2]]
        # 将掩码小于等于0的区域的分数体积置为0
        score_volume[tomogram_mask <= 0] = 0

    # 如果提供了粒子直径
    if particle_diameter is not None:
        # 计算粒子半径的像素值
        particle_radius_px = int((particle_diameter / 2) / job.voxel_size)
    # 如果作业中提供了粒子直径
    elif job.particle_diameter is not None:
        particle_radius_px = int((job.particle_diameter / 2) / job.voxel_size)
        logging.info(
            "No particle diameter was provided, so using the diameter "
            "specified previously to mask out areas around peaks. Take care for "
            "strongly elongated particles as it might prevent correct "
            "annotation when they arrange parallel to each other and close together."
        )
    else:
        raise ValueError(
            "You need to specify a particle diameter to mask out areas around each "
            "peak during extraction!"
        )

    # 对分数体积的边缘进行掩码处理
    score_volume[0:particle_radius_px, :, :] = 0
    score_volume[:, 0:particle_radius_px, :] = 0
    score_volume[:, :, 0:particle_radius_px] = 0
    score_volume[-particle_radius_px:, :, :] = 0
    score_volume[:, -particle_radius_px:, :] = 0
    score_volume[:, :, -particle_radius_px:] = 0

    # 获取模板匹配作业的标准差
    sigma = job.job_stats["std"]
    # 获取模板匹配作业的搜索空间
    search_space = job.job_stats["search_space"]
    # 如果未提供截止值
    if cut_off is None:
        # 根据Rickgauer等人（2017年，eLife）的公式计算截止值
        cut_off = erfcinv((2 * n_false_positives) / search_space) * np.sqrt(2) * sigma
        logging.info(f"cut off for particle extraction: {cut_off}")
    # 如果提供的截止值小于0
    elif cut_off < 0:
        logging.warning(
            "Provided extraction score cut-off is smaller than 0. Changing to 0 as "
            "that is smallest allowed value."
        )
        cut_off = 0

    # 计算用于屏蔽的盒子大小
    cut_box = int(particle_radius_px) * 2 + 1
    # 创建球形掩码
    cut_mask = (spherical_mask(cut_box, particle_radius_px, cut_box // 2) == 0) * 1

    # 初始化STAR文件所需的数据
    pixel_size = job.voxel_size
    tomogram_id = job.tomo_id

    # 如果需要Relion5兼容性
    if relion5_compat and tomogram_id.startswith("rec_"):
        tomogram_id = tomogram_id[4:]

    # 初始化数据列表和分数列表
    data = []
    scores = []

    # 循环提取粒子
    for _ in tqdm(range(n_particles)):
        # 找到分数体积中的最大值索引
        ind = np.unravel_index(score_volume.argmax(), score_volume.shape)
        # 获取最大值
        lcc_max = score_volume[ind]

        # 如果最大值小于等于截止值，停止提取
        if lcc_max <= cut_off:
            break

        # 将最大值添加到分数列表中
        scores.append(lcc_max)

        # 根据CCPEM的旋转约定，将角度转换为Relion使用的顺时针ZYZ旋转
        rotation = convert_euler(
            [-1 * a for a in angle_list[int(angle_volume[ind])]],
            order_in="ZXZ",
            order_out="ZYZ",
            degrees_in=False,
            degrees_out=True,
        )

        # 计算粒子的位置
        location = [i + o for i, o in zip(job.search_origin, ind)]

        # 将粒子信息添加到数据列表中
        data.append(
            (
                location[0],  # CoordinateX
                location[1],  # CoordinateY
                location[2],  # CoordinateZ
                rotation[0],  # AngleRot
                rotation[1],  # AngleTilt
                rotation[2],  # AnglePsi
                lcc_max,  # LCCmax
                cut_off,  # Extraction cut off
                sigma,  # Add sigma of template matching search, LCCmax/sigma = SNR
                pixel_size,  # DetectorPixelSize
                tomogram_id,  # MicrographName
            )
        )

        # 屏蔽已提取粒子周围的区域
        start = [i - particle_radius_px for i in ind]
        score_volume[
            start[0] : start[0] + cut_box,
            start[1] : start[1] + cut_box,
            start[2] : start[2] + cut_box,
        ] *= cut_mask

    # 创建包含粒子信息的数据框
    output = pd.DataFrame(
        data,
        columns=[
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnLCCmax",
            "rlnCutOff",
            "rlnSearchStd",
            "rlnDetectorPixelSize",
            "rlnMicrographName",
        ],
    )

    # 如果需要Relion5兼容性
    if relion5_compat:
        # 计算断层图像的中心位置
        dims = np.array(job.tomo_shape)
        center = dims / 2
        # 将坐标转换为相对于中心的埃为单位
        output["rlnCoordinateX"], output["rlnCoordinateY"], output["rlnCoordinateZ"] = (
            (output["rlnCoordinateX"] - center[0]) * job.voxel_size,
            (output["rlnCoordinateY"] - center[1]) * job.voxel_size,
            (output["rlnCoordinateZ"] - center[2]) * job.voxel_size,
        )
        # 重命名列名
        column_change = {
            "rlnCoordinateX": "rlnCenteredCoordinateXAngst",
            "rlnCoordinateY": "rlnCenteredCoordinateYAngst",
            "rlnCoordinateZ": "rlnCenteredCoordinateZAngst",
            "rlnMicrographName": "rlnTomoName",
            "rlnDetectorPixelSize": "rlnTomoTiltSeriesPixelSize",
        }
        output = output.rename(columns=column_change)

    # 如果绘图可用且需要创建绘图
    if plotting_available and create_plot:
        # 绘制分数的直方图
        y, bins = np.histogram(scores, bins=plot_bins)
        x = (bins[1:] + bins[:-1]) / 2
        hist_step = bins[1] - bins[0]
        x_ext = np.concatenate((np.linspace(x[0] - 5 * hist_step, x[0], 10), x))
        noise_amplitude = (search_space / (sigma * np.sqrt(2 * np.pi))) * hist_step
        y_background = noise_amplitude * np.exp(-(x_ext**2) / (2 * sigma**2))

        # 创建绘图对象
        fig, ax = plt.subplots()
        # 绘制提取的分数散点图
        ax.scatter(x, y, label="extracted", marker="o")
        # 绘制背景高斯曲线
        ax.plot(x_ext, y_background, label="background", color="tab:orange")
        # 绘制截止线
        ax.axvline(cut_off, color="gray", linestyle="dashed", label="cut-off")
        # 设置y轴范围
        ax.set_ylim(bottom=0, top=2 * max(y))
        # 设置y轴标签
        ax.set_ylabel("Occurence")
        # 设置x轴标签
        ax.set_xlabel(r"${LCC}_{max}$")
        # 显示图例
        ax.legend()
        # 调整布局
        plt.tight_layout()
        # 保存绘图
        plt.savefig(
            job.output_dir.joinpath(f"{job.tomo_id}_extraction_graph.svg"),
            dpi=600,
            transparent=False,
            bbox_inches="tight",
        )

    return output, scores
