# 导入NumPy库，用于数值计算
import numpy as np
# 导入traceback模块，用于获取异常的堆栈信息
import traceback
# 导入itertools模块，用于高效迭代操作
import itertools
# 从SciPy库的optimize模块导入curve_fit函数，用于曲线拟合
from scipy.optimize import curve_fit
# 从SciPy库的special模块导入erf函数，用于误差函数计算
from scipy.special import erf

try:
    # 尝试导入matplotlib.pyplot库，用于绘图
    import matplotlib.pyplot as plt
    # 尝试导入seaborn库，用于美化绘图
    import seaborn as sns
except ModuleNotFoundError:
    # 若未找到matplotlib和seaborn库，抛出运行时错误
    raise RuntimeError(
        "ROC estimation can only be done when matplotlib and seaborn are installed."
    )
# 设置seaborn的绘图风格
sns.set(context="talk", style="ticks")


class ScoreHistogramFdr:
    def __init__(self):
        """
        初始化ScoreHistogramFdr类，创建一个包含两个子图的图形。
        一个子图用于绘制得分直方图，另一个用于绘制FDR-召回率曲线。
        """
        # 创建一个大小为(5 * 2, 5)的图形
        self.fig = plt.figure(figsize=(5 * 2, 5))
        # 在图形中添加第一个子图，用于绘制得分直方图
        self.hist_ax = self.fig.add_subplot(1, 2, 1)
        # 在图形中添加第二个子图，用于绘制FDR-召回率曲线
        self.fdr_ax = self.fig.add_subplot(1, 2, 2)

    def draw_histogram(self, scores, nbins=30, return_bins=False):
        """
        绘制得分的直方图。

        参数:
        scores (list): 得分列表
        nbins (int, 可选): 直方图的 bins 数量，默认为 30
        return_bins (bool, 可选): 是否返回直方图的 y 值和 x 轴 bins，默认为 False

        返回:
        如果 return_bins 为 True，则返回直方图的 y 值和 x 轴 bins
        """
        # 绘制直方图，返回直方图的 y 值、x 轴 bins 和直方图对象
        y, x_hist, _ = self.hist_ax.hist(
            scores, bins=nbins, histtype="step", color="grey"
        )
        # 设置 x 轴标签为 LCC_max
        self.hist_ax.set_xlabel(r"${LCC}_{max}$")
        # 设置 x 轴的范围为 x 轴 bins 的最小值到最大值
        self.hist_ax.set_xlim(x_hist[0], x_hist[-1])
        # 设置 y 轴标签为频率
        self.hist_ax.set_ylabel("Frequency")
        if return_bins:
            # 如果 return_bins 为 True，返回直方图的 y 值和 x 轴 bins
            return y, x_hist

    def draw_bimodal(self, x, y1, y2, ymax=None):
        """
        绘制双峰模型和高斯粒子分布。

        参数:
        x (list): x 轴数据
        y1 (list): 双峰模型的 y 轴数据
        y2 (list): 高斯粒子分布的 y 轴数据
        ymax (float, 可选): y 轴的最大值，默认为 None
        """
        # 绘制双峰模型的曲线
        self.hist_ax.plot(
            x, y1, lw=3.5, alpha=0.9, color="tab:blue"
        )  # , label='Bimodal model')
        # 绘制高斯粒子分布的曲线
        self.hist_ax.plot(
            x, y2, lw=4, alpha=0.9, color="tab:orange"
        )  # , label='True positives')
        if ymax is not None:
            # 如果提供了 ymax，设置 y 轴的范围为 0 到 ymax
            self.hist_ax.set_ylim(0, ymax)
        # 暂时不显示图例
        # self.hist_ax.legend(loc='upper right')

    def draw_score_threshold(self, x, ymax):
        """
        绘制得分阈值的竖线。

        参数:
        x (float): 得分阈值
        ymax (float): y 轴的最大值
        """
        # 绘制得分阈值的竖线
        self.hist_ax.vlines(
            x, 0, ymax, linestyle="dashed", label=f"Cutoff: {x:.2f}", color="black"
        )
        # 显示图例
        self.hist_ax.legend(loc="upper right")

    def draw_fdr_recall(self, fdr, recall, optimal_id, ruc):
        """
        绘制 FDR-召回率曲线。

        参数:
        fdr (list): FDR 值列表
        recall (list): 召回率值列表
        optimal_id (int): 最优阈值的索引
        ruc (float): 矩形下面积（RUC）
        """
        # 绘制 FDR-召回率曲线的散点图
        self.fdr_ax.scatter(fdr, recall, facecolors="none", edgecolors="gray", s=25)
        # 绘制最优阈值的散点图
        self.fdr_ax.scatter(
            fdr[optimal_id],
            recall[optimal_id],
            s=25,
            color="black",
            label=f"RUC: {ruc:.2f}",
        )
        # 绘制对角线
        self.fdr_ax.plot([0, 1], [0, 1], ls="--", c=".3", lw=1)
        # 设置 x 轴标签为 FDR
        self.fdr_ax.set_xlabel("FDR")
        # 设置 y 轴标签为召回率
        self.fdr_ax.set_ylabel("Recall")
        # 设置 x 轴的范围为 0 到 1
        self.fdr_ax.set_xlim(0, 1)
        # 设置 x 轴的刻度为 0, 0.2, 0.4, 0.6, 0.8, 1
        self.fdr_ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # 设置 y 轴的范围为 0 到 1
        self.fdr_ax.set_ylim(0, 1)
        # 设置 y 轴的刻度为 0, 0.2, 0.4, 0.6, 0.8, 1
        self.fdr_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # 显示图例
        self.fdr_ax.legend(loc="lower right")

    def write(self, filename, quality=300, transparency=False, bbox="tight"):
        """
        将图形保存到文件。

        参数:
        filename (str): 文件名
        quality (int, 可选): 图像质量，默认为 300
        transparency (bool, 可选): 是否使用透明背景，默认为 False
        bbox (str, 可选): 边界框设置，默认为 "tight"
        """
        # 调整子图布局
        plt.tight_layout()
        # 保存图形到文件
        plt.savefig(filename, dpi=quality, transparent=transparency, bbox_inches=bbox)

    def display(self):
        """
        显示图形。
        """
        # 调整子图布局
        plt.tight_layout()
        # 显示图形
        plt.show()


def check_square_fdr(fdr, recall, epsilon=1e-3):
    """
    检查 FDR 是否近似为方形。

    参数:
    fdr (list): FDR 值列表
    recall (list): 召回率值列表
    epsilon (float, 可选): 接近 0 和 1 的容差，默认为 1e-3

    返回:
    bool: 如果 FDR 近似为方形，则返回 True，否则返回 False
    """
    # 如果函数是方形的，fdr 和 recall 应该分别包含非常接近 0 和 1 的值
    union = [
        (f, r)
        for f, r in zip(fdr, recall)
        if ((np.abs(0.0 - f) < epsilon) and (np.abs(1.0 - r) < epsilon))
    ]
    return bool(union)


def distance_to_diag(fdr, recall):
    """
    计算每个 FDR-召回率组合到对角线的距离。

    参数:
    fdr (list): FDR 值列表
    recall (list): 召回率值列表

    返回:
    list: 每个 FDR-召回率组合到对角线的距离列表
    """
    # 对角线上的两个点
    lp1, lp2 = (0, 0), (1, 1)
    # 存储距离的列表
    distance = []
    for f, r in zip(fdr, recall):
        # 计算每个 FDR-召回率组合到对角线的距离
        d = np.abs(
            (lp2[0] - lp1[0]) * (lp1[1] - r) - (lp1[0] - f) * (lp2[1] - lp1[1])
        ) / np.sqrt((lp2[0] - lp1[0]) ** 2 + (lp2[1] - lp1[1]) ** 2)
        distance.append(d)
    return distance


def calculate_histogram(scores, num_steps):
    """
    根据给定的峰值索引构造 x 和 y 数组。

    参数:
    scores (list): 得分列表
    num_steps (int): 步数

    返回:
    tuple: 包含 x 和 y 数组的元组
    """
    # 对得分列表进行排序，从低到高
    scores.sort()
    # 获取得分列表的最小值
    min = scores[0]
    # 获取得分列表的最大值
    max = scores[-1]
    # 计算每个步骤的步长
    step = (max - min) / num_steps
    x = []
    for i in range(num_steps):
        # 计算 x 轴的坐标
        x.append(min + i * step)
    # 添加最大值到 x 轴坐标列表
    x.append(max)
    y = []
    for i in range(num_steps):
        # 获取当前区间的下限
        lower = x[i]
        # 获取当前区间的上限
        upper = x[i + 1]
        # 计算得分在当前区间内的数量
        n = len([v for v in scores if lower <= v <= upper])
        y.append(n)
    return x, y


def evaluate_estimates(estimated_positions, ground_truth_positions, tolerance):
    """
    评估估计位置与真实位置的匹配情况。

    参数:
    estimated_positions (numpy.ndarray): 估计位置的数组
    ground_truth_positions (numpy.ndarray): 真实位置的数组
    tolerance (float): 容差

    返回:
    list: 每个估计位置是否匹配的列表
    """
    from scipy.spatial.distance import cdist
    # 获取估计位置的数量
    n_estimates = estimated_positions.shape[0]
    # 计算估计位置与真实位置之间的欧几里得距离矩阵
    matrix = cdist(estimated_positions, ground_truth_positions, metric="euclidean")
    # 初始化匹配结果列表
    correct = [0] * n_estimates
    for i in range(n_estimates):
        if matrix[i].min() < tolerance:
            # 如果估计位置与某个真实位置的距离小于容差，则标记为匹配
            correct[i] = 1
    return correct


def fdr_recall(correct_particles, scores):
    """
    计算 FDR 和召回率。

    参数:
    correct_particles (list): 正确粒子的列表
    scores (list): 得分列表

    返回:
    tuple: 包含 FDR 和召回率列表的元组
    """
    # 确保得分列表是递减的
    assert all(i > j for i, j in itertools.pairwise(scores)), print(
        "Scores list should be decreasing."
    )
    # 计算正确粒子的总数
    n_true_positives = sum(correct_particles)
    # 初始化真阳性和假阳性的数量
    true_positives, false_positives = 0, 0
    # 初始化 FDR 和召回率列表
    fdr, recall = [], []
    for correct, score in zip(correct_particles, scores):
        if correct:
            # 如果当前粒子是正确的，增加真阳性的数量
            true_positives += 1
        else:
            # 如果当前粒子是错误的，增加假阳性的数量
            false_positives += 1
        if n_true_positives == 0:
            # 如果没有真阳性，召回率为 0
            recall.append(0)
        else:
            # 计算召回率
            recall.append(true_positives / n_true_positives)
        # 计算 FDR
        fdr.append(false_positives / (true_positives + false_positives))
    return fdr, recall


def get_distance(line, point):
    """
    计算点到直线的距离。

    参数:
    line (tuple): 直线的斜率和截距
    point (tuple): 点的坐标

    返回:
    float: 点到直线的距离
    """
    # 获取直线的斜率和截距
    a1, b1 = line
    # 获取点的坐标
    x, y = point
    # 计算垂线的斜率
    a2 = -(1 / a1)
    # 计算垂线的截距
    b2 = y - a2 * x
    # 计算交点的 x 坐标
    x_int = (b2 - b1) / (a1 - a2)
    # 计算交点的 y 坐标
    y_int = a2 * x_int + b2
    # 计算点到交点的距离
    return np.sqrt((x_int - x) ** 2 + (y_int - y) ** 2)


def distance_to_random(fdr, recall):
    """
    计算每个 FDR-召回率组合到随机线的距离，并返回最大距离和对应的索引。

    参数:
    fdr (list): FDR 值列表
    recall (list): 召回率值列表

    返回:
    tuple: 包含最大距离和对应的索引的元组
    """
    # 初始化 AUC 列表
    auc = [0] * len(fdr)
    for i in range(len(fdr)):
        # 计算每个 FDR-召回率组合到随机线的距离
        d = get_distance((1, 0), (fdr[i], recall[i]))
        if recall[i] > fdr[i]:
            # 如果召回率大于 FDR，AUC 为距离
            auc[i] = d
        else:
            # 如果召回率小于等于 FDR，AUC 为负距离
            auc[i] = -d
    # 返回最大距离和对应的索引
    return max(auc), np.argmax(auc)


# ========== functions for fitting ==========
# 定义用于拟合的高斯函数
def gauss(x, mu, sigma, amp):
    """
    高斯函数。

    参数:
    x (float): 自变量
    mu (float): 均值
    sigma (float): 标准差
    amp (float): 幅度

    返回:
    float: 高斯函数的值
    """
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# 计算高斯函数的积分
def gauss_integral(sigma, amp):
    """
    计算高斯函数的积分。

    参数:
    sigma (float): 标准差
    amp (float): 幅度

    返回:
    float: 高斯函数的积分值
    """
    # 均值不影响积分
    return amp * np.abs(sigma) * np.sqrt(2 * np.pi)


# 定义用于拟合的双峰函数
def bimodal(x, mu1, sigma1, amp1, mu2, sigma2, amp2):
    """
    双峰函数，由两个高斯函数相加得到。

    参数:
    x (float): 自变量
    mu1 (float): 第一个高斯函数的均值
    sigma1 (float): 第一个高斯函数的标准差
    amp1 (float): 第一个高斯函数的幅度
    mu2 (float): 第二个高斯函数的均值
    sigma2 (float): 第二个高斯函数的标准差
    amp2 (float): 第二个高斯函数的幅度

    返回:
    float: 双峰函数的值
    """
    return gauss(x, mu1, sigma1, amp1) + gauss(x, mu2, sigma2, amp2)


def plist_quality_gaussian_fit(
    lcc_max_values,
    score_volume,
    particle_peak_index,
    force_peak=False,
    output_figure_name=None,
    crop_hist=False,
    num_bins=30,
    n_tomograms=1,
):
    """
    对粒子列表的质量进行高斯拟合，并绘制相关图形。

    参数:
    lcc_max_values (list): LCC 最大值列表
    score_volume (numpy.ndarray): 得分体积数组
    particle_peak_index (int): 粒子峰值索引
    force_peak (bool, 可选): 是否强制粒子分布的峰值在指定索引处，默认为 False
    output_figure_name (str, 可选): 输出图形的文件名，默认为 None
    crop_hist (bool, 可选): 是否裁剪直方图，默认为 False
    num_bins (int, 可选): 直方图的 bins 数量，默认为 30
    n_tomograms (int, 可选): 断层图像的数量，默认为 1

    返回:
    None
    """
    # 读取得分，将 LCC 最大值列表排序并转换为 NumPy 数组
    correlation_scores = np.array(sorted(lcc_max_values, reverse=True))
    # 创建 ScoreHistogramFdr 对象
    plot = ScoreHistogramFdr()
    # 绘制得分直方图，并返回直方图的 y 值和 x 轴 bins
    y, x_hist = plot.draw_histogram(
        correlation_scores, nbins=num_bins, return_bins=True
    )
    try:
        # ===== fit bimodal distribution =====
        # 调整 x 轴坐标到每个 bin 的中心，使 x 和 y 的长度相等
        x = (x_hist[1:] + x_hist[:-1]) / 2
        # 计算直方图的步长
        hist_step = x_hist[1] - x_hist[0]
        # 计算噪声高斯分布的标准差
        noise_sigma = score_volume.std()
        # 计算噪声高斯分布的均值
        noise_mean = score_volume.mean()
        # 计算噪声的大小
        noise_size = score_volume.size * n_tomograms
        # 计算噪声高斯分布的 A 值
        noise_a = ((noise_size) / (noise_sigma * np.sqrt(2 * np.pi))) * hist_step
        # 定义预期值
        expected = (noise_sigma, x[particle_peak_index], 0.1, y[particle_peak_index])
        if force_peak:
            # 如果强制粒子分布的峰值在指定索引处，设置边界
            bounds = (
                [noise_sigma, x[particle_peak_index] - 0.01, 0, 0],
                [noise_sigma * 1.5, x[particle_peak_index] + 0.01, 0.1, y[1]],
            )
        else:
            # 否则，设置默认边界
            bounds = (
                [noise_sigma, x[int(len(x) * 0.25)], 0, 0],
                [noise_sigma * 1.5, x[-1], 0.1, y[1]],
            )
        # 定义参数名称
        params_names = ["sigma_1", "mu_2", "sigma_2", "A_2"]
        # 跳过第一个位置，因为那里的噪声峰值可能不正确
        params, cov = curve_fit(
            lambda x, p1, p2, p3, p4: bimodal(x, noise_mean, p1, noise_a, p2, p3, p4),
            x[1:],
            y[1:],
            p0=expected,
            bounds=bounds,
            maxfev=2000,
        )
        # 计算每个参数的拟合标准差
        sigma = np.sqrt(np.diag(cov))
        # 打印双峰模型的拟合信息
        print("\nfit of the bimodal model:")
        print("\testimated\t\tsigma")
        for n, p, s in zip(params_names, params, sigma):
            print(f"{n}\t{p:.3f}\t\t{s:.3f}")
        print("\n")
        # 定义噪声和粒子分布的参数
        noise, population = ((noise_mean, params[0], noise_a), tuple(params[1:4]))
        # 计算双峰模型的 y 值
        y_bimodal = bimodal(x, *noise, *population)
        # 计算高斯粒子分布的 y 值
        y_gauss = gauss(x, *population)
        if crop_hist:
            # 如果裁剪直方图，设置 y 轴的最大值
            plot.draw_bimodal(x, y_bimodal, y_gauss, ymax=3 * population[2])
        else:
            # 否则，正常绘制双峰模型和高斯粒子分布
            plot.draw_bimodal(x, y_bimodal, y_gauss)
        # ===== Generate a ROC curve =====
        # 设置 ROC 曲线的步数
        roc_steps = 50
        # 生成 ROC 曲线的 x 轴坐标
        x_roc = np.flip(np.linspace(x[0], x[-1], roc_steps))
        # 计算直方图步长与 ROC 步长的比值
        roc_step = (x[-1] - x[0]) / roc_steps
        delta = hist_step / roc_step
        # 初始化假阳性的总数
        n_false_positives = 0.0
        # 初始化召回率和 FDR 列表
        recall = []
        fdr = []
        # 计算高斯粒子分布的积分
        population_integral = gauss_integral(population[1], population[2]) / hist_step
        print(
            f" - estimation total number of true positives: {population_integral:.1f}"
        )
        # 定义累积分布函数
        def cdf(x):
            return 0.5 * (1 + erf((x - population[0]) / (np.sqrt(2) * population[1])))
        # 定义噪声高斯函数
        def gauss_noise(x):
            return gauss(x, *noise)
        for x_i in x_roc:
            # 计算真阳性的数量
            n_true_positives = (1 - cdf(x_i)) * population_integral
            # 计算假阳性的数量
            n_false_positives += gauss_noise(x_i) / delta
            # 计算召回率
            recall.append(n_true_positives / population_integral)
            # 计算 FDR
            fdr.append(n_false_positives / (n_true_positives + n_false_positives))
        # 计算矩形下面积（RUC）
        recall = np.array(recall)
        fdr = np.array(fdr)
        rectangles = recall * (1 - fdr)
        # 找到最优阈值的索引和最大 RUC
        cutoff, ruc = rectangles.argmax(), rectangles.max()
        # 在分布直方图上绘制阈值线
        plot.draw_score_threshold(x_roc[cutoff], max(y))
        print(f" - optimal correlation coefficient threshold is {x_roc[cutoff]:.3f}")
        print(
            (
                " - this threshold approximately selects "
                f"{(1 - cdf(x_roc[cutoff])) * population_integral:.1f} particles",
            )
        )
        # 绘制 FDR-召回率曲线
        plot.draw_fdr_recall(fdr, recall, cutoff, ruc)
        print("Rectangle Under Curve (RUC): ", ruc)
    except (RuntimeError, ValueError):
        # 如果出现运行时错误或值错误，打印异常信息
        traceback.print_exc()
    if output_figure_name is None:
        # 如果未提供输出图形的文件名，显示图形
        plot.display()
    else:
        if output_figure_name.suffix not in [".svg", ".png"]:
            # 如果输出文件名的后缀不是 .svg 或 .png，添加 .png 后缀
            output_figure_name = output_figure_name + ".png"
        # 保存图形到文件
        plot.write(output_figure_name)
