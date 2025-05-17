# 导入cupy库，用于GPU加速计算
import cupy as cp
# 导入cupy的类型注解模块
import cupy.typing as cpt
# 导入numpy的类型注解模块
import numpy.typing as npt
# 导入voltools库，用于处理3D体积数据
import voltools as vt
# 导入垃圾回收模块
import gc
# 从cupyx的scipy.fft模块导入rfftn和irfftn函数，用于快速傅里叶变换
from cupyx.scipy.fft import rfftn, irfftn
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 从pytom_tm.correlation模块导入mean_under_mask和std_under_mask函数
from pytom_tm.correlation import mean_under_mask, std_under_mask
# 从pytom_tm.template模块导入phase_randomize_template函数
from pytom_tm.template import phase_randomize_template
# 导入 numpy 库并简称为 np，用于数值计算
import numpy as np
import numba as nb
# 从scipy.spatial.transform模块导入Rotation类
from scipy.spatial.transform import Rotation as R
import time  # 添加时间模块
# 添加新的计算
from pytom_tm.theta_mask_optimized import create_new_mask_from_spheres_optimized
import mrcfile


class TemplateMatchingPlan:
    def __init__(
        self,
        volume: npt.NDArray[float],
        template: npt.NDArray[float],
        mask: npt.NDArray[float],
        device_id: int,
        wedge: npt.NDArray[float] | None = None,
        phase_randomized_template: npt.NDArray[float] | None = None,
    ):
        """
        初始化模板匹配计划。所有必要的cupy数组将被分配到GPU上。

        参数
        ----------
        volume: npt.NDArray[float]
            表示搜索断层图像的3D numpy数组
        template: npt.NDArray[float]
            表示搜索模板的3D numpy数组，是一个大小为sx的方形盒子
        mask: npt.NDArray[float]
            表示搜索掩码的3D numpy数组，与模板尺寸相同
        device_id: int
            用于加载数组的GPU设备ID
        wedge: Optional[npt.NDArray[float]], default None
            包含模板傅里叶空间权重的3D numpy数组，
            它应该是傅里叶缩减形式，尺寸为(sx, sx, sx // 2 + 1)
        phase_randomized_template: Optional[npt.NDArray[float]], default None
            使用模板的相位随机化版本初始化计划，用于噪声校正
        """
        # 获取搜索体积的形状
        volume_shape = volume.shape
        # 将搜索体积转换为cupy数组，并指定数据类型和存储顺序
        cp_vol = cp.asarray(volume, dtype=cp.float32, order="C")
        # 计算搜索体积的实值快速傅里叶变换的共轭
        self.volume_rft_conj = rfftn(cp_vol).conj()
        # 计算搜索体积平方的实值快速傅里叶变换的共轭
        self.volume_sq_rft_conj = rfftn(cp_vol**2).conj()
        # 显式的傅里叶变换计划不再必要，因为cupy会在后台生成一个计划，其计时效果相当

        # 用于存储局部标准差的数组
        self.std_volume = cp.zeros(volume_shape, dtype=cp.float32)

        # 掩码数据
        # 将掩码转换为cupy数组
        self.mask = cp.asarray(mask, dtype=cp.float32, order="C")
        # 创建掩码的静态体积对象，用于插值操作
        self.mask_texture = vt.StaticVolume(
            self.mask, interpolation="filt_bspline", device=f"gpu:{device_id}"
        )
        # 创建填充后的掩码数组，初始化为零
        self.mask_padded = cp.zeros(volume_shape, dtype=cp.float32)
        # 计算掩码的权重，通常是掩码元素的总和
        self.mask_weight = self.mask.sum()

        # 初始化模板数据
        # 将模板转换为cupy数组
        self.template = cp.asarray(template, dtype=cp.float32, order="C")
        # 创建模板的静态体积对象，用于插值操作
        self.template_texture = vt.StaticVolume(
            self.template, interpolation="filt_bspline", device=f"gpu:{device_id}"
        )
        # 创建填充后的模板数组，初始化为零
        self.template_padded = cp.zeros(volume_shape, dtype=cp.float32)

        # 模板的傅里叶二进制楔形权重
        # 如果提供了楔形权重，则将其转换为cupy数组，否则为None
        self.wedge = (
            cp.asarray(wedge, order="C", dtype=cp.float32)
            if wedge is not None
            else None
        )

        # 初始化结果体积
        # 用于存储相关性系数图
        self.ccc_map = cp.zeros(volume_shape, dtype=cp.float32)
        # 用于存储分数，初始化为负无穷大
        self.scores = cp.ones(volume_shape, dtype=cp.float32) * -1000
        # 用于存储角度，初始化为负无穷大
        self.angles = cp.ones(volume_shape, dtype=cp.float32) * -1000

        # 随机相位模板纹理和噪声分数
        self.random_phase_template_texture = None
        self.noise_scores = None
        # 如果提供了相位随机化模板，则创建相应的静态体积对象和噪声分数数组
        if phase_randomized_template is not None:
            self.random_phase_template_texture = vt.StaticVolume(
                cp.asarray(phase_randomized_template, dtype=cp.float32, order="C"),
                interpolation="filt_bspline",
                device=f"gpu:{device_id}",
            )
            self.noise_scores = cp.ones(volume_shape, dtype=cp.float32) * -1000

        # 等待流完成工作
        cp.cuda.stream.get_current_stream().synchronize()

    def clean(self) -> None:
        """
        从GPU的内存池中移除所有存储的cupy数组。
        """
        # 获取默认的GPU内存池
        gpu_memory_pool = cp.get_default_memory_pool()
        # 删除所有存储的cupy数组
        del (
            self.volume_rft_conj,
            self.volume_sq_rft_conj,
            self.mask,
            self.mask_texture,
            self.mask_padded,
            self.template,
            self.template_texture,
            self.template_padded,
            self.wedge,
            self.ccc_map,
            self.scores,
            self.angles,
            self.std_volume,
        )
        # 强制进行垃圾回收
        gc.collect()
        # 释放内存池中的所有块
        gpu_memory_pool.free_all_blocks()


class TemplateMatchingGPU:
    def __init__(
        self,
        job_id: str,
        device_id: int,
        volume: npt.NDArray[float],
        template: npt.NDArray[float],
        mask: npt.NDArray[float],
        angle_list: list[tuple[float, float, float]],
        angle_ids: list[int],
        mask_is_spherical: bool = True,
        wedge: npt.NDArray[float] | None = None,
        stats_roi: tuple[slice, slice, slice] | npt.NDArray[float] | None = None,
        noise_correction: bool = False,
        rng_seed: int = 321,
        sphere_list : list[tuple] | None = None,
        search_origin: list[int] | None = None,
        search_size: list[int] | None = None,
        volume_origin_shape: tuple | None = None,
        mask_volume: npt.NDArray[float] | None = None,
    ):
        """
        初始化模板匹配运行。

        其他优秀的实现请参考：
        - STOPGAP: https://github.com/wan-lab-vanderbilt/STOPGAP
        - pyTME: https://github.com/KosinskiLab/pyTME

        据我所知，断层图像共轭傅里叶变换的预计算是由STOPGAP引入的！
        此外，他们还引入了与模板的相位随机化版本同时进行匹配的方法。https://doi.org/10.1107/S205979832400295X

        参数
        ----------
        job_id: str
            用于作业识别的字符串
        device_id: int
            用于运行作业的GPU设备ID
        volume: npt.NDArray[float]
            断层图像的3D numpy数组
        template: npt.NDArray[float]
            模板的3D numpy数组，是一个大小为sx的方形盒子
        mask: npt.NDArray[float]
            掩码的3D numpy数组，与模板尺寸相同
        angle_list: list[tuple[float, float, float]]
            包含3个浮点数的元组列表，表示欧拉角旋转
        angle_ids: list[int]
            实际搜索的angle_list的索引列表，这可以是完整列表的子集
        mask_is_spherical: bool, default True
            如果掩码是球形的，则为True（默认），如果是非球形掩码，则设置为False，这会增加计算时间
        wedge: Optional[npt.NDArray[float]], default None
            包含模板傅里叶空间权重的3D numpy数组，
            它应该是傅里叶缩减形式，尺寸为(sx, sx, sx // 2 + 1)
        stats_roi: Optional[tuple[slice, slice, slice]], default None
            用于计算搜索体积统计信息的感兴趣区域，默认将采用整个搜索体积
        noise_correction: bool, default False
            使用模板的相位随机化版本初始化模板匹配，用于从分数图中减去背景噪声；代价是更多的GPU内存和计算时间
        rng_seed: int, default 321
            相位随机化中的随机数生成器种子
        """
        # 使用指定的GPU设备
        cp.cuda.Device(device_id).use()

        # 作业ID
        self.job_id = job_id
        # GPU设备ID
        self.device_id = device_id
        # 作业是否激活
        self.active = True
        # 作业是否完成
        self.completed = False
        # 掩码是否为球形
        self.mask_is_spherical = mask_is_spherical
        # 角度列表
        self.angle_list = angle_list
        # 角度ID列表
        self.angle_ids = angle_ids
        # 搜索统计信息
        self.stats = {"search_space": 0, "variance": 0.0, "std": 0.0}
        # 如果未提供统计信息的感兴趣区域，则使用整个搜索体积
        if stats_roi is None:
            self.stats_roi = (slice(None), slice(None), slice(None))
        else:
            self.stats_roi = stats_roi
        # 是否进行噪声校正
        self.noise_correction = noise_correction
        # 球心信息
        self.sphere_list = sphere_list
        # 分片信息
        self.search_origin = search_origin
        self.search_size = search_size
        # 原搜索体积信息
        self.volume_origin_shape = volume_origin_shape
        # 标记了球心区域的（子）体积
        self.mask_volume = mask_volume

        # 创建模板的"随机噪声"版本
        # 如果需要噪声校正，则生成相位随机化的模板，否则为None
        shuffled_template = (
            phase_randomize_template(template, rng_seed) if noise_correction else None
        )

        # 创建模板匹配计划
        self.plan = TemplateMatchingPlan(
            volume,
            template,
            mask,
            device_id,
            wedge=wedge,
            phase_randomized_template=shuffled_template,
        )

    def run(self) -> tuple[npt.NDArray[float], npt.NDArray[float], dict]:
        """
        运行模板匹配作业。只在 stats_roi 定义的区域内计算，如果使用 tomogram_mask，
        则只对 mask 内的区域进行计算。
        图示来说明整个模板匹配过程：
        1. 初始状态
        搜索体积 (100x100x100)            模板 (20x20x20)          掩码 (20x20x20)
        +-----------------------+       +------------+           +------------+
        |                       |       |            |           |            |
        |      +--------+       |       |  蛋白质分子 |           |   球形区域  |
        |      |        |       |       |            |           |            |
        |      |  细胞质 |       |       +------------+           +------------+
        |      |        |       |
        |      +--------+       |
        |                       |
        +-----------------------+

        2. 角度迭代（只展示一个角度）

        a. 对掩码进行旋转（如果不是球形）
            
            原始掩码                   旋转后的掩码
            +------------+           +------------+
            |   O        |           |            |
            |            |  旋转45°   |     O      |
            |            |  ------> |            |
            +------------+           +------------+

        b. 计算局部标准差
        
            搜索体积中每个点的标准差（使用掩码区域内的值计算）
            +------------------------+
            |   X X X X X X X X X X  |
            |   X X X X X X X X X X  |
            |   X X X X X X X X X X  |
            |   X X X X X X X X X X  |
            |   X X X X X X X X X X  |
            +------------------------+
            其中每个X代表该点在掩码内的标准差值

        c. 旋转模板
        
            原始模板                   旋转后的模板
            +------------+           +------------+
            |  /\        |           |            |
            |  \/        |  旋转45°   |    /\      |
            |            |  ------> |    \/      |
            +------------+           +------------+

        d. 计算互相关
        
            将旋转后的模板放置在搜索体积的每个可能位置，计算归一化互相关
            +------------------------+
            |   C C C C C C C C C C  |
            |   C C C C C C C C C C  |
            |   C C C C C H C C C C  |
            |   C C C C C C C C C C  |
            |   C C C C C C C C C C  |
            +------------------------+
            其中C代表相关性值，H代表较高的相关性

        e. 更新最佳分数和角度
        
            如果当前相关性高于之前的最高分数，则更新该点的分数和角度
            分数图                      角度图
            +------------------------+  +------------------------+
            |   S S S S S S S S S S  |  |   A A A A A A A A A A  |
            |   S S S S S S S S S S  |  |   A A A A A A A A A A  |
            |   S S S S S H S S S S  |  |   A A A A A 45 A A A A |
            |   S S S S S S S S S S  |  |   A A A A A A A A A A  |
            |   S S S S S S S S S S  |  |   A A A A A A A A A A  |
            +------------------------+  +------------------------+

        3. 重复步骤2，迭代所有角度

        4. 最终结果（示例）

        分数图 (XY视图)                  角度图 (XY视图)
        +------------------------+  +------------------------+
        |   L L L L L L L L L L  |  |   * * * * * * * * * *  |
        |   L L M M M M M L L L  |  |   * * 30 45 60 * * * *  |
        |   L M M H H H M M L L  |  |   * * 20 30 30 * * * *  |
        |   L L M M M M M L L L  |  |   * * 45 60 90 * * * *  |
        |   L L L L L L L L L L  |  |   * * * * * * * * * *  |
        +------------------------+  +------------------------+
        
        L=低分数, M=中等分数, H=高分数  数字=角度ID, *=任意角度
        -------
        返回
        -------
        results: tuple[npt.NDArray[float], npt.NDArray[float], dict]
            结果是一个包含三个元素的元组：
                - score_map：在所有搜索角度上局部归一化的最大分数；
                    一个与搜索体积大小相同的3D numpy数组
                - angle_map：与相关分数最大值对应的角度列表的索引；
                    一个与搜索体积大小相同的3D numpy数组
                - 一个包含三个浮点数的搜索统计信息字典；
                    'search_space'、'variance'和 'std'
        """
        # 打印作业进度信息
        print(f"Progress job_{self.job_id} on device {self.device_id:d}:")

        # 模板的尺寸和中心坐标
        # 示例：
        # 如果模板大小为 [20, 20, 20]，则：
        # sxt = syt = szt = 20 (尺寸)
        # cxt = cyt = czt = 10 (中心坐标)
        # mx = my = mz = 0 (偶数尺寸)
        # 如果搜索体积大小为 [100, 100, 100]，则：
        # sxv = syv = szv = 100
        # cxv = cyv = czv = 50
        sxt, syt, szt = self.plan.template.shape
        cxt, cyt, czt = sxt // 2, syt // 2, szt // 2
        # 模板尺寸的奇偶性
        mx, my, mz = sxt % 2, syt % 2, szt % 2

        # 搜索体积的尺寸和中心坐标
        sxv, syv, szv = self.plan.template_padded.shape
        cxv, cyv, czv = sxv // 2, syv // 2, szv // 2

        # 创建用于填充的切片
        # 示例：
        # 如果模板中心是 [10, 10, 10]，搜索体积中心是 [50, 50, 50]，且模板大小为偶数：
        # pad_index = (slice(40, 60), slice(40, 60), slice(40, 60))
        # 如果模板大小有奇数维度，比如 [21, 20, 20]，则：
        # pad_index = (slice(40, 61), slice(40, 60), slice(40, 60))
        pad_index = (
            slice(cxv - cxt, cxv + cxt + mx),
            slice(cyv - cyt, cyv + cyt + my),
            slice(czv - czt, czv + czt + mz),
        )

        # 计算感兴趣区域的掩码
        # 示例：
        # 如果 stats_roi = (slice(20, 40), slice(20, 40), slice(20, 40))，则只有这个立方体区域内的点会被设为 True，其余点为 False
         # 计算偏移量
        shift = cp.floor(cp.array(self.plan.scores.shape) / 2).astype(int) + 1
        # 创建初始掩码
        if isinstance(self.stats_roi, (np.ndarray, cp.ndarray)):
            # 如果stats_roi是数组，直接将其转换为cupy布尔数组
            roi_mask = cp.asarray(self.stats_roi > 0, dtype=bool)
        else:
            # 原始逻辑：如果stats_roi是切片或None
            roi_mask = cp.zeros(self.plan.scores.shape, dtype=bool)
            if self.stats_roi is not None:
                roi_mask[self.stats_roi] = True
            else:
                roi_mask[:] = True
        # 翻转并滚动掩码
        roi_mask = cp.flip(cp.roll(roi_mask, -shift, (0, 1, 2)))
        # 计算感兴趣区域的大小
        roi_size = self.plan.scores[roi_mask].size

        # 如果掩码是球形的，则只需要计算一次局部标准差
        # 球形掩码的优势：对于球形掩码，无论旋转如何，形状不变，所以只需计算一次标准差，可以节省大量计算时间。
        if self.mask_is_spherical:
            # 将掩码填充到指定位置
            self.plan.mask_padded[pad_index] = self.plan.mask
            # 计算局部标准差
            self.plan.std_volume = (
                std_under_mask_convolution(
                    self.plan.volume_rft_conj,
                    self.plan.volume_sq_rft_conj,
                    self.plan.mask_padded,
                    self.plan.mask_weight,
                )
                * self.plan.mask_weight
            )

        # 使用tqdm进度条跟踪迭代
        # 示例：
        # 如果 angle_list = [(0,0,0), (0,0,45), (0,45,0), ...]
        # 则循环中 rotation 依次为 (0,0,0), (0,0,45), ...
        for i in tqdm(range(len(self.angle_ids))):
            # 获取角度ID和旋转角度
            angle_id, rotation = self.angle_ids[i], self.angle_list[i]

            sphere_list_mask = None
            # 如果有球心数据，处理扇形范围
            if self.sphere_list is not None:
                sphere_list_mask = create_new_mask_from_spheres_optimized(
                    sphere_mask=self.mask_volume,
                    sphere_list=self.sphere_list,
                    volume_shape=self.volume_origin_shape,
                    theta=rotation,
                    search_origin=self.search_origin,
                    search_size=self.search_size,
                    delta_theta=np.pi/4,
                    theta_unit='rad',
                    rotation_order='rzxz'
                )
                # 将numpy数组转换为cupy数组，然后再翻转和滚动
                sphere_list_mask_cp = cp.asarray(sphere_list_mask, dtype=bool)
                # 翻转并滚动掩码
                roi_mask = cp.flip(cp.roll(sphere_list_mask_cp, -shift, (0, 1, 2)))
                # 计算感兴趣区域的大小
                roi_size = cp.sum(roi_mask).item()  # 使用cp.sum替代索引计算
                
                # # 输出sphere_list_mask到MRC文件
                # np.transpose(sphere_list_mask, (2,1,0))
                # with mrcfile.new('sphere_list_mask.mrc', overwrite=True) as mrc:
                #     mrc.set_data(sphere_list_mask.astype(np.float32))
            
            # 如果掩码不是球形的，则需要为每个旋转重新计算局部标准差
            # 非球形掩码的例子：
            # 例如对于一个扁平的圆盘形掩码，不同旋转角度下掩码覆盖的区域不同，所以每次旋转都需要重新计算标准差。
            if not self.mask_is_spherical:
                # 旋转掩码
                self.plan.mask_texture.transform(
                    rotation=(rotation[0], rotation[1], rotation[2]),
                    rotation_order="rzxz",
                    output=self.plan.mask,
                    rotation_units="rad",
                )
                # 将旋转后的掩码填充到指定位置
                self.plan.mask_padded[pad_index] = self.plan.mask
                # 重新计算局部标准差，这是一个昂贵的步骤
                self.plan.std_volume = (
                    std_under_mask_convolution(
                        self.plan.volume_rft_conj,
                        self.plan.volume_sq_rft_conj,
                        self.plan.mask_padded,
                        self.plan.mask_weight,
                    )
                    * self.plan.mask_weight
                )

            # 旋转模板
            self.plan.template_texture.transform(
                rotation=(rotation[0], rotation[1], rotation[2]),
                rotation_order="rzxz",
                output=self.plan.template,
                rotation_units="rad",
            )
            
            # 输出template到MRC文件
            np.transpose(self.plan.template, (2,1,0))
            with mrcfile.new('template.mrc', overwrite=True) as mrc:
                mrc.set_data(cp.asnumpy(self.plan.template).astype(np.float32))

            # 进行相关性计算
            self.correlate(pad_index)

            # 更新分数和角度列表
            # 使用 CUDA 内核函数更新结果，示例：
            # 如果在某点，当前角度的相关性为0.8，而之前最高分数是0.7，则更新该点的分数为0.8，角度为当前角度ID
            # 如果当前相关性为0.6，低于之前的0.7，则保持原值不变
            update_results_kernel(
                self.plan.scores,
                self.plan.ccc_map * roi_mask,
                angle_id,
                self.plan.scores,
                self.plan.angles,
            )

            # 累加相关性系数图的方差
            # 计算当前角度下相关性图（仅在ROI区域内）的平方和，累加到方差统计中。这对于评估搜索结果的质量很有用。
            self.stats["variance"] += (
                square_sum_kernel(self.plan.ccc_map * roi_mask) / roi_size
            )

            # 如果需要噪声校正
            if self.noise_correction:
                # 旋转噪声模板
                self.plan.random_phase_template_texture.transform(
                    rotation=(rotation[0], rotation[1], rotation[2]),
                    rotation_order="rzxz",
                    output=self.plan.template,
                    rotation_units="rad",
                )

                # 进行相关性计算
                self.correlate(pad_index)

                # 更新噪声分数
                update_noise_template_results_kernel(
                    self.plan.noise_scores,
                    self.plan.ccc_map,
                    self.plan.noise_scores,
                )

        # 进行噪声校正
        if self.noise_correction:
            # 从分数图中减去噪声分数，然后加上噪声的平均值，以确保统计信息一致
            self.plan.scores = (
                self.plan.scores - self.plan.noise_scores
            ) + self.plan.noise_scores.mean()

        # 恢复正确的方向
        # 这段代码：
        # 对分数和角度图进行翻转和滚动，恢复正确的空间方向
        # 计算最终的统计信息：搜索空间大小、方差和标准差
        self.plan.scores = cp.roll(cp.flip(self.plan.scores), shift, axis=(0, 1, 2))
        self.plan.angles = cp.roll(cp.flip(self.plan.angles), shift, axis=(0, 1, 2))

        # 计算搜索空间
        self.stats["search_space"] = int(roi_size * len(self.angle_ids))
        # 计算方差
        self.stats["variance"] = float(self.stats["variance"] / len(self.angle_ids))
        # 计算标准差
        self.stats["std"] = float(cp.sqrt(self.stats["variance"]))

        # 将结果打包回CPU
        # 最后，将结果从GPU内存转移到CPU内存，清理GPU资源，并返回结果元组。
        results = (self.plan.scores.get(), self.plan.angles.get(), self.stats)

        # 清除所有使用的GPU内存
        self.plan.clean()

        return results

    def correlate(self, padding_index: tuple[slice, slice, slice]):
        """
        计算模板和断层图像的相关性。

        参数
        ----------
        padding_index: tuple[slice, slice, slice]
            加权和归一化后填充模板的位置
        """
        
        # 如果提供了楔形权重
        # 如果提供了楔形权重（通常用于处理缺失楔形问题），将其应用于模板。这是在傅里叶空间中完成的，通过：
        # 将模板转换到傅里叶空间 (rfftn)
        # 乘以楔形权重
        # 转换回实空间 (irfftn)
        if self.plan.wedge is not None:
            # 在旋转后将楔形权重应用于模板
            self.plan.template = irfftn(
                rfftn(self.plan.template) * self.plan.wedge, s=self.plan.template.shape
            )

        # 归一化并掩码模板
        # 计算模板在掩码区域内的均值
        mean = mean_under_mask(
            self.plan.template, self.plan.mask, mask_weight=self.plan.mask_weight
        )
        # 计算模板在掩码区域内的标准差
        std = std_under_mask(
            self.plan.template, self.plan.mask, mean, mask_weight=self.plan.mask_weight
        )
        # 对模板进行归一化和掩码操作
        self.plan.template = ((self.plan.template - mean) / std) * self.plan.mask

        # 将模板粘贴到中心位置
        self.plan.template_padded[padding_index] = self.plan.template

        # 计算体积和模板之间的快速局部相关性函数，
        # 归一化是体积中每个点在掩码区域内的标准差
        self.plan.ccc_map = (
            irfftn(
                self.plan.volume_rft_conj * rfftn(self.plan.template_padded),
                s=self.plan.template_padded.shape,
            )
            / self.plan.std_volume
        )


def std_under_mask_convolution(
    volume_rft_conj: cpt.NDArray[float],
    volume_sq_rft_conj: cpt.NDArray[float],
    padded_mask: cpt.NDArray[float],
    mask_weight: float,
) -> cpt.NDArray[float]:
    """
    计算体积中每个位置在掩码下的局部标准差。
    计算在傅里叶空间中进行，因为这是体积和掩码之间的卷积。

    参数
    ----------
    volume_rft_conj: cpt.NDArray[float]
        搜索体积的实值快速傅里叶变换的共轭
    volume_sq_rft_conj: cpt.NDArray[float]
        搜索体积平方的实值快速傅里叶变换的共轭
    padded_mask: cpt.NDArray[float]
        已填充到体积尺寸的模板掩码
    mask_weight: float
        掩码的权重，通常计算为掩码的总和

    返回
    -------
    std_v: cpt.NDArray[float]
        体积中局部标准差的数组
    """
    # 计算填充后掩码的实值快速傅里叶变换
    padded_mask_rft = rfftn(padded_mask)
    # 计算局部标准差
    std_v = (
        irfftn(volume_sq_rft_conj * padded_mask_rft, s=padded_mask.shape) / mask_weight
        - (irfftn(volume_rft_conj * padded_mask_rft, s=padded_mask.shape) / mask_weight)
        ** 2
    )
    # 防止潜在的负数平方根和除以零
    std_v[std_v <= cp.float32(1e-18)] = 1
    # 计算平方根
    std_v = cp.sqrt(std_v)
    return std_v


"""
如果找到新的最大值，则更新分数和角度。
"""
update_results_kernel = cp.ElementwiseKernel(
    "float32 scores, float32 ccc_map, float32 angles",
    "float32 scores_out, float32 angles_out",
    "if (scores < ccc_map) {scores_out = ccc_map; angles_out = angles;}",
    "update_results",
)


"""
更新噪声模板的分数。
"""
update_noise_template_results_kernel = cp.ElementwiseKernel(
    "float32 scores, float32 ccc_map",
    "float32 scores_out",
    "if (scores < ccc_map) {scores_out = ccc_map;}",
    "update_noise_template_results",
)


"""
计算体积中平方和。假设均值为0，这使得操作更快。
"""
square_sum_kernel = cp.ReductionKernel(
    "T x",  # 输入参数
    "T y",  # 输出参数
    "x * x",  # 预处理表达式
    "a + b",  # 归约操作
    "y = a",  # 归约后输出处理
    "0",  # 单位值
    "variance",  # 内核名称
)

def create_new_mask_from_spheres_optimized_self(
    sphere_list, 
    volume_shape, 
    theta,  # 现在假设theta是rzxz旋转顺序和弧度单位
    search_origin,
    search_size,
    delta_theta=np.pi/4,  # 默认值改为弧度单位，pi/4 = 45度
    theta_unit='rad',  # 改为rad作为默认单位
    rotation_order='rzxz'  # 改为rzxz作为默认旋转顺序
):
    
    # 如果传入角度单位，转换为弧度
    if theta_unit == 'deg':
        delta_theta = np.deg2rad(delta_theta)
        theta = np.deg2rad(theta)
    
    # 计算主方向向量 - 使用rzxz顺序和弧度单位
    # scipy.spatial.transform.Rotation中，'zxz'相当于'rzxz'
    rot = R.from_euler('zxz', theta)
    direction = rot.apply(np.array([0, 0, 1]))
    
    # 初始化掩码
    mask = np.zeros(volume_shape, dtype=np.float32)
    
    # Numba加速的核心函数
    @nb.njit(parallel=True)
    def process_spheres(mask, spheres, direction, delta_theta):
        # 遍历每个球
        for s in range(len(spheres)):
            cx, cy, cz, rmin, rmax = spheres[s]
            
            # 限制处理区域
            x_min = max(0, int(cx - rmax) - 1)
            x_max = min(mask.shape[0], int(cx + rmax) + 2)
            y_min = max(0, int(cy - rmax) - 1)
            y_max = min(mask.shape[1], int(cy + rmax) + 2)
            z_min = max(0, int(cz - rmax) - 1)
            z_max = min(mask.shape[2], int(cz + rmax) + 2)
            
            # 并行处理x维度
            for x in nb.prange(x_min, x_max):
                for y in range(y_min, y_max):
                    for z in range(z_min, z_max):
                        # 计算向量和距离
                        vx = x - cx
                        vy = y - cy
                        vz = z - cz
                        dist_sq = vx*vx + vy*vy + vz*vz
                        dist = np.sqrt(dist_sq)
                        
                        # 检查是否在球壳范围内
                        if rmin <= dist <= rmax and dist > 0:
                            # 计算与方向向量的夹角
                            dot_product = (vx*direction[0] + vy*direction[1] + vz*direction[2]) / dist
                            # 防止数值误差
                            if dot_product > 1.0:
                                dot_product = 1.0
                            elif dot_product < -1.0:
                                dot_product = -1.0
                                
                            alpha = np.arccos(dot_product)
                            # 检查是否在角度范围内
                            if alpha <= delta_theta:
                                mask[x, y, z] = 1.0
        return mask
    
    # 将spheres列表转换为numba兼容的数组
    spheres_array = np.array(sphere_list)
    
    # 调用加速函数 - 耗时高 2s 左右
    mask = process_spheres(mask, spheres_array, direction, delta_theta)
    
    # 分片返回
    result = mask[
        search_origin[0] : search_origin[0] + search_size[0],
        search_origin[1] : search_origin[1] + search_size[1],
        search_origin[2] : search_origin[2] + search_size[2],
    ]
    
    non_zero_count = np.count_nonzero(result)
    
    return result

# if __name__ == "__main__":
#     # 从 pytom_tm.io 模块导入多个函数和异常类，用于文件输入输出
#     from pytom_tm.io import read_mrc_meta_data, read_mrc, write_mrc, UnequalSpacingError
#     import mrcfile
#     """测试球壳掩码生成函数"""
#     # 1. 读取体积文件获取形状
#     volume_path = "d:/work/my/wyj/match-pick/pytom-match-pick/tests/newdata/Position_50_2_6.24Apx.mrc"
#     print(f"读取体积文件: {volume_path}")
#     try:
#         volume = read_mrc(volume_path)
#         print(f"体积形状: {volume.shape}")
#         volume_shape = volume.shape
#     except Exception as e:
#         print(f"体积读取失败: {e}")
#         sys.exit(1)

#     # 2. 设置参数
#     sphere_list_file = "d:/work/my/wyj/match-pick/pytom-match-pick/tests/newdata/metadata/vesicles.txt"
    
#     # 读取球心列表
#     sphere_list = []
#     with open(sphere_list_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 4:
#                 x, y, z, rmax = map(float, parts[:4])
#                 rmin = rmax * 0.5
#                 sphere_list.append((x, y, z, rmin, rmax))

#     theta = (45, 0, 0)  # 欧拉角
#     delta_theta = 45    # 角度范围

#     # 使用完整的体积大小
#     search_origin = (0, 0, 0)
#     search_size = volume_shape

#     # 3. 调用函数
#     print("生成球壳掩码...")
#     try:
#         mask = create_new_mask_from_spheres_optimized_self(
#             sphere_list=sphere_list,
#             volume_shape=volume_shape,
#             theta=theta,
#             search_origin=search_origin,
#             search_size=search_size,
#             delta_theta=delta_theta,
#             theta_unit='deg',
#             euler_order='zyx'
#         )
#         print(f"掩码生成成功，形状: {mask.shape}")

#         # 4. 保存掩码为MRC文件
#         output_path = "d:/work/my/wyj/match-pick/pytom-match-pick/tests/output/sphere_mask_test.mrc"

#         output_ar = np.transpose(mask, (2,1,0))
#         with mrcfile.new(output_path, overwrite=True) as mrc:
#             mrc.set_data(output_ar)
#         print(f"已保存掩码为MRC文件: {output_path}")
#     except Exception as e:
#         print(f"掩码生成失败: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)