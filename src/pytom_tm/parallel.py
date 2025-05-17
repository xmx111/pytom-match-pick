# 导入 numpy 的类型注解模块，用于类型提示
import numpy.typing as npt
# 导入 multiprocessing 模块，用于实现多进程编程
import multiprocessing as mp
# 导入 logging 模块，用于记录日志
import logging
# 导入 queue 模块，用于处理队列
import queue
# 导入 time 模块，用于处理时间相关操作
import time
# 导入 contextlib 模块，用于创建上下文管理器
import contextlib
# 从 multiprocessing.managers 模块导入 BaseProxy 类，用于代理对象
from multiprocessing.managers import BaseProxy
# 从 functools 模块导入 reduce 函数，用于对序列进行累积操作
from functools import reduce
# 从 pytom_tm.tmjob 模块导入 TMJob 类，用于处理模板匹配任务
from pytom_tm.tmjob import TMJob
# 从 pytom_tm.utils 模块导入 mute_stdout_stderr 函数，用于静音标准输出和标准错误输出
from pytom_tm.utils import mute_stdout_stderr

try:
    # 尝试将多进程启动方法设置为 "spawn"，以便为 cupy 设置正确的 GPU
    mp.set_start_method("spawn")
except RuntimeError:
    # 如果设置失败，忽略该错误
    pass


def gpu_runner(
    gpu_id: int,
    task_queue: BaseProxy,
    result_queue: BaseProxy,
    log_level: int,
    unittest_mute: bool,
) -> None:
    """
    启动一个 GPU 运行器，每个运行器应初始化为一个 multiprocessing.Process()，
    并管理在单个 GPU 上运行的任务。每个运行器将从任务队列中获取任务，
    并在任务完成后将结果放入结果队列。当任务队列为空时，gpu_runner 将停止。

    参数
    ----------
    gpu_id: int
        分配给运行器的 GPU 索引
    task_queue: mp.managers.BaseProxy
        来自 multiprocessing 的共享队列，包含要运行的任务
    result_queue: mp.manager.BaseProxy
        来自 multiprocessing 的共享队列，用于存储已完成的任务结果
    log_level: int
        日志记录的级别
    unittest_mute: Bool
        可选的运行器静音选项，用于防止单元测试时终端输出过多，仅用于开发
    """
    if unittest_mute:
        # 如果需要静音，使用 mute_stdout_stderr 上下文管理器
        mute_context = mute_stdout_stderr
    else:
        # 否则，使用空上下文管理器
        mute_context = contextlib.nullcontext
    with mute_context():
        # 配置日志记录的级别
        logging.basicConfig(level=log_level)
        while True:
            try:
                # 尝试从任务队列中立即获取一个任务
                job = task_queue.get_nowait()
                # 启动任务并将结果立即放入结果队列
                result_queue.put_nowait(job.start_job(gpu_id, return_volumes=False))
            except queue.Empty:
                # 如果任务队列为空，跳出循环
                break


def run_job_parallel(
    main_job: TMJob,
    volume_splits: tuple[int, int, int],
    gpu_ids: list[int, ...],
    unittest_mute: bool = False,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    在单个或多个 GPU 上并行运行一个任务。考虑 tomogram_mask 来优化搜索。
    
    参数
    ----------
    main_job: pytom_tm.tmjob.TMJob
        一个来自 pytom_tm 的 TMJob 对象，包含搜索所需的所有数据
    volume_splits: tuple[int, int, int]
        长度为 3 的元组，分别表示 x、y 和 z 方向的分割数
    gpu_ids: list[int, ...]
        用于分布任务的 GPU 索引列表
    unittest_mute: bool, 默认值为 False
        布尔值，用于静音生成的进程的终端输出，仅在单元测试时设置为 True

    返回
    -------
    result: tuple[npt.NDArray[float], npt.NDArray[float]]
        包含 LCCmax 和角度 ID 的体积数组
    """
    # 计算分割后的总块数
    n_pieces = reduce(lambda x, y: x * y, volume_splits)
    # 初始化任务列表
    jobs = []

    # =================== 分割为子任务 ===============
    if n_pieces == 1:
        if len(gpu_ids) > 1:
            # 如果有多个 GPU 且分割块数为 1，按角度搜索分割任务
            jobs = main_job.split_rotation_search(len(gpu_ids))
        else:
            # 如果只有一个 GPU，将整个任务添加到任务列表
            jobs.append(main_job)
    elif n_pieces > 1:
        # 先按体积分割任务
        volume_jobs = main_job.split_volume_search(volume_splits)
        
        # 如果使用了 tomogram_mask，需要检查每个子任务是否有有效区域
        if main_job.has_tomogram_mask:
            # 移除没有有效区域的子任务
            volume_jobs = [j for j in volume_jobs if j is not None]
        
        # 如果没有有效的子任务，返回空结果
        if not volume_jobs:
            # 创建空的结果数组
            empty_shape = main_job.data.shape
            return (
                np.zeros(empty_shape, dtype=np.float32),
                np.zeros(empty_shape, dtype=np.float32)
            )
        
        # 计算每个子体积任务的角度搜索分割因子
        rotation_split_factor = max(1, len(gpu_ids) // len(volume_jobs))
        if rotation_split_factor >= 2:
            # 对每个子体积任务进一步按角度搜索分割
            for j in volume_jobs:
                jobs += j.split_rotation_search(rotation_split_factor)
        else:
            # 仅按体积分割任务
            jobs = volume_jobs
    else:
        # 如果分割块数无效，抛出异常
        raise ValueError("Invalid number of pieces in split volume")

    # ================== 执行任务 =========================
    if len(jobs) == 1:
        # 如果只有一个任务，直接在第一个 GPU 上运行并返回结果
        return main_job.start_job(gpu_ids[0], return_volumes=True)
    elif len(jobs) >= len(gpu_ids):
        # 如果任务数大于等于 GPU 数，使用多进程并行执行任务
        results = []
        with mp.Manager() as manager:
            # 创建任务队列，用于存储待执行的任务
            task_queue = manager.Queue()
            # 创建结果队列，用于存储已完成的任务结果
            result_queue = manager.Queue()
            # 将所有任务放入任务队列
            [task_queue.put_nowait(j) for j in jobs]
            # 创建并启动多个进程
            procs = [
                mp.Process(
                    target=gpu_runner,
                    args=(
                        g,
                        task_queue,
                        result_queue,
                        main_job.log_level,
                        unittest_mute,
                    ),
                )
                for g in gpu_ids
            ]
            [p.start() for p in procs]
            while True:
                while not result_queue.empty():
                    # 从结果队列中获取结果并添加到结果列表
                    results.append(result_queue.get_nowait())
                if len(results) == len(jobs):
                    # 如果所有任务的结果都已获取，记录日志并跳出循环
                    logging.debug("Got all results from the child processes")
                    break
                for p in procs:
                    # 检查每个进程的状态，如果有进程异常退出，终止所有进程并抛出异常
                    if not p.is_alive() and p.exitcode == 1:
                        [x.terminate() for x in procs]
                        raise RuntimeError(
                            "One or more of the processes stopped unexpectedly."
                        )
                # 暂停 1 秒后继续检查
                time.sleep(1)
            # 等待所有进程结束
            [p.join() for p in procs]
            # 记录日志表示所有进程已终止
            logging.debug("Terminated the processes")
        # 合并分割后的任务结果
        return main_job.merge_sub_jobs(stats=results)
    else:
        # 如果任务数小于 GPU 数，抛出异常
        raise ValueError(
            "For some reason there are more gpu_ids than split job, this should never "
            "happen."
        )
