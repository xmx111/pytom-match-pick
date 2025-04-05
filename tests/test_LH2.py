from pytom_tm import entry_points
import pathlib

def run_template_matching(
    template_file: str,
    mask_file: str,
    volume_file: str,
    output_dir: str,
    tilt_file: str,
    defocus_file: str,
    dose_file: str,
    gpu_ids: list[int]
):
    """
    运行LH2的模板匹配程序

    参数:
        template_file: 模板文件路径
        mask_file: 掩码文件路径
        volume_file: 体积文件路径
        output_dir: 输出目录
        tilt_file: 倾斜角度文件路径
        defocus_file: 散焦文件路径
        dose_file: 剂量文件路径
        gpu_ids: GPU设备ID列表
    """
    # 构建命令行参数列表
    args = [
        "pytom_match_template.py",
        "-t", str(pathlib.Path(template_file)),
        "-m", str(pathlib.Path(mask_file)),
        "-v", str(pathlib.Path(volume_file)),
        "-d", str(pathlib.Path(output_dir)),
        "--angular-search", "3.00",  # LH2需要更高的采样精度
        "--voxel-size", "6.24",
        "-g", *[str(gpu_id) for gpu_id in gpu_ids],  # 展开GPU ID列表
        "-a", str(pathlib.Path(tilt_file)),
        "--defocus", str(pathlib.Path(defocus_file)),
        "--dose-accumulation", str(pathlib.Path(dose_file)),
        "--per-tilt-weighting",
        "--amplitude-contrast", "0.07",
        "--spherical-aberration", "2.7",
        "--voltage", "300",
        "--tomogram-ctf-model", "phase-flip",
        "--z-axis-rotational-symmetry", "8",  # LH2的C8对称性
        "-s", "3", "3", "1"  # 体积分割参数
    ]

    # 调用模板匹配函数
    entry_points.match_template(args)

# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    template_file = "newdata/template6_24/LH2_template.mrc"  # LH2模板文件路径
    mask_file = "newdata/template6_24/LH2_mask.mrc"         # LH2掩码文件路径
    volume_file = "newdata/Position_50_2_6.24Apx.mrc"  # 体积文件路径
    output_dir = "output"                         # 输出目录
    tilt_file = "newdata/metadata/Position_50_2.tlt"      # 倾斜角度文件路径
    defocus_file = "newdata/metadata/Position_50_2_defocus.txt"  # 散焦文件路径
    dose_file = "newdata/metadata/Position_50_2_dose.txt"  # 剂量文件路径
    gpu_ids = [0]  # GPU ID列表

    # 运行模板匹配
    run_template_matching(
        template_file=template_file,
        mask_file=mask_file,
        volume_file=volume_file,
        output_dir=output_dir,
        tilt_file=tilt_file,
        defocus_file=defocus_file,
        dose_file=dose_file,
        gpu_ids=gpu_ids
    )