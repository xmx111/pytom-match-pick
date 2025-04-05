import pathlib
# 从pytom_tm.io模块导入读取散焦文件、剂量文件和倾斜角度文件的函数
from pytom_tm.io import read_defocus_file, read_dose_file, read_tlt_file

# 定义用于tomo_104的剂量和CTF参数
# 球面像差系数
cs = 2.7
# 振幅衬度
amp = 0.08
# 加速电压
vol = 200
# 读取散焦数据，文件路径为当前文件所在目录下的Data文件夹中的test_imod.defocus文件
defocus_data = read_defocus_file(
    pathlib.Path(__file__).parent.joinpath("Data").joinpath("test_imod.defocus")
)
# 初始化CTF参数列表
CTF_PARAMS = []
# 遍历散焦数据
for d in defocus_data:
    # 为每个散焦值创建一个CTF参数的字典，并添加到CTF_PARAMS列表中
    CTF_PARAMS.append(
        {
            # 散焦值
            "defocus": d,
            # 振幅衬度
            "amplitude_contrast": amp,
            # 加速电压
            "voltage": vol,
            # 球面像差系数
            "spherical_aberration": cs,
            # 相位偏移角度（度）
            "phase_shift_deg": 0.0,
        }
    )

# 读取累积剂量数据，文件路径为当前文件所在目录下的Data文件夹中的test_dose.txt文件
ACCUMULATED_DOSE = read_dose_file(
    pathlib.Path(__file__).parent.joinpath("Data").joinpath("test_dose.txt")
)
# 读取倾斜角度数据，文件路径为当前文件所在目录下的Data文件夹中的test_angles.rawtlt文件
TILT_ANGLES = read_tlt_file(
    pathlib.Path(__file__).parent.joinpath("Data").joinpath("test_angles.rawtlt")
)
