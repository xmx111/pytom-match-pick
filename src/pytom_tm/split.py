import mrcfile
import numpy as np
import os
from pathlib import Path

def split_mrc(input_path, output_dir=None):
    """
    将MRC文件分割成4个相等的部分，只在X和Y轴切分
    
    参数:
        input_path: 输入MRC文件路径
        output_dir: 输出目录，默认为输入文件所在目录
    
    返回:
        生成的四个文件的路径列表
    """
    # 如果没有指定输出目录，使用输入文件所在目录
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件的基本名称（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    mark_path = 'd:/work/my/wyj/match-pick/pytom-match-pick/tests/output/vesicle_mask.mrc'
    mark_mrc = mrcfile.open(mark_path)
    mark_name = 'vesicle_mask'
    # 读取MRC文件
    with mrcfile.open(input_path) as mrc:
        # 获取数据体积
        data = mrc.data
        mark_data = mark_mrc.data
        # 获取数据维度
        depth, height, width = data.shape
        
        # 计算每个维度的中点
        mid_height = height // 2
        mid_width = width // 2
        
        # 四个区域的数据
        # 左上
        data_1 = data[:, :mid_height, :mid_width]
        # 右上
        data_2 = data[:, :mid_height, mid_width:]
        # 左下
        data_3 = data[:, mid_height:, :mid_width]
        # 右下
        data_4 = data[:, mid_height:, mid_width:]

        # 四个区域的数据
        # 左上
        mark_data_1 = mark_data[:, :mid_height, :mid_width]
        # 右上
        mark_data_2 = mark_data[:, :mid_height, mid_width:]
        # 左下
        mark_data_3 = mark_data[:, mid_height:, :mid_width]
        # 右下
        mark_data_4 = mark_data[:, mid_height:, mid_width:]
        
        # 准备输出文件路径
        output_files = []
        for i, sub_data in enumerate([data_1, data_2, data_3, data_4], 1):
            output_file = os.path.join(output_dir, f"{base_name}_part{i}.mrc")
            output_files.append(output_file)
            
            # 保存为新的MRC文件
            with mrcfile.new(output_file, overwrite=True) as new_mrc:
                # 复制原始数据的属性
                new_mrc.voxel_size = mark_mrc.voxel_size
                # 设置新的数据
                new_mrc.set_data(sub_data)

        for i, sub_data in enumerate([mark_data_1, mark_data_2, mark_data_3, mark_data_4], 1):
            output_file = os.path.join(output_dir, f"{mark_name}_part{i}.mrc")
            output_files.append(output_file)
            
            # 保存为新的MRC文件
            with mrcfile.new(output_file, overwrite=True) as new_mrc:
                # 复制原始数据的属性
                new_mrc.voxel_size = mark_mrc.voxel_size
                # 设置新的数据
                new_mrc.set_data(sub_data)
    
    return output_files

def main():
    # 调用分割函数
    output_files = split_mrc('d:/work/my/wyj/match-pick/pytom-match-pick/tests/newdata/Position_50_2_6.24Apx.mrc', 'd:/work/my/wyj/match-pick/pytom-match-pick/tests/output')
    
    # 打印输出文件路径
    print("已生成以下文件：")
    for file_path in output_files:
        print(f"- {file_path}")

if __name__ == "__main__":
    main()
    