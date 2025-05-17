import numpy as np
import mrcfile
from tqdm import tqdm

volume_shape = (1024, 1024, 250)
output_path = r'd:/work/my/wyj/match-pick/pytom-match-pick/tests/output/vesicle_mask2.mrc'

f = open(r'd:/work/my/wyj/match-pick/pytom-match-pick/tests/newdata/metadata/vesicles.txt', 'r')
file_cont = [i.split('\t') for i in f.readlines()]
f.close()
vesicle_list = [[float(i) for i in raw] for raw in file_cont]
for vesicle in vesicle_list:
    vesicle.append(vesicle[-1]*0.5)
mask = np.zeros(volume_shape, dtype=np.float32)
# for vesicle in tqdm(vesicle_list):
#     cx,cy,cz,rmax,rmin = vesicle
#     search_sub = [[int(cx-rmax),int(cx+rmax)],
#                   [int(cx-rmax),int(cx+rmax)],
#                   [int(cx-rmax),int(cx+rmax)]]
#     for x in range(int(cx-rmax)-1,int(cx+rmax)+2):
#         for y in range(int(cy-rmax)-1,int(cy+rmax)+2):
#             for z in range(int(cz-rmax)-1,int(cz+rmax)+2):
#                 dist = np.linalg.norm(np.array([x-cx,y-cy,z-cz]), 2)
#                 if (dist>=rmin) and (dist<=rmax):
#                     mask[x][y][z] = 1
# 对每个球体进行处理
for vesicle in tqdm(vesicle_list):
    # 获取球心坐标和半径
    cx, cy, cz, rmax, rmin = vesicle
    
    # 计算搜索范围（添加1个像素的边界）
    x_min = max(0, int(cx - rmax) - 1)
    x_max = min(volume_shape[0], int(cx + rmax) + 2)
    y_min = max(0, int(cy - rmax) - 1)
    y_max = min(volume_shape[1], int(cy + rmax) + 2)
    z_min = max(0, int(cz - rmax) - 1)
    z_max = min(volume_shape[2], int(cz + rmax) + 2)
    
    # 只在球体周围的立方体区域内计算距离
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
                if rmin <= dist <= rmax:
                    mask[x, y, z] = 1
#G: change axis order before export 
output_ar = np.transpose(mask, (2,1,0))
f = mrcfile.new(output_path, overwrite=True)
f.set_data(output_ar)
f.voxel_size = 6.24
f.update_header_from_data()
f.close()