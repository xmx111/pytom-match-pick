import numpy as np
from scipy.spatial.transform import Rotation as R
import time as t
import cupy as cp
def create_new_mask_from_spheres_optimized(
    sphere_mask,     # 先前生成的vesicle_mask.mrc
    sphere_list,     #R: flip by axis 1 to match volume
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
    direction = rot.apply(np.array([-1, 0, 0]))  # R: -z axis pointing to the surface of vesicles 
    u_direction = direction / np.linalg.norm(direction)  # 单位化方向向量
    u_direction = np.asarray(u_direction)  # 转换为cupy数组
    # 初始化掩码
    mask = cp.zeros(volume_shape, dtype=cp.float32)
    # 确保sphere_list是numpy数组
    sphere_list = np.array(sphere_list)
    max_r = int(np.max(sphere_list[:, -1]))  # 获取最大半径

    # 生成公共theta_mask
    box_range = np.arange(-max_r - 1, max_r + 1, dtype=int)
    xv, yv, zv = np.meshgrid(box_range, box_range, box_range, indexing='ij')  # R: no Cartesian indexing
    dot_map = xv * u_direction[0] + yv * u_direction[1] + zv * u_direction[2]
    norm_map = np.sqrt(xv**2 + yv**2 + zv**2) + 1e-10
    cos_map = dot_map / norm_map
    public_theta_mask = cos_map >= np.cos(delta_theta)
    public_theta_mask = public_theta_mask.astype(int)
    public_theta_mask = cp.asarray(public_theta_mask)  # 转换为cupy数组
    def process_spheres(mask, spheres):
        # 遍历每个球
        for s in range(len(spheres)): 
            cx,cy,cz,rmin,rmax = spheres[s]
            mask[cx-rmax:cx+rmax, 
                 cy-rmax:cy+rmax, 
                 cz-rmax:cz+rmax] += public_theta_mask[max_r-rmax:max_r+rmax,
                                                       max_r-rmax:max_r+rmax,
                                                       max_r-rmax:max_r+rmax]
        return mask

    # 将spheres列表转换为numba兼容的数组
    spheres_array = np.array(sphere_list).astype(int)
    # 调用加速函数
    mask = process_spheres(mask, spheres_array)
    mask[mask > 1] = 1

    mask = mask[
        search_origin[0] : search_origin[0] + search_size[0],
        search_origin[1] : search_origin[1] + search_size[1],
        search_origin[2] : search_origin[2] + search_size[2],
    ]

    # 将sphere_mask转为cupy数组
    sphere_mask_cp = cp.asarray(sphere_mask)
    theta_mask = mask*sphere_mask_cp
    theta_mask[theta_mask !=0] = 1  # 将非零值设为1
    theta_mask = cp.asnumpy(theta_mask)
    

    return theta_mask

if __name__ == "__main__":
    '''
    test code
    '''
    import mrcfile

    sphere_mask = mrcfile.open('/home/good/Documents/test/vesicle_mask_bin.mrc', mode='r').data.T
    sphere_mask_cp = cp.asarray(sphere_mask)
    sphere_file = '/home/good/WGL/vesicle_picking_test/vesicles.txt'

    sphere_list = []
    with open(sphere_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y, z, rmax = map(float, parts[:4])
                rmin = rmax * 0.5
                self.sphere_list.append((x, y, z, rmin, rmax))
    
    volume_shape = np.array(sphere_mask.shape)
    theta = np.array([-0.6981317007977318,
                    1.577963267948966,
                    0.0])
    search_origin = (0,0,0)
    search_size = (0,0,0)
    delta_theta=np.pi/4
    start_time = t.time()
    theta_mask = create_new_mask_from_spheres_optimized(
        sphere_mask_cp,
        sphere_list,
        volume_shape,
        theta,
        search_origin,
        search_size,
        delta_theta=delta_theta,
    )
    end_time = t.time()
    print(f"Time taken: {end_time - start_time} seconds")