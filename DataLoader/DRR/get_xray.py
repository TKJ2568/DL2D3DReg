"""
鉴于投影方式和计算光线穿透的过程的不同，将compute_cross_voxel从ct_projector分离出来
1 基于循环的逐页透视投影
2 基于矩阵的透视投影
3 平行投影
"""
import numpy as np
import cupy as cp
import concurrent.futures


def compute_cross_voxel(slice_list, max_axis, o_src, rays, ct_vox, vox_sum_col):
    """
    "circular_perspective"循环投影模式
    :param slice_list: 沿主光轴的切片索引
    :param max_axis: 主光轴
    :param o_src: 光源在CT体素下的坐标
    :param rays: 每条射线在CT体素下的方向向量
    :param ct_vox: CT体素
    :param vox_sum_col:
    :return: X光在每一页体素的累计值，X光射线的权重
    """
    if max_axis == 0:
        ax_ord = [0, 1, 2]
    elif max_axis == 1:
        ax_ord = [1, 0, 2]
    else:
        ax_ord = [2, 0, 1]
    # 归一化ray,保证其主光轴方向为单位矢量
    ray_a = rays[:, ax_ord[1]] / rays[:, ax_ord[0]]
    ray_b = rays[:, ax_ord[2]] / rays[:, ax_ord[0]]
    _ct_vox = ct_vox.transpose(ax_ord)
    for slice_index in slice_list:
        _slice = _ct_vox[slice_index.round().astype(np.int16)]
        sli_d = slice_index - o_src[ax_ord[0]]
        col_a = (ray_a * sli_d + (o_src[ax_ord[1]] + 0.5).astype(np.float32)).astype(np.int16)
        col_b = (ray_b * sli_d + (o_src[ax_ord[2]] + 0.5).astype(np.float32)).astype(np.int16)
        val_msk = (col_a >= 0) & (col_a < ct_vox.shape[ax_ord[1]]) & (col_b >= 0) & (col_b < ct_vox.shape[ax_ord[2]])
        # print(np.sum(_slice[col_a[val_msk], col_b[val_msk]]))
        vox_sum_col[int(slice_index)][val_msk] = _slice[col_a[val_msk], col_b[val_msk]]
    ray_weighs = np.sqrt(ray_a ** 2 + ray_b ** 2 + 1)
    return np.sum(vox_sum_col, axis=0), ray_weighs


# 定义一个函数，它接受页面索引和对应的索引，并返回体素值
def get_voxel_value(_slice, col_a, col_b, val_msk, vox_col):
    vox_col[val_msk] = _slice[col_a[val_msk], col_b[val_msk]]
    return vox_col


def compute_cross_voxel_matrix(slices, max_axis, o_src, rays, ct_vox, vox_sum_col):
    """
    "matrix_perspective",必须要在cuda下使用才有优势，同时建议开始多线程并行赋值
    :param slices: 沿主光轴的切片索引
    :param max_axis: 主光轴
    :param o_src: 光源在CT体素下的坐标
    :param rays: 每条射线在CT体素下的方向向量
    :param ct_vox: CT体素
    :param vox_sum_col:
    :return: X光在每一页体素的累计值，X光射线的权重
    """
    vox_sum_col = cp.array(vox_sum_col)
    if max_axis == 0:
        ax_ord = [0, 1, 2]
    elif max_axis == 1:
        ax_ord = [1, 0, 2]
    else:
        ax_ord = [2, 0, 1]
        # 归一化ray,保证其主光轴方向为单位矢量
    ray_a = rays[:, ax_ord[1]] / rays[:, ax_ord[0]]
    ray_b = rays[:, ax_ord[2]] / rays[:, ax_ord[0]]
    sli_d = cp.array(slices - o_src[max_axis])
    ray_a = cp.array(ray_a[:, None])
    ray_b = cp.array(ray_b[:, None])
    col_a = (ray_a * sli_d[None, :] + (o_src[ax_ord[1]] + 0.5).astype(np.float32)).astype(np.int16).T
    col_b = (ray_b * sli_d[None, :] + (o_src[ax_ord[2]] + 0.5).astype(np.float32)).astype(np.int16).T
    val_msk = cp.array((col_a >= 0) & (col_a < ct_vox.shape[ax_ord[1]]) \
                       & (col_b >= 0) & (col_b < ct_vox.shape[ax_ord[2]]))
    _ct_vox = cp.array(ct_vox.transpose(ax_ord))

    # 把每一页体素中所有射线的累计量获取到
    # time1 = time.time()
    # 初始化线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 使用 executor.submit 提交任务到线程池，并使用 as_completed 来迭代完成的任务
        futures = [executor.submit(get_voxel_value, _ct_vox[i],
                                   col_a[i], col_b[i], val_msk[i], vox_sum_col[i]) for i in cp.arange(len(sli_d))]
        # 等待所有任务完成，并收集结果
        for future in concurrent.futures.as_completed(futures):
            # 注意：这里假设 sli_d 和 vox_col 的索引是对应的
            vox_sum_col[futures.index(future)] = future.result()
    # print("投影循环用时", time.time() - time1)
    ray_weighs = cp.sqrt(ray_a ** 2 + ray_b ** 2 + 1).ravel()
    return cp.sum(vox_sum_col, axis=0).get(), ray_weighs.get()


def orthogonal_projection(slice_list, max_axis, o_src, rays, d_s2p=800, ct_vox=None, vox_sum_col=None):
    """
    "orthogonal_projection"正交投影模式，假设为平行光，目前用不到。d_s2p目前的默认值是光源坐标系下，
    在CT体素坐标系下可能需要根据主光轴的不同产生变化（当voxel_space做过归一化之后就不用考虑这个了）
    之后会把orthogonal_projection函数融合进compute_cross_voxel。目前是仅用循环简单实现，后续考虑矩阵实现。
    :param slice_list: 沿主光轴的切片索引
    :param max_axis: 主光轴
    :param o_src: 光源在CT体素下的坐标
    :param rays: 每条射线在CT体素下的方向向量
    :param d_s2p: 定义在CT体素坐标系下光源到像平面的距离
    :param ct_vox: CT体素
    :param vox_sum_col:
    :return: X光在每一页体素的累计值，X光射线的权重
    """
    if max_axis == 0:
        ax_ord = [0, 1, 2]
    elif max_axis == 1:
        ax_ord = [1, 0, 2]
    else:
        ax_ord = [2, 0, 1]
    # 归一化ray,保证其主光轴方向为单位矢量
    ray_a = (rays[:, ax_ord[1]] / rays[:, ax_ord[0]]) * d_s2p
    ray_b = (rays[:, ax_ord[2]] / rays[:, ax_ord[0]]) * d_s2p
    _ct_vox = ct_vox.transpose(ax_ord)
    for slice_index in slice_list:
        _slice = _ct_vox[slice_index.round().astype(np.int16)]
        col_a = (ray_a + (o_src[ax_ord[1]] + 0.5).astype(np.float32)).astype(np.int16)
        col_b = (ray_b + (o_src[ax_ord[2]] + 0.5).astype(np.float32)).astype(np.int16)
        val_msk = (col_a >= 0) & (col_a < ct_vox.shape[ax_ord[1]]) & (col_b >= 0) & (col_b < ct_vox.shape[ax_ord[2]])
        vox_sum_col[int(slice_index)][val_msk] = _slice[col_a[val_msk], col_b[val_msk]]
    ray_weighs = np.sqrt(ray_a ** 2 + ray_b ** 2 + 1)
    return np.sum(vox_sum_col, axis=0), ray_weighs
