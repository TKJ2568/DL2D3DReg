import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.axes_size import AxesX


def top_n_indices(arr, n):
    """返回一维数组中前n个最大值的索引"""
    if n <= 0:
        return np.array([], dtype=int)
    if n >= len(arr):
        return np.arange(len(arr))
    return np.argsort(arr)[-n:][::-1]  # 取最后n个并反转顺序

def get_img(pose, projector):
    im = projector.project(pose[: 3], pose[3:], mode='not_display')
    im = Image.fromarray(im)
    return im

def get_real_pose(pose, init_pose):
    pose[:3] += init_pose
    formatted_result = [f"{x:.2f}" for x in np.round(pose, 2)]
    return f"[{', '.join(formatted_result)}]"

class AbnormalAnalyser:
    def __init__(self, init_pose, projector, **kwargs):
        self.init_pose = init_pose
        self.projector = projector
        self.abnormal_img_num = kwargs.get("abnormal_img_num", 10)
        self.tru = []
        self.pre = {}
        self.init_pre_dict(kwargs.get("model_name_list", []))
        # 每个样本的异常分数
        self.abnormal_score = []
        self.save_path = kwargs.get("save_path", "./") + "/test_results"
        self.save_name = kwargs.get("save_name", "异常值可视化.png")

    def init_pre_dict(self, model_name_list):
        for model_name in model_name_list:
            self.pre[model_name] = []

    def add_one_result(self, model_name_list, predict_list):
        for i in range(len(model_name_list)):
            self.pre[model_name_list[i]].append(predict_list[i])

    def abnormal_visualize(self):
        self.abnormal_score = np.sum(np.array(self.abnormal_score), axis=0)
        top_n = top_n_indices(self.abnormal_score, self.abnormal_img_num)

        col_titles = ["ground_truth"] + list(self.pre.keys())
        img_matrix = []
        sub_titles_matrix = []

        for i in top_n:
            tru_img = get_img(self.tru[i], self.projector)
            tru_pos = get_real_pose(self.tru[i], self.init_pose)
            pre_img_list = []
            pre_pos_list = []
            for pre_list in self.pre.values():
                pre_img = get_img(pre_list[i], self.projector)
                pre_img_list.append(pre_img)
                pre_pos = get_real_pose(pre_list[i], self.init_pose)
                pre_pos_list.append(pre_pos)
            images = [tru_img] + pre_img_list
            sub_titles = [tru_pos] + pre_pos_list
            img_matrix.append(images)
            sub_titles_matrix.append(sub_titles)

        # 可视化部分
        num_cols = len(col_titles)  # 列数为模型数量 + ground truth
        num_rows = len(top_n)  # 行数为异常样本数量

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
        if num_rows == 1:  # 如果只有一行，需要将 axes 转换为二维数组
            axes = np.expand_dims(axes, axis=0)

        # 设置列标题（模型名）
        for ax, col_title in zip(axes[0], col_titles):
            ax.set_title(col_title, fontsize=8, fontweight='bold')

        # 填充图像和子标题
        for row_idx, (images, sub_titles) in enumerate(zip(img_matrix, sub_titles_matrix)):
            for col_idx, (img, sub_title) in enumerate(zip(images, sub_titles)):
                ax = axes[row_idx, col_idx]
                assert isinstance(ax, Axes)
                ax.imshow(img, cmap='gray')
                ax.axis('off')  # 关闭坐标轴
                # 设置子标题（显示在图像底部）
                ax.text(0.5, -0.10, sub_title, fontsize=5.5, ha='center', transform=ax.transAxes)

        # 调整布局以避免重叠
        plt.tight_layout()
        # 渲染图表
        save_file_path = os.path.join(str(self.save_path), self.save_name)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')