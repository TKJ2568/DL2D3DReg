import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# 创建一个 2 行 3 列的子图布局
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 列标题
col_titles = ['Column 1', 'Column 2', 'Column 3']

# 遍历第一行的子图，设置列标题
for ax, col_title in zip(axes[0], col_titles):
    ax.set_title(col_title, fontsize=14, fontweight='bold')

# 填充图像和子标题
for row_idx in range(2):  # 遍历两行
    for col_idx in range(3):  # 遍历三列
        ax = axes[row_idx, col_idx]

        # 根据行列索引选择不同的数据绘制
        if col_idx == 0:
            ax.plot(x, y1, color='blue')
        elif col_idx == 1:
            ax.plot(x, y2, color='green')
        elif col_idx == 2:
            ax.plot(x, y3, color='red')

        # 设置子标题（显示在图像底部）
        sub_title = f"Subplot ({row_idx + 1}, {col_idx + 1})"
        ax.text(0.5, -0.15, sub_title, fontsize=10, ha='center', transform=ax.transAxes)

        # 可选：关闭坐标轴以更简洁
        ax.set_xticks([])
        ax.set_yticks([])

# 调整布局以避免重叠
plt.tight_layout()

# 显示图像
plt.show()