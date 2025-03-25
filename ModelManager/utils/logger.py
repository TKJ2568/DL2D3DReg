import numpy as np
from pyecharts.charts import Line, Grid
from pyecharts import options as opts
import csv
import copy

def normalize_data(data):
    """
    归一化数据到 [0, 1] 范围
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return copy.deepcopy((data - min_val) / (max_val - min_val))

def create_line_chart(x_data, y_data, y_label, title=None, y_axis_name=None,
                      legend_pos_top=None):
    line = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis(y_label, y_data, is_smooth=True,
                   symbol_size=8,
                   is_hover_animation=False,
                   label_opts=opts.LabelOpts(is_show=False),
                   linestyle_opts=opts.LineStyleOpts(width=1.5))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title, pos_left="center") if title else None,
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(
                name='epoch',
                type_="value",
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=True),
            ),
            yaxis_opts=opts.AxisOpts(name=y_axis_name),
            legend_opts=opts.LegendOpts(pos_left="left", pos_top=legend_pos_top),  # 设置 legend 的位置
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                feature={
                    "dataZoom": {"yAxisIndex": "none"},
                    "restore": {},
                    "saveAsImage": {},
                },
            ),
        )
    )
    return line

class Logger:
    def __init__(self):
        """
        初始化训练监控器。
        """
        self.epochs = []  # 存储轮次
        self.losses = []  # 存储损失值
        self.precisions = []  # 存储精度

    def add_entry(self, epoch, loss, precision):
        """
        添加一条训练记录。

        :param epoch: int, 当前轮次
        :param loss: float, 当前损失值
        :param precision: float, 当前精度
        """
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.precisions.append(precision)

    def visualization(self, figure_config):
        """
        可视化训练过程中的损失值和精度，并保存结果。
        """
        self.losses = np.array(self.losses).astype(np.float64)
        self.precisions = np.array(self.precisions).astype(np.float64)

        # 创建损失值折线图
        loss_line = create_line_chart(
            self.epochs, self.losses, figure_config["y_label_loss"],
            title=figure_config["title"], y_axis_name="Loss",
            legend_pos_top="3%",  # legend 放在顶部
        )

        # 创建精度折线图
        precision_line = create_line_chart(
            self.epochs, self.precisions, figure_config["y_label_precision"],
            y_axis_name="Precision",
            legend_pos_top="33%",  # legend 放在中部
        )

        # 归一化数据
        normalized_losses = normalize_data(self.losses)
        normalized_precisions = normalize_data(self.precisions)

        # 创建归一化后的损失值折线图
        norm_loss_line = create_line_chart(
            self.epochs, normalized_losses, figure_config["y_label_loss"],
            y_axis_name="Normalized Loss",
            legend_pos_top="63%",  # legend 放在底部
        )

        # 创建归一化后的精度折线图
        norm_precision_line = create_line_chart(
            self.epochs, normalized_precisions, figure_config["y_label_precision"],
            y_axis_name="Normalized Precision",
        )
        # 将归一化后的两条曲线叠加到同一个图上
        overlap_chart = norm_loss_line.overlap(norm_precision_line)

        # 使用 Grid 组件将三个图表放在同一页面中
        grid = (
            Grid(init_opts=opts.InitOpts(width=figure_config["figure_size"][0], height=figure_config["figure_size"][1]))
            .add(
                loss_line,
                grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="10%", height="20%")
            )
            .add(
                precision_line,
                grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="40%", height="20%")
            )
            .add(
                overlap_chart,
                grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="70%", height="20%")
            )
        )

        # 全局 datazoom_opts 配置
        grid.options.update(
            dataZoom=[
                opts.DataZoomOpts(
                    is_show=True,
                    is_realtime=True,
                    start_value=30,
                    end_value=70,
                    xaxis_index=[0, 1, 2],  # 作用于所有子图的 x 轴
                )
            ]
        )

        # 保存可视化结果
        grid.render(figure_config["save_path"])

    def save_to_csv(self, file_path):
        """
        将日志保存到 CSV 文件。

        :param file_path: str, CSV 文件保存路径
        """
        # 确保数据长度一致
        if not (len(self.epochs) == len(self.losses) == len(self.precisions)):
            raise ValueError("epochs, losses, and precisions 的长度不一致，无法保存到 CSV 文件。")

        # 打开文件并写入数据
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(["Epoch", "Loss", "Precision"])
            # 写入数据
            for epoch, loss, precision in zip(self.epochs, self.losses, self.precisions):
                writer.writerow([epoch, loss, precision])