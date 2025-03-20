from pyecharts import options as opts
from pyecharts.charts import Line
import numpy as np

# 示例数据
x_data = [10, 20, 30, 40, 50]
y_data = np.array([10, 20, 15, 25, 30]).astype(np.float64)

# 创建折线图
line = Line()

# 添加数据
line.add_xaxis(x_data)
line.add_yaxis("系列名称", y_data)

# 设置全局配置
line.set_global_opts(
    xaxis_opts=opts.AxisOpts(
        name="X轴",  # X轴名称
        type_="value",  # X轴类型为数值
    ),
    yaxis_opts=opts.AxisOpts(
        name="Y轴",  # Y轴名称
        type_="value",  # Y轴类型为数值
    ),
)

# 渲染图表
line.render("output/line_chart.html")