from pyecharts import options as opts
from pyecharts.charts import Boxplot
import numpy as np

# 模拟数据：3个模型在4个评价指标下的测试结果（每组20次测试）
np.random.seed(42)
models = ["模型A", "模型B", "模型C"]
metrics = ["准确率", "召回率", "F1值", "AUC"]

# 生成模拟数据：model_results[模型][指标] = 数据列表
model_results = {
    "模型A": {
        "准确率": np.random.normal(0.85, 0.05, 20).tolist(),
        "召回率": np.random.normal(0.78, 0.08, 20).tolist(),
        "F1值": np.random.normal(0.82, 0.06, 20).tolist(),
        "AUC": np.random.normal(0.88, 0.04, 20).tolist(),
    },
    "模型B": {
        "准确率": np.random.normal(0.82, 0.06, 20).tolist(),
        "召回率": np.random.normal(0.82, 0.07, 20).tolist(),
        "F1值": np.random.normal(0.81, 0.05, 20).tolist(),
        "AUC": np.random.normal(0.85, 0.05, 20).tolist(),
    },
    "模型C": {
        "准确率": np.random.normal(0.88, 0.04, 20).tolist(),
        "召回率": np.random.normal(0.75, 0.09, 20).tolist(),
        "F1值": np.random.normal(0.79, 0.07, 20).tolist(),
        "AUC": np.random.normal(0.90, 0.03, 20).tolist(),
    }
}

# 创建箱线图
boxplot = Boxplot()

# 添加x轴（评价指标）
boxplot.add_xaxis(metrics)

# 为每个模型添加数据系列
for model in models:
    # 准备该模型在所有指标下的数据
    model_data = [model_results[model][metric] for metric in metrics]
    boxplot.add_yaxis(
        model,
        boxplot.prepare_data(model_data),
        itemstyle_opts=opts.ItemStyleOpts(
            color=np.random.rand(3).tolist()  # 随机颜色
        )
    )

# 设置全局选项
boxplot.set_global_opts(
    title_opts=opts.TitleOpts(title="多模型在不同评价指标下的表现比较"),
    yaxis_opts=opts.AxisOpts(
        name="得分",
        min_=0.6,
        max_=1.0,
        splitarea_opts=opts.SplitAreaOpts(is_show=True)
    ),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)

boxplot.render("output/boxplot_models_by_metrics.html")