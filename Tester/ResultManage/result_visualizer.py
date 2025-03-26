import os

from pyecharts import options as opts
from pyecharts.charts import Bar, Boxplot, Grid
from pyecharts.commons.utils import JsCode


class ResultVisualizer:
    def __init__(self, test_name_list, **kwargs):
        self.test_name_list = test_name_list
        self.save_path = kwargs.get("save_path", "./") + "/test_results"
        self.save_name = kwargs.get("save_name", "测试评价指标对比.html")
        self.figure_config = kwargs.get("figure_config")

    def visualize_results(self, logger_dicts):
        model_names = list(logger_dicts.keys())

        # 初始化箱线图
        boxplot = Boxplot()

        # 添加x轴数据
        boxplot.add_xaxis(self.test_name_list)

        # 循环添加每个模型的y轴数据
        for model_name in model_names:
            model_data = logger_dicts[model_name].get_normalized_result()
            boxplot.add_yaxis(
                series_name=model_name,
                y_axis=boxplot.prepare_data(model_data)
            )

        # 设置箱线图全局选项
        boxplot.set_global_opts(
            title_opts=opts.TitleOpts(title="多模型在不同评价指标下的表现比较"),
            yaxis_opts=opts.AxisOpts(
                name="得分",
                splitarea_opts=opts.SplitAreaOpts(is_show=True)
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
            legend_opts=opts.LegendOpts(pos_top="0%"),
        )

        # 初始化条形图
        bar = Bar()

        # 添加x轴数据
        bar.add_xaxis(self.test_name_list)

        # 循环添加每个模型的y轴数据
        for model_name in model_names:
            model_data = logger_dicts[model_name].get_mean_result()
            bar.add_yaxis(
                series_name=model_name,
                y_axis=model_data,
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="top",
                    formatter=JsCode("function(x){return x.value.toFixed(2);}")
                )
            )

        # 使用 Grid 将两张图表组合在一起
        grid = (
            Grid(init_opts=opts.InitOpts(width=self.figure_config["figure_size"][0],
                                         height=self.figure_config["figure_size"][1]))
            .add(
                boxplot,
                grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="10%", height="40%")
            )
            .add(
                bar,
                grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="60%", height="40%")  # 条形图放在上半部分
            )
        )

        # 渲染图表
        save_file_path = os.path.join(str(self.save_path), self.save_name)
        grid.render(save_file_path)
