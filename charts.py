import pyecharts
from pyecharts import options as opts
from pyecharts.options.global_options import ThemeType

def accLine(xaxisBatchList, totalAccList, totalTestAccList, saveName):
    accLine = pyecharts.charts.Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    accLine.add_xaxis(xaxis_data=xaxisBatchList)
    accLine.add_yaxis("train acc", totalAccList, is_connect_nones=True, is_smooth=True, is_symbol_show=False,
                      is_hover_animation=True)
    accLine.add_yaxis("value acc", totalTestAccList, is_connect_nones=True, is_smooth=True, is_symbol_show=False,
                      is_hover_animation=True)
    accLine.set_global_opts(title_opts=pyecharts.options.TitleOpts(title="Accuracy Variation"),
                            tooltip_opts=pyecharts.options.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                            xaxis_opts=pyecharts.options.AxisOpts(name="batch_count", name_location="middle", name_gap=50,
                                                                  splitline_opts=pyecharts.options.SplitLineOpts(is_show=True)),
                            yaxis_opts=pyecharts.options.AxisOpts(name="accuracy", name_location="middle", name_gap=50,
                                                                  splitline_opts=pyecharts.options.SplitLineOpts(is_show=True)))
    accLine.render(saveName + ".html")

def lossLine(xaxisBatchList, totalLossList, totalTestLossList, saveName):
    lossLine = pyecharts.charts.Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    lossLine.add_xaxis(xaxis_data=xaxisBatchList)
    lossLine.add_yaxis("train loss", totalLossList, is_connect_nones=True, is_smooth=True, is_symbol_show=False,
                       is_hover_animation=True)
    lossLine.add_yaxis("value loss", totalTestLossList, is_connect_nones=True, is_smooth=True, is_symbol_show=False,
                       is_hover_animation=True)
    lossLine.set_global_opts(title_opts=pyecharts.options.TitleOpts(title="Loss curve"),
                             tooltip_opts=pyecharts.options.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                             xaxis_opts=pyecharts.options.AxisOpts(name="batch_count", name_location="middle", name_gap=50,
                                                                   splitline_opts=pyecharts.options.SplitLineOpts(is_show=True)),
                             yaxis_opts=pyecharts.options.AxisOpts(name="loss", name_location="middle", name_gap=50,
                                                                   splitline_opts=pyecharts.options.SplitLineOpts(is_show=True)))
    lossLine.render(saveName + ".html")

def f1Line(xaxisBatchList, totalF1scoreList, totalTestF1scoreList, saveName):
    f1Line = pyecharts.charts.Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    f1Line.add_xaxis(xaxis_data=xaxisBatchList)
    f1Line.add_yaxis("train f1 score", totalF1scoreList, is_connect_nones=True, is_smooth=True, is_symbol_show=False,
                     is_hover_animation=True)
    f1Line.add_yaxis("value f1 score", totalTestF1scoreList, is_connect_nones=True, is_smooth=True,
                     is_symbol_show=False, is_hover_animation=True)
    f1Line.set_global_opts(title_opts=pyecharts.options.TitleOpts(title="f1 curve"),
                           tooltip_opts=pyecharts.options.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                           xaxis_opts=pyecharts.options.AxisOpts(name="batch_count", name_location="middle", name_gap=50,
                                                                 splitline_opts=pyecharts.options.SplitLineOpts(is_show=True)),
                           yaxis_opts=pyecharts.options.AxisOpts(name="f1", name_location="middle", name_gap=50,
                                                                 splitline_opts=pyecharts.options.SplitLineOpts(is_show=True)))
    f1Line.render(saveName + ".html")

def aucLine(xaxisBatchList, totalAucscoreList, totalTestAucscoreList, saveName):
    aucLine = pyecharts.charts.Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    aucLine.add_xaxis(xaxis_data=xaxisBatchList)
    aucLine.add_yaxis("train auc", totalAucscoreList, is_connect_nones=True, is_smooth=True,
                      is_symbol_show=False,
                      is_hover_animation=True,
                      )
    aucLine.add_yaxis("value auc", totalTestAucscoreList, is_connect_nones=True, is_smooth=True,
                      is_symbol_show=False,
                      is_hover_animation=True,
                      )
    aucLine.set_global_opts(title_opts=pyecharts.options.TitleOpts(title="aucScore curve"),
                            tooltip_opts=pyecharts.options.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                            xaxis_opts=pyecharts.options.AxisOpts(name="batch_count", name_location="middle", name_gap=50,
                                                                  splitline_opts=pyecharts.options.SplitLineOpts(is_show=True)),
                            yaxis_opts=pyecharts.options.AxisOpts(name="auc", name_location="middle", name_gap=50,
                                                                  splitline_opts=pyecharts.options.SplitLineOpts(is_show=True)))
    aucLine.render(saveName + ".html")