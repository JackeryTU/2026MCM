# P2 独立视觉与语义核查记录

审查范围为冻结的16张逻辑图。每张彩色PNG和对应 _qa 灰度PNG均实际打开查看，非依据作者记录直接判定。完整复现已于2026-09-11完成，入口退出0；随后独立核对新生成的16张彩色PNG与16张灰度PNG，解码像素数组均与已读权威图完全一致，最大像素差0，16张SVG均无嵌入栅格。证据见同目录 numeric_physical_comparison.json；此记录本身不代替全链回执。仅使用路径、大小、时间和实际数值，不使用任何文件摘要。

| 逻辑图 | 独立查看结论 |
|---|---|
|raw_q1_typical_profiles|负荷/PV与价格分面，单位、实虚线和阶梯可读，灰度不丢主信息。|
|process_q1_price_dispatch|144点离散关系，颜色是时刻而非连线因果；rho仅描述。|
|result_q1_optimal_schedule|充放电正负区分，SOC有145个边界点并显示24h，与循环日定义一致。|
|raw_q2_annual_netload|365×144网格，净负荷含负值，色标对称且亮度单调；灰度保留次序。|
|raw_q2_pv_load_relation|使用全部52560个原始时段的hexbin，不以密集重叠散点冒充样本数量；对数数量色标完整。|
|process_q2_forecast_validation|48096预测–实测时段，坐标同尺度，45度参考、MAE/RMSE/R2及色标不遮挡数据。|
|result_q2_annual_risk|334日原值与7日中心均线以粗细区分；中心均线仅用于描述，不能作为决策输入。|
|result_q2_baselines|334日费用总额零起点，合同与应急分量在灰度仍有亮度区别。|
|raw_q3_forecast_error|4次发布×24提前期MAE，不能把夜间零值解释成日间预报精准；跨年缺失按可得目标计数。|
|process_q3_contract_updates|0时虚线与最终实线可区分，6/12/18h标线清楚；实际净负荷不是储能后需求，4.84MWh是绝对调整总量。|
|result_q3_adjustment_benefit|q3_0h与q3_6h相同0时信息的334日配对；45度参考与77.8%费用下降天数，不声称每日均改善。|
|result_q3_update_frequency|点图不把类别连成连续关系，圆/方在灰度可辨；2h星号为因果合成修正，下图明确替代结算仅同策略重计费。|
|raw_q4_price_variability|12月箱线，中位线、须、刻度可读；样本数与1.5IQR须约定由图契约补足。|
|process_q4_price_dispatch|48096时段hexbin，净放电单位kWh/10min，rho=.44标注“事后描述”，灰度色阶清楚。|
|result_q4_strategy_comparison|4主策略总额/应急双面板，零起点、纹理在灰度可辨；13.2%/13.1%是总体策略差，需保留图契约中的非纯频率因果说明。|
|result_q4_alpha_sensitivity|alpha连续敏感性参数用线辅助读数，两策略实虚线/点形区分，.8竖线事前固定，不能事后重选。|

未见文字乱码、标签裁切、遮挡关键数值或未说明双轴；16图覆盖四题的原始/过程/结果三类。PNG按约定尺寸300dpi，SVG应保留矢量；具体程序检查结果以最终独立JSON与流水线严格格式/覆盖日志为准。纸面最终缩放、LaTeX嵌入后的字号与印刷质量留待W1/W2。

源码语义核对：microgrid_core.py 的 Battery 属性固定容量比例10/90/50%；predict_price 仅前30日中位数与已实现18槽偏差；safe_update_trajectory 只取过去日期的当日目标残差、先非负整点再插值；nowcast_trajectory 仅最新已发布官方信息和已实现观测；run_q3_day 始终以不可变0时p0计费，每个最终执行槽一次结算；run_experiments.py 的缓存只缓存因果预测轨迹，不缓存SOC或调度结果。

补充说明：主方法是边际分位数轨迹压缩后的LP，不能写成完整随机MPC或已证实覆盖保证；alpha=.8为事前规则，不是本年度调优最优值；应急倍率敏感性保持策略不变，仅重计费；容量变化为设备尺寸的归一化敏感性，不改主模型初值。
