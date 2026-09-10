2026 C题参考支撑材料

本目录与“参考初稿.pdf”配套，仅供学习、核对和人工改写，不代表已经提交。

目录：
1. code：模型、年度计算、实验、绘图、审计和测试源文件。
2. utils：项目使用的辅助模块。
3. data：题目与原始附件的只读副本。
4. results：五份结果工作簿、核心汇总、代表日表及实验总表。
5. figures：彩色 PNG 和可编辑 SVG 图。
6. AI工具使用详情.pdf：AI辅助过程及待人工核验事项。

推荐环境：
Python 3.14；NumPy 2.4.1；pandas 3.0.0；SciPy 1.17.0；
scikit-learn 1.8.0；openpyxl 3.1.5；Matplotlib 3.10.8；Pillow 12.2.0。

从本目录运行数值主链：
python -X utf8 code/run_all.py --project-root . --alpha 0.8
python -X utf8 code/run_experiments.py --project-root .

项目的 reproduce.py 还会调用数学建模技能提供的绘图与格式检查脚本。
使用该统一入口前，请将环境变量 MATH_MODELING_SKILL_ROOT 指向本机相应技能目录。
没有该辅助工具时，核心数值求解仍可由上述两个命令运行，但绘图和格式检查入口需要相应脚本。

时间口径：输入时刻按10分钟区间终点解释；正式统计为2025-02-01至12-31共334日，
计算从1月1日开始并连续传递SOC。

重要边界：2小时方案使用最新已发布官方预报与已观测前缀构造合成更新，
不是额外官方预报；替代结算仅对同一策略重计费，不是重新优化。
