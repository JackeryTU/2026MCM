# 支撑材料说明

本目录为 2026 年数学建模竞赛（C 题：微网购电优化）论文的支撑材料，
包含程序代码、文献资料、AI 使用报告、结果与图表，以及本说明文档。

## 一、目录结构

| 目录 | 内容 |
|---|---|
| 01_程序代码/ | 论文所用全部 Python 源码，分 py/（求解与结果汇总）与 utils/（绘图）两个子目录 |
| 02_文献资料/ | 参考文献全文（3 篇开放获取 PDF）与 文献清单.md（19 条题录、DOI、下载状态） |
| 03_AI使用报告/ | AI报告/AI工具使用详情.md/.tex/.pdf，AI 工具使用过程与范围说明 |
| 04_结果与图表/ | results/（result1–4.xlsx、全部 csv/json 结果）与 figures/（论文插图 png/svg） |
| 05_说明文档/ | 本文件 |

## 二、程序代码说明

正式求解与绘图代码原位于项目根目录 py/ 与 utils/，现已整体迁入
01_程序代码/。各文件用途：

- py/microgrid_core.py：微网物理模型与数据装载（SOC、供需平衡、分时电价）。
- py/solution_lib.py：四个问题的求解核心（连续 LP、报童分位、滚动 MPC、CVaR）。
- py/问题1_求解.py、py/问题2_求解.py、py/问题3_求解.py、py/问题4_求解.py：
  对应四问的主求解脚本。
- py/问题3_其他时刻增量.py、py/收口计算.py、py/汇总结果.py、
  py/补充敏感性计算.py、py/终端价值敏感性.py：敏感性、汇总与收口计算。
- py/make_figures.py、py/run_lite.py：绘图入口与轻量复现入口。
- utils/figure_style.py 及 utils/fig_*.py：论文插图样式与各图生成脚本。

## 三、复现方式

代码迁移后，脚本中的相对路径（data/、results/、figures/）需指向项目根目录，
建议在项目根目录下运行，或将本目录 04_结果与图表/ 内的数据回链到根目录。
运行时序为：先执行 py/问题1_求解.py … py/问题4_求解.py，再执行
py/汇总结果.py 汇总结果，最后由 py/make_figures.py 与 utils/run_figures.py 生成插图。

论文正文编译：

      cd 参考初稿-正常版-LaTeX
      latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

注意：论文插图由正文 ../figures/*.png 引用，figures/ 仍保留在项目根目录，
本目录 04_结果与图表/figures/ 为备份副本。

## 四、生成或校验

- 参考文献库：参考初稿-正常版-LaTeX/references.bib。
- 结果校验记录：04_结果与图表/results/复现清单.json、
  04_结果与图表/results/csv/结果文件校验.json。
- 本轮仅整理与归档，未重新运行求解代码，论文数值与图表均为既有产出。

