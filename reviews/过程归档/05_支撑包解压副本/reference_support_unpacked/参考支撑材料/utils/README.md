# 绘图辅助工具

这里保存从 `math-modeling-skill` 复制并由求解代码调用的共用绘图工具，不保存最终图片。

- Python 项目使用 Skill 提供的 `plot_style.py` 等样式或导出辅助工具。
- MATLAB 项目使用 `apply_publication_style.m`、`audit_publication_figure.m` 和 `export_publication_figure.m`。
- 每问的绘图主逻辑仍写在根目录对应的 `问题X_求解.py` 或 `.m` 中。
- 生成的 SVG、PNG 和 `图表面板.html` 统一写入根目录 `figures/`；自动生成的灰度质检预览可写入 `figures/_qa/`。
- 从 Skill 更新工具时，应记录来源版本；不要直接修改 Skills 安装目录中的原文件。
