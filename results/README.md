# 当前权威结果目录

本目录是项目当前唯一的权威数值结果目录。论文中的参数、表格和数值结论应以这里的实际运行结果为准。

- 当前结果来自 P2 通过后的编程复现流程。
- 当前活动记录使用 `results/` 作为输出目录，不再使用 `results_versioned/`。
- 复现命令为：`python -X utf8 reproduce_programmer_v2.py --project-root . --output-dir results --figures-dir figures`
- 本目录中的 JSON 记录采用文件路径、字节数和修改时间等复现信息；按项目约定不计算、不比较哈希。
- 程序生成的数据不应手工改写；如模型或参数发生变化，应重新运行程序并同步更新审查记录。

`reviews/过程归档/04_独立复现/programmer_v2_p2/` 中的 `results_versioned/` 属于历史隔离复现证据，不是当前活动结果目录。
