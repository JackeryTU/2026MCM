from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "sections" / "appendix_required_tables.tex"


def number(value: str) -> str:
    return f"{float(value):.2f}"


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / "csv" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(items: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        result[item["日期"]].append(item)
    return result


def purchase_table(question: int, date: str, items: list[dict[str, str]]) -> str:
    key = "实际购电_kWh" if question == 2 else "最终合同_q_kWh"
    first, second = items[:3], items[3:]
    line = lambda group: " & ".join(f"{x['时段']} & {number(x[key])}" for x in group) + r" \\" + "\n"
    return (
        r"\begin{table}[H]\centering\scriptsize" "\n"
        f"\\caption{{问题{question}：{date}指定时段购电量}}\n"
        r"\begin{tabular}{cc|cc|cc}\toprule" "\n"
        r"时间段&购电量/kWh&时间段&购电量/kWh&时间段&购电量/kWh\\\midrule" "\n"
        + line(first) + line(second) + r"\bottomrule\end{tabular}\end{table}" "\n"
    )


def storage_table(question: int, date: str, items: list[dict[str, str]]) -> str:
    pairs = [items[index:index + 2] for index in range(0, len(items), 2)]
    line = lambda pair: " & ".join(f"{x['时间段']} & {number(x['充电量_kWh'])} & {number(x['放电量_kWh'])}" for x in pair) + r" \\" + "\n"
    return (
        r"\begin{table}[H]\centering\scriptsize" "\n"
        f"\\caption{{问题{question}：{date}分段充放电量及始末储电量}}\n"
        r"\begin{tabular}{ccc|ccc}\toprule" "\n"
        r"时间段&充电量/kWh&放电量/kWh&时间段&充电量/kWh&放电量/kWh\\\midrule" "\n"
        + "".join(line(pair) for pair in pairs)
        + rf"\midrule 日初储电量&\multicolumn{{2}}{{c|}}{{{number(items[0]['日初储电量_kWh'])}}}&日末储电量&\multicolumn{{2}}{{c}}{{{number(items[-1]['日末储电量_kWh'])}}}\\\bottomrule\end{{tabular}}\end{{table}}" "\n"
    )


def emergency_table(question: int, items: list[dict[str, str]]) -> str:
    body = "\n".join(f"{x['日期']}&{x['购电时间段']}&{number(x['购电量_kWh'])}\\\\" for x in items)
    return (
        r"\begin{table}[H]\centering\scriptsize" "\n"
        f"\\caption{{问题{question}：指定日期紧急购电量}}\n"
        r"\begin{tabular}{lll}\toprule 日期&紧急购电时间段&购电量/kWh\\\midrule" "\n"
        + body + "\n" + r"\bottomrule\end{tabular}\end{table}" "\n"
    )


def section(question: int) -> str:
    purchases = grouped(rows(f"问题{question}_表1_指定时段.csv"))
    storage = grouped(rows(f"问题{question}_表2_四小时聚合.csv"))
    emergency = rows(f"问题{question}_表3_紧急购电.csv")
    out = [f"\\subsection{{问题{question}：表1、表2与表3}}\n"]
    for date in purchases:
        out.append(purchase_table(question, date, purchases[date]))
        out.append(storage_table(question, date, storage[date]))
    out.append(emergency_table(question, emergency))
    return "".join(out)


OUT.write_text(section(2) + "\n" + section(3), encoding="utf-8")
