# LaTeX 排版规则

> 排版是论文的"门面"。编译报错、图表乱跑、公式错位都会扣印象分。
> 本文件给出全套 LaTeX 排版规则，配合 `数学建模论文通用LaTeX模板.tex` 使用。

---

## 一、页面与字体

### 页面设置
```latex
\usepackage{geometry}
\geometry{left=2.5cm, right=2.5cm, top=2.5cm, bottom=2.5cm}
```
> 国赛要求 A4 纸，页边距 2.5 cm。

### 字体设置
```latex
\documentclass[12pt,a4paper,fontset=fandol]{ctexart}
\usepackage{newtxtext}  % 英文 Times 风格
\usepackage{newtxmath}  % 数学公式 Times 风格
```
- **正文**：12pt 宋体
- **英文**：Times New Roman
- **数学**：Times 风格
- **fontset=fandol**：CTeX 自带，无需额外字体文件

### 行距与缩进
```latex
\renewcommand{\baselinestretch}{1.5}  % 1.5 倍行距
\setlength{\parindent}{2em}            % 首行缩进 2 字符
\setlength{\parskip}{0.5em}            % 段间距
```

---

## 二、章节标题

### 标题格式
```latex
\ctexset{
    section/number=\chinese{section}、,           % 一、二、三、
    section/format=\large\bfseries\centering,      % 居中加粗
    subsection/number=\arabic{subsection},         % 1 2 3
    subsubsection/number=\arabic{subsection}.\arabic{subsubsection}
}
```

### 标题层级
| 层级 | 命令 | 示例 |
|------|------|------|
| 一级 | `\section{问题重述}` | 一、问题重述 |
| 二级 | `\subsection{问题一分析}` | 5.1 问题一分析 |
| 三级 | `\subsubsection{目标函数}` | 5.1.1 目标函数 |

> **铁律**：一级标题用中文数字（国赛要求），不超过 3 级标题。

---

## 三、图表规则

### 图题在下、表题在上
```latex
\captionsetup[table]{position=above}   % 表题在上
\captionsetup[figure]{position=below}  % 图题在下
```

### 图片插入
```latex
\begin{figure}[H]  % H 强制当前位置
    \centering
    \includegraphics[width=0.9\textwidth]{figures/fig1_xxx.png}
    \caption{图片标题}
    \label{fig:xxx}
\end{figure}
```

### 三线表
```latex
\begin{table}[H]
    \centering
    \renewcommand{\arraystretch}{1.4}  % 行高
    \caption{表格标题}
    \label{tab:xxx}
    \begin{tabular}{ccc}
        \toprule
        列1 & 列2 & 列3 \\
        \midrule
        数据 & 数据 & 数据 \\
        \bottomrule
    \end{tabular}
\end{table}
```

### 三线表铁律
1. 用 `booktabs` 包：`\toprule` `\midrule` `\bottomrule`
2. **无竖线**（不要 `|`）
3. **无内部横线**（除表头外不画 `\hline`）
4. `\renewcommand{\arraystretch}{1.4}` 行高 1.4 倍

### 跨页长表格
```latex
\begin{longtable}{cccc}
    \caption{完整数据表} \label{tab:long} \\
    \toprule
    列1 & 列2 & 列3 & 列4 \\
    \midrule
    \endfirsthead
    \multicolumn{4}{l}{续表 \thetable} \\
    \toprule
    列1 & 列2 & 列3 & 列4 \\
    \midrule
    \endhead
    \midrule
    \multicolumn{4}{r}{接下页} \\
    \endfoot
    \bottomrule
    \endlastfoot
    数据 & 数据 & 数据 & 数据 \\
\end{longtable}
```

---

## 四、公式规则

### 单行公式
```latex
\begin{equation}
    F = \sum_{i=1}^{n} c_i x_i
    \label{eq:obj}
\end{equation}
```

### 多行公式
```latex
\begin{align}
    \min F &= \sum_{i,t} H_i z_{i,t} + S_i y_{i,t} \label{eq:obj1} \\
    \text{s.t.} \quad z_{i,t} &= z_{i,t-1} + x_{i,t} - D_{i,t} \label{eq:inv}
\end{align}
```

### 公式引用
```latex
如式(\ref{eq:obj})所示...
% 或用 cleveref
如\cref{eq:obj}所示...  % 自动生成"式(1)"
```

### 公式铁律
1. **每个公式必须编号**（除极简单推导）
2. **公式中变量必须解释**：公式下方用"其中 $x$ 表示..."
3. **不要用 `$$...$$`**（已过时），用 `equation` 或 `\[ \]`

---

## 五、代码块

### Python 代码
```latex
\begin{lstlisting}[caption={核心求解代码}, label={code:solver}]
import numpy as np
import pulp

model = pulp.LpProblem("MILP", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", (items, weeks), lowBound=0, cat='Integer')
model += pulp.lpSum(H[i]*z[i][t] for i in items for t in weeks)
model.solve(pulp.LpULP_CBC_CMD(msg=0, timeLimit=120))
\end{lstlisting}
```

### 代码块设置（lstset）
```latex
\lstset{
    language=Python,
    basicstyle=\small\ttfamily,
    keywordstyle=\color{blue},
    commentstyle=\color{green!60!black},
    stringstyle=\color{red!70!black},
    numbers=left,
    numberstyle=\tiny\color{gray},
    frame=single,
    breaklines=true,        % 自动换行
    tabsize=4,
    literate={_}{{\_}}1,    % 下划线修复
    captionpos=b            % 标题在下方
}
```

### 代码铁律
1. **代码块必标题 + 标签**
2. **核心代码入正文，完整代码入附录**
3. **下划线报错**：加 `literate={_}{{\_}}1`

---

## 六、列表与枚举

### 无序列表
```latex
\begin{itemize}
    \item 第一项；
    \item 第二项。
\end{itemize}
```

### 有序列表
```latex
\begin{enumerate}
    \item 第一步；
    \item 第二步。
\end{enumerate}
```

### 紧凑列表
```latex
\setlist{nosep, leftmargin=2em}  % 全局紧凑
% 或单次
\begin{itemize}[nosep, leftmargin=2em]
    \item ...
\end{itemize}
```

---

## 七、引用与文献

### GB/T 7714 格式
```latex
\begin{thebibliography}{99}
    \bibitem{1} 姜启源, 谢金星, 叶俊. 数学模型(第五版)[M]. 北京: 高等教育出版社, 2018.
    \bibitem{2} Holland J H. Adaptation in Natural and Artificial Systems[M]. MIT Press, 1992.
    \bibitem{3} Box G E P, Jenkins G M. Time Series Analysis: Forecasting and Control[M]. Holden-Day, 1976.
    \bibitem{4} 作者. 论文名[J]. 期刊名, 年份, 卷号(期号): 页码.
\end{thebibliography}
```

### 引用命令
```latex
本文采用 MILP\cite{1}。
% 多引用
\cite{1,2,3}
```

### 文献铁律
1. **数量**：5-15 篇
2. **类型**：教材 + 经典文献 + 近 5 年论文
3. **格式**：GB/T 7714，[M]专著 [J]期刊 [Z]其他

---

## 八、AI 使用声明（2026 国赛新规）

### 位置
**正文之后、参考文献之前**。

### 模板
```latex
\section*{AI 工具使用声明}
\addcontentsline{toc}{section}{AI 工具使用声明}

本队伍在本次竞赛中\textbf{【使用了】} AI 工具。

\begin{itemize}
    \item \textbf{工具名称与版本}：ChatGPT-4 / Claude-3.5
    \item \textbf{使用场景}：语言润色 / 代码调试 / 模型思路梳理
    \item \textbf{典型交互实例}：（附 1~3 次核心交互记录）
    \item \textbf{人工修改与核验}：所有 AI 输出均经人工核实修正，未直接照搬。
\end{itemize}
```

> 详细规范见 `../AI使用规范.txt`。

---

## 九、常见编译错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `Undefined control sequence` | 命令未定义 | 检查宏包是否加载 |
| `Missing $ inserted` | 公式符号在正文 | 用 `$...$` 包裹 |
| `! LaTeX Error: File not found` | 图片路径错 | 检查 `figures/xxx.png` 是否存在 |
| `Overfull \hbox` | 表格/公式超宽 | 用 `\small` 或 `\resizebox` |
| 下划线报错 | 代码中 `_` | 加 `literate={_}{{\_}}1` |
| 中文乱码 | 编码非 UTF-8 | 文件存为 UTF-8，编译用 xelatex |
| 图表跑丢 | 浮动体漂移 | 用 `[H]` 强制当前位置 |

---

## 十、编译命令

```bash
# 推荐：xelatex（支持中文）
xelatex main.tex
xelatex main.tex  # 跑两次解决交叉引用

# 或用 latexmk 自动多次编译
latexmk -xelatex main.tex

# 清理临时文件
latexmk -c main.tex
```

---

## 十一、排版检查清单

- [ ] 一级标题中文数字（一、二、三）？
- [ ] 图题在下、表题在上？
- [ ] 三线表无竖线？
- [ ] 公式全部编号？
- [ ] 代码块有标题与标签？
- [ ] 参考文献 GB/T 7714？
- [ ] AI 声明在参考文献前？
- [ ] 页边距 2.5 cm？
- [ ] 1.5 倍行距？
- [ ] 首行缩进 2 字符？
- [ ] 交叉引用无 `??`？

---

## 十二、一句话总结

> **xelatex + fandol + booktabs + 三线表无竖线 + 图下表上 + 公式编号 + AI 声明位置正确。**
