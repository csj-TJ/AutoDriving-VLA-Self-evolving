"""Compile the retained editable XeLaTeX report into the latest PDF."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "项目报告.md"
OUTPUT = ROOT / "report" / "自动驾驶VLA自进化研究报告.tex"
PDF_STEM = "自动驾驶VLA自进化研究报告_最新版"


PREAMBLE = r"""\documentclass[UTF8,a4paper,12pt,fontset=fandol]{ctexart}
\usepackage[left=2.6cm,right=2.4cm,top=2.5cm,bottom=2.5cm]{geometry}
\usepackage{booktabs,longtable,array,ragged2e}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{xurl}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage{caption}
\usepackage{microtype}
\setCJKmonofont{FandolFang-Regular}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black,
  pdftitle={基于自我反思与 Critic 反馈的自动驾驶视觉语言动作模型自进化研究}}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.25em}
\setlength{\headheight}{15pt}
\setlength{\LTpre}{0.5em}
\setlength{\LTpost}{0.8em}
\renewcommand{\arraystretch}{1.18}
\setlist{nosep,leftmargin=2.5em}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{自动驾驶 VLA 自进化研究}
\fancyfoot[C]{--- \thepage\ ---}
\renewcommand{\headrulewidth}{0.4pt}
\newcolumntype{P}[1]{>{\RaggedRight\arraybackslash}p{#1}}

\begin{document}
\begin{titlepage}
  \centering
  \vspace*{2.2cm}
  {\zihao{1}\bfseries 基于自我反思与 Critic 反馈的\\[0.35em]
  自动驾驶视觉语言动作模型自进化研究\par}
  \vspace{2.0cm}
  {\zihao{3}课程实践项目报告（选题 4）\par}
  \vspace{0.8cm}
  {\zihao{4}基座方向：OpenDriveVLA / Vision-Language-Action\par}
  {\zihao{4}实验版本：OpenDriveVLA 冻结基座轻量实验 Demo v0.3\par}
  \vfill
  {\zihao{4}2026 年 8 月\par}
\end{titlepage}
\pagenumbering{Roman}
\tableofcontents
\clearpage
\pagenumbering{arabic}
\onehalfspacing
"""


POSTAMBLE = "\\end{document}\n"


def escape_text(value: str) -> str:
    """Escape prose while preserving inline code spans and HTTP(S) URLs."""
    tokens: list[str] = []

    def protect(rendered: str) -> str:
        marker = f"@@LATEXTOKEN{len(tokens)}@@"
        tokens.append(rendered)
        return marker

    def render_code(match: re.Match[str]) -> str:
        code = match.group(1)
        if any(ord(char) > 127 for char in code) or " " in code:
            code_replacements = {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                " ": r"\ ",
                "/": r"/\allowbreak{}",
            }
            safe = "".join(code_replacements.get(char, char) for char in code)
            return protect(r"\texttt{" + safe + "}")
        return protect(r"\path{" + code.replace("}", r"\}") + "}")

    value = re.sub(r"`([^`]+)`", render_code, value)
    value = re.sub(
        r"https?://[^\s，。；]+",
        lambda match: protect(r"\url{" + match.group(0) + "}"),
        value,
    )
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "/": r"/\allowbreak{}",
        "κ": r"\ensuremath{\kappa}",
        "α": r"\ensuremath{\alpha}",
        "↑": r"\ensuremath{\uparrow}",
        "↓": r"\ensuremath{\downarrow}",
    }
    value = "".join(replacements.get(char, char) for char in value)
    for index, rendered in enumerate(tokens):
        marker = escape_text_marker(f"@@LATEXTOKEN{index}@@")
        value = value.replace(marker, rendered)
    return value


def escape_text_marker(marker: str) -> str:
    return marker.replace("@", "@")


def table_spec(columns: int) -> str:
    if columns == 8:
        return "@{}P{0.14\\textwidth}P{0.09\\textwidth}P{0.09\\textwidth}P{0.09\\textwidth}P{0.09\\textwidth}P{0.08\\textwidth}P{0.10\\textwidth}P{0.09\\textwidth}@{}"
    if columns == 5:
        return "@{}P{0.14\\textwidth}P{0.12\\textwidth}P{0.15\\textwidth}P{0.14\\textwidth}P{0.31\\textwidth}@{}"
    if columns == 4:
        return "@{}P{0.17\\textwidth}P{0.25\\textwidth}P{0.20\\textwidth}P{0.26\\textwidth}@{}"
    if columns == 3:
        return "@{}P{0.22\\textwidth}P{0.49\\textwidth}P{0.17\\textwidth}@{}"
    width = 0.88 / columns
    return "@{}" + f"P{{{width:.3f}\\textwidth}}" * columns + "@{}"


def parse_table(lines: list[str], start: int) -> tuple[list[str], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
        index += 1
    if len(rows) < 2:
        return [escape_text(lines[start]) + "\n"], start + 1
    header = rows[0]
    body = rows[2:]
    columns = len(header)
    rendered = ["\\begin{center}\n", "\\small\n", f"\\begin{{longtable}}{{{table_spec(columns)}}}\n"]
    rendered.append("\\toprule\n")
    rendered.append(" & ".join(r"\textbf{" + escape_text(cell) + "}" for cell in header) + r" \\" + "\n")
    rendered.append("\\midrule\n\\endfirsthead\n")
    rendered.append("\\toprule\n")
    rendered.append(" & ".join(r"\textbf{" + escape_text(cell) + "}" for cell in header) + r" \\" + "\n")
    rendered.append("\\midrule\n\\endhead\n")
    rendered.append("\\midrule\n\\multicolumn{" + str(columns) + r"}{r}{续下页} \\" + "\n\\endfoot\n")
    rendered.append("\\bottomrule\n\\endlastfoot\n")
    for row in body:
        row = row + [""] * (columns - len(row))
        rendered.append(" & ".join(escape_text(cell) for cell in row[:columns]) + r" \\" + "\n")
    rendered.extend(["\\end{longtable}\n", "\\end{center}\n"])
    return rendered, index


def convert(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = [PREAMBLE]
    index = 0
    paragraph: list[str] = []
    list_kind: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append(escape_text(" ".join(part.strip() for part in paragraph)) + "\n\n")
            paragraph = []

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            output.append(f"\\end{{{list_kind}}}\n\n")
            list_kind = None

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if index == 0 and stripped.startswith("# "):
            index += 1
            continue
        if index < 7 and (not stripped or "课程实践项目报告" in stripped or "基座方向" in stripped or "实验版本" in stripped or stripped.startswith("日期：")):
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            close_list()
            index += 1
            continue
        heading = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            raw_title = heading.group(2)
            if raw_title == "摘要":
                output.append("\\section*{摘要}\n\\addcontentsline{toc}{section}{摘要}\n")
                index += 1
                continue
            if raw_title == "参考文献与代码":
                output.append("\\section*{参考文献与代码}\n\\addcontentsline{toc}{section}{参考文献与代码}\n")
                index += 1
                continue
            if raw_title.startswith("附录 A："):
                output.append("\\appendix\n\\section{" + escape_text(raw_title.split("：", 1)[1]) + "}\n")
                index += 1
                continue
            raw_title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", raw_title)
            title = escape_text(raw_title)
            command = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
            output.append(f"\\{command}{{{title}}}\n")
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|\-]+\|$", lines[index + 1].strip()):
            flush_paragraph()
            close_list()
            rendered, index = parse_table(lines, index)
            output.extend(rendered)
            continue
        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if unordered or ordered:
            flush_paragraph()
            wanted = "itemize" if unordered else "enumerate"
            if list_kind != wanted:
                close_list()
                output.append(f"\\begin{{{wanted}}}\n")
                list_kind = wanted
            output.append("  \\item " + escape_text((unordered or ordered).group(1)) + "\n")
            index += 1
            continue
        if re.match(r"^\[\d+\]\s+", stripped):
            flush_paragraph()
            close_list()
            output.append(escape_text(stripped) + "\\par\n")
            index += 1
            continue
        paragraph.append(stripped.removesuffix("  "))
        index += 1

    flush_paragraph()
    close_list()
    output.append(POSTAMBLE)
    return "".join(output)


def compile_pdf() -> None:
    engine = shutil.which("xelatex")
    if not engine:
        raise SystemExit("xelatex was not found. Install MiKTeX or TeX Live first.")
    command = [engine, "-interaction=nonstopmode", "-halt-on-error", f"-jobname={PDF_STEM}", OUTPUT.name]
    for _ in range(2):
        completed = subprocess.run(
            command,
            cwd=OUTPUT.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            tail = completed.stdout.decode(errors="replace")[-4000:]
            raise SystemExit(f"XeLaTeX failed with code {completed.returncode}:\n{tail}")
    log_path = OUTPUT.parent / f"{PDF_STEM}.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    missing = log_text.count("Missing character:")
    overfull = log_text.count("Overfull \\hbox")
    if missing:
        raise SystemExit(f"XeLaTeX produced {missing} missing-glyph warnings; PDF was not accepted.")
    print(f"XeLaTeX QA: missing glyphs={missing}, overfull boxes={overfull}")
    for suffix in (".aux", ".log", ".out", ".toc"):
        artifact = OUTPUT.parent / f"{PDF_STEM}{suffix}"
        if artifact.exists():
            artifact.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the retained Chinese XeLaTeX project report.")
    parser.add_argument("--compile", action="store_true", help="Run XeLaTeX twice and create the PDF.")
    args = parser.parse_args()
    if not OUTPUT.is_file():
        raise SystemExit(f"Missing LaTeX source: {OUTPUT}")
    print(f"Using {OUTPUT.relative_to(ROOT)}")
    if args.compile:
        compile_pdf()
        print(f"Wrote {(OUTPUT.parent / f'{PDF_STEM}.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
