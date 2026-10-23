#!/usr/bin/env python
"""可复现分析入口脚本（关灯科技 · 实战任务）。

用法（仓库根目录）：
    uv run analyze.py                      # 自动读取 data/ 下唯一的 CSV
    uv run analyze.py --data data/xx.csv   # 显式指定 CSV
    uv run analyze.py --out output         # 指定输出目录（默认 output/）

拿到任务书后：
  1. 先跑一遍本脚本 —— 无论 CSV 长什么样，都会产出一份「数据体检报告」
     （形状 / 缺失 / 分布 / 时间范围），写入 output/overview.txt，图表写入 output/。
  2. 在 main() 底部标有 [TODO] 的位置写真正的业务分析；
     README 中引用的每一个数字，都必须能由本脚本的输出复算出来。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境也能出图

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体回退链（Windows 优先；找不到时 matplotlib 自动回退，不影响出图）
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
MAX_PRINT_LEN = 240  # 单行打印截断长度，防止长文本刷屏


def _cut(s: str, limit: int = MAX_PRINT_LEN) -> str:
    return s if len(s) <= limit else s[: limit - 1] + "…"


def find_default_csv() -> Path | None:
    """data/ 下恰好一个 CSV 时返回它；多个时列出让考生指定。"""
    if not DATA_DIR.is_dir():
        return None
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    if len(csvs) > 1:
        print("[!] data/ 下有多个 CSV，请用 --data 指定要分析的那个：")
        for p in csvs:
            print(f"    {p.name}")
        sys.exit(2)
    return None


def datetime_like(df: pd.DataFrame) -> dict[str, pd.Series]:
    """在非数值列里探测日期时间列（解析成功率 >80% 且不止一个取值）。"""
    found: dict[str, pd.Series] = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col]
        non_null = int(s.notna().sum())
        if non_null == 0 or s.nunique(dropna=True) <= 1:
            continue
        try:
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            continue
        if parsed.notna().sum() / non_null > 0.8:
            found[col] = parsed
    return found


def build_overview(df: pd.DataFrame, src_name: str) -> str:
    """生成数据体检报告（README 可直接引用其中数字）。"""
    lines: list[str] = []

    def add(s: str = "") -> None:
        lines.append(s)

    add(f"# 数据体检报告 —— {src_name}")
    add("")
    add(f"- 形状：**{df.shape[0]} 行 × {df.shape[1]} 列**")
    add(f"- 完全重复行：{int(df.duplicated().sum())}")
    add(f"- 整体缺失单元格占比：{df.isna().mean().mean() * 100:.2f}%")
    add("")
    add("## 列概览")
    tbl = pd.DataFrame(
        {
            "类型": df.dtypes.astype(str),
            "非空": df.notna().sum(),
            "缺失%": (df.isna().mean() * 100).round(2),
            "唯一值": df.nunique(),
        }
    )
    tbl.index.name = "列"
    add(tbl.to_markdown())
    add("")

    num = df.select_dtypes(include="number")
    if not num.empty:
        add("## 数值列分布")
        desc = num.describe().T.round(2)
        desc.index.name = "列"
        add(desc.to_markdown())
        add("")

    cat = df.select_dtypes(include=["object", "string"])
    if not cat.empty:
        add("## 分类列取值（Top5）")
        for c in cat.columns:
            vc = cat[c].value_counts(dropna=True).head(5)
            tops = "；".join(f"{k}（{v} 次）" for k, v in vc.items())
            add(f"- **{c}**（{cat[c].nunique(dropna=True)} 个取值）: {_cut(tops)}")
        add("")

    dts = datetime_like(df)
    if dts:
        add("## 时间范围（自动探测）")
        for c, s in dts.items():
            span = s.max() - s.min()
            add(f"- {c}: {s.min().date()} ~ {s.max().date()}（跨度 {span.days} 天）")
        add("")

    return "\n".join(lines)


def save_charts(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """把通用图表存到 out_dir，返回生成的文件名列表。"""
    saved: list[str] = []

    num = df.select_dtypes(include="number")
    plot_cols = [c for c in num.columns if num[c].nunique(dropna=True) > 1][:12]
    if plot_cols:
        ncols = min(3, len(plot_cols))
        nrows = math.ceil(len(plot_cols) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows))
        flat = np.atleast_1d(axes).ravel()
        for ax, c in zip(flat, plot_cols):
            vals = num[c].dropna()
            ax.hist(vals, bins=min(40, max(10, int(vals.nunique()))))
            ax.set_title(str(c), fontsize=10)
            ax.tick_params(labelsize=8)
        for ax in flat[len(plot_cols):]:
            ax.set_visible(False)
        fig.suptitle("数值列分布", fontsize=12)
        fig.tight_layout()
        p = out_dir / "numeric_hist.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p.name)

    miss = (df.isna().mean() * 100).round(2)
    miss = miss[miss > 0]
    if not miss.empty:
        fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(miss) + 1.6))
        miss.sort_values().plot.barh(ax=ax, color="#4f46e5")
        ax.set_xlabel("缺失比例（%）")
        ax.set_title("各列缺失比例", fontsize=11)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        p = out_dir / "missing_ratio.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p.name)

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="对发放的 CSV 做可复现分析")
    ap.add_argument(
        "--data", type=Path, default=None,
        help="CSV 路径；缺省时自动使用 data/ 下唯一的 CSV",
    )
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录（默认 output/）")
    args = ap.parse_args()

    src = args.data if args.data is not None else find_default_csv()
    if src is None or not Path(src).exists():
        print(f"[x] 未找到 CSV（{args.data or 'data/ 下没有 .csv 文件'}）。")
        print("    请把任务页下载的 CSV 放入 data/，或用 --data 指定路径。")
        sys.exit(1)

    out_dir: Path = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取 {Path(src).name} …")
    try:
        df = pd.read_csv(src, low_memory=False)
    except Exception as e:  # noqa: BLE001 —— 给出可读的错误信息即可
        print(f"[x] CSV 读取失败：{e}")
        sys.exit(1)

    report = build_overview(df, Path(src).name)
    print()
    print(report)
    report_path = out_dir / "overview.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    saved = save_charts(df, out_dir)

    # ======================= [TODO] 业务分析 =======================
    # 看到任务书后，在这里写真正的分析（引用数字全部从 df 计算并打印），例如：
    #     g = df.groupby("渠道")["金额"].agg(["sum", "mean", "count"])
    #     print(g.to_markdown())
    #     fig.savefig(out_dir / "evidence_xxx.png")
    # ===============================================================

    print()
    print(f"[√] 文本报告：{report_path}")
    if saved:
        print(f"[√] 图表：{', '.join(saved)}（位于 {out_dir}/）")


if __name__ == "__main__":
    main()
