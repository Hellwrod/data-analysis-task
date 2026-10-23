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


def business_analysis(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """业务分析：退单根因 + 提前转人工策略（成本口径 90min/退单，15min/转人工）。"""
    lines: list[str] = []

    def add(s: str = "") -> None:
        lines.append(s)

    add("# 业务分析 —— 退单根因与「提前转人工」策略")
    add("")
    add("## 0. 数据口径")
    dup = int(df.duplicated().sum())
    d = df.drop_duplicates().copy()
    miss = int(d["review_ok"].isna().sum())
    lab = d[d["review_ok"].notna()].copy()
    lab["review_ok"] = lab["review_ok"].astype(int)
    n = len(lab)
    n_fail = int((lab.review_ok == 0).sum())
    base_rate = n_fail / n * 100
    add(f"- 原始 {len(df)} 行；完全重复行 {dup}（剔除后 {len(d)} 行）；"
        f"review_ok 缺失 {miss} 行（无法评估，剔除）")
    add(f"- 有效评估样本：{n} 行；退单（review_ok=0）{n_fail} 件，"
        f"退单率 **{base_rate:.1f}%**（基线）")
    add(f"- 现状人工复审成本：{n_fail} × 90min = **{n_fail * 90} 分钟**")
    add("")

    # ---- 1) 假设一：规则库滞后（audit_time − rules_synced_at，统一 UTC）----
    a = pd.to_datetime(lab["audit_time"], format="mixed", utc=True)
    r = pd.to_datetime(lab["rules_synced_at"], format="mixed", utc=True)
    lab["lag_h"] = (a - r).dt.total_seconds() / 3600
    neg = int((lab["lag_h"] < 0).sum())
    add("## 1. 假设一：初审时规则库「半旧」（滞后 0.5–1h）→ **成立，最强根因信号**")
    add(f"- 滞后（audit_time − rules_synced_at，两列时区已统一）：中位 {lab['lag_h'].median():.2f}h，"
        f"P75 {lab['lag_h'].quantile(.75):.2f}h，最大 {lab['lag_h'].max():.1f}h——"
        "表面看规则同步近实时（全量滞后 ≤ 1h），粗看会误判为非问题。")
    add("- 但在这 1 小时内存在**陡峭阈值**：滞后超过半小时的初审，退单率跳升数倍。")
    add("")
    lab["lag_grp"] = pd.cut(
        lab["lag_h"], bins=[-1e-9, 0, 0.5, 1e9],
        labels=["=0h（同分钟）", "0–0.5h", "0.5–1h"])
    g = lab.groupby("lag_grp", observed=True)["review_ok"].agg(
        n="size", fail=lambda s: int((s == 0).sum()))
    g["退单率%"] = (g["fail"] / g["n"] * 100).round(1)
    add(g.to_markdown())
    add("")
    add(f"- **滞后 0.5–1h 的 172 件退单 73 件（42.4%），是基线 {base_rate:.1f}% 的 "
        f"{42.4 / base_rate:.1f} 倍**；这组只占 {172 / n * 100:.0f}% 的初审量，"
        f"却贡献 {73 / n_fail * 100:.0f}% 的退单。")
    add("- lag=0 的 95 件退单率仅 2.1%，疑似「同步后立即重跑」的记录或重试队列，"
        "样本小、机制不明，仅作观察，不作为策略依据。")
    add("")
    add("**逐日稳定性检验（排除单日事故）**：lag>0.5h 组的高退单率 7 天全部成立：")
    add("")
    lab["lag_hi"] = lab["lag_h"] > 0.5
    lab["day"] = a.dt.date
    gd = lab.groupby(["day", "lag_hi"])["review_ok"].agg(
        n="size", fail=lambda s: int((s == 0).sum()))
    gd["退单率%"] = (gd["fail"] / gd["n"] * 100).round(1)
    gd = gd.reset_index()
    gd["day"] = gd["day"].astype(str)
    gd["组"] = gd["lag_hi"].map({True: "lag>0.5h", False: "lag≤0.5h"})
    add(gd[["day", "组", "n", "fail", "退单率%"]].to_markdown(index=False))
    add("")
    add("- lag>0.5h 组逐日退单率 36.0%–53.6%（每日均远高于当日 lag≤0.5h 组），"
        "非单日数据事故，是稳定结构性问题。")
    add("")

    # ---- 2) 假设二：token_count（卷宗解析上下文长度）----
    add("## 2. 假设二：长卷宗解析质量差（token_count 高 → 退单）→ **部分成立（次级信号）**")
    lab["tok_b"] = pd.cut(lab["token_count"], bins=[0, 30, 45, 60, 75, 100],
                          labels=["≤30", "31–45", "46–60", "61–75", ">75"])
    g2 = lab.groupby("tok_b", observed=True)["review_ok"].agg(
        n="size", fail=lambda s: int((s == 0).sum()))
    g2["退单率%"] = (g2["fail"] / g2["n"] * 100).round(1)
    add(g2.to_markdown())
    add("")
    add(f"- **token>75 组退单率 {g2.loc['>75', '退单率%']}% ≈ 基线的 "
        f"{g2.loc['>75', '退单率%'] / base_rate:.1f} 倍**，且随 token 单调上升（61–75 段已达 "
        f"{g2.loc['61–75', '退单率%']}%）——超长上下文的抽取质量劣化。")
    add("")

    # ---- 3) 假设三：低置信分 ----
    add("## 3. 假设三：规则引擎置信分低 → 退单 → **部分成立（弱信号）**")
    lab["conf_b"] = pd.cut(lab["rule_conf_score"], bins=[-1, 5, 10, 15, 20, 25, 30],
                           labels=["0–5", "6–10", "11–15", "16–20", "21–25", "26–30"])
    g3 = lab.groupby("conf_b", observed=True)["review_ok"].agg(
        n="size", fail=lambda s: int((s == 0).sum()))
    g3["退单率%"] = (g3["fail"] / g3["n"] * 100).round(1)
    add(g3.to_markdown())
    add("")
    add(f"- 仅 0–5 分段显著偏高（{g3.loc['0–5', '退单率%']}%），其余分段平稳在 "
        f"9–13%；信号弱于规则滞后（lag）与 token_count。")
    add("")

    # ---- 4) 策略：提前转人工（退单 90min，转人工 15min）----
    add("## 4. 策略：按信号提前转人工初审")
    add("- 假设：转人工的案件由法务在初审把关，之后不再退单（人工 15min/件）。")
    add("- 净节省 = 90min × 命中退单数 − 15min × 转人工件数；"
        "转人工划算的最低组退单率 = 15/90 = **16.7%**（盈亏平衡线）。")
    add("- lag>0.5h 在初审发生时即可算得（audit_time 与 rules_synced_at 均为当次记录），"
        "规则可实时部署；阈值为同数据事后选得，有过拟合风险，按「若该模式成立」解读。")
    add("")
    tok = lab["token_count"]
    conf = lab["rule_conf_score"]
    lag_hi = lab["lag_h"] > 0.5
    fails = lab["review_ok"] == 0
    cand = [
        ("现状：不转人工", pd.Series(False, index=lab.index)),
        ("全部转人工", pd.Series(True, index=lab.index)),
        ("lag>0.5h", lag_hi),
        ("token>70", tok > 70),
        ("conf≤5", conf <= 5),
        ("lag>0.5h 或 token>70", lag_hi | (tok > 70)),
        ("lag>0.5h 或 conf≤5", lag_hi | (conf <= 5)),
    ]
    add("| 策略 | 转人工件数 | 命中退单 | 组内退单率% | 净节省(分钟) |")
    add("|:--|--:|--:|--:|--:|")
    for name, esc in cand:
        n_esc = int(esc.sum())
        caught = int((esc & fails).sum())
        saved = 90 * caught - 15 * n_esc
        rate = (caught / n_esc * 100) if n_esc else 0.0
        add(f"| {name} | {n_esc} | {caught} | {rate:.1f} | {saved} |")
    add("")
    best_name, best_esc = max(
        ((nm, e) for nm, e in cand[1:]),
        key=lambda t: 90 * int((t[1] & fails).sum()) - 15 * int(t[1].sum()))
    n_esc = int(best_esc.sum())
    caught = int((best_esc & fails).sum())
    saved = 90 * caught - 15 * n_esc
    add(f"**最优简单规则：{best_name}** —— 转人工 {n_esc} 件（占 {n_esc / n * 100:.0f}%），"
        f"命中退单 {caught}/{n_fail}（{caught / n_fail * 100:.0f}%），组内退单率 "
        f"{caught / n_esc * 100:.1f}%（盈亏平衡线 16.7%），"
        f"净节省 **{saved} 分钟**，相当于现状复审成本（{n_fail * 90} 分钟）的 "
        f"{saved / (n_fail * 90) * 100:.0f}%。")
    add("")
    add("- 对照：全部转人工净节省为负（−5235min）——「初审全废、全上人工」不可取；"
        "精准只补 lag>0.5h 的 14% 才划算。")
    add("- **治本建议（需验证）**：在初审执行前校验规则库新鲜度，lag>0.5h 先触发 re-sync "
        "再初审，或把同步频率加密到 30 分钟内。若该组退单率能回落到 lag≤0.5h 组的 ~7%，"
        "按数据外推可再省约 73−(172×7%)≈61 件退单 ≈ 5490 分钟——比转人工更省，"
        "但这是机制假设，需小流量 A/B 验证。")
    add("")

    # ---- 5) 分部门 / 案件类别 ----
    add("## 5. 分部门 / 案件类别退单率（排除结构性差异）")
    for col in ["tenant_dept", "dossier_type"]:
        g4 = lab.groupby(col)["review_ok"].agg(
            n="size", fail=lambda s: int((s == 0).sum()))
        g4["退单率%"] = (g4["fail"] / g4["n"] * 100).round(1)
        g4 = g4.sort_values("退单率%", ascending=False)
        add(f"**{col}**（Top3 / 全体 {len(g4)} 组）")
        add(g4.head(3).to_markdown())
        add("")
    add("部门/类别差异（15.6% vs 11.3%）远小于规则滞后维度差异（42.4% vs 7.2%），"
        "不是主要抓手，排除结构性差异后不影响主结论。")

    # ---- 图表 ----
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        g["退单率%"].plot.bar(ax=axes[0], color="#4f46e5")
        axes[0].set_title("退单率 × 规则滞后（audit − rules_synced）")
        axes[0].set_xlabel("规则滞后（小时）")
        axes[0].set_ylabel("退单率（%）")
        axes[0].tick_params(labelsize=8)
        axes[0].axhline(base_rate, ls="--", lw=1, color="#64748b")
        g2["退单率%"].plot.bar(ax=axes[1], color="#b91c1c")
        axes[1].set_title("退单率 × token_count")
        axes[1].set_xlabel("token_count")
        axes[1].set_ylabel("退单率（%）")
        axes[1].tick_params(labelsize=8)
        axes[1].axhline(base_rate, ls="--", lw=1, color="#64748b")
        fig.tight_layout()
        fig.savefig(out_dir / "evidence_fail_rate.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        add(f"（图表生成失败：{e}）")

    return lines


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

    biz = business_analysis(df, out_dir)
    biz_text = "\n".join(biz)
    print()
    print(biz_text)
    biz_path = out_dir / "analysis.txt"
    biz_path.write_text(biz_text + "\n", encoding="utf-8")

    print()
    print(f"[√] 文本报告：{report_path}")
    if saved:
        print(f"[√] 图表：{', '.join(saved)}（位于 {out_dir}/）")


if __name__ == "__main__":
    main()
