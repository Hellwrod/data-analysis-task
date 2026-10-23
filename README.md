# data-analysis-task

> 关灯科技 · 实战任务交付仓库。只使用任务页发放的 CSV 数据（放在 `data/`），
> 所有引用数字均可用 `analyze.py` 对发放数据复算。

## 一句话结论

[TODO] 根问题是 ……；我做了 ……；证明了 ……

## 我的判断：看到了什么，根问题是什么

- [TODO] 数据体检：样本量、时间跨度、关键字段（数字见 `output/overview.txt`）
- [TODO] 看到的现象 / 异常：
- [TODO] 根问题判断：

## 做了什么

1. [TODO]
2. [TODO]

## 怎么证明有用

- [TODO] 指标：从 X 变为 Y（口径：……；由 `analyze.py` 输出可直接核对）
- 引用的数字全部来自发放数据 + `analyze.py` 的输出，无外部数据。

## 假设与取舍（不明确处按我的判断执行并记录于此）

- [TODO] 例：字段 X 按 …… 理解；时间口径取 ……

## 如何复现

环境：Python ≥ 3.12（推荐 [uv](https://docs.astral.sh/uv/)；没有 uv 用 pip 等价安装依赖）。

```bash
uv sync              # 等价于：pip install pandas matplotlib tabulate
# 把任务页下载的 CSV 放到 data/ 下（只用这份发放数据）
uv run analyze.py    # 等价于：python analyze.py
```

输出：关键数字打印在终端，并写入 `output/overview.txt`；图表写入 `output/`。

## 仓库结构

```
├── analyze.py      # 唯一入口：数据体检 + 业务分析（[TODO] 段）+ 出图
├── data/           # 发放的 CSV 放这里（只用这份）
├── output/         # 脚本产物：overview.txt、图表
├── README.md       # 本文件
├── AI_LOG.md       # AI 交互记录
├── pyproject.toml  # 依赖清单（uv.lock 锁定版本）
```
