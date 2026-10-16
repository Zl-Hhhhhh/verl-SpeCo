# 第 1 项实验归档

本目录归档 2026-10-15 完成的 actor0→5 配对诊断：8,192 个候选、2,048 个选中位置，drafter 优化器更新为 0。
结论及指标定义见 [RESULTS.md](RESULTS.md)，环境准备、启动问题和处理记录见 [RUN.md](RUN.md)。文中阶段状态均指本项完成时；本次归档不包含后续 A/B/C/D 训练结果。

## 仓库内的产物

- `candidate_scores.jsonl` / `.parquet`：所有候选的逐位置标量，包含 S、U、m、两种 TV、actor 熵、绝对位置及原 trace 接受情况。
- `selected_target_metrics.jsonl` / `.parquet`：选中位置在 η=0.5、1 下的两类目标指标，以及同 prefix 的特征漂移指标。
- `target_diagnostics.json`、`result_audit.json`、`paired_panel_manifest.json`：汇总统计、分母、目标数学检查及数据隔离记录。
- 三张图、内存采样、执行日志、环境检查、解析后的配置及测试日志：结果审阅和标量审计依据。
- `artifact_manifest.json`：本机原始目录的文件大小及 SHA256，包括未提交的大文件；它不是本次 Git 提交的文件清单。
- `reproducibility/manifest.json`：运行起点、源文件哈希、硬件和内存保护设置。实际提交的复现代码位于仓库的 `tools/`、`verl_speco/` 和 `outputs/experiment_setup/directional_repair_diagnostic_20261015/`。

## 留在本机的原始分布

64 个 `paired_rank*.pt` 共约 5.9 GiB，保留在本目录的本机副本，未加入 Git；其文件名和校验值记录在 `artifact_manifest.json`。
这些文件保存选中位置完整词表的 p−、p+、q(F+)、重复 q 和 q(F−)。从完整概率重新检查非负性、归一化和 TV 改善恒等式，需要这些原始文件。
仓库内的 JSONL 保留标量的完整浮点精度；Parquet 禁用字典编码，`selected` 存为整数 0/1，规避本机 Arrow 回读问题。

## 离线复核

在提供 NumPy 和 Matplotlib 的环境中，从仓库根目录运行：

```bash
python3 tools/report_directional_diagnostic.py \
  outputs/directional_repair_diagnostic_20261015/halfnode_r1 \
  --report /tmp/directional_repair_results.md
```

该命令只读取已保存的标量和运行记录，核对覆盖范围并重生成审计、图表和报告，不启动模型。
原始概率的重分析入口为 `outputs/experiment_setup/directional_repair_diagnostic_20261015/analyze.py`。
