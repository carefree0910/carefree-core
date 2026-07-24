# 性能基线

这里的 runner 用于记录可复现的性能基线，不修改被测实现，也不把 GitHub
共享 runner 上的墙钟波动当作硬性通过条件。

## 运行

安装项目的 benchmark 依赖后，在仓库根目录执行：

```bash
python -m benchmarks.run \
  --profile smoke \
  --output .artifacts/benchmarks/smoke.json \
  --validate
```

完整的固定参数基线使用：

```bash
python -m benchmarks.run \
  --profile baseline \
  --output .artifacts/benchmarks/baseline.json \
  --validate
```

`smoke` 每项记录 7 个样本，`baseline` 每项记录 15 个样本。报告目录已被
gitignore；需要评审性能变更时，应同时提供变更前、变更后的 JSON，而不是只提供
单个百分比。

## 测量契约

- runner 先用一个独立子进程校验行为摘要，再为每个计时样本启动全新 Python
  子进程；校验进程不进入统计。
- 冷导入使用只预载少量标准库的专用 worker；报告中的
  `worker_module_count_before_import` 明确记录计时起点已有的模块数。
- 所有样本使用同一 seed，并固定 Python hash、OMP、MKL、OpenBLAS 和 NumExpr
  线程设置。CPU suite 明确隐藏 CUDA。
- 计时使用 `time.perf_counter_ns()`。微基准在单个样本内使用固定迭代次数，
  报告同时保存总时间和每次迭代时间。
- 冷导入的新增模块数属于样本观测值，不属于行为黄金值；共享的 Matplotlib
  缓存可能改变首次校验与后续测量的模块加载细节。
- 不删除离群值。统计量是中位数、原始 MAD，以及采用 R-7 线性插值的 Q1、Q3
  和 IQR。
- Linux 峰值 RSS 来自 `/proc/self/status` 的 `VmHWM`，范围是“整个 worker
  启动到测量结束”，包含解释器、导入与 setup。它不是算子独占内存。
- PyTorch 没有稳定的公开 CPU allocator 峰值接口，因此 Torch CPU 峰值透明地
 复用进程峰值 RSS，并在报告中标为 `process_peak_rss`；不能将其解释为 tensor
  allocator 的独占内存。CPU 场景不需要设备同步，CUDA 字段保持 `null`。
- 子进程有硬超时；超时或异常时会终止整个进程组、保留日志，并写出
  `*.partial.json`，正式报告只会在全部场景成功后原子替换。
- `smoke` 另有 12 分钟总预算，低于 CI job 的 20 分钟外层超时；收到 SIGINT
  或 SIGTERM 时，runner 会先回收当前子进程组。

报告格式由 [report-v1.schema.json](report-v1.schema.json) 定义，
`--validate` 同时执行 JSON Schema 与跨字段校验，并从原始样本重新计算统计量。

## 场景

场景注册表位于 [registry.py](registry.py)，覆盖：

- `core`、`core.learn` 冷导入；
- 异步 DataLoader 完整耗尽；
- 推理结果累积与变长填充；
- `Trainer.train_step`；
- MoE dispatch/combine；
- EMA 更新与 eval/train swap；
- 内存中的 toolkit Pipeline build；
- `filter_kw`；
- 模型 `summary`。

每项都有固定的 `smoke` 小参数和 `baseline` 中参数。按照当前任务边界，这里不包含
`core/flow`、网络请求、Pipeline 序列化或 ensemble/checkpoint 融合。当前 v1 是
CPU suite；未来增加 GPU/大模型场景时必须使用独立 `gpu` tag、记录设备与 CUDA
内存同步策略，并且不能进入默认 CPU smoke。

## CI

普通 pytest 不执行完整性能套件。GitHub Actions 仅在独立 job 中运行一次 CPU
`smoke` 并上传 JSON 与失败日志；它校验协议和行为，不设置性能阈值。完整
`baseline` 由开发者在固定机器上手动或定时生成。
