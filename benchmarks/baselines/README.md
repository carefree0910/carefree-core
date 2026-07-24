# 基线报告约定

运行时报告默认写入被 gitignore 的 `.artifacts/benchmarks/`。报告本身已经记录
commit、dirty 状态、Python、OS、CPU、依赖版本、constraints 摘要、seed、参数、
原始样本和离散程度。

只有在需要维护长期固定机器基线时，才把人工确认的报告复制到本目录。提交时应：

1. 文件名包含机器标签、profile 和日期；
2. 在评审说明中同时给出变更前与变更后的报告；
3. 保留原始样本，不删除离群值；
4. 不把共享 GitHub runner 的时间设成硬阈值。

本目录的说明文件不代表已经接受任何具体性能数字。
