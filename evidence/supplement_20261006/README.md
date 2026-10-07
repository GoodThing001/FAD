# 2026-10-06 补实验证据

本轮重新训练和逐预测核对；不将旧结果复算记为新训练。真实GBSA、docking、MD新增均为0。

|运行|指标行|预测文件核验|发现轮次核验|状态|
|---|---:|---:|---:|---|
|local_supplement_active_confirm_20261006|2400|2400|2000|DONE|
|local_supplement_active_dev_20261006|720|720|600|DONE|
|local_supplement_history_confirm_20261006|500|500|0|DONE|
|local_supplement_history_dev_20261006|155|155|0|DONE|
|srv_supplement_defect_dev_20261006|120|120|0|DONE|
|srv_supplement_defect_force_20261006|120|120|0|DONE|
|srv_supplement_paper_aggregate_20261006|120|120|0|DONE|

完整逐种子、配对CI、胜平负、分歧校准及文件SHA256见 [index.json](index.json)。
冻结配方与种子见 [协议](../../docs/项目记录/补实验冻结协议_20261006.md)。
用户随后明确授权服务器上传与运行，首个NUPACK官方核验失败包保留；修订原因及范围见冻结协议§2.1。
8项默认环境测试通过；另2项PyTorch CPU契约测试通过（人工张量仅用于程序检查，不是实验结果）。
强制保留结构诊断为首轮结果后提出的消融，见协议§2.2；两份FAILED包不纳入正式统计。
预训练四组DONE聚合，域内对三对照的晋级条件未达，不追加这组配方的100种子确认。
服务器终态只读审计见 [记录](supplement_remote_validation_20261006.json)，本轮实验无遗留进程；原论文完整外域语料来源未重新认证。
NUPACK正式重算：srv_supplement_defect_compute_v2_20261006，2000行，44个官方对照，最大绝对误差9.81e-06。
