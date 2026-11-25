# TextStyleDet_vlm — 第四章轻量级主动聚焦智能体（AZ-Agent）工作区（Audit-Hardened v2）

本工作区独立于 `E:\textstyleDet\textstyleDet_rn`（全程强制 `sys.dont_write_bytecode = True` + 运行前后只读树完整性快照校验，对原仓库与数据集保持 **100% 零修改、零字节码/缓存写入**），在 `cvpr` Conda 环境下构建第四章 **AZ-Agent** 的诊断、训练与评测管线。

## 目录结构

```text
E:\textstyleDet\TextStyleDet_vlm\
├── run_phase0.py                 # Phase 0 严格审计版评测入口（Manifest SHA256 断点强校验 + 只读边界快照）
├── verify_audit_fixes.py         # 红队审计回归单测套件（验证 P0-01/P0-02/P1-01/P1-05/P3-01）
├── DESIGN_SYNC_PHASE0.md         # 与 SmolVLM.md 的精确公式、参数版本化同步说明（P2-04）
├── zoom_agent\
│   ├── __init__.py               # 顶层字节码锁与模块导出
│   ├── runtime_shim.py           # 只读导入桥接 + 调用即抛 NotImplementedError 的安全 fcntl 桩
│   ├── detector_wrapper.py       # 冻结 StyleDetV6 封装（逐窗口 batch=1 独立画布推理 + 分块省显存 ROI）
│   ├── candidate_grid.py         # N=18 多尺度重叠离散候选窗口池（12 标准网格 + 6 道路远景中带窗）
│   ├── state_encoder.py          # 无参数多源线索构造器（M_rpn + M_det；可训练投影层留归 Phase 1 PolicyHead）
│   ├── wbf_fusion.py             # 内切边截断过滤（<5px）+ 同视图互斥防吞框 WBF + 对称过滤 NMS
│   ├── selectors.py              # RPN 灰区启发式选窗器（零 GT）& 静态 GT 错题覆盖选窗器（SFT 参考）
│   ├── eval_utils.py             # 内存级 VOC07 11点多IoU AP 评测、最大余数百分比舍入与原子文件写入
│   └── phase0_diagnose.py        # 严格先执行零GT路径、再执行两级离线Oracle诊断的七路评测核心
└── outputs\
    └── phase0\
        ├── source_only_calibration_fp32.json      # 源域 daytime_clear/train 独立超参标定溯源记录
        ├── phase0_v2_fp32_audited_results.json    # 四目标域全量标准 FP32 审计版原始数据（14,976 张）
        └── phase0_v2_fp32_audited_report.md       # 脚本 100% 自动生成的审计版 Phase 0 验收报告
```

## 复现与回归验证命令（`cvpr` 环境）

```powershell
# 1. 运行红队审计回归单测套件
& "E:\anaconda3\envs\cvpr\python.exe" -B "E:\textstyleDet\TextStyleDet_vlm\verify_audit_fixes.py"

# 2. 全量运行四目标域标准 FP32 严格隔离版 Phase 0 评测
& "E:\anaconda3\envs\cvpr\python.exe" -B "E:\textstyleDet\TextStyleDet_vlm\run_phase0.py" `
  --domains daytime_foggy night_rainy dusk_rainy Night-Sunny `
  --tag phase0_v2_fp32_audited
```
