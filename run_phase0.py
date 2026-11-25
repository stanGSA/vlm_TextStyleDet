#!/usr/bin/env python
"""Audit-Hardened CLI Runner for Phase 0: Error Diagnosis & Dual-View Benchmark.

Enforces:
  - ``sys.dont_write_bytecode = True`` before all imports (P1-05)
  - Read-only directory integrity check before & after execution (P1-05)
  - Cryptographic ``experiment_manifest`` SHA256 verification on checkpoint resume (P0-02)
  - Atomic JSON/Markdown writes & 100% deterministic report generation from JSON (P2-03)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import platform
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision

from zoom_agent.detector_wrapper import FrozenTextStyleDetWrapper
from zoom_agent.eval_utils import (
    SKY_DEFAULT_SPLITS,
    SKY_TARGET_DOMAINS,
    atomic_write_text,
    sha256_file,
    sha256_json,
)
from zoom_agent.phase0_diagnose import (
    COMPARED_METHODS,
    build_phase0_components,
    evaluate_domain_phase0,
)

DEFAULT_CKPT = (
    r"E:\textstyleDet\textstyleDet_rn\native_101_v3_and_art_rn101_v11_best"
    r"\native_101\native_101_multi_size_v3\best_model.pth"
)
DEFAULT_CLIP_RN101 = r"E:\textstyleDet\data\CLIP\RN101.pt"
DEFAULT_SKY_DIR = r"E:\textstyleDet\data\sky"
DEFAULT_OUT_DIR = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase0"


def snapshot_readonly_tree(root: Path, suffixes: tuple[str, ...]) -> Dict[str, tuple[int, int]]:
    """Record (size, mtime_ns) of matching files to verify zero mutations during run."""
    snap: Dict[str, tuple[int, int]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            st = p.stat()
            snap[str(p.relative_to(root))] = (int(st.st_size), int(st.st_mtime_ns))
    return snap


def build_experiment_manifest(
    wrapper: FrozenTextStyleDetWrapper,
    args: argparse.Namespace,
    workspace_root: Path,
) -> Dict:
    comps = build_phase0_components(
        k_max=args.k_max,
        trunc_margin_px=args.trunc_margin_px,
        max_detections=args.max_detections,
    )
    code_files = [workspace_root / "run_phase0.py", *sorted((workspace_root / "zoom_agent").glob("*.py"))]
    code_hashes = {str(p.relative_to(workspace_root)): sha256_file(p) for p in code_files if p.is_file()}

    return {
        "schema_version": 2,
        "checkpoint_path": str(wrapper.checkpoint_path),
        "checkpoint_sha256": sha256_file(wrapper.checkpoint_path),
        "clip_roi_weights_path": str(wrapper.clip_roi_weights_path),
        "clip_roi_weights_sha256": sha256_file(wrapper.clip_roi_weights_path),
        "data_dir": str(Path(args.data_dir).resolve()),
        "max_images": int(args.max_images),
        "detector_config": wrapper.config_dict(),
        "candidate_grid": comps["grid"].config_dict(),
        "state_encoder": comps["state_encoder"].config_dict(),
        "rpn_heuristic_selector": comps["heur_selector"].config_dict(),
        "static_oracle_selector": comps["static_oracle_selector"].config_dict(),
        "reward_pool_selector": comps["reward_pool_selector"].config_dict(),
        "fuser_wbf": comps["fuser_wbf"].config_dict(),
        "fuser_wbf_notrunc": comps["fuser_wbf_notrunc"].config_dict(),
        "fuser_nms": comps["fuser_nms"].config_dict(),
        "reward_proxy_formula": "2.0*TP50 + 2.5*TP75 + 1.5*sum_IoU - 0.8*score_FP - 0.02*K (score>=0.12)",
        "source_only_calibration_note": (
            "All detector/selector/fusion hyperparameters fixed on source domain daytime_clear prior to target evaluation."
        ),
        "code_sha256": code_hashes,
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda": str(torch.version.cuda),
            "device_name": torch.cuda.get_device_name(wrapper.device) if wrapper.device.type == "cuda" else "cpu",
        },
    }


def generate_markdown_report(summary: Dict, out_md: Path) -> None:
    """Deterministically generate the entire Phase-0 Markdown report from ``summary`` JSON."""
    domains_data: Dict[str, Dict] = summary["domains"]
    class_names: List[str] = summary["class_names"]
    manifest = summary["experiment_manifest"]
    det_cfg = manifest["detector_config"]

    methods_meta = [
        ("baseline", "TextStyleDet 基线 (No-Crop)"),
        ("rpn_heuristic_wbf", "+ RPN 灰区启发式裁剪 (零 GT, WBF)"),
        ("static_oracle_nms", "+ 静态错题覆盖 Oracle + 对称 NMS (SFT 参考消融)"),
        ("static_oracle_wbf_notrunc", "+ 静态错题覆盖 Oracle + WBF (无内切截断过滤)"),
        ("static_oracle_wbf", "**+ 静态错题覆盖 Oracle + WBF (SFT 伪标签参考)**"),
        ("reward_oracle_nms", "+ 受限奖励动作 Oracle + 对称 NMS (诊断参考)"),
        ("reward_oracle_wbf", "**+ 受限奖励动作 Oracle + WBF (RL 环境反馈参考)**"),
    ]

    lines: List[str] = []
    lines.append("# Phase 0 严格审计版诊断报告：TextStyleDet 误差拆解与双视图融合摸底\n")
    lines.append(f"- **实验配置指纹（Manifest SHA256）**：`{summary['manifest_digest']}`")
    lines.append(f"- **冻结检测器权重**：`{summary['checkpoint']}` (Epoch `{summary['checkpoint_epoch']}`, 训练期保存指标 `{summary['checkpoint_saved_best_metric']:.4f}`)")
    lines.append(
        f"- **推理精度与隔离模式**：`{'BF16 Autocast' if summary['use_bf16'] else '标准 FP32'}`，"
        f"`crop_isolation={det_cfg['crop_isolation_mode']}`，`crop_max_proposals={det_cfg['crop_max_proposals']}`，"
        f"`max_detections={manifest['fuser_wbf']['max_detections']}`"
    )
    lines.append(
        f"- **推理显存实测（PyTorch CUDA）**：峰值已分配 **`{summary['max_peak_allocated_gib']:.3f} GiB`**，"
        f"峰值缓存池保留 **`{summary['max_peak_reserved_gib']:.3f} GiB`**（注：仅代表冻结推理显存，SFT/GRPO 训练显存需在后续训练步实测）"
    )
    lines.append(
        "- **评测口径声明**：`mAP@0.5:0.95` 采用与上游 `StyleDetV6 / SE-COT` 完全一致的 VOC2007 11-point AP 在 `IoU=0.50:0.05:0.95` 上的均值；"
        "所有含 `Oracle` 行均使用对应域 GT 仅作离线诊断参考，严禁用于目标域调参或模型选择。\n"
    )

    if summary.get("target_mean"):
        tm = summary["target_mean"]
        lines.append("## 一、四目标域等权宏平均总览（Day-Foggy / Dusk-Rainy / Night-Sunny / Night-Rainy）\n")
        lines.append(
            "| 方案 | Target mAP@0.5 | Target mAP@0.75 | Target mAP@0.5:0.95 | 弱势类 AP50 | 弱势类 AP75 | 宏平均 $\\bar{K}$ (微平均) | 端到端实测 / 折算 FPS |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        base_ap50 = tm["baseline"]["AP50"]
        base_ap75 = tm["baseline"]["AP75"]
        for m_key, m_label in methods_meta:
            m = tm[m_key]
            d50 = m["AP50"] - base_ap50
            d75 = m["AP75"] - base_ap75
            s50 = f"{m['AP50']:.2f}" if m_key == "baseline" else f"{m['AP50']:.2f} ({d50:+.2f})"
            s75 = f"{m['AP75']:.2f}" if m_key == "baseline" else f"{m['AP75']:.2f} ({d75:+.2f})"
            k_str = f"{m['macro_avg_crops']:.2f} ({m['micro_avg_crops']:.2f})"
            fps_tag = (
                f"{m['measured_e2e_fps']:.2f} (实测)"
                if m.get("measured_e2e_fps") is not None
                else f"{m['projected_policy_fps']:.2f} (折算)"
            )
            lines.append(
                f"| {m_label} | {s50} | {s75} | {m['AP']:.2f} | "
                f"{m['weak_AP50']:.2f} | {m['weak_AP75']:.2f} | {k_str} | {fps_tag} |"
            )
        lines.append("")

    lines.append("## 二、冻结基线跨天气误差诊断（RPN 提案覆盖率与全 GT 一对一互斥结果划分）\n")
    lines.append(
        "> 说明：下表后四列为 `score >= 0.10` 一对一贪心匹配下对**全部非 difficult GT** 的互斥穷尽划分（最大余数法四舍五入展示加和严格为 `100.00%`，精确整数计数保存在 JSON 中）；括号内补充粗定位框占**已检出 TP@0.5** 的内部比例。\n"
    )
    lines.append(
        "| 域 (Split) | 图像数 | 总 GT 数 | RPN R@100 | RPN R@1000 | 弱势类 RPN R@1000 | 小目标 RPN R@1000 | 漏检且RPN未覆盖 | 漏检但RPN已覆盖 | 粗定位TP (`0.5<=IoU<0.75`) | 紧致TP (`IoU>=0.75`) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for dom, d_info in domains_data.items():
        diag = d_info["diagnosis"]
        rpn = diag["rpn_proposal_coverage"]
        part = diag["global_gt_1to1_partition_at_score010"]
        lines.append(
            f"| `{dom}` ({d_info['split']}) | {d_info['num_images']} | {diag['num_gt']} | "
            f"{rpn['R100_50']:.2f}% | {rpn['R1000_50']:.2f}% | {rpn['weak_R1000_50']:.2f}% | {rpn['small_R1000_50']:.2f}% | "
            f"{part['fn_rpn_unrecovered_pct_of_all_gt']:.2f}% | {part['fn_roi_or_nms_dropped_pct_of_all_gt']:.2f}% | "
            f"{part['tp_coarse_50_75_pct_of_all_gt']:.2f}% (占TP {part['coarse_share_within_detected_tp50_pct']:.1f}%) | "
            f"{part['tp_tight_75_pct_of_all_gt']:.2f}% |"
        )
    lines.append("")

    lines.append("## 三、各域详细指标对比（含公平同动作消融与独立动作上限参考）\n")
    for idx_d, (dom, d_info) in enumerate(domains_data.items(), start=1):
        tb = d_info["timing_and_budget"]
        sels = tb["selectors"]
        lines.append(f"### 3.{idx_d} `{dom}` (`{d_info['split']}`, {d_info['num_images']} 张)")
        lines.append(
            f"- **端到端实测延迟**：基线均值 `{tb['baseline_e2e_latency']['mean_ms']:.1f} ms` (`p50={tb['baseline_e2e_latency']['p50_ms']:.1f} ms`, `p95={tb['baseline_e2e_latency']['p95_ms']:.1f} ms`, `{tb['baseline_e2e_latency']['measured_e2e_fps']:.2f} FPS`)；"
            f"启发式均值 `{tb['rpn_heuristic_e2e_latency']['mean_ms']:.1f} ms` (`{tb['rpn_heuristic_e2e_latency']['measured_e2e_fps']:.2f} FPS`)；单窗独立重检均值 `{tb['mean_isolated_single_crop_ms']:.1f} ms`。"
        )
        lines.append(
            f"- **选窗统计**：启发式 $\\bar{{K}}={sels['rpn_heuristic']['avg_crops']:.2f}$（`No-Crop` `{sels['rpn_heuristic']['no_crop_pct']:.1f}%`，远景窗选择频率 `{sels['rpn_heuristic']['horizon_window_frequency_pct']:.1f}%`）；"
            f"静态错题 Oracle $\\bar{{K}}={sels['static_oracle']['avg_crops']:.2f}$（`No-Crop` `{sels['static_oracle']['no_crop_pct']:.1f}%`）；"
            f"受限奖励 Oracle (WBF) $\\bar{{K}}={sels['reward_oracle_wbf']['avg_crops']:.2f}$（`No-Crop` `{sels['reward_oracle_wbf']['no_crop_pct']:.1f}%`，远景窗选择频率 `{sels['reward_oracle_wbf']['horizon_window_frequency_pct']:.1f}%`）。\n"
        )
        lines.append("| 方案 | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | 弱势类 AP50 | " + " | ".join(f"`{c}`" for c in class_names) + " |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | " + " | ".join(":---:" for _ in class_names) + " |")
        for m_key, m_label in methods_meta:
            m = d_info["metrics"][m_key]
            cls_str = " | ".join(f"{m['classwise_ap50'][c]:.2f}" for c in class_names)
            lines.append(
                f"| {m_label} | {m['AP50']:.2f} | {m['AP75']:.2f} | {m['AP']:.2f} | {m['weak_AP50']:.2f} | {cls_str} |"
            )
        lines.append("")

    lines.append("## 四、尺度分层一对一匹配召回率对比（`score >= 0.10`, 1-to-1 Greedy Matching）\n")
    lines.append(
        "| 域 | 小目标 `<32²` R@0.5 (基线→启发式→静态Oracle→奖励Oracle) | 小目标 R@0.75 (基线→启发式→静态Oracle→奖励Oracle) | 中目标 `32²~96²` R@0.75 (基线→启发式→静态Oracle→奖励Oracle) |"
    )
    lines.append("| :--- | :---: | :---: | :---: |")
    for dom, d_info in domains_data.items():
        sr = d_info["diagnosis"]["size_stratified_1to1_recall_at_score010"]
        b, h, so, ro = sr["baseline"], sr["rpn_heuristic_wbf"], sr["static_oracle_wbf"], sr["reward_oracle_wbf"]
        lines.append(
            f"| `{dom}` | {b['small_R50']:.1f}% → {h['small_R50']:.1f}% → {so['small_R50']:.1f}% → **{ro['small_R50']:.1f}%** | "
            f"{b['small_R75']:.1f}% → {h['small_R75']:.1f}% → {so['small_R75']:.1f}% → **{ro['small_R75']:.1f}%** | "
            f"{b['medium_R75']:.1f}% → {h['medium_R75']:.1f}% → {so['medium_R75']:.1f}% → **{ro['medium_R75']:.1f}%** |"
        )
    lines.append("")

    # 5. Deterministic Data-Driven Conclusions
    lines.append("## 五、基于实测数据的客观审计结论（脚本自动计算生成）\n")
    if summary.get("target_mean"):
        tm = summary["target_mean"]
        b50, b75, bmap = tm["baseline"]["AP50"], tm["baseline"]["AP75"], tm["baseline"]["AP"]
        h50, h75 = tm["rpn_heuristic_wbf"]["AP50"], tm["rpn_heuristic_wbf"]["AP75"]
        sn50, sn75 = tm["static_oracle_nms"]["AP50"], tm["static_oracle_nms"]["AP75"]
        snt50, snt75, sntmap = (
            tm["static_oracle_wbf_notrunc"]["AP50"],
            tm["static_oracle_wbf_notrunc"]["AP75"],
            tm["static_oracle_wbf_notrunc"]["AP"],
        )
        sw50, sw75, swmap = tm["static_oracle_wbf"]["AP50"], tm["static_oracle_wbf"]["AP75"], tm["static_oracle_wbf"]["AP"]
        rw50, rw75, rwmap = tm["reward_oracle_wbf"]["AP50"], tm["reward_oracle_wbf"]["AP75"], tm["reward_oracle_wbf"]["AP"]

        lines.append(
            f"1. **FP32 基线复现与零泄漏隔离验证**：四目标域标准 FP32 基线 `mAP@0.5 = {b50:.4f}`（对比原训练保存的 `best_metric = {summary['checkpoint_saved_best_metric']:.4f}`，差值 `{b50 - summary['checkpoint_saved_best_metric']:+.4f}`）；"
            f"所有局部窗口均在 `batch_size=1` 独立画布下执行且启发式在读取 GT 前完成，消除跨窗 Padding 耦合。"
        )
        lines.append(
            f"2. **同动作无偏融合消融（`Static Oracle` 固定 $\\bar{{K}}={tm['static_oracle_wbf']['macro_avg_crops']:.2f}$）**："
            f"在完全相同的静态选窗动作与对称候选过滤下，`WBF` 相比 `NMS` 在 `mAP@0.5` 变化 `{sw50 - sn50:+.2f}`（`{sn50:.2f} → {sw50:.2f}`），"
            f"在 `mAP@0.75` 变化 `{sw75 - sn75:+.2f}`（`{sn75:.2f} → {sw75:.2f}`）；"
            f"启用内切边截断过滤相比无过滤：`mAP@0.5` `{sw50 - snt50:+.2f}`、`mAP@0.75` `{sw75 - snt75:+.2f}`、`mAP@0.5:0.95` `{swmap - sntmap:+.2f}`。"
        )
        lines.append(
            f"3. **两级参考对比（启发式 vs 静态错题覆盖 vs 受限奖励筛选）**："
            f"零 GT 启发式取得 `mAP@0.5 = {h50:.2f} ({h50 - b50:+.2f})` / `mAP@0.75 = {h75:.2f} ({h75 - b75:+.2f})`（宏平均 $\\bar{{K}}={tm['rpn_heuristic_wbf']['macro_avg_crops']:.2f}$）；"
            f"静态错题覆盖参考（Stage-1 伪标签形态）为 `{sw50:.2f} ({sw50 - b50:+.2f})` / `{sw75:.2f} ({sw75 - b75:+.2f})`（$\\bar{{K}}={tm['static_oracle_wbf']['macro_avg_crops']:.2f}$）；"
            f"加入环境奖励正增益筛选后（受限奖励动作参考）达 **`{rw50:.2f} ({rw50 - b50:+.2f})` / `{rw75:.2f} ({rw75 - b75:+.2f})`**、`mAP@0.5:0.95 = {rwmap:.2f} ({rwmap - bmap:+.2f})`，"
            f"同时宏平均裁剪次数降至 **`{tm['reward_oracle_wbf']['macro_avg_crops']:.2f}`**（微平均 `{tm['reward_oracle_wbf']['micro_avg_crops']:.2f}`）。"
            f"这表明在静态盲区定位之外抑制无效/误报裁剪具有显著增益空间，后续需在 Phase 2/3 中通过源域训练实证 SFT 与 GRPO 的泛化闭环。"
        )

    atomic_write_text(out_md, "\n".join(lines) + "\n")


def _fix_typo_field_name_safe(sel_stats: Dict) -> None:
    for st in sel_stats.values():
        if "horizon_window_Frequency_pct" in st:
            st["horizon_window_frequency_pct"] = st.pop("horizon_window_Frequency_pct")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit-Hardened AZ-Agent Phase 0 Benchmark")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--clip_weights", type=str, default=DEFAULT_CLIP_RN101)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_SKY_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["daytime_foggy", "night_rainy", "dusk_rainy", "Night-Sunny"],
    )
    parser.add_argument("--k_max", type=int, default=2)
    parser.add_argument("--trunc_margin_px", type=float, default=5.0)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--roi_chunk_size", type=int, default=256)
    parser.add_argument("--crop_max_proposals", type=int, default=300)
    parser.add_argument("--max_images", type=int, default=0, help="0 = full split; >0 = bounded run")
    parser.add_argument("--use_bf16", action="store_true", help="Default False (standard FP32 matching upstream eval)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tag", type=str, default="phase0_v2_fp32_audited")
    args = parser.parse_args()

    if len(set(args.domains)) != len(args.domains):
        raise ValueError(f"Duplicate domains in --domains: {args.domains}")

    workspace_root = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot protected read-only directories prior to loading/running
    rn_root = Path(r"E:\textstyleDet\textstyleDet_rn")
    sky_root = Path(args.data_dir)
    rn_snap_before = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_before = snapshot_readonly_tree(sky_root, (".pkl",))

    print(f"[Phase 0 v2] Loading frozen TextStyleDet from: {args.checkpoint}")
    wrapper = FrozenTextStyleDetWrapper(
        checkpoint_path=args.checkpoint,
        clip_roi_weights_path=args.clip_weights,
        rn_root=rn_root,
        device=args.device,
        use_amp_bf16=args.use_bf16,
        roi_chunk_size=args.roi_chunk_size,
        crop_max_proposals=args.crop_max_proposals,
    )
    manifest = build_experiment_manifest(wrapper, args, workspace_root)
    manifest_digest = sha256_json(manifest)

    print(
        f"[Phase 0 v2] Manifest={manifest_digest[:16]}... | {wrapper.image_backbone}/{wrapper.detector_head} "
        f"| precision={'BF16' if wrapper.use_amp_bf16 else 'FP32'} | crop_props={wrapper.crop_max_proposals} "
        f"| max_dets={args.max_detections}"
    )

    json_path = out_dir / f"{args.tag}_results.json"
    md_path = out_dir / f"{args.tag}_report.md"

    cached_domains: Dict[str, Dict] = {}
    if json_path.is_file():
        prev = json.loads(json_path.read_text(encoding="utf-8"))
        prev_digest = prev.get("manifest_digest")
        if prev_digest != manifest_digest:
            raise RuntimeError(
                f"Refusing to mix incompatible checkpoint cache at {json_path}: "
                f"existing manifest_digest={prev_digest} != current={manifest_digest}. "
                "Use a new --tag or delete the stale output file explicitly."
            )
        cached_domains = dict(prev.get("domains") or {})

    domains_out: Dict[str, Dict] = {}
    for dom in args.domains:
        if dom in cached_domains and cached_domains[dom].get("manifest_digest") == manifest_digest:
            print(f"[Phase 0 v2] Verified exact manifest match; reusing completed domain: {dom}")
            domains_out[dom] = cached_domains[dom]
            continue

        split = SKY_DEFAULT_SPLITS.get(dom, "test")
        res = evaluate_domain_phase0(
            wrapper=wrapper,
            data_dir=args.data_dir,
            domain=dom,
            split=split,
            k_max=args.k_max,
            trunc_margin_px=args.trunc_margin_px,
            max_detections=args.max_detections,
            max_images=args.max_images,
        )
        _fix_typo_field_name_safe(res["timing_and_budget"]["selectors"])
        res["manifest_digest"] = manifest_digest
        domains_out[dom] = res

        m_base = res["metrics"]["baseline"]
        m_heur = res["metrics"]["rpn_heuristic_wbf"]
        m_stat = res["metrics"]["static_oracle_wbf"]
        m_rew = res["metrics"]["reward_oracle_wbf"]
        sels = res["timing_and_budget"]["selectors"]
        print(
            f"  -> [{dom}/{split}] "
            f"Base={m_base['AP50']:.2f}/{m_base['AP75']:.2f} | "
            f"Heur={m_heur['AP50']:.2f}/{m_heur['AP75']:.2f} (K={sels['rpn_heuristic']['avg_crops']:.2f}) | "
            f"StaticOrac={m_stat['AP50']:.2f}/{m_stat['AP75']:.2f} (K={sels['static_oracle']['avg_crops']:.2f}) | "
            f"RewardOrac={m_rew['AP50']:.2f}/{m_rew['AP75']:.2f} (K={sels['reward_oracle_wbf']['avg_crops']:.2f}) | "
            f"Alloc={res['gpu_memory_gib']['peak_allocated_gib']:.2f}GiB",
            flush=True,
        )

        partial_payload = {
            "manifest_digest": manifest_digest,
            "experiment_manifest": manifest,
            "checkpoint": str(wrapper.checkpoint_path),
            "checkpoint_epoch": wrapper.checkpoint_epoch,
            "checkpoint_saved_best_metric": wrapper.checkpoint_best_metric,
            "class_names": list(wrapper.class_names),
            "k_max": args.k_max,
            "max_images": args.max_images,
            "use_bf16": wrapper.use_amp_bf16,
            "max_peak_allocated_gib": max(
                (d["gpu_memory_gib"]["peak_allocated_gib"] for d in domains_out.values()), default=0.0
            ),
            "max_peak_reserved_gib": max(
                (d["gpu_memory_gib"]["peak_reserved_gib"] for d in domains_out.values()), default=0.0
            ),
            "target_mean": None,
            "domains": domains_out,
        }
        atomic_write_text(json_path, json.dumps(partial_payload, indent=2, ensure_ascii=False))

    # Verify zero mutations on protected read-only directories
    rn_snap_after = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_after = snapshot_readonly_tree(sky_root, (".pkl",))
    readonly_audit = {
        "textstyle_rn_mutated_or_added": sorted(
            k for k in set(rn_snap_before) | set(rn_snap_after) if rn_snap_before.get(k) != rn_snap_after.get(k)
        ),
        "sky_pkl_mutated_or_added": sorted(
            k for k in set(sky_pkl_before) | set(sky_pkl_after) if sky_pkl_before.get(k) != sky_pkl_after.get(k)
        ),
    }
    if readonly_audit["textstyle_rn_mutated_or_added"] or readonly_audit["sky_pkl_mutated_or_added"]:
        raise RuntimeError(f"Read-only boundary violation detected: {readonly_audit}")

    target_mean = None
    if all(d in domains_out for d in SKY_TARGET_DOMAINS):
        target_mean = {}
        total_target_imgs = sum(domains_out[d]["num_images"] for d in SKY_TARGET_DOMAINS)

        method_to_sel_key = {
            "baseline": None,
            "rpn_heuristic_wbf": "rpn_heuristic",
            "static_oracle_nms": "static_oracle",
            "static_oracle_wbf_notrunc": "static_oracle",
            "static_oracle_wbf": "static_oracle",
            "reward_oracle_nms": "reward_oracle_nms",
            "reward_oracle_wbf": "reward_oracle_wbf",
        }

        for m_key in COMPARED_METHODS:
            sel_k = method_to_sel_key[m_key]
            if sel_k is None:
                macro_k = 0.0
                micro_k = 0.0
                meas_fps = float(
                    np.mean([domains_out[d]["timing_and_budget"]["baseline_e2e_latency"]["measured_e2e_fps"] for d in SKY_TARGET_DOMAINS])
                )
                proj_fps = None
            else:
                macro_k = float(
                    np.mean([domains_out[d]["timing_and_budget"]["selectors"][sel_k]["avg_crops"] for d in SKY_TARGET_DOMAINS])
                )
                total_crops_4d = sum(
                    domains_out[d]["timing_and_budget"]["selectors"][sel_k]["total_crops"] for d in SKY_TARGET_DOMAINS
                )
                micro_k = total_crops_4d / max(total_target_imgs, 1)
                if sel_k == "rpn_heuristic":
                    meas_fps = float(
                        np.mean([domains_out[d]["timing_and_budget"]["rpn_heuristic_e2e_latency"]["measured_e2e_fps"] for d in SKY_TARGET_DOMAINS])
                    )
                    proj_fps = None
                else:
                    meas_fps = None
                    proj_fps = float(
                        np.mean([
                            domains_out[d]["timing_and_budget"]["selectors"][sel_k]["projected_policy_fps_at_same_k"]
                            for d in SKY_TARGET_DOMAINS
                        ])
                    )

            target_mean[m_key] = {
                "AP": round(float(np.mean([domains_out[d]["metrics"][m_key]["AP"] for d in SKY_TARGET_DOMAINS])), 4),
                "AP50": round(float(np.mean([domains_out[d]["metrics"][m_key]["AP50"] for d in SKY_TARGET_DOMAINS])), 4),
                "AP75": round(float(np.mean([domains_out[d]["metrics"][m_key]["AP75"] for d in SKY_TARGET_DOMAINS])), 4),
                "weak_AP": round(
                    float(np.mean([domains_out[d]["metrics"][m_key]["weak_AP"] for d in SKY_TARGET_DOMAINS])), 4
                ),
                "weak_AP50": round(
                    float(np.mean([domains_out[d]["metrics"][m_key]["weak_AP50"] for d in SKY_TARGET_DOMAINS])), 4
                ),
                "weak_AP75": round(
                    float(np.mean([domains_out[d]["metrics"][m_key]["weak_AP75"] for d in SKY_TARGET_DOMAINS])), 4
                ),
                "macro_avg_crops": round(macro_k, 4),
                "micro_avg_crops": round(micro_k, 4),
                "measured_e2e_fps": round(meas_fps, 2) if meas_fps is not None else None,
                "projected_policy_fps": round(proj_fps, 2) if proj_fps is not None else None,
            }

        print("\n================ TARGET 4-DOMAIN MACRO SUMMARY ================")
        for m_key, vals in target_mean.items():
            fps_disp = (
                f"{vals['measured_e2e_fps']:.2f}(meas)"
                if vals["measured_e2e_fps"] is not None
                else f"{vals['projected_policy_fps']:.2f}(proj)"
            )
            print(
                f"  {m_key:26s}: AP50={vals['AP50']:.2f} | AP75={vals['AP75']:.2f} | "
                f"mAP={vals['AP']:.2f} | WeakAP50={vals['weak_AP50']:.2f} | "
                f"K_macro={vals['macro_avg_crops']:.2f} (micro={vals['micro_avg_crops']:.2f}) | FPS={fps_disp}"
            )

    summary = {
        "manifest_digest": manifest_digest,
        "experiment_manifest": manifest,
        "readonly_boundary_audit": readonly_audit,
        "checkpoint": str(wrapper.checkpoint_path),
        "checkpoint_epoch": wrapper.checkpoint_epoch,
        "checkpoint_saved_best_metric": wrapper.checkpoint_best_metric,
        "class_names": list(wrapper.class_names),
        "k_max": args.k_max,
        "max_images": args.max_images,
        "use_bf16": wrapper.use_amp_bf16,
        "max_peak_allocated_gib": max(
            (d["gpu_memory_gib"]["peak_allocated_gib"] for d in domains_out.values()), default=0.0
        ),
        "max_peak_reserved_gib": max(
            (d["gpu_memory_gib"]["peak_reserved_gib"] for d in domains_out.values()), default=0.0
        ),
        "target_mean": target_mean,
        "domains": domains_out,
    }

    atomic_write_text(json_path, json.dumps(summary, indent=2, ensure_ascii=False))
    generate_markdown_report(summary, md_path)
    print(f"\n[Phase 0 v2] Saved verified JSON -> {json_path}")
    print(f"[Phase 0 v2] Saved verified Report -> {md_path}")


if __name__ == "__main__":
    main()
