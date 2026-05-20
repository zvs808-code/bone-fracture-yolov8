"""
╔══════════════════════════════════════════════════════════════════╗
║  AutoML 自动调优 — Optuna + YOLOv8m  【分段训练版】              ║
║                                                                  ║
║  特性：                                                          ║
║  • 每晚运行到指定时间（默认 07:00）自动暂停                       ║
║  • 进度存入 SQLite，第二晚自动续接，不丢失任何 trial              ║
║  • 最终训练同样支持中断续训（YOLO last.pt resume）               ║
║  • 自动检测当前处于哪个阶段（搜索中 / 搜索完 / 最终训练）        ║
╚══════════════════════════════════════════════════════════════════╝

每晚启动命令（只需运行这一条，自动判断从哪里续接）：
    python automl_optuna.py

查看今晚进度：
    Get-Content automl_out.txt -Tail 30 -Wait

强制立即停止（明早不想等时间到）：
    Ctrl+C  （进度已保存，下次启动自动续接）
"""

import os, json, shutil, datetime, time, gc
from pathlib import Path

import optuna
from optuna.storages import RDBStorage
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════
#  ★ 用户配置区 — 按需修改
# ═══════════════════════════════════════════════════════════════
BASE_MODEL   = str(Path(__file__).parent / "runs_v4/yolov8m_v4/weights/best.pt")
DATA_YAML    = str(Path(__file__).parent / "final_fracture_v3.yaml")  # 扩充数据集（37,269张）

N_TRIALS     = 20       # 总搜索次数（扩增至20，找最优超参）
EPOCHS_TUNE  = 7        # 每 trial 训练 epoch
EPOCHS_FINAL = 100      # 最终最佳参数完整训练 epoch（增至100）

DEVICE       = 0        # GPU 编号

# 每天自动停止时间（早上几点暂停，留时间给你用电脑）
STOP_HOUR    = 23       # 23:00 自动暂停（全天可跑）
STOP_MINUTE  = 0

# 路径（不用改）
STUDY_DIR  = Path(__file__).parent / "automl_runs"
REPORT_DIR = Path(__file__).parent / "automl_report"
DB_PATH    = STUDY_DIR / "optuna_study.db"          # 进度数据库（断点续训关键）
STATE_FILE = STUDY_DIR / "automl_state.json"        # 阶段状态记录
# ═══════════════════════════════════════════════════════════════

STUDY_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────────
#  工具函数
# ───────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def should_stop_now() -> bool:
    """用户要求：一直训练直到完成，不自动停止。"""
    return False


def time_until_stop() -> str:
    now  = datetime.datetime.now()
    stop = now.replace(hour=STOP_HOUR, minute=STOP_MINUTE, second=0, microsecond=0)
    if stop <= now:
        stop += datetime.timedelta(days=1)
    delta = stop - now
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"phase": "search", "final_resumed": False}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ───────────────────────────────────────────────────────────────
#  Optuna 目标函数
# ───────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """每次 trial 返回验证集 mAP@50。"""

    # 检查时间：若快到停止时间，终止搜索
    if should_stop_now():
        log("⏰ 到达停止时间，暂停搜索（进度已保存）")
        raise optuna.exceptions.OptunaError("TIME_LIMIT_REACHED")

    imgsz        = trial.suggest_categorical("imgsz",       [640])  # 限640防OOM
    lr0          = trial.suggest_float(      "lr0",          1e-5, 5e-3, log=True)
    lrf          = trial.suggest_float(      "lrf",          0.005, 0.2)
    momentum     = trial.suggest_float(      "momentum",     0.85, 0.98)
    weight_decay = trial.suggest_float(      "weight_decay", 1e-5, 5e-3, log=True)
    warmup_epochs= trial.suggest_int(        "warmup_epochs",1, 5)
    box          = trial.suggest_float(      "box",          4.0, 12.0)
    cls          = trial.suggest_float(      "cls",          0.1,  1.0)
    dfl          = trial.suggest_float(      "dfl",          1.0,  3.0)
    patience     = trial.suggest_int(        "patience",     5,   20)
    hsv_v        = trial.suggest_float(      "hsv_v",        0.2,  0.6)
    scale        = trial.suggest_float(      "scale",        0.1,  0.5)
    erasing      = trial.suggest_float(      "erasing",      0.0,  0.4)

    run_name = f"trial_{trial.number:03d}"
    log(f"▶ Trial {trial.number+1}/{N_TRIALS}  imgsz={imgsz}  lr0={lr0:.2e}  "
        f"box={box:.1f}  patience={patience}  (剩余 {time_until_stop()})")

    model   = YOLO(BASE_MODEL)
    results = model.train(
        data          = DATA_YAML,
        epochs        = EPOCHS_TUNE,
        imgsz         = imgsz,
        device        = DEVICE,
        project       = str(STUDY_DIR),
        name          = run_name,
        exist_ok      = True,
        lr0           = lr0,
        lrf           = lrf,
        momentum      = momentum,
        weight_decay  = weight_decay,
        warmup_epochs = warmup_epochs,
        box           = box,
        cls           = cls,
        dfl           = dfl,
        patience      = patience,
        fliplr=0, degrees=0, shear=0, perspective=0,
        mosaic=0, mixup=0, copy_paste=0,
        hsv_h=0.005, hsv_s=0.1,
        hsv_v         = hsv_v,
        scale         = scale,
        erasing       = erasing,
        verbose       = False,
        batch         = 6,     # 降至6防止最后epoch OOM
        workers       = 0,     # 0=主进程加载，彻底避免内存爆满
        amp           = True,  # 混合精度，节省显存
        cache         = False,
    )

    metrics = results.results_dict
    map50   = float(metrics.get("metrics/mAP50(B)", 0.0))
    log(f"  ✓ Trial {trial.number+1} 完成  mAP50 = {map50:.4f}")

    # 强制释放内存，防止多 trial 累积耗尽 RAM
    try:
        import torch
        del model, results
        gc.collect()
        torch.cuda.empty_cache()
        log(f"  🧹 内存已回收")
    except Exception:
        pass

    return map50


# ───────────────────────────────────────────────────────────────
#  低置信度区域分析报告
# ───────────────────────────────────────────────────────────────
def run_low_conf_report(model_path: str, conf_thresh: float = 0.35):
    log("📊 生成低置信度区域分析报告...")
    model   = YOLO(model_path)
    records = []

    for r in model.predict(source=DATA_YAML, conf=0.01, iou=0.5,
                           save=False, verbose=False, stream=True):
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf)
            if conf < conf_thresh:
                xywhn = box.xywhn[0].tolist()
                records.append({
                    "image":   Path(r.path).name,
                    "conf":    round(conf, 4),
                    "cx_norm": round(xywhn[0], 4),
                    "cy_norm": round(xywhn[1], 4),
                    "w_norm":  round(xywhn[2], 4),
                    "h_norm":  round(xywhn[3], 4),
                })

    edge = sum(1 for r in records
               if r["cx_norm"] < 0.05 or r["cx_norm"] > 0.95
               or r["cy_norm"] < 0.05 or r["cy_norm"] > 0.95)

    report = {
        "generated_at":       datetime.datetime.now().isoformat(),
        "conf_threshold":     conf_thresh,
        "total_low_conf":     len(records),
        "edge_region_count":  edge,
        "edge_region_pct":    round(edge / max(len(records), 1) * 100, 1),
        "detections":         records[:200],
    }
    out = REPORT_DIR / "low_conf_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"  ✅ 低置信度检测数：{len(records)}  边缘占比：{report['edge_region_pct']}%")
    log(f"  报告路径：{out}")


# ───────────────────────────────────────────────────────────────
#  主流程
# ───────────────────────────────────────────────────────────────
def main():
    state = load_state()

    # ══════════════════════════════════════════════════════════
    #  阶段 1：Optuna 搜索（可能跨多晚）
    # ══════════════════════════════════════════════════════════
    if state["phase"] == "search":

        storage = RDBStorage(url=f"sqlite:///{DB_PATH}")
        study = optuna.create_study(
            study_name  = "fracture_yolov8m_automl",
            storage     = storage,
            load_if_exists = True,          # ← 关键：存在则续接
            direction   = "maximize",
            sampler     = optuna.samplers.TPESampler(seed=42),
        )

        completed = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])

        log("=" * 58)
        log(f"  AutoML 搜索阶段  [{completed}/{N_TRIALS} trials 已完成]")
        log(f"  今晚停止时间：{STOP_HOUR:02d}:{STOP_MINUTE:02d}  "
            f"剩余可用：{time_until_stop()}")
        log("=" * 58)

        if completed >= N_TRIALS:
            log("✅ 所有 trials 已完成，跳转至最终训练阶段")
            state["phase"] = "final"
            save_state(state)
        else:
            remaining = N_TRIALS - completed
            log(f"  继续搜索，剩余 {remaining} trials...")
            try:
                study.optimize(
                    objective,
                    n_trials   = remaining,
                    catch       = (Exception,),   # 遇到 TIME_LIMIT 不崩溃
                    show_progress_bar = False,
                )
            except KeyboardInterrupt:
                log("⌨  手动中断，进度已保存")
            except Exception as e:
                log(f"  搜索中断：{e}")

            # 重新检查完成数
            completed_now = len([t for t in study.trials
                                  if t.state == optuna.trial.TrialState.COMPLETE])
            log(f"\n  本晚完成 {completed_now - completed} trials  "
                f"累计 {completed_now}/{N_TRIALS}")

            if completed_now >= N_TRIALS:
                log("🏆 搜索阶段全部完成！")
                state["phase"] = "final"
                save_state(state)
            else:
                log(f"  明晚继续运行脚本，自动从第 {completed_now+1} trial 续接")
                _print_best_so_far(study)
                return   # 今晚结束，明晚再来

    # 如果搜索刚完成，显示最佳参数
    if state["phase"] == "final":
        storage = RDBStorage(url=f"sqlite:///{DB_PATH}")
        study   = optuna.load_study(
            study_name = "fracture_yolov8m_automl",
            storage    = storage,
        )
        _print_best_so_far(study)

        best = study.best_trial
        params_path = REPORT_DIR / "best_params.json"
        params_path.write_text(json.dumps(
            {"best_map50": best.value, "params": best.params},
            indent=2, ensure_ascii=False
        ))

    # ══════════════════════════════════════════════════════════
    #  阶段 2：最终完整训练（支持中断续训）
    # ══════════════════════════════════════════════════════════
    if state["phase"] == "final":

        final_dir    = STUDY_DIR / "automl_best"
        last_pt      = final_dir / "weights" / "last.pt"
        best_pt_dst  = Path(__file__).parent / "automl_best.pt"

        # 检查是否已训练完成
        if best_pt_dst.exists() and state.get("final_done"):
            log("✅ 最终训练已完成，直接生成报告")
            run_low_conf_report(str(best_pt_dst))
            log("\n🎉 AutoML 全部完成！")
            return

        log("\n" + "=" * 58)
        if should_stop_now():
            log("⏰ 已到停止时间，最终训练将在明晚自动开始")
            log(f"  明晚运行同一命令即可自动续接最终训练")
            return

        log(f"  最终训练阶段  ({EPOCHS_FINAL} epochs)")
        log(f"  今晚停止时间：{STOP_HOUR:02d}:{STOP_MINUTE:02d}  "
            f"剩余：{time_until_stop()}")
        log("=" * 58)

        # 加载最佳超参数
        params_path = REPORT_DIR / "best_params.json"
        p = json.loads(params_path.read_text())["params"]

        # 判断是续训还是新训
        if last_pt.exists() and not state.get("final_done"):
            log(f"  🔄 检测到 last.pt，从断点续训...")
            model = YOLO(str(last_pt))
            final_results = model.train(resume=True)
        else:
            log(f"  🚀 开始最终训练（最佳参数：imgsz={p['imgsz']} lr0={p['lr0']:.2e}）")
            model = YOLO(BASE_MODEL)
            final_results = model.train(
                data          = DATA_YAML,
                epochs        = EPOCHS_FINAL,
                imgsz         = p["imgsz"],
                device        = DEVICE,
                project       = str(STUDY_DIR),
                name          = "automl_best",
                exist_ok      = True,
                batch         = 8,
                workers       = 2,
                lr0           = p["lr0"],
                lrf           = p["lrf"],
                momentum      = p["momentum"],
                weight_decay  = p["weight_decay"],
                warmup_epochs = p["warmup_epochs"],
                box           = p["box"],
                cls           = p["cls"],
                dfl           = p["dfl"],
                patience      = 30,            # 更宽松的 early stopping
                close_mosaic  = 15,            # 最后15 epoch关mosaic稳定收敛
                # 开启完整 augmentation（最终训练用）
                fliplr        = 0.5,
                degrees       = 5.0,
                hsv_h         = 0.015,
                hsv_s         = 0.4,
                hsv_v         = p["hsv_v"],
                scale         = p["scale"],
                erasing       = p["erasing"],
                mosaic        = 1.0,
                mixup         = 0.1,
                verbose       = True,
                amp           = True,
            )

        # 复制 best.pt
        best_pt_src = final_dir / "weights" / "best.pt"
        if best_pt_src.exists():
            shutil.copy(best_pt_src, best_pt_dst)
            state["final_done"] = True
            save_state(state)
            log(f"\n✅ 最佳模型：{best_pt_dst}")
            run_low_conf_report(str(best_pt_dst))
            log("\n🎉 AutoML 全部完成！")
            log(f"   权重文件：{best_pt_dst}")
            log(f"   参数报告：{REPORT_DIR / 'best_params.json'}")
            log(f"   低置信报告：{REPORT_DIR / 'low_conf_report.json'}")
        else:
            log("⏰ 今晚训练被时间中断，明晚运行同一命令自动续接")


def _print_best_so_far(study):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        log("  （还无已完成的 trial）")
        return
    best = study.best_trial
    log(f"\n  🏆 目前最佳：Trial #{best.number}  mAP50 = {best.value:.4f}")
    log("  最佳参数：")
    for k, v in best.params.items():
        log(f"    {k:20s} = {v}")


if __name__ == "__main__":
    main()
