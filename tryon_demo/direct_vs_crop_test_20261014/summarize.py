#!/usr/bin/env python3
"""汇总：补充指标（背景漂移/身份相似度）+ 最终对比网格 + 更新 timing.json。"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import proc  # noqa: E402


def load(p):
    return cv2.imread(os.path.join(HERE, p))


def main():
    person = load("00_inputs/person_complex_bg.png")
    _, p_a = proc.remove_bg(person)

    def bg_drift(im):
        if im is None:
            return None
        r = im if im.shape[:2] == person.shape[:2] else cv2.resize(im, (person.shape[1], person.shape[0]))
        outside = p_a <= 0.05
        d = np.abs(r.astype(np.float32) - person.astype(np.float32)).mean(2)
        return round(float(d[outside].mean()), 2)

    targets = {
        "A1_native_first": "02_A_direct/A1_native_res0.png",
        "A1f_native_final": "02_A_direct/A1f_native_final.png",
        "A2_1024_first": "02_A_direct/A2_res1024.png",
        "A2r_1024_clean": "02_A_direct/A2_res1024_rerun.png",
        "A3_512": "02_A_direct/A3_res512.png",
        "A4_759": "02_A_direct/A4_res759.png",
        "B3_pasteback": "03_B_crop_paste/B3_pasteback.png",
        "B3p_inpaint": "03_B_crop_paste/B3p_pasteback_inpaint.png",
    }
    metrics = {"bg_drift_outside_person(越小越好)": {}, "身份相似度(与原人像)": {}}
    for name, p in targets.items():
        im = load(p)
        if im is None:
            print("skip", p)
            continue
        metrics["bg_drift_outside_person(越小越好)"][name] = bg_drift(im)
        s = proc.identity_similarity(im, person)
        if s is not None:
            metrics["身份相似度(与原人像)"][name] = s
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    tj = os.path.join(HERE, "timing.json")
    data = json.load(open(tj, encoding="utf-8"))
    data["metrics_final"] = metrics
    for f in ("timing_supplement.json", "timing_a1f.json"):
        fp = os.path.join(HERE, f)
        if os.path.exists(fp):
            try:
                data.update(json.load(open(fp, encoding="utf-8")))
            except Exception:
                pass
    with open(tj, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 最终对比网格
    def panel(img, label, H=900):
        sc = H / img.shape[0]
        p_ = cv2.resize(img, (max(1, int(img.shape[1] * sc)), H))
        bar = np.zeros((44, p_.shape[1], 3), np.uint8)
        cv2.putText(bar, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, p_])

    row = [
        panel(person, "input person (complex bg)"),
        panel(load("01_cutout/person_crop_white.png"), "B cutout"),
        panel(load("02_A_direct/A2_res1024_rerun.png"), "A2 direct 1024 (193s)"),
        panel(load("02_A_direct/A4_res759.png"), "A4 direct 759 (92s)"),
        panel(load("03_B_crop_paste/B3_pasteback.png"), "B3 crop+paste (144s)"),
        panel(load("03_B_crop_paste/B3p_pasteback_inpaint.png"), "B3p leftover-fix"),
    ]
    grid = np.hstack(row)
    if grid.shape[1] > 5600:
        sc = 5600 / grid.shape[1]
        grid = cv2.resize(grid, (int(grid.shape[1] * sc), int(grid.shape[0] * sc)))
    cv2.imwrite(os.path.join(HERE, "04_compare", "compare_grid_final.png"), grid)
    print("已写 04_compare/compare_grid_final.png 与 timing.json")


if __name__ == "__main__":
    main()
