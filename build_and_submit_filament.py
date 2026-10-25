import os
import sys
import time
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")
ARTIFACT_DIR = r"C:\Users\swsz9\.gemini\antigravity\brain\c9e5e770-b6cf-4fe7-a4b3-7e0ca322c2a8"
ENS_DIR = os.path.abspath("SolarFilament2026/ens")
SUB_V1 = os.path.abspath("SolarFilament2026/submission_v1_honest_3arch_pq.csv")
SUB_V2 = os.path.abspath("SolarFilament2026/submission_v2_high_conf_pq.csv")


def rle_from_string(s: str):
    cnts = []
    p, n = 0, len(s)
    while p < n:
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_to_string(cnts):
    out = []
    for i, x in enumerate(cnts):
        if i > 2:
            x -= cnts[i - 2]
        more = 1
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def rle_to_intervals(s: str):
    cnts = rle_from_string(s)
    pos = 0
    iv = []
    area = 0
    for i, c in enumerate(cnts):
        if i % 2 == 1 and c > 0:
            iv.append((pos, pos + c))
            area += c
        pos += c
    return iv, area


def intervals_intersection(iv1, iv2):
    i, j = 0, 0
    inter = 0
    while i < len(iv1) and j < len(iv2):
        a1, b1 = iv1[i]
        a2, b2 = iv2[j]
        lo = a1 if a1 > a2 else a2
        hi = b1 if b1 < b2 else b2
        if hi > lo:
            inter += hi - lo
        if b1 < b2:
            i += 1
        else:
            j += 1
    return inter


def intervals_iou(item1, item2):
    iv1, area1 = item1[0], item1[1]
    iv2, area2 = item2[0], item2[1]
    if area1 == 0 or area2 == 0:
        return 0.0
    # Quick bounding-index check
    if iv1[-1][1] <= iv2[0][0] or iv2[-1][1] <= iv1[0][0]:
        return 0.0
    inter = intervals_intersection(iv1, iv2)
    if inter == 0:
        return 0.0
    return inter / (area1 + area2 - inter)


def encode_1d_mask(mask_1d, total_len=2048 * 2048):
    diffs = np.flatnonzero(mask_1d[1:] != mask_1d[:-1]) + 1
    boundaries = np.concatenate(([0], diffs, [total_len]))
    lengths = np.diff(boundaries).tolist()
    if mask_1d[0]:
        lengths = [0] + lengths
    return rle_to_string(lengths)


def majority_vote_rle(rle_strings, img_used, total_len=2048 * 2048):
    votes = np.zeros(total_len, dtype=np.uint8)
    threshold = 2 if len(rle_strings) >= 3 else 1
    # If len == 2, use primary (rle_strings[0])
    target_rles = rle_strings if len(rle_strings) >= 3 else [rle_strings[0]]
    for s in target_rles:
        cnts = rle_from_string(s)
        pos = 0
        for i, c in enumerate(cnts):
            if i % 2 == 1 and c > 0:
                votes[pos : pos + c] += 1
            pos += c
    mask_1d = (votes >= threshold) & (~img_used)
    area = int(mask_1d.sum())
    if area < 50:
        return None, 0
    img_used |= mask_1d
    return encode_1d_mask(mask_1d, total_len), area


def load_by_image(csv_path):
    df = pd.read_csv(csv_path)
    by_img = {}
    for _, row in df.iterrows():
        fid = str(row["filament_id"])
        rle = str(row["segmentation_rle"])
        img_id = fid.rsplit("_", 1)[0]
        iv, area = rle_to_intervals(rle)
        by_img.setdefault(img_id, []).append((iv, area, rle))
    return by_img


def main():
    print("Loading 3 Solar Filament instance segmentation models (Phuong TTA-4, Noman MaskRCNN+UNet, Ektarr YOLO11s+UNet)...")
    ph_map = load_by_image(os.path.join(ENS_DIR, "sf_phuong_submission.csv"))
    no_map = load_by_image(os.path.join(ENS_DIR, "sf_noman_submission.csv"))
    ek_map = load_by_image(os.path.join(ENS_DIR, "sf_ektarr_submission.csv"))

    all_imgs = sorted(set(ph_map) | set(no_map) | set(ek_map))
    print(f"Total unique test images with detections: {len(all_imgs)}")

    v1_rows = []
    v2_rows = []
    img_scores = []

    for img_id in all_imgs:
        ph_list = ph_map.get(img_id, [])
        no_list = no_map.get(img_id, [])
        ek_list = ek_map.get(img_id, [])

        kept_rles = []
        used_no = set()
        used_ek = set()
        img_ious = []
        img_used = np.zeros(2048 * 2048, dtype=bool)

        # 1. Start from Phuong TTA-4 topology-safe instances
        for ph_item in ph_list:
            best_no_iou, best_no_idx = 0.0, -1
            for j, no_item in enumerate(no_list):
                if j not in used_no:
                    iou = intervals_iou(ph_item, no_item)
                    if iou > best_no_iou:
                        best_no_iou, best_no_idx = iou, j

            best_ek_iou, best_ek_idx = 0.0, -1
            for k, ek_item in enumerate(ek_list):
                if k not in used_ek:
                    iou = intervals_iou(ph_item, ek_item)
                    if iou > best_ek_iou:
                        best_ek_iou, best_ek_idx = iou, k

            matched_rles = [ph_item[2]]
            if best_no_iou >= 0.30:
                used_no.add(best_no_idx)
                matched_rles.append(no_list[best_no_idx][2])
            if best_ek_iou >= 0.30:
                used_ek.add(best_ek_idx)
                matched_rles.append(ek_list[best_ek_idx][2])

            # Keep if supported by >=2 models OR high-confidence filament (>= 450 px)
            if len(matched_rles) >= 2 or ph_item[1] >= 450:
                fused_rle, area = majority_vote_rle(matched_rles, img_used)
                if fused_rle is not None:
                    kept_rles.append((fused_rle, len(matched_rles), 0.5 * (best_no_iou + best_ek_iou)))
                    img_ious.append(0.5 * (best_no_iou + best_ek_iou))

        # 2. Check if Noman + Ektarr agree strongly (IoU >= 0.55) on a filament missed by Phuong
        for j, no_item in enumerate(no_list):
            if j in used_no:
                continue
            best_ek_iou, best_ek_idx = 0.0, -1
            for k, ek_item in enumerate(ek_list):
                if k not in used_ek:
                    iou = intervals_iou(no_item, ek_item)
                    if iou > best_ek_iou:
                        best_ek_iou, best_ek_idx = iou, k
            if best_ek_iou >= 0.55 and no_item[1] >= 450:
                used_no.add(j)
                used_ek.add(best_ek_idx)
                fused_rle, area = majority_vote_rle([no_item[2], ek_list[best_ek_idx][2]], img_used)
                if fused_rle is not None:
                    kept_rles.append((fused_rle, 2, best_ek_iou))
                    img_ious.append(best_ek_iou)

        for idx, (rle_str, _, _) in enumerate(kept_rles, 1):
            v1_rows.append((f"{img_id}_{idx}", rle_str))

        # Build v3: Full Phuong TTA-4 recall + 2-of-3 boundary refinement + strict zero overlap
        img_used_v3 = np.zeros(2048 * 2048, dtype=bool)
        used_no_v3 = set()
        used_ek_v3 = set()
        v3_idx = 0
        for ph_item in ph_list:
            best_no_iou, best_no_idx = 0.0, -1
            for j, no_item in enumerate(no_list):
                if j not in used_no_v3:
                    iou = intervals_iou(ph_item, no_item)
                    if iou > best_no_iou:
                        best_no_iou, best_no_idx = iou, j
            best_ek_iou, best_ek_idx = 0.0, -1
            for k, ek_item in enumerate(ek_list):
                if k not in used_ek_v3:
                    iou = intervals_iou(ph_item, ek_item)
                    if iou > best_ek_iou:
                        best_ek_iou, best_ek_idx = iou, k
            m_rles = [ph_item[2]]
            if best_no_iou >= 0.40:
                used_no_v3.add(best_no_idx)
                m_rles.append(no_list[best_no_idx][2])
            if best_ek_iou >= 0.40:
                used_ek_v3.add(best_ek_idx)
                m_rles.append(ek_list[best_ek_idx][2])
            fused_rle, area = majority_vote_rle(m_rles, img_used_v3)
            if fused_rle is not None:
                v3_idx += 1
                v2_rows.append((f"{img_id}_{v3_idx}", fused_rle))

    pd.DataFrame(v1_rows, columns=["filament_id", "segmentation_rle"]).to_csv(SUB_V1, index=False)
    print(f"Saved {SUB_V1}: {len(v1_rows)} instances across {len(all_imgs)} images ({os.path.getsize(SUB_V1):,} bytes)")
    SUB_V3 = os.path.abspath("SolarFilament2026/submission_v3_full_recall_3arch_pq.csv")
    pd.DataFrame(v2_rows, columns=["filament_id", "segmentation_rle"]).to_csv(SUB_V3, index=False)
    print(f"Saved {SUB_V3}: {len(v2_rows)} instances across {len(all_imgs)} images ({os.path.getsize(SUB_V3):,} bytes)")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        ctx = browser.contexts[0]
        for pg in ctx.pages:
            if "filament-segmentation-2026" in pg.url:
                pg.close()

        page = ctx.new_page()
        page.set_viewport_size({"width": 1440, "height": 1000})

        for sub_idx, (sub_path, desc) in enumerate([
            (SUB_V1, "v1 Zero-Overlap 3-Architecture Panoptic Consensus (Mask R-CNN + U-Net 4-Flip TTA + YOLO11s-seg 2-of-3 Pixel Majority Voting)"),
            (SUB_V3, "v3 Full-Recall Topology-Safe 3-Architecture Boundary Refinement (Mask R-CNN + U-Net TTA-4 + YOLO11s-seg IoU>=0.40 Pixel Majority)"),
        ], 1):
            page.goto("https://www.kaggle.com/competitions/filament-segmentation-2026", wait_until="domcontentloaded", timeout=60000)
            time.sleep(5)
            sub_btn = page.locator('button:has-text("Submit Prediction"), a:has-text("Submit Prediction")')
            if sub_btn.count() > 0:
                sub_btn.first.click()
                time.sleep(4)

            file_inputs = page.locator('input[type="file"]')
            if file_inputs.count() > 0:
                file_inputs.first.set_input_files(sub_path)
                for _ in range(30):
                    time.sleep(1)
                    sub_submit = page.locator('button:has-text("Submit")')
                    ready = False
                    for i in range(sub_submit.count()):
                        b = sub_submit.nth(i)
                        if b.is_visible() and (b.inner_text() or "").strip() == "Submit" and not b.is_disabled():
                            ready = True
                            break
                    if ready:
                        break

                ta = page.locator('textarea[placeholder*="Describe"], textarea[aria-label*="Description"], textarea')
                if ta.count() > 0 and ta.first.is_visible():
                    ta.first.fill(desc)
                    time.sleep(1)

                sub_submit = page.locator('button:has-text("Submit")')
                for i in range(sub_submit.count()):
                    b = sub_submit.nth(i)
                    if b.is_visible() and (b.inner_text() or "").strip() == "Submit" and not b.is_disabled():
                        b.click()
                        print(f"[{sub_idx}] Clicked Submit on Solar Filament ({os.path.basename(sub_path)})!")
                        time.sleep(10)
                        break

        page.goto("https://www.kaggle.com/competitions/filament-segmentation-2026/submissions", wait_until="domcontentloaded", timeout=60000)
        for _ in range(8):
            time.sleep(5)
            txt = page.locator("body").inner_text()
            if "Pending" not in txt:
                break
            page.reload(wait_until="domcontentloaded")
        time.sleep(3)
        shot = os.path.join(ARTIFACT_DIR, "kaggle_filament_submitted.png")
        page.screenshot(path=shot, full_page=True)
        print("Saved screenshot:", shot)
        print(page.locator("body").inner_text()[800:2600])
        page.close()


if __name__ == "__main__":
    main()
