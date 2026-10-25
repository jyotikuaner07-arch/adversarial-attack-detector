#!/usr/bin/env python3
"""
eval_lab.py — Standalone evaluation script for AdverSense.
Evaluates FGSM, PGD, and DeepFool attacks against Standard and Robust victim models,
and evaluates clean false-positive rate on MNIST test images.
Does not start Flask.
"""

import os
import json
import torch

from pipeline import (
    classify,
    detect,
    test_defences,
    victim_standard,
    victim_robust,
    mnist_test,
)


def evaluate_attacks():
    attacks = ['FGSM', 'PGD', 'DeepFool']
    models = [('standard', victim_standard), ('robust', victim_robust)]

    rows = []
    any_heldout_missing = False

    for atk in attacks:
        index_path = f'attacks/{atk}_index.json'
        pt_path = f'attacks/{atk}_adv_images.pt'

        if not os.path.exists(index_path) or not os.path.exists(pt_path):
            print(f"⚠️ Warning: Missing files for {atk}")
            continue

        with open(index_path, 'r') as f:
            all_records = json.load(f)

        # Evaluate over success == true entries capped at 50 per attack
        records = [r for r in all_records if r.get('success')][:50]
        adv_images = torch.load(pt_path, map_location='cpu')

        # Check for held-out detection from _heldout.json and _heldout.pt when present
        heldout_index_path = f'attacks/{atk}_index_heldout.json'
        heldout_pt_path = f'attacks/{atk}_adv_images_heldout.pt'

        if os.path.exists(heldout_index_path) and os.path.exists(heldout_pt_path):
            with open(heldout_index_path, 'r') as f_h:
                all_heldout_records = json.load(f_h)
            heldout_adv_images = torch.load(heldout_pt_path, map_location='cpu')

            heldout_sample = [
                (i, r) for i, r in enumerate(all_heldout_records)
                if r.get('success')
            ][:50]

            if heldout_sample:
                heldout_det = 0
                for i, hr in heldout_sample:
                    img_idx = i if i < len(heldout_adv_images) else (hr['index'] - 500)
                    h_tensor = heldout_adv_images[img_idx].unsqueeze(0)
                    is_adv, _ = detect(h_tensor)
                    if is_adv:
                        heldout_det += 1
                heldout_det_rate = (heldout_det / len(heldout_sample) * 100)
                heldout_str = f"{heldout_det_rate:.1f}%"
            else:
                heldout_str = "0.0%"
        else:
            heldout_str = "not available"
            any_heldout_missing = True

        for model_name, model_obj in models:
            if model_obj is None:
                rows.append({
                    'attack': atk,
                    'model': model_name,
                    'n': len(records),
                    'asr_str': '—',
                    'det_rate': 0.0,
                    'heldout_str': heldout_str,
                    'n_fooled': 0,
                    'rec_str': '—',
                    'error': 'Model unavailable'
                })
                continue

            n = len(records)
            atk_success = 0
            detected = 0
            n_fooled = 0
            recovered = 0

            for r in records:
                idx = r['index']
                true_label = str(r['true'])
                tensor_idx = idx if (0 <= idx < len(adv_images)) else all_records.index(r)
                tensor = adv_images[tensor_idx].unsqueeze(0)

                # 1. Classification & attack success check
                pred, _ = classify(tensor, victim_model=model_obj)
                if pred != true_label:
                    atk_success += 1
                    n_fooled += 1

                    # 2. Defences & recovery rate: evaluated ONLY over fooled samples
                    def_results = test_defences(tensor, victim_model=model_obj)
                    if any(res['prediction'] == true_label for res in def_results):
                        recovered += 1

                # 3. Detector output (evaluated across all N attack samples)
                is_adv, _ = detect(tensor)
                if is_adv:
                    detected += 1

            # Attack success rate formatting:
            # For rows where the evaluated model is the same model the attacks were crafted against (standard),
            # the value is 100% by selection (only success==true entries are iterated). Print "—*" instead.
            if model_name == 'standard':
                asr_str = "—*"
            else:
                asr = (atk_success / n * 100) if n > 0 else 0.0
                asr_str = f"{asr:.1f}%"

            det_rate = (detected / n * 100) if n > 0 else 0.0

            # Recovery rate formatting with N fooled as denominator
            if n_fooled > 0:
                rec_rate = (recovered / n_fooled * 100)
                rec_str = f"{rec_rate:.1f}%"
            else:
                rec_str = "—"

            rows.append({
                'attack': atk,
                'model': model_name,
                'n': n,
                'asr_str': asr_str,
                'det_rate': det_rate,
                'heldout_str': heldout_str,
                'n_fooled': n_fooled,
                'rec_str': rec_str,
            })

    return rows, any_heldout_missing


def evaluate_clean_fpr(n_samples=200):
    fp_count = 0
    total = min(n_samples, len(mnist_test))
    for i in range(total):
        tensor = mnist_test[i][0].unsqueeze(0)
        is_adv, _ = detect(tensor)
        if is_adv:
            fp_count += 1
    fp_rate = (fp_count / total * 100) if total > 0 else 0.0
    return total, fp_count, fp_rate


def main():
    rows, any_heldout_missing = evaluate_attacks()
    clean_n, clean_fp, clean_fpr = evaluate_clean_fpr(200)

    # Format evaluation table
    lines = []
    header_line = f"{'Attack':<12} {'Model':<12} {'N':>4} {'Attack Success Rate':>21} {'Detection Rate':>16} {'Held-out Detection':>20} {'N fooled':>10} {'Recovery Rate':>15}"
    sep = "=" * len(header_line)
    sub_sep = "-" * len(header_line)

    lines.append(sep)
    lines.append(header_line)
    lines.append(sub_sep)

    for r in rows:
        det_str = f"{r['det_rate']:.1f}%"
        lines.append(
            f"{r['attack']:<12} {r['model']:<12} {r['n']:>4} "
            f"{r['asr_str']:>21} {det_str:>16} {r['heldout_str']:>20} "
            f"{r['n_fooled']:>10} {r['rec_str']:>15}"
        )

    lines.append(sub_sep)
    lines.append(f"Clean false-positive rate (N={clean_n}): {clean_fpr:.1f}% ({clean_fp}/{clean_n} flagged)")
    lines.append("")
    lines.append("* Attack success rate is 100% by selection (only success==true entries crafted against the standard model are evaluated).")
    if any_heldout_missing:
        lines.append("Note: Held-out detection (indices 500-999) is not available; regenerating attacks over 500-999 is required.")
    lines.append(sep)

    table_text = "\n".join(lines)
    print(table_text)

    os.makedirs('results', exist_ok=True)
    out_path = 'results/lab_eval_table.txt'
    with open(out_path, 'w') as f:
        f.write(table_text + "\n")
    print(f"\n✅ Results written to {out_path}")


if __name__ == '__main__':
    main()
