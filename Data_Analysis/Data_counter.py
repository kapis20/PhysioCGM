import os
import glob
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# CHANGE THIS to the folder that contains all your c1s01_processed, c1s02_processed, etc.
ROOT_DIR  = r"/mnt/d/datasets/BigDataset/PhysioCGM/PhysioCGM/" 

def safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0


def summarize_subject(folder):
    subject_name = os.path.basename(folder).replace("_processed", "")
    pkl_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))

    n_files = 0
    n_cgm = 0
    n_ecg_windows = 0
    n_ppg_windows = 0
    total_ecg_samples = 0
    total_ppg_samples = 0
    glucose_vals = []

    bad_files = 0

    for fpath in tqdm(pkl_files, desc=f"Processing {subject_name}"):
        try:
            with open(fpath, "rb") as f:
                sample = pickle.load(f)
        except Exception:
            bad_files += 1
            continue

        n_files += 1

        # CGM / glucose
        glucose = sample.get("glucose", None)
        if glucose is not None:
            n_cgm += 1
            glucose_vals.append(glucose)

        # ECG
        ecg_wave = (
            sample.get("zephyr", {})
                  .get("ECG", {})
                  .get("EcgWaveform", None)
        )
        ecg_len = safe_len(ecg_wave)
        if ecg_len > 0:
            n_ecg_windows += 1
            total_ecg_samples += ecg_len

        # PPG = E4 BVP
        ppg_wave = (
            sample.get("e4", {})
                  .get("BVP", {})
                  .get("BVP", None)
        )
        ppg_len = safe_len(ppg_wave)
        if ppg_len > 0:
            n_ppg_windows += 1
            total_ppg_samples += ppg_len

    return {
        "subject": subject_name,
        "processed_pkl_files": n_files,
        "bad_pkl_files": bad_files,
        "cgm_readings": n_cgm,
        "ecg_windows_nonempty": n_ecg_windows,
        "ppg_windows_nonempty": n_ppg_windows,
        "total_ecg_samples": total_ecg_samples,
        "total_ppg_samples": total_ppg_samples,
        "ecg_hours_est": total_ecg_samples / 250 / 3600 if total_ecg_samples else 0,
        "ppg_hours_est": total_ppg_samples / 64 / 3600 if total_ppg_samples else 0,
        "min_glucose": np.min(glucose_vals) if glucose_vals else np.nan,
        "max_glucose": np.max(glucose_vals) if glucose_vals else np.nan,
    }


def main():
    processed_folders = sorted(
        os.path.join(ROOT_DIR, f)
        for f in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, f)) and f.endswith("_processed")
    )

    if not processed_folders:
        print("No *_processed folders found.")
        return

    max_workers = min(len(processed_folders), os.cpu_count() or 1)

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(summarize_subject, folder): folder
            for folder in processed_folders
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
            folder = futures[future]
            try:
                result = future.result()
                rows.append(result)
                print(f"Done: {os.path.basename(folder)}")
            except Exception as e:
                print(f"Failed: {os.path.basename(folder)} -> {e}")

    df = pd.DataFrame(rows).sort_values("subject")
    print("\nSummary:\n")
    print(df.to_string(index=False))

    out_csv = os.path.join(ROOT_DIR, "processed_summary_parallel.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    main()