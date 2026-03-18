import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def to_datetime_array(x):
    t = pd.to_datetime(x, errors="coerce")
    if isinstance(t, pd.Series):
        t = t.to_numpy()
    return np.asarray(t)


def clean_time_signal(t, x):
    t = np.asarray(t)
    x = np.asarray(x, dtype=np.float32)

    if len(t) == 0 or len(x) == 0:
        return None, None

    n = min(len(t), len(x))
    t = t[:n]
    x = x[:n]

    valid = (~pd.isna(t)) & np.isfinite(x)
    t = t[valid]
    x = x[valid]

    if len(t) < 2:
        return None, None

    order = np.argsort(t.astype("datetime64[ns]").astype(np.int64))
    t = t[order]
    x = x[order]

    return t, x


def crop_to_overlap(t, x, overlap_start, overlap_end):
    mask = (t >= overlap_start) & (t <= overlap_end)
    t_crop = t[mask]
    x_crop = x[mask]

    if len(t_crop) < 2 or len(x_crop) < 2:
        return None, None

    return t_crop, x_crop


def process_one_pkl(pkl_path, subject_id, output_subject_dir, min_overlap_sec,
                    glucose_min, glucose_max):
    try:
        with open(pkl_path, "rb") as f:
            sample = pickle.load(f)
    except Exception as e:
        return {
            "status": "bad_file",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
            "error": str(e),
        }

    glucose = sample.get("glucose", None)
    cgm_idx = sample.get("Index", None)
    cgm_timestamp = sample.get("Timestamp", None)

    if glucose is None:
        return {
            "status": "missing_glucose",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
        }

    try:
        glucose = float(glucose)
    except Exception:
        return {
            "status": "bad_glucose",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
        }

    if glucose_min is not None and glucose < glucose_min:
        return {
            "status": "filtered_glucose",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
            "glucose": glucose,
        }

    if glucose_max is not None and glucose > glucose_max:
        return {
            "status": "filtered_glucose",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
            "glucose": glucose,
        }

    try:
        ecg_block = sample["zephyr"]["ECG"]
        ppg_block = sample["e4"]["BVP"]
    except Exception:
        return {
            "status": "missing_modality",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
            "glucose": glucose,
        }

    ecg_t = to_datetime_array(ecg_block.get("Time", []))
    ecg_x = np.asarray(ecg_block.get("EcgWaveform", []), dtype=np.float32)

    ppg_t = to_datetime_array(ppg_block.get("Time", []))
    ppg_x = np.asarray(ppg_block.get("BVP", []), dtype=np.float32)

    ecg_t, ecg_x = clean_time_signal(ecg_t, ecg_x)
    ppg_t, ppg_x = clean_time_signal(ppg_t, ppg_x)

    if ecg_t is None or ppg_t is None:
        return {
            "status": "empty_signal",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
            "glucose": glucose,
        }

    overlap_start = max(ecg_t[0], ppg_t[0])
    overlap_end = min(ecg_t[-1], ppg_t[-1])

    overlap_sec = float((overlap_end - overlap_start) / np.timedelta64(1, "s"))

    if overlap_sec < min_overlap_sec:
        return {
            "status": "short_overlap",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
            "glucose": glucose,
            "overlap_sec": overlap_sec,
        }

    ecg_t_crop, ecg_x_crop = crop_to_overlap(ecg_t, ecg_x, overlap_start, overlap_end)
    ppg_t_crop, ppg_x_crop = crop_to_overlap(ppg_t, ppg_x, overlap_start, overlap_end)

    if ecg_t_crop is None or ppg_t_crop is None:
        return {
            "status": "crop_failed",
            "subject": subject_id,
            "source_pkl": os.path.basename(pkl_path),
            "glucose": glucose,
        }

    ecg_t_rel_s = ((ecg_t_crop - overlap_start) / np.timedelta64(1, "s")).astype(np.float32)
    ppg_t_rel_s = ((ppg_t_crop - overlap_start) / np.timedelta64(1, "s")).astype(np.float32)

    segment_name = f"{int(cgm_idx) if cgm_idx is not None else os.path.splitext(os.path.basename(pkl_path))[0]}.npz"
    segment_path = os.path.join(output_subject_dir, "segments", segment_name)
    os.makedirs(os.path.dirname(segment_path), exist_ok=True)

    np.savez_compressed(
        segment_path,
        ecg_t_rel_s=ecg_t_rel_s,
        ecg=ecg_x_crop,
        ppg_t_rel_s=ppg_t_rel_s,
        ppg=ppg_x_crop,
        glucose=np.float32(glucose),
        cgm_idx=np.int64(cgm_idx if cgm_idx is not None else -1),
        cgm_timestamp=str(cgm_timestamp),
        overlap_start=str(pd.Timestamp(overlap_start)),
        overlap_end=str(pd.Timestamp(overlap_end)),
        n_ecg=np.int32(len(ecg_x_crop)),
        n_ppg=np.int32(len(ppg_x_crop)),
    )

    return {
        "status": "ok",
        "subject": subject_id,
        "source_pkl": os.path.basename(pkl_path),
        "segment_file": segment_path,
        "cgm_idx": cgm_idx,
        "cgm_timestamp": cgm_timestamp,
        "glucose": glucose,
        "overlap_start": str(pd.Timestamp(overlap_start)),
        "overlap_end": str(pd.Timestamp(overlap_end)),
        "overlap_sec": overlap_sec,
        "n_ecg": int(len(ecg_x_crop)),
        "n_ppg": int(len(ppg_x_crop)),
    }


def process_one_subject(subject_folder, output_root, min_overlap_sec,
                        glucose_min, glucose_max):
    subject_id = os.path.basename(subject_folder).replace("_processed", "")
    output_subject_dir = os.path.join(output_root, subject_id)
    os.makedirs(output_subject_dir, exist_ok=True)

    pkl_files = sorted(glob.glob(os.path.join(subject_folder, "*.pkl")))

    results = []
    for pkl_path in tqdm(pkl_files, desc=f"Processing {subject_id}"):
        result = process_one_pkl(
            pkl_path=pkl_path,
            subject_id=subject_id,
            output_subject_dir=output_subject_dir,
            min_overlap_sec=min_overlap_sec,
            glucose_min=glucose_min,
            glucose_max=glucose_max,
        )
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_subject_dir, "metadata.csv"), index=False)

    summary = {
        "subject": subject_id,
        "n_input_pkls": len(pkl_files),
        "n_ok": int((df["status"] == "ok").sum()) if len(df) else 0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build matched ECG-PPG-BGL dataset without resampling.")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Folder containing c1s01_processed, c1s02_processed, ...")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output folder")
    parser.add_argument("--min_overlap_sec", type=float, default=30.0,
                        help="Minimum ECG/PPG overlap required")
    parser.add_argument("--glucose_min", type=float, default=None,
                        help="Optional minimum glucose filter")
    parser.add_argument("--glucose_max", type=float, default=None,
                        help="Optional maximum glucose filter")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of subject-level parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subject_folders = sorted(
        os.path.join(args.root_dir, x)
        for x in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, x)) and x.endswith("_processed")
    )

    if not subject_folders:
        print("No *_processed folders found.")
        return

    summaries = []

    if args.workers <= 1:
        for folder in subject_folders:
            summary = process_one_subject(
                subject_folder=folder,
                output_root=args.output_dir,
                min_overlap_sec=args.min_overlap_sec,
                glucose_min=args.glucose_min,
                glucose_max=args.glucose_max,
            )
            summaries.append(summary)
            print(f"Done {summary['subject']}: {summary['n_ok']} matched segments")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one_subject,
                    folder,
                    args.output_dir,
                    args.min_overlap_sec,
                    args.glucose_min,
                    args.glucose_max,
                ): folder
                for folder in subject_folders
            }

            for future in as_completed(futures):
                folder = futures[future]
                try:
                    summary = future.result()
                    summaries.append(summary)
                    print(f"Done {summary['subject']}: {summary['n_ok']} matched segments")
                except Exception as e:
                    print(f"Failed {os.path.basename(folder)}: {e}")

    summary_df = pd.DataFrame(summaries).sort_values("subject")
    summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()