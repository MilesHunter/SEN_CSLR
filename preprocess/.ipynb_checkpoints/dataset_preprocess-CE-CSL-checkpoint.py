import argparse
import math
import random
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


CSV_SPLITS = ("train", "dev", "test")
TRAINING_SPLITS = ("train", "dev")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess CE-CSL videos and labels for SEN_CSLR training."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=".",
        help="Root directory of CE-CSL, containing label/ and video/.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CSL",
        help="Dataset name used under ./preprocess/. Keep default=CSL for compatibility with existing training code.",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default="./dataset/CSL",
        help="Output dataset root for extracted frames.",
    )
    parser.add_argument(
        "--output-res",
        type=str,
        default="256x256px",
        help="Output frame resolution, e.g. 256x256px.",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=1.0,
        help="Fraction of each split to keep. Example: 0.01 means 1%%.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Random seed used when sampling subset rows.",
    )
    parser.add_argument(
        "--process-image",
        "-p",
        action="store_true",
        default=False,
        help="Extract and resize video frames into target-path/features/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-extract frames even if target frame folder already exists.",
    )
    return parser.parse_args()


def parse_resolution(resolution):
    values = "".join(ch if ch.isdigit() else " " for ch in resolution).split()
    if len(values) != 2:
        raise ValueError(f"Invalid output resolution: {resolution}")
    width, height = (int(value) for value in values)
    return width, height


def normalize_gloss(gloss_text):
    if pd.isna(gloss_text):
        return ""
    tokens = [token.strip() for token in str(gloss_text).split("/")]
    tokens = [token for token in tokens if token]
    return " ".join(tokens)


def sanitize_segment(value):
    cleaned = str(value).strip().replace("\\", "_").replace("/", "_")
    cleaned = cleaned.replace(" ", "_")
    if not cleaned:
        raise ValueError("Encountered empty path segment while building sample path.")
    return cleaned


def resolve_video_path(video_root, split_name, translator, sample_id):
    translator_dir = video_root / split_name / str(translator)
    for extension in VIDEO_EXTENSIONS:
        candidate = translator_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    matches = sorted(path for path in translator_dir.glob(f"{sample_id}.*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot find video for sample '{sample_id}' under '{translator_dir}'."
    )


def sample_subset(dataframe, ratio, seed):
    if ratio <= 0 or ratio > 1:
        raise ValueError("subset-ratio must be within (0, 1].")
    if ratio == 1:
        return dataframe.reset_index(drop=True)
    if len(dataframe) == 0:
        return dataframe.copy()
    sample_size = max(1, math.ceil(len(dataframe) * ratio))
    if 0 < ratio < 1:
        sample_size = max(2, sample_size)
    sample_size = min(len(dataframe), sample_size)
    return dataframe.sample(n=sample_size, random_state=seed).sort_index().reset_index(drop=True)


def extract_frames(video_path, output_dir, output_size, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_dir.glob("*.jpg"))
    if existing_frames and not overwrite:
        return len(existing_frames)

    if overwrite:
        for existing_frame in existing_frames:
            existing_frame.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width, height = output_size
    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frame_path = output_dir / f"img_{frame_count:05d}.jpg"
        if not cv2.imwrite(str(frame_path), resized):
            capture.release()
            raise RuntimeError(f"Failed to write frame: {frame_path}")
        frame_count += 1
    capture.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frame_count


def build_split_info(dataframe, split_name, video_root, target_root, output_size, process_image, overwrite):
    split_info = {}
    skipped_samples = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Build {split_name}"):
        sample_id = str(row["Number"]).strip()
        translator = str(row["Translator"]).strip()
        sentence = "" if pd.isna(row["Chinese Sentences"]) else str(row["Chinese Sentences"]).strip()
        if not sample_id or sample_id.lower() == "nan":
            raise ValueError(f"Invalid Number value in {split_name}.csv row {index + 2}")
        if not translator or translator.lower() == "nan":
            raise ValueError(f"Invalid Translator value in {split_name}.csv row {index + 2}")
        gloss = normalize_gloss(row["Gloss"])
        if not gloss:
            skipped_samples.append((sample_id, "empty gloss"))
            continue

        video_path = resolve_video_path(video_root, split_name, translator, sample_id)
        frame_folder_name = "__".join(
            [sanitize_segment(split_name), sanitize_segment(translator), sanitize_segment(sample_id)]
        )
        frame_output_dir = target_root / "features" / f"fullFrame-{output_size[0]}x{output_size[1]}px" / frame_folder_name

        if process_image:
            num_frames = extract_frames(video_path, frame_output_dir, output_size, overwrite)
        else:
            existing_frames = sorted(frame_output_dir.glob("*.jpg"))
            if not existing_frames:
                raise FileNotFoundError(
                    f"Frames not found for '{sample_id}'. Run with --process-image first or point target-path to prepared frames."
                )
            num_frames = len(existing_frames)

        split_info[len(split_info)] = {
            "fileid": sample_id,
            "folder": frame_folder_name,
            "signer": translator,
            "label": gloss,
            "num_frames": num_frames,
            "original_info": sample_id,
        }

    if skipped_samples:
        print(f"Skipped {len(skipped_samples)} sample(s) in {split_name} because of empty gloss.")
    return split_info


def update_gloss_dict(total_dict, info_dict):
    for sample in info_dict.values():
        for gloss in sample["label"].split():
            total_dict[gloss] = total_dict.get(gloss, 0) + 1


def save_groundtruth(info_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        for sample in info_dict.values():
            handle.write(
                f"{sample['fileid']} 1 {sample['signer']} 0.0 1.79769e+308 {sample['label']}\n"
            )


def validate_columns(dataframe, split_name):
    required_columns = {"Number", "Translator", "Chinese Sentences", "Gloss"}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {split_name}.csv: {', '.join(missing_columns)}"
        )


def main():
    args = parse_args()
    random.seed(args.subset_seed)

    dataset_root = Path(args.dataset_root).resolve()
    label_root = dataset_root / "label"
    video_root = dataset_root / "video"
    target_root = Path(args.target_path).resolve()
    preprocess_root = Path(__file__).resolve().parent / args.dataset
    preprocess_root.mkdir(parents=True, exist_ok=True)

    output_size = parse_resolution(args.output_res)
    sign_dict = {}

    for split_name in CSV_SPLITS:
        csv_path = label_root / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")

        dataframe = pd.read_csv(csv_path)
        validate_columns(dataframe, split_name)
        dataframe = sample_subset(dataframe, args.subset_ratio, args.subset_seed)

        split_info = build_split_info(
            dataframe=dataframe,
            split_name=split_name,
            video_root=video_root,
            target_root=target_root,
            output_size=output_size,
            process_image=args.process_image,
            overwrite=args.overwrite,
        )

        np.save(preprocess_root / f"{split_name}_info.npy", cast(Any, split_info), allow_pickle=True)
        save_groundtruth(split_info, preprocess_root / f"{args.dataset}-groundtruth-{split_name}.stm")
        if split_name in TRAINING_SPLITS:
            update_gloss_dict(sign_dict, split_info)
        print(f"Saved {split_name}: {len(split_info)} sample(s)")

    sorted_sign_dict = sorted(sign_dict.items(), key=lambda item: item[0])
    save_dict = {
        gloss: [index + 1, count]
        for index, (gloss, count) in enumerate(sorted_sign_dict)
    }
    np.save(preprocess_root / "gloss_dict.npy", cast(Any, save_dict), allow_pickle=True)

    print(f"Saved gloss dictionary with {len(save_dict)} classes to {preprocess_root / 'gloss_dict.npy'}")
    print(f"Frame root: {target_root / 'features' / f'fullFrame-{output_size[0]}x{output_size[1]}px'}")
    print("Done.")


if __name__ == "__main__":
    main()
