from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def uniform_sample_indices(total_frames: int, num_samples: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    positions = np.linspace(0, total_frames - 1, num=num_samples)
    return np.clip(np.rint(positions).astype(np.int64), 0, total_frames - 1)


def _decode_all_frames(capture: cv2.VideoCapture, to_rgb: bool) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def sample_video_frames(video_path: Path, num_samples: int = 60, to_rgb: bool = True) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            indices = uniform_sample_indices(total_frames=total_frames, num_samples=num_samples)
            sampled: dict[int, np.ndarray] = {}
            for index in sorted(set(indices.tolist())):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    sampled = {}
                    break
                if to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled[index] = frame

            if sampled:
                ordered_frames = [sampled[int(index)] for index in indices]
                return np.stack(ordered_frames, axis=0)

            capture.release()
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not reopen video file: {video_path}")

        all_frames = _decode_all_frames(capture, to_rgb=to_rgb)
        if not all_frames:
            raise RuntimeError(f"No decodable frames in video file: {video_path}")

        indices = uniform_sample_indices(total_frames=len(all_frames), num_samples=num_samples)
        return np.stack([all_frames[int(index)] for index in indices], axis=0)
    finally:
        capture.release()


def save_sampled_frames_npy(
    video_path: Path,
    output_path: Path,
    num_samples: int = 60,
    to_rgb: bool = True,
) -> np.ndarray:
    frames = sample_video_frames(video_path=video_path, num_samples=num_samples, to_rgb=to_rgb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), frames)
    return frames
