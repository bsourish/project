# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf

## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "traindata")
TEST_DIR = os.path.join(BASE_DIR, "test", "TestData")


def _list_videos(preferred_dir, fallback_dirs=None):
    def _list_in_dir(d):
        if not os.path.isdir(d):
            return []
        return [f for f in os.listdir(d) if f.lower().endswith(".mp4")]

    videos = _list_in_dir(preferred_dir)
    if videos:
        return preferred_dir, videos

    if fallback_dirs:
        for d in fallback_dirs:
            videos = _list_in_dir(d)
            if videos:
                return d, videos
            # If this is a container folder, search recursively once
            if os.path.isdir(d):
                found = []
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith(".mp4"):
                            found.append(os.path.join(root, f))
                if found:
                    return None, found  # full paths

    return preferred_dir, []

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract multiple evenly spaced color frames for robustness
def _get_sample_frames_color(video_path, num_samples=5):
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            indices = [0]
        else:
            if num_samples < 1:
                num_samples = 1
            step = max(total_frames // num_samples, 1)
            indices = [min(i * step, total_frames - 1) for i in range(num_samples)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frames.append(frame)

        if not frames:
            raise RuntimeError(f"Could not read frames from {video_path}")
        return frames
    finally:
        cap.release()


def _label_from_filename(filename):
    name = os.path.splitext(filename)[0]
    if "_PRACTICE_" in name:
        return name.split("_PRACTICE_")[0]
    return name


train_feature_vectors = []
train_labels = []

if os.path.isdir(TRAIN_DIR):
    extractor = HandShapeFeatureExtractor.get_instance()
    train_videos = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".mp4")]
    train_videos.sort()
    for fname in train_videos:
        fpath = os.path.join(TRAIN_DIR, fname)
        frames = _get_sample_frames_color(fpath, num_samples=5)
        feats = [extractor.extract_feature(fr).reshape(-1) for fr in frames]
        feature = np.mean(feats, axis=0)
        train_feature_vectors.append(feature)
        train_labels.append(_label_from_filename(fname))

train_feature_vectors = np.array(train_feature_vectors)
train_labels = np.array(train_labels)




# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
test_feature_vectors = []
test_labels = []
test_filenames = []

test_dir, test_videos = _list_videos(
    TEST_DIR,
    fallback_dirs=[
        os.path.join(BASE_DIR, "TestData"),
        os.path.join(BASE_DIR, "test"),
        os.path.join(BASE_DIR, "test", "testdata"),
        os.path.join(BASE_DIR, "test", "TestData"),
    ],
)

if test_videos:
    extractor = HandShapeFeatureExtractor.get_instance()
    # If test_videos are full paths from recursive search, keep as-is.
    if test_dir is not None:
        test_videos.sort()
        video_paths = [os.path.join(test_dir, f) for f in test_videos]
        video_names = test_videos
    else:
        video_paths = test_videos
        video_names = [os.path.basename(p) for p in test_videos]

    for fpath, fname in zip(video_paths, video_names):
        frames = _get_sample_frames_color(fpath, num_samples=5)
        feats = [extractor.extract_feature(fr).reshape(-1) for fr in frames]
        feature = np.mean(feats, axis=0)
        test_feature_vectors.append(feature)
        test_labels.append(_label_from_filename(fname))
        test_filenames.append(fname)

test_feature_vectors = np.array(test_feature_vectors)
test_labels = np.array(test_labels)




# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
output_label_map = {
    "Num0": 0,
    "Num1": 1,
    "Num2": 2,
    "Num3": 3,
    "Num4": 4,
    "Num5": 5,
    "Num6": 6,
    "Num7": 7,
    "Num8": 8,
    "Num9": 9,
    "FanDown": 10,  # Decrease Fan Speed
    "FanOff": 11,
    "FanOn": 12,
    "FanUp": 13,    # Increase Fan Speed
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16,
}

results = []

if train_feature_vectors.size > 0 and test_feature_vectors.size > 0:
    train_vectors = train_feature_vectors
    test_vectors = test_feature_vectors

    train_norms = np.linalg.norm(train_vectors, axis=1)
    train_norms[train_norms == 0] = 1e-8

    for test_vec in test_vectors:
        test_norm = np.linalg.norm(test_vec)
        if test_norm == 0:
            test_norm = 1e-8
        cosine_sim = np.dot(train_vectors, test_vec) / (train_norms * test_norm)
        cosine_dist = 1.0 - cosine_sim
        best_idx = int(np.argmin(cosine_dist))
        best_label = train_labels[best_idx]
        if best_label not in output_label_map:
            raise ValueError(f"Unknown gesture label: {best_label}")
        results.append(output_label_map[best_label])

results_path = "Results.csv"
with open(results_path, "w") as f:
    for label in results:
        f.write(f"{label}\n")
