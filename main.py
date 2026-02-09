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

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
def _get_middle_frame_gray(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        else:
            middle = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Could not read frame from {video_path}")
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return gray
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
        frame_gray = _get_middle_frame_gray(fpath)
        feature = extractor.extract_feature(frame_gray)
        train_feature_vectors.append(feature.reshape(-1))
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

if os.path.isdir(TEST_DIR):
    extractor = HandShapeFeatureExtractor.get_instance()
    test_videos = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".mp4")]
    test_videos.sort()
    for fname in test_videos:
        fpath = os.path.join(TEST_DIR, fname)
        frame_gray = _get_middle_frame_gray(fpath)
        feature = extractor.extract_feature(frame_gray)
        test_feature_vectors.append(feature.reshape(-1))
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
