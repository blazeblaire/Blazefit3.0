# video_features.py (requires mediapipe & opencv)
import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None
import numpy as np
import pandas as pd

def extract_pose_features(video_path):
    if mp is None:
        # mediator fallback for environments without mediapipe
        return pd.DataFrame([{'stride_frequency':0.0,'hip_variance':0.0}])
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            frames.append({'left_hip_y': left_hip.y,
                           'right_hip_y': right_hip.y,
                           'left_ankle_y': left_ankle.y,
                           'right_ankle_y': right_ankle.y})
    cap.release()
    df = pd.DataFrame(frames)
    if df.empty:
        return pd.DataFrame([{'stride_frequency':0.0,'hip_variance':0.0}])
    df['ankle_diff'] = (df['left_ankle_y'] - df['right_ankle_y']).fillna(0)
    crossings = np.sum(np.abs(np.diff(np.sign(df['ankle_diff'].values)))>0)
    duration = len(df) / fps
    stride_freq = crossings / duration if duration>0 else 0.0
    hip_var = df[['left_hip_y','right_hip_y']].var().mean()
    return pd.DataFrame([{'stride_frequency': float(stride_freq), 'hip_variance': float(hip_var)}])

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    feats = extract_pose_features(args.video)
    feats.to_csv(args.out, index=False)
