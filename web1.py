# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import os
import json
import math
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Basketball Shot Tracker")

st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

    /* Overall background */
    .stApp {
        background: linear-gradient(135deg, #0d0d0d, #1a1a1a);
        color: #f2f2f2;
        font-family: 'Orbitron', sans-serif;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(90deg, #ff6600, #ff3300);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 {
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    /* Card style containers */
    .block-container {
        padding-top: 1rem;
    }
    .stCard {
        background: #1e1e1e;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
        margin-bottom: 2rem;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #ff6600, #ff3300);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #ff3300, #ff6600);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111;
        padding: 1rem;
    }
    section[data-testid="stSidebar"] h2 {
        color: #ff6600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section
st.markdown(
    """
    <div class="hero">
        <h1>Basketball Shot Tracker</h1>
        <p>Upload your shot. Get pro-level feedback. Train smarter.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Helpers: geometry / angle
# ----------------------------
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba * mag_bc == 0:
        return None
    cosine_angle = max(min(dot / (mag_ba * mag_bc), 1), -1)
    return math.degrees(math.acos(cosine_angle))

# ----------------------------
# Coaching advice
# ----------------------------
def give_advice(report):
    advice = []

    # Launch angle
    if "launch_angle" in report:
        angle = report["launch_angle"]
        if angle < 40:
            advice.append(f"Your release angle was {angle:.1f}°. Try releasing higher (ideal is 45–55°).")
        elif angle > 55:
            advice.append(f"Your release angle was {angle:.1f}°. Try lowering it slightly to stay in the 45–55° range.")
        else:
            advice.append(f"Nice! Your release angle was {angle:.1f}°, which is in the ideal range (45–55°).")

    # Arc height
    if report.get("arc_height_px") is not None:
        arc = report["arc_height_px"]
        if arc < 50:
            advice.append("The arc of your shot was quite flat. Try following through more to add height.")
        elif arc > 150:
            advice.append("Your shot had a very high arc. This can reduce consistency — try smoothing it out.")
        else:
            advice.append("Your arc height looked solid.")

    # Elbow advice (study-based)
    prep_target = 85.1
    release_target = 159.6

    if "right_elbow_prep_angle" in report and report["right_elbow_prep_angle"] is not None:
        prep = report["right_elbow_prep_angle"]
        if abs(prep - prep_target) <= 10:
            advice.append(f"Prep phase elbow angle was {prep:.1f}°, close to the proficient ~85°.")
        elif prep < prep_target:
            advice.append(f"Your prep elbow angle was {prep:.1f}° (a bit too tight). Open it closer to ~85°.")
        else:
            advice.append(f"Your prep elbow angle was {prep:.1f}° (too open). Aim for ~85° bend before release.")

    if "right_elbow_release_angle" in report and report["right_elbow_release_angle"] is not None:
        rel = report["right_elbow_release_angle"]
        if abs(rel - release_target) <= 10:
            advice.append(f"Release elbow angle was {rel:.1f}°, right near the proficient ~160°.")
        elif rel < release_target:
            advice.append(f"Your release elbow angle was {rel:.1f}° (too low). Extend more fully to reach ~160°.")
        else:
            advice.append(f"Your release elbow angle was {rel:.1f}° (overextended). Aim to finish closer to ~160°.")

    return advice

# ----------------------------
# Load models once (cached)
# ----------------------------
@st.cache_resource
def load_models(ball_weights: str = "yolov8n.pt", pose_weights: str = "yolov8n-pose.pt"):
    st.session_state["model_load_time"] = datetime.now().isoformat()
    ball_model = YOLO(ball_weights)
    pose_model = YOLO(pose_weights)
    return ball_model, pose_model

# ----------------------------
# Video processing (your pipeline)
# ----------------------------
def process_video(video_path, ball_model, pose_model, show_progress=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = out_tmp.name
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    trajectory = []
    analysis_data = []

    body_connections = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
        (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)
    ]

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    progress = st.progress(0) if show_progress and total_frames else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        frame_rec = {
            "frame": frame_num,
            "ball": None,
            "nearest_wrist": None,
            "wrist_ball_dist": None,
            "right_elbow_angle": None,
            "left_elbow_angle": None
        }

        # Ball detection
        ball_results = ball_model(frame)[0]
        for box in getattr(ball_results, "boxes", []):
            cls_id = int(box.cls[0])
            label = ball_model.names[cls_id]
            if label == "sports ball" or label.lower().count("ball"):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = (x2 - x1), (y2 - y1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trajectory.append((cx, cy))
                frame_rec["ball"] = {"cx": cx, "cy": cy, "w": w, "h": h}
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                break

        # Draw trajectory
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

        # Pose detection
        pose_results = pose_model(frame)[0]
        nearest_wrist = None
        nearest_dist = None

        for person in getattr(pose_results, "keypoints", []):
            try:
                keypoints = person.xy[0].tolist()
            except Exception:
                continue

            # Skeleton
            for i, j in body_connections:
                if i < len(keypoints) and j < len(keypoints):
                    xi, yi = int(keypoints[i][0]), int(keypoints[i][1])
                    xj, yj = int(keypoints[j][0]), int(keypoints[j][1])
                    if xi > 0 and yi > 0 and xj > 0 and yj > 0:
                        cv2.line(frame, (xi, yi), (xj, yj), (0, 255, 255), 2)

            # Elbow angles
            def record_elbow(a_id, b_id, c_id, side):
                if all(k < len(keypoints) for k in [a_id, b_id, c_id]):
                    a, b, c = keypoints[a_id], keypoints[b_id], keypoints[c_id]
                    if all(p[0] > 0 and p[1] > 0 for p in [a, b, c]):
                        ang = calculate_angle(a, b, c)
                        if ang is not None:
                            bx, by = int(b[0]), int(b[1])
                            cv2.putText(frame, f"{int(ang)}°", (bx + 10, by - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            frame_rec[f"{side}_elbow_angle"] = round(ang, 2)

            record_elbow(6, 8, 10, "right")
            record_elbow(5, 7, 9, "left")

            # Nearest wrist to ball
            if frame_rec["ball"] is not None:
                cx, cy = frame_rec["ball"]["cx"], frame_rec["ball"]["cy"]
                for wid in (9, 10):
                    if wid < len(keypoints):
                        wx, wy = keypoints[wid][0], keypoints[wid][1]
                        if wx > 0 and wy > 0:
                            d = math.hypot(wx - cx, wy - cy)
                            if nearest_dist is None or d < nearest_dist:
                                nearest_dist = d
                                nearest_wrist = (int(wx), int(wy))

        if nearest_wrist is not None:
            frame_rec["nearest_wrist"] = [nearest_wrist[0], nearest_wrist[1]]
            frame_rec["wrist_ball_dist"] = round(float(nearest_dist), 3)
            cv2.circle(frame, tuple(nearest_wrist), 5, (255, 255, 0), -1)
            cv2.line(frame, (frame_rec["ball"]["cx"], frame_rec["ball"]["cy"]),
                     tuple(nearest_wrist), (255, 255, 0), 1)

        analysis_data.append(frame_rec)
        out.write(frame)

        if progress:
            progress.progress(frame_num / total_frames)

    cap.release()
    out.release()

    # Write JSON
    out_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(out_json.name, "w") as f:
        json.dump({
            "video": video_path,
            "fps": fps,
            "processed_at": datetime.now().isoformat(),
            "frames": analysis_data
        }, f, indent=2)

    return output_video_path, out_json.name, analysis_data

# ----------------------------
# Shot analysis
# ----------------------------
def analyze_shot(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    fps = data.get("fps", 30.0)

    ball_frames = [fr for fr in frames if fr.get("ball") is not None]
    if len(ball_frames) < 6:
        return {"error": f"Only {len(ball_frames)} detections found. Need >=6."}

    xs = np.array([fr["ball"]["cx"] for fr in ball_frames], dtype=float)
    ys = np.array([fr["ball"]["cy"] for fr in ball_frames], dtype=float)
    ws = np.array([fr["ball"]["w"]  for fr in ball_frames], dtype=float)
    hs = np.array([fr["ball"]["h"]  for fr in ball_frames], dtype=float)
    dists = np.array([fr.get("wrist_ball_dist", np.nan) for fr in ball_frames], dtype=float)
    r = 0.25*(ws+hs)

    # release detection heuristic
    k = 1.7
    N = 3
    def finite_second_diff(y):
        if len(y) < 3: return np.array([])
        return y[2:] - 2*y[1:-1] + y[:-2]

    release_idx = None
    sep_mask = dists > (k * r)
    for i in range(len(xs) - N):
        if np.all(sep_mask[i:i+N]) and not np.any(np.isnan(dists[i:i+N])):
            release_idx = i
            break

    stab_win = 5
    stab_tol = 3.0
    if release_idx is not None and release_idx + stab_win + 2 < len(ys):
        yseg = ys[release_idx:release_idx + stab_win + 2]
        acc = finite_second_diff(yseg)
        if len(acc) >= 3 and np.std(acc) > stab_tol:
            for j in range(release_idx+1, min(release_idx+6, len(ys)-stab_win-2)):
                yseg = ys[j:j + stab_win + 2]
                acc = finite_second_diff(yseg)
                if len(acc) >= 3 and np.std(acc) <= stab_tol:
                    release_idx = j
                    break

    if release_idx is None:
        acc = finite_second_diff(ys)
        if len(acc) >= 5:
            win = 5
            stds = np.array([np.std(acc[i:i+win]) for i in range(len(acc)-win+1)])
            cand = np.where(stds < stab_tol)[0]
            release_idx = int(cand[0]) if len(cand) else 1
        else:
            release_idx = 1

    # launch angle
    if 1 <= release_idx < len(xs)-1:
        dx = xs[release_idx + 1] - xs[release_idx - 1]
        dy_img = ys[release_idx + 1] - ys[release_idx - 1]
        dy = -dy_img
        launch_angle = math.degrees(math.atan2(dy, dx))
    else:
        dx = xs[1] - xs[0]
        dy = -(ys[1] - ys[0])
        launch_angle = math.degrees(math.atan2(dy, dx))

    arc_height = None
    xp = yp = None
    if release_idx is not None and release_idx < len(xs) - 3:
        xs_rel = xs[release_idx:]
        ys_rel = ys[release_idx:]
        coeffs = np.polyfit(xs_rel, ys_rel, 2)
        a, b, c = coeffs
        xp = np.linspace(xs_rel.min(), xs_rel.max(), 100)
        yp = a * xp**2 + b * xp + c
        arc_height = float(ys_rel.max() - ys_rel.min())

    # Elbow angles summary
    right_angles = [fr["right_elbow_angle"] for fr in frames if fr.get("right_elbow_angle") is not None]
    avg_right_prep = None
    avg_right_release = None
    if right_angles:
        prep_window = right_angles[max(0, release_idx-5):release_idx]
        release_window = right_angles[release_idx:release_idx+3]
        if prep_window:
            avg_right_prep = float(np.mean(prep_window))
        if release_window:
            avg_right_release = float(np.mean(release_window))

    figs = []
    fig1 = plt.figure(figsize=(6,4))
    plt.scatter(xs, ys, label="All detections", alpha=0.5)
    if xp is not None:
        plt.plot(xp, yp, label="Fitted parabola")
        plt.scatter(xs[release_idx], ys[release_idx], s=80, marker='x', label="Release")
    plt.gca().invert_yaxis()
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.title("Ball Trajectory")
    plt.legend()
    figs.append(fig1)

   report = {
        "release_idx": int(release_idx),
        "launch_angle": float(launch_angle),
        "arc_height_px": arc_height,
        "num_ball_points": len(xs),
        "right_elbow_prep_angle": avg_right_prep,
        "right_elbow_release_angle": avg_right_release,
    }
 return {"report": report, "figs": figs}
        "release_idx": int(release_idx),
        "
