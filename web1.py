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
# Load models once (cached)
# ----------------------------
@st.cache_resource
def load_models(ball_weights: str = "yolov8n.pt", pose_weights: str = "yolov8n-pose.pt"):
    st.session_state["model_load_time"] = datetime.now().isoformat()
    ball_model = YOLO(ball_weights)
    pose_model = YOLO(pose_weights)
    return ball_model, pose_model

# ----------------------------
# Video processing (adapted from your script)
# ----------------------------
def process_video(video_path, ball_model, pose_model, show_progress=True):
    cap = cv2.VideoCapture(video_path)
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

    # loop frames
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
        ball_box = None
        for box in getattr(ball_results, "boxes", []):
            cls_id = int(box.cls[0])
            label = ball_model.names[cls_id]
            if label == "sports ball" or label.lower().count("ball"):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = (x2 - x1), (y2 - y1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trajectory.append((cx, cy))
                ball_box = (x1, y1, x2, y2)
                frame_rec["ball"] = {"cx": cx, "cy": cy, "w": w, "h": h}
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                break

        # draw trajectory
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

        # Pose detection
        pose_results = pose_model(frame)[0]

        nearest_wrist = None
        nearest_dist = None

        # The exact access pattern depends on ultralytics version; this mirrors your working code.
        for person in getattr(pose_results, "keypoints", []):
            # person.xy appears in your original; we try to reuse that
            try:
                keypoints = person.xy[0].tolist()
            except Exception:
                continue

            # skeleton
            for i, j in body_connections:
                if i < len(keypoints) and j < len(keypoints):
                    xi, yi = int(keypoints[i][0]), int(keypoints[i][1])
                    xj, yj = int(keypoints[j][0]), int(keypoints[j][1])
                    if xi > 0 and yi > 0 and xj > 0 and yj > 0:
                        cv2.line(frame, (xi, yi), (xj, yj), (0, 255, 255), 2)

            # elbow angles
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

            # nearest wrist to ball center
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

       #update progress bar
    if progress:
            progress.progress(frame_num / total_frames)

    cap.release()
    out.release()

    # write JSON
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
# Shot analysis (adapted)
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

    # release detection (same heuristic)
    k = 1.6
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
        # fallback
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

    # create plots (matplotlib figures)
    figs = []
    # trajectory + fit
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
    }
    return {"report": report, "figs": figs}

# ----------------------------
# Streamlit UI
# ----------------------------
st.header("Basketball shot tracker — Upload your Shot")

col1, col2 = st.columns([2,1])

with col2:
    model_size = st.selectbox("Model size (smaller = faster)", ["yolov8n", "yolov8s", "yolov8m", "yolov8x"], index=0)
    ball_weights = f"{model_size}.pt" if not model_size.endswith("-pose") else model_size
    pose_weights = f"{model_size}-pose.pt"
    st.write("Model files used:", ball_weights, "and", pose_weights)
    st.markdown("**Tip:** if this is slow, choose `yolov8n`.")

# load models (cached)
with st.spinner("Loading models (cached)..."):
    ball_model, pose_model = load_models(ball_weights, pose_weights)

with col1:
    video_file = st.file_uploader("Upload video (mp4/mov/avi) — or drag & drop", type=["mp4","mov","avi"])
    json_file = st.file_uploader("Or upload existing *_data.json to analyze", type=["json"])

# if JSON uploaded, analyze directly
if json_file is not None and video_file is None:
    tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_json.write(json_file.read())
    tmp_json.close()
    with st.spinner("Analyzing uploaded JSON..."):
        res = analyze_shot(tmp_json.name)
    if "error" in res:
        st.error(res["error"])
    else:
        st.success("Analysis complete")
        st.json(res["report"])
        for fig in res["figs"]:
            st.pyplot(fig)

# if video uploaded, run full pipeline
if video_file is not None:
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video.write(video_file.read())
    tmp_video.close()
    st.write("Saved uploaded video to:", tmp_video.name)

    if st.button("Run tracking + analysis"):
        with st.spinner("Running tracking (this can be slow — model inference on CPU may take a while)..."):
            tracked_path, json_path, analysis_data = process_video(tmp_video.name, ball_model, pose_model)
        st.success("Tracking finished")
        st.video(open(tracked_path, "rb").read())

        # show JSON download
        with open(json_path, "rb") as f:
            json_bytes = f.read()
        st.download_button("Download tracking JSON", data=json_bytes, file_name=os.path.basename(json_path), mime="application/json")

        # analyze
        with st.spinner("Running arc analysis..."):
            res = analyze_shot(json_path)
        if "error" in res:
            st.error(res["error"])
        else:
            st.json(res["report"])
            for fig in res["figs"]:
                st.pyplot(fig)

st.markdown("---")
st.caption("If it's broke, fix it")
