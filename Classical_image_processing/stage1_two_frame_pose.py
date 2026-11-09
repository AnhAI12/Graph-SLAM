"""
Stage 1: relative camera pose from TWO RGB-D images (ORB + depth back-projection + PnP RANSAC).

Usage:
  python stage1_two_frame_pose.py live              # capture 2 frames from a D455, then compute pose
  python stage1_two_frame_pose.py files             # reuse the frames saved in ./captures

Live controls: SPACE = grab frame (first A, then B), q = quit.
Tip: for frame B, move the camera a little (5-30 cm, small rotation) and keep the scene overlapping.

Install:  pip install opencv-python numpy pyrealsense2
"""
import argparse
import json
import os

import cv2
import numpy as np


# ---------------------------------------------------------------- capture
def capture_live(out_dir):
    import pyrealsense2 as rs

    os.makedirs(out_dir, exist_ok=True)
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)

    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()  # usually 0.001 m
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    info = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "depth_scale": depth_scale}
    json.dump(info, open(os.path.join(out_dir, "intrinsics.json"), "w"), indent=2)
    print("Intrinsics:", info)
    print("Distortion coeffs (color):", list(intr.coeffs))

    align = rs.align(rs.stream.color)  # put depth pixels on the color image grid
    labels, i = ["A", "B"], 0
    try:
        while i < 2:
            frames = align.process(pipe.wait_for_frames())
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())  # uint16, raw units

            preview = color.copy()
            cv2.putText(preview, f"SPACE = grab frame {labels[i]}   q = quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("color", preview)
            cv2.imshow("depth (aligned)", depth_vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                return False
            if k == 32:  # SPACE
                cv2.imwrite(os.path.join(out_dir, f"{labels[i]}_color.png"), color)
                cv2.imwrite(os.path.join(out_dir, f"{labels[i]}_depth.png"), depth)  # 16-bit PNG
                print(f"Saved frame {labels[i]}")
                i += 1
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    return True


# ---------------------------------------------------------------- core algorithm
def estimate_pose(gray_a, depth_a, gray_b, K, depth_scale, min_z=0.3, max_z=6.0):
    """
    Returns a dict with T_ba (4x4) such that  X_b = R * X_a + t,
    i.e. it maps 3D points from camera-A coordinates into camera-B coordinates.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Step 1: detect + describe
    orb = cv2.ORB_create(nfeatures=2000)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None:
        raise RuntimeError("No features found in one of the images.")
    print(f"Keypoints: A={len(kp_a)}  B={len(kp_b)}")

    # Step 2: match (Hamming distance for binary ORB descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_a, des_b), key=lambda m: m.distance)
    print(f"Raw matches: {len(matches)}")

    # Step 3: lift frame-A keypoints to 3D with depth (this is what replaces the essential matrix)
    h, w = depth_a.shape
    obj_pts, img_pts, valid = [], [], []
    for m in matches:
        u, v = kp_a[m.queryIdx].pt
        ui, vi = min(int(round(u)), w - 1), min(int(round(v)), h - 1)
        z = float(depth_a[vi, ui]) * depth_scale
        if not (min_z < z < max_z):  # skip holes (z = 0) and very far / noisy depth
            continue
        obj_pts.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
        img_pts.append(kp_b[m.trainIdx].pt)
        valid.append(m)
    print(f"Matches with valid depth: {len(valid)}")
    if len(valid) < 20:
        raise RuntimeError("Too few matches with valid depth. Try more texture / closer frames.")

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    # Step 4: PnP + RANSAC (rejects wrong matches AND solves for the pose)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj_pts, img_pts, K, None,
        iterationsCount=300, reprojectionError=3.0, confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 15:
        raise RuntimeError("PnP failed (too few inliers).")
    inl = inl.ravel()

    # Optional refinement using inliers only
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts[inl], img_pts[inl], K, None, rvec, tvec)

    # Quality check: reprojection error of the inliers (pixels)
    proj, _ = cv2.projectPoints(obj_pts[inl], rvec, tvec, K, None)
    err = np.linalg.norm(proj.reshape(-1, 2) - img_pts[inl], axis=1)

    R = cv2.Rodrigues(rvec)[0]
    T_ba = np.eye(4)
    T_ba[:3, :3], T_ba[:3, 3] = R, tvec.ravel()
    return {"T_ba": T_ba, "inliers": inl, "valid_matches": valid, "kp_a": kp_a, "kp_b": kp_b,
            "reproj_mean": float(err.mean()), "n_valid": len(valid)}


def report(res):
    T_ba = res["T_ba"]
    T_ab = np.linalg.inv(T_ba)  # pose of camera B expressed in camera A's frame
    R, t = T_ab[:3, :3], T_ab[:3, 3]
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    print("\n================ RESULT ================")
    print(f"Inliers: {len(res['inliers'])} / {res['n_valid']}  "
          f"(ratio {len(res['inliers']) / res['n_valid']:.2f})")
    print(f"Mean reprojection error of inliers: {res['reproj_mean']:.2f} px")
    print("\nCamera B pose in camera A frame (x right, y down, z forward):")
    print("R =\n", np.round(R, 4))
    print("t (meters) =", np.round(t, 4))
    print(f"\nTranslation distance: {np.linalg.norm(t):.3f} m")
    print(f"Rotation angle:       {angle:.2f} deg")
    print("========================================")
    print("Sanity check: does this match how far / how much you actually moved the camera?")


def show_matches(color_a, color_b, res):
    inl_matches = [res["valid_matches"][i] for i in res["inliers"]]
    vis = cv2.drawMatches(color_a, res["kp_a"], color_b, res["kp_b"], inl_matches[:100], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("inlier matches (max 100 shown) - press any key", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- main
def run(out_dir):
    info = json.load(open(os.path.join(out_dir, "intrinsics.json")))
    K = np.array([[info["fx"], 0, info["cx"]], [0, info["fy"], info["cy"]], [0, 0, 1]],
                 dtype=np.float64)
    ca = cv2.imread(os.path.join(out_dir, "A_color.png"))
    cb = cv2.imread(os.path.join(out_dir, "B_color.png"))
    da = cv2.imread(os.path.join(out_dir, "A_depth.png"), cv2.IMREAD_UNCHANGED)  # keep 16-bit
    if ca is None or cb is None or da is None:
        raise FileNotFoundError(f"Missing images in {out_dir}. Run in 'live' mode first.")

    res = estimate_pose(cv2.cvtColor(ca, cv2.COLOR_BGR2GRAY), da,
                        cv2.cvtColor(cb, cv2.COLOR_BGR2GRAY), K, info["depth_scale"])
    report(res)
    show_matches(ca, cb, res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["live", "files"])
    ap.add_argument("--dir", default="captures")
    a = ap.parse_args()
    if a.mode == "live":
        if not capture_live(a.dir):
            raise SystemExit("Capture cancelled.")
    run(a.dir)
