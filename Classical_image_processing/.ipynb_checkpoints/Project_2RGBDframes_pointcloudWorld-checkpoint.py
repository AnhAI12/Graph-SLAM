import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# =====================================================================
# STEP 0: THAM SỐ NỘI CAMERA (INTRINSICS K) - lấy từ stream MÀU của D455
# =====================================================================
fx = 386.376220703125  # Tiêu cự theo chiều ngang (pixel)
fy = 385.9943542480469  # Tiêu cự theo chiều dọc (pixel)
cx = 325.576416015625  # Tâm quang học u0 (pixel)
cy = 241.9173126220703  # Tâm quang học v0 (pixel)

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

# Ảnh depth 16-bit lưu mm -> nhân 0.001 để ra mét
DEPTH_FACTOR = 0.001  # m/unit

# Lọc depth: D455 nhiễu mạnh khi xa, và = 0 khi không đo được
MIN_Z = 0.3  # m
MAX_Z = 4.0  # m

# Tham số bản đồ
VOXEL_SIZE = 0.01  # m, downsample point cloud (1 cm)
EVAL_THRESHOLD = 0.03  # m, khoảng cách tối đa để tính 2 điểm là "khớp" khi đánh giá

# Công tắc
SHOW_DEBUG = True  # hiện ảnh keypoints / matches
USE_ICP = True  # refine pose PnP bằng ICP rồi so sánh


# =====================================================================
# STEP 1: FEATURE DETECTION & DESCRIPTION (ORB)
# =====================================================================
def extract_features(img_color):
  """Feature extraction Keypoints và Descriptors by ORB."""
  orb = cv2.ORB_create(nfeatures=5000)
  keypoints, descriptors = orb.detectAndCompute(img_color, None)
  return keypoints, descriptors


# =====================================================================
# STEP 2: FEATURE MATCHING (KNN + RATIO TEST)
# =====================================================================
def match_features(desc1, desc2):
  """Ghép cặp đặc trưng giữa 2 ảnh dùng khoảng cách Hamming + ratio test."""
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
  knn_matches = bf.knnMatch(desc1, desc2, k=2)

  good_matches = []
  for pair in knn_matches:
    if len(pair) < 2:
      continue
    m, n = pair
    if m.distance < 0.75 * n.distance:
      good_matches.append(m)

  return good_matches


# =====================================================================
# HÀM PHỤ TRỢ: CHIẾU NGƯỢC 2D -> 3D DỰA VÀO ẢNH DEPTH
# =====================================================================
def pixel_to_3d(u, v, depth_img, K):
  """Chuyển pixel (u, v) + depth thành tọa độ 3D (X, Y, Z) trong hệ camera."""
  h, w = depth_img.shape
  u_int, v_int = int(round(u)), int(round(v))

  if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
    return None

  z_raw = depth_img[v_int, u_int]
  if z_raw == 0 or np.isnan(z_raw):
    return None  # Không có dữ liệu chiều sâu tại pixel này

  z = z_raw * DEPTH_FACTOR
  if not (MIN_Z < z < MAX_Z):  # [MỚI] bỏ điểm quá gần / quá xa (nhiễu)
    return None

  x = (u - K[0, 2]) * z / K[0, 0]
  y = (v - K[1, 2]) * z / K[1, 1]

  return np.array([x, y, z], dtype=np.float64)


# =====================================================================
# STEP 3 & 4: GEOMETRIC VERIFICATION & POSE ESTIMATION (PnP RANSAC)
# =====================================================================
def estimate_pose_rgbd(kp1, kp2, matches, depth1, K):
  """Ước lượng R, t từ Frame 1 sang Frame 2 bằng PnP + RANSAC.

  Kết quả thỏa:  X_frame2 = R @ X_frame1 + t
  """
  pts_3d = []  # Tọa độ 3D các điểm thuộc Frame 1
  pts_2d = []  # Tọa độ 2D tương ứng của điểm đó trên Frame 2
  valid_matches = []

  for m in matches:
    u1, v1 = kp1[m.queryIdx].pt
    u2, v2 = kp2[m.trainIdx].pt

    p3d = pixel_to_3d(u1, v1, depth1, K)

    if p3d is not None:
      pts_3d.append(p3d)
      pts_2d.append([u2, v2])
      valid_matches.append(m)

  pts_3d = np.array(pts_3d, dtype=np.float64)
  pts_2d = np.array(pts_2d, dtype=np.float64)

  if len(pts_3d) < 6:
    print("Không đủ điểm 3D hợp lệ để giải PnP!")
    return None, None, None, None

  success, rvec, tvec, inliers = cv2.solvePnPRansac(
      objectPoints=pts_3d,
      imagePoints=pts_2d,
      cameraMatrix=K,
      distCoeffs=None,
      flags=cv2.SOLVEPNP_AP3P,
      reprojectionError=3.0,
      confidence=0.99,
  )

  if not success or inliers is None:
    print(" PnP RANSAC thất bại!")
    return None, None, None, None

  # [MỚI] Tinh chỉnh lại pose chỉ bằng các inliers (chính xác hơn kết quả RANSAC thô)
  idx = inliers.ravel()
  rvec, tvec = cv2.solvePnPRefineLM(pts_3d[idx], pts_2d[idx], K, None, rvec, tvec)

  R, _ = cv2.Rodrigues(rvec)
  t = tvec

  return R, t, valid_matches, inliers


# =====================================================================
# [MỚI] STEP 5: TẠO POINT CLOUD TỪ ẢNH RGB-D (vector hóa, không dùng vòng lặp)
# =====================================================================
def depth_to_pointcloud(color, depth, K):
  """Biến 1 cặp (RGB, depth) thành point cloud màu trong hệ camera của frame đó."""
  if color.shape[:2] != depth.shape[:2]:
    raise ValueError(
        f"Ảnh màu {color.shape[:2]} và depth {depth.shape[:2]} khác kích thước. "
        "Depth phải được align vào ảnh màu (rs.align) và không được resize."
    )

  h, w = depth.shape
  v, u = np.indices((h, w))
  z = depth.astype(np.float64) * DEPTH_FACTOR
  mask = (z > MIN_Z) & (z < MAX_Z)  # bỏ hố (z=0), điểm quá gần / quá xa

  x = (u - K[0, 2]) * z / K[0, 0]
  y = (v - K[1, 2]) * z / K[1, 1]
  pts = np.stack([x, y, z], axis=-1)[mask]
  cols = color[..., ::-1][mask].astype(np.float64) / 255.0  # BGR -> RGB

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pts)
  pcd.colors = o3d.utility.Vector3dVector(cols)
  return pcd


def make_T(R, t):
  """Ghép R (3x3), t (3x1) thành ma trận 4x4."""
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = np.asarray(t).ravel()
  return T


# =====================================================================
# [MỚI] STEP 6: ĐÁNH GIÁ & ICP REFINE
# =====================================================================
def evaluate(src, tgt, T_tgt_src, label):
  """fitness: tỉ lệ điểm khớp (cao tốt). rmse: sai số trung bình (thấp tốt)."""
  r = o3d.pipelines.registration.evaluate_registration(
      src, tgt, EVAL_THRESHOLD, T_tgt_src
  )
  print(f"  {label:<22} fitness = {r.fitness:.3f}   inlier RMSE = {r.inlier_rmse * 100:.2f} cm")
  return r


def refine_icp(src, tgt, T_init):
  """ICP point-to-plane, bắt đầu từ pose PnP."""
  search = o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
  src.estimate_normals(search)
  tgt.estimate_normals(search)
  res = o3d.pipelines.registration.registration_icp(
      src, tgt, 0.05, T_init,
      o3d.pipelines.registration.TransformationEstimationPointToPlane(),
      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
  )
  return res.transformation


def copy_pcd(pcd):
  """Open3D PointCloud không có .clone() -> dùng deepcopy."""
  return copy.deepcopy(pcd)


def show_cloud(clouds, title):
  """Camera hệ OpenCV có y hướng xuống, z hướng tới -> lật lại cho dễ nhìn."""
  flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
  disp = [copy_pcd(c).transform(flip) for c in clouds]
  axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
  disp.append(axes)
  o3d.visualization.draw_geometries(disp, window_name=title)


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def run(data_path, show=True):
  color1 = cv2.imread(os.path.join(data_path, 'color_0.png'))
  color2 = cv2.imread(os.path.join(data_path, 'color_1.png'))
  depth1 = cv2.imread(os.path.join(data_path, 'depth_0.png'), cv2.IMREAD_UNCHANGED)
  depth2 = cv2.imread(os.path.join(data_path, 'depth_1.png'), cv2.IMREAD_UNCHANGED)
  if any(x is None for x in (color1, color2, depth1, depth2)):
    raise FileNotFoundError(f"Thiếu ảnh trong {data_path} (color_0/1.png, depth_0/1.png)")
  print("depth dtype:", depth1.dtype, "(phải là uint16)")

  # --- STEP 1: Feature Extraction ---
  kp1, desc1 = extract_features(color1)
  kp2, desc2 = extract_features(color2)
  print(f"Frame 1: {len(kp1)} keypoints | Frame 2: {len(kp2)} keypoints")

  # --- STEP 2: Feature Matching ---
  matches = match_features(desc1, desc2)
  print(f"Số cặp khớp (sau ratio test): {len(matches)}")

  # --- STEP 3 & 4: Pose Estimation qua PnP RANSAC ---
  R, t, valid_matches, inliers = estimate_pose_rgbd(kp1, kp2, matches, depth1, K)
  if R is None:
    return None

  T_21 = make_T(R, t)  # đưa điểm từ hệ camera 1 sang hệ camera 2
  T_12 = np.linalg.inv(T_21)  # đưa điểm từ hệ camera 2 về hệ camera 1 (= hệ world)
  angle = np.degrees(np.linalg.norm(cv2.Rodrigues(T_12[:3, :3])[0]))

  print("\nTƯ THẾ CAMERA 2 TRONG HỆ CAMERA 1 (x phải, y xuống, z tới):")
  print("t (m) =", np.round(T_12[:3, 3], 4), f"| khoảng dịch = {np.linalg.norm(T_12[:3, 3]):.3f} m")
  print(f"Góc xoay = {angle:.2f} độ")
  print(f"Inliers: {len(inliers)} / {len(valid_matches)}")

  if show:
    inlier_matches = [valid_matches[i[0]] for i in inliers]
    img_matches = cv2.drawMatches(
        color1, kp1, color2, kp2, inlier_matches[:200], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"Inlier matches ({min(200, len(inlier_matches))} of {len(inlier_matches)} shown)")
    plt.axis("off")
    plt.show()

  # --- STEP 5: Point cloud mỗi frame (trong hệ camera của chính nó) ---
  pcd1 = depth_to_pointcloud(color1, depth1, K).voxel_down_sample(VOXEL_SIZE)
  pcd2 = depth_to_pointcloud(color2, depth2, K).voxel_down_sample(VOXEL_SIZE)
  print(f"\nPoint cloud sau downsample: frame 1 = {len(pcd1.points)}, frame 2 = {len(pcd2.points)}")

  # --- STEP 6: Đánh giá độ khớp của 2 cloud (world = hệ camera 1) ---
  # evaluate_registration(source, target, thr, T): T biến source -> target
  print("\nĐÁNH GIÁ ĐỘ KHỚP (cloud 2 -> cloud 1):")
  evaluate(pcd2, pcd1, np.eye(4), "Không dùng pose (I)")
  evaluate(pcd2, pcd1, T_12, "Pose PnP")

  T_best, label = T_12, "PnP"
  if USE_ICP:
    T_icp = refine_icp(pcd2, pcd1, T_12)
    r_icp = evaluate(pcd2, pcd1, T_icp, "PnP + ICP refine")
    r_pnp = o3d.pipelines.registration.evaluate_registration(pcd2, pcd1, EVAL_THRESHOLD, T_12)
    if r_icp.fitness >= r_pnp.fitness:  # chỉ nhận ICP nếu không làm tệ đi
      T_best, label = T_icp, "PnP+ICP"
    dt = np.linalg.norm(T_icp[:3, 3] - T_12[:3, 3]) * 100
    print(f"  ICP đã chỉnh vị trí thêm {dt:.2f} cm so với PnP")

  # --- STEP 7: Đưa cloud 2 về world và gộp vào global map ---
  pcd2_world = copy_pcd(pcd2).transform(T_best)  # X_world = T_12 @ X_cam2
  global_map = pcd1 + pcd2_world
  global_map, _ = global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
  global_map = global_map.voxel_down_sample(VOXEL_SIZE)
  o3d.io.write_point_cloud("global_map.ply", global_map)
  print(f"\nGlobal map ({label}): {len(global_map.points)} điểm -> đã lưu global_map.ply")

  if show:
    # (a) Kiểm tra căn chỉnh: cloud 1 màu đỏ, cloud 2 màu xanh
    # Nếu đúng, tường/sàn/bàn phải hòa vào nhau thành một lớp.
    a = copy_pcd(pcd1).paint_uniform_color([1, 0.2, 0.2])
    b_no = copy_pcd(pcd2).paint_uniform_color([0.2, 0.2, 1])
    b_yes = copy_pcd(pcd2_world).paint_uniform_color([0.2, 0.2, 1])
    show_cloud([a, b_no], "KHÔNG dùng pose (2 lớp lệch nhau là bình thường)")
    show_cloud([a, b_yes], f"Sau khi dùng pose {label} (vật tĩnh phải khớp)")
    # (b) Bản đồ màu thật
    show_cloud([global_map], "Global map (màu thật)")

  return T_best


if __name__ == "__main__":
  DATA_PATH = os.path.join(os.getcwd(), 'images')
  run(DATA_PATH, show=SHOW_DEBUG)