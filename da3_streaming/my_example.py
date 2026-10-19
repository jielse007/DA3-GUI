import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QSplitter, QTextEdit, QGroupBox, QTabWidget,
    QFrame, QSlider, QCheckBox, QComboBox, QLineEdit,
    QGridLayout, QScrollArea
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer, QTime, QRunnable, QThreadPool, QObject
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette

import os, glob, shutil, socket, threading, time
import numpy as np
import trimesh
from datetime import datetime


# ============================================================
# 信号发射器（QRunnable 不能直接有信号，通过这个中转）
# ============================================================
class UploadEmitter(QObject):
    progress = Signal(int)
    finished = Signal(str)
    error    = Signal(str)

class DA3Emitter(QObject):
    finished = Signal(str, str)
    progress = Signal(str, int)
    error   = Signal(str)

# ============================================================
# 文件复制任务（QRunnable，由线程池管理）
# ============================================================
class FileCopyRunnable(QRunnable):
    def __init__(self, file_list, dest_dir, emitter):
        super().__init__()
        self.file_list = file_list
        self.dest_dir  = dest_dir
        self.emitter   = emitter

    def run(self):
        try:
            total = len(self.file_list)
            for i, f in enumerate(self.file_list):
                shutil.copy(f, os.path.join(self.dest_dir, os.path.basename(f)))
                self.emitter.progress.emit(int((i + 1) / total * 100))
                QThread.msleep(5)
            self.emitter.finished.emit(self.dest_dir)
        except Exception as e:
            self.emitter.error.emit(str(e))


# ============================================================
# 视频抽帧任务（QRunnable，由线程池管理）
# ============================================================
class VideoFrameRunnable(QRunnable):
    def __init__(self, video_path, dest_dir, emitter, fps=2, max_width=640):
        super().__init__()
        self.video_path = video_path
        self.dest_dir   = dest_dir
        self.emitter    = emitter
        self.fps        = fps
        self.max_width  = max_width

    def run(self):
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.emitter.error.emit(f"无法打开视频: {self.video_path}")
                return
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            interval = max(1, int(video_fps / self.fps))
            saved = 0
            for idx in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                if idx % interval == 0:
                    h, w = frame.shape[:2]
                    if w > self.max_width:
                        frame = cv2.resize(frame, (self.max_width, int(h * w / self.max_width if h else 1)))
                    cv2.imwrite(os.path.join(self.dest_dir, f"frame_{saved:06d}.png"), frame)
                    saved += 1
                self.emitter.progress.emit(int(idx / total_frames * 100))
                QThread.msleep(1)
            cap.release()
            self.emitter.progress.emit(100)
            self.emitter.finished.emit(self.dest_dir)
        except Exception as e:
            import traceback
            self.emitter.error.emit(f"{e}\n{traceback.format_exc()}")


# ============================================================
# 持久摄像头深度计算线程（常驻，复用线程，避免频繁创建销毁）
# ============================================================
class PersistentCamWorker(QThread):
    finished = Signal(str)  # DA3 输出目录
    error    = Signal(str)

    def __init__(self, cached_model):
        super().__init__()
        self.cached_model = cached_model
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_frame = None
        self._running = True

    def submit_frame(self, frame_bgr):
        with self._lock:
            self._latest_frame = frame_bgr.copy()
            self._cond.notify()

    def stop(self):
        with self._lock:
            self._running = False
            self._cond.notify()

    def run(self):
        import tempfile, cv2, importlib.util
        while True:
            with self._lock:
                while self._latest_frame is None and self._running:
                    self._cond.wait()
                if not self._running:
                    break
                frame = self._latest_frame
                self._latest_frame = None

            try:
                # 保存帧到临时目录（DA3_Streaming 需要文件夹输入）
                tmp_dir = tempfile.mkdtemp(prefix="cam_da3_")
                img_path = os.path.join(tmp_dir, "frame_000000.png")
                cv2.imwrite(img_path, frame)

                # 加载 DA3_Streaming（仅首次，之后复用 sys.modules 缓存）
                _here = os.path.dirname(os.path.abspath(__file__))
                _src = os.path.join(_here, "..", "src")
                sys.path.insert(0, _src)
                sys.path.insert(0, _here)

                _spec = importlib.util.spec_from_file_location(
                    "da3_streaming_cam", os.path.join(_here, "da3_streaming.py"))
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                from loop_utils.config_utils import load_config
                sys.path.remove(_here)

                # 创建 DA3_Streaming 实例并运行（与主流程完全一致）
                config = load_config("da3_streaming/configs/base_config.yaml")
                da3 = _mod.DA3_Streaming(tmp_dir, os.path.join(tmp_dir, "out"), config)
                da3.run()
                da3.close()

                # 发出 DA3 输出目录
                output_dir = os.path.join(tmp_dir, "out")
                self.finished.emit(output_dir)
            except Exception as e:
                import traceback
                self.error.emit(f"{e}\n{traceback.format_exc()}")


# ============================================================
# RANSAC 地面平面检测 + 坐标轴，合并到 GLB（纯 numpy，无额外依赖）
# ============================================================
# 上轴：0=X, 1=Y, 2=Z
# DA3 输出在 npz_output_process.py 中做了 YZ 翻转（OpenCV→OpenGL），
# OpenGL 与 Babylon.js 惯例均为 Y 朝上，因此这里默认 Y。
# ⚠️ 若地面网格显示成"竖直的墙"，说明上轴判断错误 → 改成 2 再试。
_UP_AXIS = 1
# 水平性阈值：平面法向量与上轴夹角 < 20° 才算"水平面"（用于排除墙）
_HORIZ_COS = float(np.cos(np.radians(20.0)))
# 打印候选平面诊断信息（确认上轴 / 调参时可打开，稳定后可关）
_GROUND_DEBUG = False


def _refine_plane(pts):
    """对 inlier 点用 PCA 精修平面，返回 (单位法向量 n, 截距 d)，平面方程 n·x + d = 0"""
    c = pts.mean(axis=0)
    # 最小奇异向量即为平面法向量
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    return n, -float(n @ c)


def _fit_plane_ransac(pts, thresh, iters=800):
    """纯 numpy 的 3D 平面 RANSAC

    每次随机取 3 点 → 叉乘求过这 3 点的平面 → 统计到该平面距离 < thresh 的点数，
    保留 inlier 最多的那个平面，最后用全部 inlier 做 PCA 精修。
    返回 (单位法向量 n, 截距 d, inlier 布尔掩码)；失败时返回 (None, None, None)。
    """
    n_pts = len(pts)
    if n_pts < 3:
        return None, None, None

    best_cnt, best_n, best_d, best_mask = 0, None, None, None
    for _ in range(iters):
        p0, p1, p2 = pts[np.random.choice(n_pts, 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:          # 三点共线，无法确定平面
            continue
        n = n / norm
        d = -float(n @ p0)
        mask = np.abs(pts @ n + d) < thresh
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_n, best_d, best_mask = cnt, n, d, mask

    if best_mask is None or best_cnt < 3:
        return None, None, None

    # 用 inlier 全集精修（RANSAC 的 3 点解噪声大，PCA 拟合更准）
    n, d = _refine_plane(pts[best_mask])
    mask = np.abs(pts @ n + d) < thresh
    return n, d, mask


def _plane_grid_mesh(n, d, pts_in, grid=20):
    """在真 3D 平面 (n·x+d=0) 上生成矩形网格面片

    真 3D 平面无法用 z=f(x,y) 表达（墙/天花板会造成 x,y 一对多），
    因此必须先在平面内构造正交基 (u, v)，再把 inlier 投影到该基底求展布。
    """
    c = pts_in.mean(axis=0)                      # 平面中心
    # 构造平面内正交基：取与法向量最不平行的坐标轴做参考
    ref = np.zeros(3)
    ref[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    rel = pts_in - c
    su, sv = rel @ u, rel @ v
    u0, u1 = float(su.min()), float(su.max())
    v0, v1 = float(sv.min()), float(sv.max())
    # 外扩 10% 边距
    mu, mv = 0.1 * (u1 - u0), 0.1 * (v1 - v0)
    u0, u1, v0, v1 = u0 - mu, u1 + mu, v0 - mv, v1 + mv

    us = np.linspace(u0, u1, grid)
    vs = np.linspace(v0, v1, grid)
    U, V = np.meshgrid(us, vs)
    verts = (c[None, :] + U.ravel()[:, None] * u[None, :]
             + V.ravel()[:, None] * v[None, :])

    faces = []
    for i in range(grid - 1):
        for j in range(grid - 1):
            a = i * grid + j
            faces.append([a, a + 1, a + grid])
            faces.append([a + 1, a + grid + 1, a + grid])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    extent = max(u1 - u0, v1 - v0)
    return mesh, extent, c


def _detect_ground_plane(pts, thresh):
    """顺序 RANSAC 剥离候选平面，再按「水平性 / 低位性 / 支撑性」选出地面

    ① 水平性  |n·up| > cos(20°)      → 排除墙
    ② 低位性  平面高度 < 点云质心高度   → 排除天花板
    ③ 支撑性  平面上方点数 >> 下方点数   → 排除天花板/悬空面
    最后在通过者中取「位置最低」的 → 地面是支撑一切的最低水平面，桌面必在其上方

    返回 (选中的平面 dict 或 None, 全部候选列表)
    """
    up = np.zeros(3); up[_UP_AXIS] = 1.0
    all_h = pts[:, _UP_AXIS]                     # 全量高度，用于低位性/支撑性统计
    # 用 1%/99% 分位数而非极值，避免离群点把高度区间撑坏
    h_lo = float(np.percentile(all_h, 1))
    h_hi = float(np.percentile(all_h, 99))
    span = max(h_hi - h_lo, 1e-6)

    work, cands = pts, []
    for _ in range(5):                            # 最多剥离 5 个平面
        if len(work) < 500:
            break
        n, d, mask = _fit_plane_ransac(work, thresh)
        if n is None or int(mask.sum()) < 500:
            break

        plane_pts = work[mask]
        if n @ up < 0:                            # 统一法向量朝上，便于比较高低
            n, d = -n, -d
        height   = float(plane_pts[:, _UP_AXIS].mean())
        cos_tilt = float(abs(n @ up))             # 1.0 = 完全水平
        # 支撑性在全量点云上统计（而非逐轮缩小的 work，否则语义随剥离进度漂移）
        above = int((all_h > height + thresh).sum())
        below = int((all_h < height - thresh).sum())

        # ① 水平性：过滤掉墙
        ok_tilt = cos_tilt > _HORIZ_COS
        # ② 低位性：地面不能是点云里最高的那一层
        #    用 h_hi 作上界而非"低于质心/中位数"——当地面就是全部点云时，
        #    地面高度恰等于中位数，任何以中位数为界的判据都会恒假而误杀
        ok_low = height < h_hi - 0.1 * span
        # ③ 支撑性：平面上方点数不少于下方 —— 地面支撑万物，天花板反之
        #    用 >= 而非倍数，保证"场景只有单一水平面"时不被误杀
        ok_supp = above >= below
        cands.append(dict(n=n, d=d, pts=plane_pts, cnt=int(mask.sum()),
                          height=height, cos_tilt=cos_tilt,
                          above=above, below=below,
                          ok=ok_tilt and ok_low and ok_supp))

        work = work[~mask]                        # 剥离该平面继续找下一个

    valid = [c for c in cands if c["ok"]]
    if not valid:
        return None, cands
    valid.sort(key=lambda c: (c["height"], -c["cnt"]))   # 最低者胜，同高取点多者
    return valid[0], cands


def _add_ground_plane(ply_path):
    """对点云做 RANSAC 地面检测，生成地面面片+坐标轴，合并到 GLB"""
    # 1. 加载点云
    scene = trimesh.load(ply_path)
    # 点云可能是 PointCloud 或 Trimesh，统一获取顶点
    def _safe_colors(arr, n_pts):
        """取顶点颜色；形状不对（无颜色/维度不符）时返回 None，避免 IndexError"""
        if arr is None:
            return None
        c = np.asarray(arr)
        return c[:, :3] if (c.ndim == 2 and len(c) == n_pts) else None

    if isinstance(scene, trimesh.Trimesh):
        pts = np.asarray(scene.vertices)
        colors = _safe_colors(getattr(scene.visual, 'vertex_colors', None), len(pts))
    elif isinstance(scene, trimesh.PointCloud):
        pts = np.asarray(scene.vertices)
        colors = _safe_colors(getattr(scene, 'colors', None), len(pts))
    else:
        return

    # 采样点云用于显示（最多 200k 点）
    if len(pts) > 200000:
        idx_show = np.random.choice(len(pts), 200000, replace=False)
        pc_pts = pts[idx_show]
        pc_col = colors[idx_show] if colors is not None and len(colors) == len(pts) else None
    else:
        pc_pts = pts
        pc_col = colors

    # 点云用 trimesh.PointCloud 保留
    pc_mesh = trimesh.PointCloud(vertices=pc_pts)
    if pc_col is not None and len(pc_col) == len(pc_pts):
        pc_mesh.colors = pc_col

    # 2. 地面检测（纯 numpy 3D RANSAC + 三条件约束）
    n_fit = min(50000, len(pts))
    sample = pts[np.random.choice(len(pts), n_fit, replace=False)]

    # 距离阈值随场景尺度自适应（DA3 输出尺度是 sim3 对齐后的任意尺度）
    diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
    thresh = float(np.clip(0.008 * diag, 0.01, 0.2))

    ground, cands = _detect_ground_plane(sample, thresh)

    if _GROUND_DEBUG:
        axis_names = "XYZ"
        print(f"[地面检测] 上轴={axis_names[_UP_AXIS]}  阈值={thresh:.4f}  "
              f"点数={len(sample)}")
        for a in range(3):
            print(f"            {axis_names[a]} 范围: "
                  f"{pts[:, a].min():.3f} ~ {pts[:, a].max():.3f}  "
                  f"(质心 {pts[:, a].mean():.3f})")
        for i, c in enumerate(cands):
            mark = "[通过]" if c["ok"] else "[淘汰]"   # 避免 GBK 控制台编码报错
            print(f"            平面{i}: 点数={c['cnt']:>6} 高度={c['height']:.3f} "
                  f"水平度={c['cos_tilt']:.3f} 上方={c['above']} 下方={c['below']} {mark}")
        if ground:
            print(f"            -> 选中: 点数={ground['cnt']} 高度={ground['height']:.3f}")
        else:
            print("            -> 未找到符合约束的地面平面，退化为场景中心坐标轴")
            print(f"               提示: 若确认场景里有地面，可能上轴判断有误，"
                  f"可把 _UP_AXIS 改成 {( _UP_AXIS + 1) % 3} 或 {( _UP_AXIS + 2) % 3} 再试")

    # 3. 地面网格 / 坐标轴
    plane_mesh = None
    if ground is not None:
        plane_mesh, extent, center = _plane_grid_mesh(
            ground["n"], ground["d"], ground["pts"])
        plane_mesh.visual.vertex_colors = [80, 200, 80, 50]
        axis_len = extent * 0.2
    else:
        # 兜底：场景中心放一组坐标轴，不画地面
        center = pts.mean(axis=0)
        axis_len = float(np.ptp(pts, axis=0).max()) * 0.3

    # 4. 坐标轴（世界轴对齐，原点落在地面中心）
    origin = np.asarray(center)
    colors_axis = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    axes_meshes = []
    directions = [[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]
    for i, (d_vec, col) in enumerate(zip(directions, colors_axis)):
        seg = trimesh.creation.cylinder(radius=axis_len * 0.02, height=axis_len, sections=8)
        if i == 0:
            seg.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
        elif i == 1:
            seg.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        seg.apply_translation(origin + np.array(d_vec) * 0.5)
        seg.visual.vertex_colors = col
        axes_meshes.append(seg)

    # 5. 用 trimesh.Scene 包装所有几何体，导出 GLB
    glb_path = os.path.splitext(ply_path)[0] + ".glb"
    export_scene = trimesh.Scene()
    export_scene.add_geometry(pc_mesh, node_name="pointcloud")
    if plane_mesh is not None:
        export_scene.add_geometry(plane_mesh, node_name="ground")
    for i, ax in enumerate(axes_meshes):
        export_scene.add_geometry(ax, node_name=f"axis_{i}")
    export_scene.export(glb_path)


# ============================================================
# 后台工作线程（线程池 QRunnable）
# ============================================================
# Open3D 查看器线程（RANSAC 地面检测 + 坐标轴）
# ============================================================
class Open3DViewer(QThread):
    hwnd_ready = Signal(int)  # 窗口创建完成后发送 HWND 给主线程嵌入

    def __init__(self, ply_path):
        super().__init__()
        self.ply_path = ply_path

    def run(self):
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(self.ply_path)
        if not pcd.has_points():
            return
        pcd = pcd.voxel_down_sample(voxel_size=0.01)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=500)
        a, b, c_pl, d = plane_model
        inlier_cloud = pcd.select_by_index(inliers)
        bbox = inlier_cloud.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()

        plane_mesh = o3d.geometry.TriangleMesh.create_box(
            ext[0] * 1.1, 0.01, ext[2] * 1.1)
        plane_mesh.translate(bbox.get_center() - [0, ext[1] / 2, 0])
        plane_mesh.paint_uniform_color([0.2, 0.7, 0.2])

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=ext.max() * 0.3, origin=bbox.get_center())

        viewer = o3d.visualization.Visualizer()
        viewer.create_window(
            window_name="DA3-Open3D  [ L:Rotate  R:Pan  Scroll:Zoom ]",
            width=800, height=600, visible=True)

        viewer.add_geometry(pcd)
        viewer.add_geometry(plane_mesh)
        viewer.add_geometry(axes)

        opt = viewer.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.15, 0.15, 0.18])
        opt.show_coordinate_frame = False

        # 用 Win32 API 找到窗口句柄（枚举窗口，匹配标题前缀）
        import ctypes
        user32 = ctypes.windll.user32
        time.sleep(0.3)

        found = []
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_cb(h, l):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(h, buf, 256)
            if buf.value.startswith("DA3-Open3D"):
                found.append(h)
                return False
            return True
        user32.EnumWindows(WNDENUMPROC(enum_cb), ctypes.c_void_p())

        if found:
            self.hwnd_ready.emit(found[0])
        viewer.run()
        viewer.destroy_window()


class DA3Runnable(QRunnable):
    def __init__(self, image_dir, output_dir, config_path, emitter, sample_ratio=1.0):
        super().__init__()
        self.image_dir    = image_dir
        self.output_dir   = output_dir
        self.config_path  = config_path
        self.emitter      = emitter
        self.sample_ratio = sample_ratio

    def _log(self, msg, pct):
        self.emitter.progress.emit(msg, pct)
        QThread.msleep(10)

    def run(self):
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            _src  = os.path.join(_here, "..", "src")
            sys.path.insert(0, _src)
            sys.path.insert(0, _here)

            import importlib.util

            _spec = importlib.util.spec_from_file_location(
                "da3_streaming", os.path.join(_here, "da3_streaming.py")
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            DA3_Streaming = _mod.DA3_Streaming
            from loop_utils.config_utils import load_config
            sys.path.remove(_here)

            config = load_config(self.config_path)

            # 获取所有图片文件路径
            imgs = (
                sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
                + sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
            )
            self._log(f"共 {len(imgs)} 帧", 10)

            # 创建DA3_Streaming实例
            da3 = DA3_Streaming(self.image_dir, self.output_dir, config)

            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            self._log("DA3 推理中...", 30)
            try:
                da3.run()
            finally:
                da3.close()
                sys.stdout = old_stdout
                # 手动清理临时目录（防止 da3.close 未覆盖的情况）
                import shutil as _shutil
                for _d in ["_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop"]:
                    _p = os.path.join(self.output_dir, _d)
                    if os.path.exists(_p):
                        _shutil.rmtree(_p, ignore_errors=True)

            self._log("融合点云并导出 GLB...", 80)

            # 加载 npz_output_process
            _npz_spec = importlib.util.spec_from_file_location(
                "npz_output_process", os.path.join(_here, "npz_output_process.py")
            )
            _npz_mod = importlib.util.module_from_spec(_npz_spec)
            _npz_spec.loader.exec_module(_npz_mod)

            results_dir = os.path.join(self.output_dir, "results_output")
            poses_file  = os.path.join(self.output_dir, "camera_poses.txt")
            ply_file    = os.path.join(self.output_dir, "combined.ply")

            if os.path.exists(results_dir) and os.path.exists(poses_file):
                _npz_mod.create_point_cloud(
                    npz_folder=results_dir,
                    pose_file=poses_file,
                    output_ply=ply_file,
                    conf_threshold_coef=0.5,
                    sample_ratio=self.sample_ratio,
                )
                self._log("完成", 100)
            else:
                self._log("完成（无融合文件）", 95)

            # === RANSAC 地面检测 + 坐标轴（增强3D显示） ===
            ply_file = os.path.join(self.output_dir, "combined.ply")
            if os.path.exists(ply_file):
                try:
                    self._log("RANSAC 检测地面...", 97)
                    _open_in_open3d(os.path.join(self.output_dir, 'combined.ply'))
                except Exception:
                    pass

            glb_files = sorted(glob.glob(os.path.join(self.output_dir, "*.glb")))
            glb_path = glb_files[-1] if glb_files else ""
            self.emitter.finished.emit(self.output_dir, glb_path)

        except Exception as e:
            import traceback
            self.emitter.error.emit(f"{e}\n{traceback.format_exc()}")


# ============================================================
# 主窗口
# ============================================================
class GradioLikeApp(QMainWindow):
    # ============================================================
    # 基本设置
    # ============================================================
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth-Anything-3") # 设置窗口标题为"Depth-Anything-3"
        self.setGeometry(40, 40, 1500, 900)     # 设置窗口位置(40,40)和大小(宽1500,高900)
                
        # ============================================================
        # 开始设置整个应用的样式表(CSS风格)
        # ============================================================
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;  /* 主窗口背景色：浅灰色 */
            }
            QGroupBox {
                font-weight: bold;           /* 分组框标题字体加粗 */
                border: 2px solid #d0d0d0;   /* 边框：2像素实线，浅灰色 */
                border-radius: 8px;          /* 边框圆角：8像素 */
                margin-top: 10px;            /* 顶部外边距：10像素 */
                padding-top: 10px;           /* 顶部内边距：10像素 */
                background-color: white;     /* 背景色：白色 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;   /* 标题位置相对于外边距 */
                left: 10px;                  /* 标题距离左边10像素 */
                padding: 0 5px 0 5px;        /* 标题内边距：上下0，左右5像素 */
                background-color: white;     /* 标题背景色：白色 */
            }
            QPushButton {
                background-color: #4CAF50;   /* 按钮背景色：绿色 */
                color: white;                /* 按钮文字颜色：白色 */
                border: none;                /* 无边框 */
                padding: 8px 16px;           /* 按钮内边距：上下8，左右16像素 */
                border-radius: 4px;          /* 按钮圆角：4像素 */
                font-weight: bold;           /* 字体加粗 */
                font-size: 13px;             /* 字体大小：13像素 */
            }
            QPushButton:hover {
                background-color: #45a049;   /* 鼠标悬停时的背景色：深绿色 */
            }
            QPushButton:pressed {
                background-color: #3d8b40;   /* 鼠标按下时的背景色：更深的绿色 */
            }
            QPushButton#danger {
                background-color: #f44336;   /* ID 为 "danger" 的按钮：红色 */
            }
            QPushButton#danger:hover {
                background-color: #da190b;   /* 红色按钮悬停时：深红色 */
            }
            QPushButton#secondary {
                background-color: #6c757d;   /* ID 为 "secondary" 的按钮：灰色 */
            }
            QPushButton#secondary:hover {
                background-color: #5a6268;   /* 灰色按钮悬停时：深灰色 */
            }
            QTextEdit, QLineEdit {
                border: 1px solid #d0d0d0;   /* 边框：1像素实线，浅灰色 */
                border-radius: 4px;          /* 圆角：4像素 */
                padding: 4px;                /* 内边距：4像素 */
                background-color: white;     /* 背景色：白色 */
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;   /* 标签页面板边框 */
                border-radius: 4px;          /* 圆角：4像素 */
                background-color: white;     /* 背景色：白色 */
            }
            QTabBar::tab {
                background-color: #f0f0f0;   /* 标签页背景色：浅灰色 */
                padding: 8px 16px;           /* 内边距：上下8，左右16像素 */
                margin-right: 2px;           /* 右边距：2像素 */
                border-top-left-radius: 4px; /* 左上角圆角：4像素 */
                border-top-right-radius: 4px;/* 右上角圆角：4像素 */
            }
            QTabBar::tab:selected {
                background-color: white;     /* 选中的标签页背景：白色 */
                border-bottom: 2px solid #4CAF50;  /* 底部边框：绿色强调线 */
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;   /* 鼠标悬停时的背景色 */
            }
            QSlider::groove:horizontal {
                height: 8px;                 /* 滑动条轨道高度：8像素 */
                background: #d0d0d0;         /* 轨道背景色：浅灰色 */
                border-radius: 4px;          /* 轨道圆角：4像素 */
            }
            QSlider::handle:horizontal {
                background: #4CAF50;         /* 滑块手柄颜色：绿色 */
                width: 18px;                 /* 手柄宽度：18像素 */
                height: 18px;                /* 手柄高度：18像素 */
                margin: -5px 0;              /* 外边距：上下-5，左右0 */
                border-radius: 9px;          /* 手柄圆角：9像素（圆形） */
            }
            QComboBox {
                border: 1px solid #d0d0d0;   /* 边框：1像素实线，浅灰色 */
                border-radius: 4px;          /* 圆角：4像素 */
                padding: 4px 8px;            /* 内边距：上下4，左右8像素 */
                background-color: white;     /* 背景色：白色 */
            }
            QComboBox:hover {
                border-color: #4CAF50;       /* 鼠标悬停时边框变绿 */
            }
            QCheckBox {
                spacing: 8px;                /* 复选框文字和勾选框之间的间距：8像素 */
            }
            .card {
                background-color: white;     /* 自定义类 "card" 的背景色：白色 */
                border: 1px solid #e0e0e0;   /* 边框：1像素实线，浅灰色 */
                border-radius: 8px;          /* 圆角：8像素 */
                padding: 12px;               /* 内边距：12像素 */
            }
            .title-label {
                font-size: 16px;             /* 自定义类 "title-label" 字体大小 */
                font-weight: bold;           /* 字体加粗 */
                color: #333;                 /* 文字颜色：深灰色 */
            }
            .subtitle-label {
                font-size: 12px;             /* 自定义类 "subtitle-label" 字体大小 */
                color: #666;                 /* 文字颜色：中灰色 */
            }
        """)

        central = QWidget()                                 # 创建一个中心部件(容器)
        self.setCentralWidget(central)                      # 将中心部件设置为窗口的中心区域
        main_layout = QVBoxLayout(central)                  # 创建垂直布局管理器，并关联到中心部件
        main_layout.setSpacing(10)                          # 设置布局内控件间距为10px
        main_layout.setContentsMargins(10, 10, 10, 10)      # 设置布局外边距：左10上10右10下10

        # self.create_header(main_layout)                     # 调用create_header方法创建头部区域并添加到主布局

        splitter = QSplitter(Qt.Horizontal)                 # 创建水平分割器(可拖拽调整左右比例)
        splitter.addWidget(self.create_left_panel())        # 添加左侧面板(调用create_left_panel方法)
        splitter.addWidget(self.create_right_panel())       # 添加右侧面板(调用create_right_panel方法)
        splitter.setSizes([200, 1200])                      # 设置左右面板初始宽度：左300px，右1100px
        main_layout.addWidget(splitter)                     # 将分割器添加到主布局

        self.create_status_bar()

        # 预加载模型（单次加载，后续推理复用）
        self._cached_model = None
        self._model_loaded = False
        self._get_cached_model()

    def _get_cached_model(self):
        """单次加载 DepthAnything3 模型，存入缓存"""
        if self._cached_model is not None:
            return self._cached_model
        import json, torch
        _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
        sys.path.insert(0, _src)
        from depth_anything_3.api import DepthAnything3
        from safetensors.torch import load_file
        with open("weights_small/config.json") as f:
            cfg = json.load(f)
        model = DepthAnything3(**cfg)
        weight = load_file("weights_small/model.safetensors")
        model.load_state_dict(weight, strict=False)
        model = model.cpu().eval()
        self._cached_model = model
        self._model_loaded = True
        return model

    # ============================================================
    # 创建顶部标题栏
    # ===========================================================
    def create_header(self, layout):
        w = QWidget()
        w.setStyleSheet("""
            background-color: white;            /* 背景色：白色 */
            border-bottom: 2px solid #e0e0e0;   /* 底部边框：2像素实线，浅灰色 */
            padding: 10px;                      /* 内边距：10像素 */
        """)
        hl = QHBoxLayout(w)     # QHBoxLayout()：创建水平布局，标题栏内容水平排列

        tl = QVBoxLayout()   # QVBoxLayout()：创建垂直布局，标题栏内容垂直排列
        #主标题
        title = QLabel("🎯 DA3 - Depth Anything V3")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        tl.addWidget(title)     # addWidget()：将标签添加到垂直布局中
        #副标题
        subtitle = QLabel("交互式三维浏览与深度预测工具")
        subtitle.setStyleSheet("font-size: 13px; color: #7f8c8d;")        
        tl.addWidget(subtitle)   # addWidget()：将副标题添加到垂直布局中

        hl.addLayout(tl)        # addLayout()：将整体垂直布局添加到水平布局中
        hl.addStretch()         # addStretch()：添加一个弹性空间，作用：将后面的按钮推到右侧

        # 定义按钮的样式字符串（便于复用）
        btn = ("""
            padding: 6px 12px; font-size: 12px; 
            QPushButton { background-color: #4CAF50; }          /* 按键背景色：绿色 */
            QPushButton:hover { background-color: #45a049; }    /* 按钮悬停时：深绿 */
        """)
        self.save_btn = QPushButton("💾 保存")
        self.save_btn.setStyleSheet(btn)
        hl.addWidget(self.save_btn)
        self.export_btn = QPushButton("📤 导出")
        self.export_btn.setStyleSheet(btn)
        hl.addWidget(self.export_btn)
        layout.addWidget(w)

    # ============================================================
    # 创建左侧面板
    # ===========================================================
    def create_left_panel(self):
        panel = QWidget()               # QWidget()：创建一个容器
        panel.setMaximumWidth(450)      # setMaximumWidth()：设置最大宽度为450px，防止左侧面板过宽
        layout = QVBoxLayout(panel)     # QVBoxLayout()：容器垂直布局，左侧面板内容垂直排列
        layout.setSpacing(10)           # setSpacing()：设置布局内控件间距为10px

        # --- 上传 ---
        g = QGroupBox("上传图片/视频")
        gl = QVBoxLayout(g)

        self.upload_btn = QPushButton("上传图片/视频")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        self.upload_btn.clicked.connect(self._on_upload)
        gl.addWidget(self.upload_btn)

        layout.addWidget(g)

        # --- 配置 ---
        cg = QGroupBox("可视化选项")
        cl = QVBoxLayout(cg)
        cl.setSpacing(8)

        top = QHBoxLayout()
        label = QLabel("过滤器百分比")
        label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        top.addWidget(label)
        top.addStretch()
        self.value_label = QLabel("[ 90% ]")
        self.value_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2196F3;"
            "background: #e8f0fe; border-radius: 4px; padding: 2px 8px;")
        top.addWidget(self.value_label)
        self.reset_btn = QPushButton("↻")
        self.reset_btn.setFixedWidth(60)
        self.reset_btn.setStyleSheet(
            "QPushButton { background-color: #6c757d; }"
            "QPushButton:hover { background: #5a6268; }"
            "QPushButton:pressed { background: #495057; }")
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        top.addWidget(self.reset_btn)
        cl.addLayout(top)

        sl = QHBoxLayout()
        min_l = QLabel("0")
        min_l.setStyleSheet("color: #999; font-size: 11px;")
        sl.addWidget(min_l)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(90)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #d0d0d0; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #2196F3; width: 16px; height: 16px;"
            "margin: -5px 0; border-radius: 8px; }"
            "QSlider::handle:horizontal:hover { background: #1976D2; }"
            "QSlider::sub-page:horizontal { background: #2196F3; border-radius: 3px; }")
        self.slider.valueChanged.connect(self.on_slider_changed)
        sl.addWidget(self.slider)
        max_l = QLabel("100")
        max_l.setStyleSheet("color: #999; font-size: 11px;")
        sl.addWidget(max_l)
        cl.addLayout(sl)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #e0e0e0;")
        cl.addWidget(sep)

        ftop = QHBoxLayout()
        flabel = QLabel("抽取帧率")
        flabel.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        ftop.addWidget(flabel)
        ftop.addStretch()
        self.fps_label = QLabel("[ 2 fps ]")
        self.fps_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2196F3;"
            "background: #e8f0fe; border-radius: 4px; padding: 2px 8px;")
        ftop.addWidget(self.fps_label)
        self.fps_reset_btn = QPushButton("↻")
        self.fps_reset_btn.setFixedWidth(60)
        self.fps_reset_btn.setStyleSheet(
            "QPushButton { background: #6c757d; color: white; border: none;"
            "padding: 5px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; }"
            "QPushButton:hover { background: #5a6268; }"
            "QPushButton:pressed { background: #495057; }")
        self.fps_reset_btn.clicked.connect(self.on_fps_reset_clicked)
        ftop.addWidget(self.fps_reset_btn)
        cl.addLayout(ftop)

        fsl = QHBoxLayout()
        fmin = QLabel("0.1")
        fmin.setStyleSheet("color: #999; font-size: 11px;")
        fsl.addWidget(fmin)
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 100)
        self.fps_slider.setValue(20)
        self.fps_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #d0d0d0; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #2196F3; width: 16px; height: 16px;"
            "margin: -5px 0; border-radius: 8px; }"
            "QSlider::handle:horizontal:hover { background: #1976D2; }"
            "QSlider::sub-page:horizontal { background: #2196F3; border-radius: 3px; }")
        self.fps_slider.valueChanged.connect(self.on_fps_slider_changed)
        fsl.addWidget(self.fps_slider)
        fmax = QLabel("10")
        fmax.setStyleSheet("color: #999; font-size: 11px;")
        fsl.addWidget(fmax)
        cl.addLayout(fsl)

        layout.addWidget(cg)

        # --- 操作 ---
        ag = QGroupBox("操作")
        al = QVBoxLayout(ag)
        self.run_btn = QPushButton("▶️ 运行推理")
        self.run_btn.setStyleSheet("padding: 12px; font-size: 14px;")
        self.run_btn.clicked.connect(self._on_run_inference)
        self.run_btn.setEnabled(False)
        al.addWidget(self.run_btn)
        row = QHBoxLayout()
        self.clear_btn = QPushButton("🗑️ 清空全部")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d; padding: 12px; font-size: 14px;
                color: white; border: none; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #5a6268; }
            QPushButton:pressed { background-color: #4e555b; }
        """)
        self.clear_btn.clicked.connect(self._on_clear_all)
        row.addWidget(self.clear_btn)
        al.addLayout(row)
        layout.addWidget(ag)

        # --- 进度 ---
        pg = QGroupBox("进度")
        pl = QVBoxLayout(pg)
        self.upload_progress_label = QLabel("上传: 等待")
        self.upload_progress_label.setStyleSheet("color: #666; font-size: 11px;")
        pl.addWidget(self.upload_progress_label)
        from PySide6.QtWidgets import QProgressBar
        self.upload_bar = QProgressBar()
        self.upload_bar.setRange(0, 100)
        self.upload_bar.setValue(0)
        self.upload_bar.setFormat("上传 %p%")
        self.upload_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #d0d0d0; border-radius: 3px;
                background: #e8e8e8; height: 12px; text-align: center; font-size: 10px; }
            QProgressBar::chunk { background-color: #2196F3; border-radius: 2px; }
        """)
        pl.addWidget(self.upload_bar)

        # 推理进度标签
        self.progress_label = QLabel("推理: 就绪")
        self.progress_label.setStyleSheet("color: #4CAF50; font-weight: bold; margin-top: 6px;")
        pl.addWidget(self.progress_label)
        self.progress_track = QFrame()
        self.progress_track.setStyleSheet("background-color: #e0e0e0; border-radius: 4px;")
        self.progress_track.setFixedHeight(8)
        pl.addWidget(self.progress_track)
        self.progress_fill = QFrame(self.progress_track)
        self.progress_fill.setStyleSheet("background-color: #4CAF50; border-radius: 4px;")
        self.progress_fill.setFixedHeight(8)
        self._set_progress(0)
        layout.addWidget(pg)

        return panel

    # ----------------------------------------------------------
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        tabs = QTabWidget()

        tabs.addTab(self._tab_pointcloud(), "3D点云")
        tabs.addTab(self._tab_depth(), "深度预测")
        tabs.addTab(self._tab_camera(), "实时显示")

        layout.addWidget(tabs)
        return panel

    def _tab_pointcloud(self):
        t = QWidget()
        l = QVBoxLayout(t)
        l.setContentsMargins(0, 0, 0, 0)
        self.pointcloud_container = QWidget()
        self.pointcloud_container.setMinimumHeight(450)
        self.pointcloud_container.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        cl = QVBoxLayout(self.pointcloud_container)
        cl.setContentsMargins(0, 0, 0, 0)
        self.pointcloud_label = QLabel("Open3D Viewer\n\nUpload images and run inference to open")
        self.pointcloud_label.setAlignment(Qt.AlignCenter)
        self.pointcloud_label.setWordWrap(True)
        self.pointcloud_label.setStyleSheet("color: #e0e0e0; font-size: 16px; padding: 20px; background: transparent;")
        cl.addWidget(self.pointcloud_label)
        l.addWidget(self.pointcloud_container)
        self._embedded_hwnd = None
        return t

    def _start_http_server(self, serve_dir, cam=False):
        """启动本地 HTTP 服务器（主流程与摄像头各用一个，互不干扰）
        cam=False: 主流程服务器（端口 8765-8790）
        cam=True : 摄像头服务器（端口 8791-8816）
        """
        srv_attr  = "_cam_http_server" if cam else "_http_server"
        port_attr = "_cam_http_port"   if cam else "_http_port"
        ports     = range(8791, 8816)  if cam else range(8765, 8790)

        # 关闭同类型的旧服务（只关自己那套，不影响另一套）
        old = getattr(self, srv_attr, None)
        if old:
            old.shutdown()
            setattr(self, srv_attr, None)
            setattr(self, port_attr, None)

        # 找空闲端口
        port = None
        for p in ports:
            try:
                s = socket.socket()
                s.bind(('', p))
                s.close()
                port = p
                break
            except OSError:
                continue

        if not port:
            return None
        setattr(self, port_attr, port)

        class CORSHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=serve_dir, **kwargs)
            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                super().end_headers()

        server = HTTPServer(('127.0.0.1', port), CORSHandler)
        setattr(self, srv_attr, server)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return port

    def _load_glb_in_viewer(self, glb_path, viewer=None, cam=False):
        """在 Babylon.js 查看器中加载点云（可指定 viewer，默认 pointcloud_viewer）
        cam=True 时使用摄像头专属 HTTP 服务器，避免与主流程互相覆盖"""
        glb_dir = os.path.dirname(os.path.abspath(glb_path))
        glb_name = os.path.basename(glb_path)
        viewer = viewer or self.pointcloud_viewer

        port = self._start_http_server(glb_dir, cam=cam)
        if not port:
            viewer.setHtml("<h3 style='color:red;text-align:center;margin-top:100px'>"
                           "无法启动 HTTP 服务器</h3>")
            return

        glb_url = f"http://127.0.0.1:{port}/{glb_name}"

        viewer.setHtml(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Viewer</title>
    <style>
        /* 全局样式：无内外边距，隐藏溢出，全屏高度，深色背景 */
        html, body {{ margin:0; padding:0; overflow:hidden; height:100%; background:#222; }}
        #canvas {{ width:100%; height:100%; outline:none; }}
        #loading {{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
              color:#ccc; font-family:sans-serif; font-size:14px; }}
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <div id="loading">Loading...</div>
    <!-- ========== 加载 Babylon.js 库 ========== -->
    <script src="https://cdn.babylonjs.com/babylon.js"></script>
    <script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
    <script>
        const canvas = document.getElementById("canvas");
        const engine = new BABYLON.Engine(canvas, true, {{ preserveDrawingBuffer: true, stencil: true, antialias: true }});
        const scene = new BABYLON.Scene(engine);
        scene.clearColor = new BABYLON.Color4(0.15, 0.15, 0.18, 1);

        // === 灯光 (Sandbox 同款) ===
        const light = new BABYLON.HemisphericLight("light",
            new BABYLON.Vector3(0, 1, 0), scene);
        light.intensity = 0.7;

        // === 相机（正视图角度，距离适中） ===
        const camera = new BABYLON.ArcRotateCamera(
            "cam",
            Math.PI/2,          // alpha: 水平旋转角度90度
            Math.PI/2.2,        // beta:  垂直旋转角度82度
            5,                  // 初始半径（加载后自动适配）
            BABYLON.Vector3.Zero(), scene);
        camera.attachControl(canvas, true);
        camera.lowerRadiusLimit = 0.0001;
        camera.upperRadiusLimit = 10000;
        camera.minZ = 0.0001;
        camera.wheelPrecision = 50;       // 滚轮缩放（越大缩放越慢（更精确））
        camera.angularSensibilityX = 1000;
        camera.angularSensibilityY = 1000;
        camera.panningSensibility = 1000;

        BABYLON.SceneLoader.ImportMesh(
            "",
            "",
            "{glb_url}",
            scene,

            function(meshes) {{
                document.getElementById("loading").style.display = "none";
                if (meshes.length > 0) {{
                    const bb = meshes[0].getBoundingInfo().boundingBox;
                    const center = bb.center;
                    const size = bb.extendSize.length();
                    camera.target = center;
                    camera.radius = Math.max(size * 0.5, 0.1);
                    camera.lowerRadiusLimit = size * 0.00001;
                }}
            }},
            function(evt) {{
                if (evt.lengthComputable) {{
                    document.getElementById("loading").textContent =
                        "Loading " + Math.round(evt.loaded*100/evt.total) + "%";
                }}
            }},
            function() {{ document.getElementById("loading").textContent = "Load Failed"; }}
        );

        engine.runRenderLoop(() => scene.render());
        window.addEventListener("resize", () => engine.resize());
    </script>
</body>
</html>""")

    def _tab_depth(self):
        t = QWidget()
        l = QVBoxLayout(t)

        # --- 导航栏 ---
        nav = QHBoxLayout()
        self.depth_prev_btn = QPushButton("< 上一张")
        self.depth_prev_btn.setObjectName("secondary")
        self.depth_prev_btn.clicked.connect(self._on_depth_prev)
        self.depth_prev_btn.setEnabled(False)
        nav.addWidget(self.depth_prev_btn)

        self.depth_combo = QComboBox()
        self.depth_combo.setEnabled(False)
        self.depth_combo.currentIndexChanged.connect(self._on_depth_combo)
        nav.addWidget(self.depth_combo, 1)

        self.depth_next_btn = QPushButton("下一张 >")
        self.depth_next_btn.setObjectName("secondary")
        self.depth_next_btn.clicked.connect(self._on_depth_next)
        self.depth_next_btn.setEnabled(False)
        nav.addWidget(self.depth_next_btn)
        l.addLayout(nav)

        # --- 图像显示区 ---
        disp = QHBoxLayout()
        # 左: RGB
        self.rgb_label = QLabel("RGB 图像")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setMinimumHeight(400)
        self.rgb_label.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border-radius: 6px; font-size: 14px;")
        disp.addWidget(self.rgb_label)
        # 右: 深度图
        self.depth_label = QLabel("深度图像")
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setMinimumHeight(400)
        self.depth_label.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border-radius: 6px; font-size: 14px;")
        disp.addWidget(self.depth_label)
        l.addLayout(disp)

        # 初始化状态
        self.depth_results_dir = None
        self.depth_frame_list = []
        self.depth_frame_idx = 0
        return t

    # ============================================================
    # 深度图导航
    # ============================================================
    def _init_depth_viewer(self, results_dir):
        """推理完成后初始化深度图查看器"""
        self.depth_results_dir = results_dir
        self.depth_frame_list = sorted(
            glob.glob(os.path.join(results_dir, "frame_*.npz")))
        self.depth_frame_list.sort(
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
        self.depth_frame_idx = 0

        n = len(self.depth_frame_list)
        self.depth_combo.clear()
        self.depth_combo.addItems([f"第 {i+1} 张" for i in range(n)])
        self.depth_combo.setEnabled(n > 0)
        self.depth_prev_btn.setEnabled(n > 1)
        self.depth_next_btn.setEnabled(n > 1)
        if n > 0:
            self._show_depth_frame(0)

    def _show_depth_frame(self, idx):
        """显示第 idx 帧的 RGB 和深度图"""
        if idx < 0 or idx >= len(self.depth_frame_list):
            return
        self.depth_frame_idx = idx
        self.depth_combo.blockSignals(True)
        self.depth_combo.setCurrentIndex(idx)
        self.depth_combo.blockSignals(False)

        npz = np.load(self.depth_frame_list[idx])
        rgb = npz["image"]
        depth = npz["depth"]

        # RGB → QPixmap
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.rgb_label.setPixmap(
            QPixmap.fromImage(qimg).scaledToWidth(
                400, Qt.SmoothTransformation))
        self.rgb_label.setText("")

        # 深度图 → 彩色 → QPixmap
        from depth_anything_3.utils.visualize import visualize_depth
        d_vis = visualize_depth(depth)
        h2, w2, _ = d_vis.shape
        qimg2 = QImage(d_vis.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        self.depth_label.setPixmap(
            QPixmap.fromImage(qimg2).scaledToWidth(
                400, Qt.SmoothTransformation))
        self.depth_label.setText("")

    def _on_depth_prev(self):
        self._show_depth_frame(self.depth_frame_idx - 1)

    def _on_depth_next(self):
        self._show_depth_frame(self.depth_frame_idx + 1)

    def _on_depth_combo(self, idx):
        self._show_depth_frame(idx)

    def _tab_camera(self):
        t = QWidget()
        l = QVBoxLayout(t)

        # 左右分栏 1:2
        panels = QHBoxLayout()
        panels.setStretch(0, 1)
        panels.setStretch(1, 3)

        # 左边：上下两栏（摄像头 | 深度图）
        left_panel = QVBoxLayout()
        self.cam_label = QLabel("摄像头未开启")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumHeight(180)
        self.cam_label.setScaledContents(True)
        self.cam_label.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border-radius: 8px; font-size: 12px;")
        left_panel.addWidget(self.cam_label)

        self.cam_depth_label = QLabel("深度图")
        self.cam_depth_label.setAlignment(Qt.AlignCenter)
        self.cam_depth_label.setMinimumHeight(180)
        self.cam_depth_label.setScaledContents(True)
        self.cam_depth_label.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border-radius: 8px; font-size: 12px;")
        left_panel.addWidget(self.cam_depth_label)

        panels.addLayout(left_panel)

        # 右边：3D 点云实时显示（Babylon.js）
        from PySide6.QtWebEngineWidgets import QWebEngineView
        self.cam_3d_view = QWebEngineView()
        self.cam_3d_view.setMinimumHeight(380)
        self.cam_3d_view.setHtml("""
            <html><body style="background:#1a1a2e;display:flex;align-items:center;
            justify-content:center;height:100vh;margin:0;font-family:sans-serif">
            <div style="text-align:center;color:#e0e0e0">
            <div style="font-size:36px;margin-bottom:12px">&#127758;</div>
            <div style="font-size:14px">3D 点云（实时）</div>
            <div style="font-size:11px;color:#888;margin-top:6px">开启摄像头后自动更新</div>
            </div></body></html>
        """)
        panels.addWidget(self.cam_3d_view)

        l.addLayout(panels)

        # 控制栏
        ctrl = QHBoxLayout()
        self.cam_toggle_btn = QPushButton("📷 开启摄像头")
        self.cam_toggle_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; padding: 6px 14px; font-size: 13px; }"
            "QPushButton:hover { background-color: #1976D2; }")
        self.cam_toggle_btn.clicked.connect(self._on_toggle_camera)
        ctrl.addWidget(self.cam_toggle_btn)

        self.cam_save_btn = QPushButton("💾 截图保存")
        self.cam_save_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; padding: 6px 14px; font-size: 13px; }"
            "QPushButton:hover { background-color: #45a049; }")
        self.cam_save_btn.clicked.connect(self._on_cam_save)
        ctrl.addWidget(self.cam_save_btn)
        ctrl.addStretch()
        l.addLayout(ctrl)

        self._cam_cap = None
        self._cam_depth_running = False
        self._cam_timer = QTimer()
        self._cam_timer.timeout.connect(self._on_cam_frame)
        self._cam_last_frame = None  # 最新帧
        self._cam_last_depth = None  # 最新深度图
        self._cam_http_server = None  # 摄像头专属 HTTP 服务器（与主流程独立）
        self._cam_http_port = None
        self._cam_prev_tmp_dir = None  # 上一帧临时目录（新帧就绪后清理）
        return t

    def _on_toggle_camera(self):
        if self._cam_cap is not None:
            self._cam_timer.stop()
            self._cam_cap.release()
            self._cam_cap = None
            self._cam_depth_running = False
            if hasattr(self, '_cam_persistent_worker') and self._cam_persistent_worker:
                self._cam_persistent_worker.stop()
                self._cam_persistent_worker.wait()
                self._cam_persistent_worker = None
            # 关闭摄像头专属 HTTP 服务器
            if hasattr(self, '_cam_http_server') and self._cam_http_server:
                self._cam_http_server.shutdown()
                self._cam_http_server = None
                self._cam_http_port = None
            self.cam_label.setPixmap(QPixmap())
            self.cam_label.setText("摄像头未开启")
            self.cam_label.setStyleSheet(
                "background-color: #1a1a2e; color: #888; border-radius: 8px; font-size: 14px;")
            self.cam_3d_view.setHtml("""
                <html><body style="background:#1a1a2e;display:flex;align-items:center;
                justify-content:center;height:100vh;margin:0;font-family:sans-serif">
                <div style="text-align:center;color:#e0e0e0">
                <div style="font-size:36px;margin-bottom:12px">&#127758;</div>
                <div style="font-size:14px">3D 点云（实时）</div></div></body></html>
            """)
            self.cam_toggle_btn.setText("📷 开启摄像头")
        else:
            import cv2
            self._cam_cap = cv2.VideoCapture(0)
            if not self._cam_cap.isOpened():
                self._cam_cap = None
                self.cam_label.setText("⚠️ 无法打开摄像头")
                return
            self.cam_label.setText("")
            self.cam_label.setStyleSheet("background-color: #000; border-radius: 8px;")
            self.cam_toggle_btn.setText("⏹ 关闭摄像头")
            # 启动持久深度计算线程（只创建一次）
            if self._cached_model:
                self._cam_persistent_worker = PersistentCamWorker(self._cached_model)
                self._cam_persistent_worker.finished.connect(self._on_cam_depth_finished)
                self._cam_persistent_worker.error.connect(self._on_cam_depth_error)
                self._cam_persistent_worker.start()
            self._cam_timer.start(30)

    def _on_cam_frame(self):
        if self._cam_cap is None:
            return
        ret, frame = self._cam_cap.read()
        if not ret:
            return
        self._cam_last_frame = frame.copy()  # 保存最新帧

        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(qimg))

        if hasattr(self, '_cam_persistent_worker') and self._cam_persistent_worker:
            self._cam_persistent_worker.submit_frame(frame)

    def _on_cam_save(self):
        """保存当前帧：frame.png + depth.png + combined.glb"""
        if self._cam_last_frame is None:
            self.status_label.setText("⚠️ 无帧可保存")
            return
        save_dir = os.path.join("output", "cam_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)

        import cv2
        # 1. RGB 帧
        cv2.imwrite(os.path.join(save_dir, "frame.png"), self._cam_last_frame)

        # 2. 彩色深度图
        if hasattr(self, '_cam_last_depth') and self._cam_last_depth is not None:
            from depth_anything_3.utils.visualize import visualize_depth
            depth_color = visualize_depth(self._cam_last_depth)
            cv2.imwrite(os.path.join(save_dir, "depth.png"),
                        depth_color[:, :, ::-1])

        # 3. 从 DA3 输出目录拷贝 combined.glb（与主流程一致）
        if hasattr(self, '_cam_last_output_dir'):
            src_glb = os.path.join(self._cam_last_output_dir, "combined.glb")
            if os.path.exists(src_glb):
                shutil.copy2(src_glb, os.path.join(save_dir, "combined.glb"))

        self.status_label.setText(f"💾 已保存: {save_dir}")

    def _on_cam_depth_finished(self, output_dir):
        """接收 DA3_Streaming 输出目录，读取深度图并显示 3D 点云"""
        self._cam_last_output_dir = output_dir  # 保存输出目录，供截图使用
        npz_dir = os.path.join(output_dir, "results_output")
        npz_files = sorted(glob.glob(os.path.join(npz_dir, "frame_*.npz")))
        if not npz_files:
            return
        data = np.load(npz_files[0])
        depth = data["depth"]
        self._cam_last_depth = depth

        from depth_anything_3.utils.visualize import visualize_depth
        d_vis = visualize_depth(depth)
        h, w, _ = d_vis.shape
        qimg = QImage(d_vis.data, w, h, w * 3, QImage.Format_RGB888)
        self.cam_depth_label.setPixmap(QPixmap.fromImage(qimg))
        self.cam_depth_label.setText("")

        try:
            self._update_cam_3d(output_dir)
        except Exception:
            pass

    def _update_cam_3d(self, output_dir):
        """DA3 输出目录 → npz → GLB → 3D 显示"""
        import importlib.util
        _here = os.path.dirname(os.path.abspath(__file__))
        _spec = importlib.util.spec_from_file_location(
            "npz_proc", os.path.join(_here, "npz_output_process.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        res_dir = os.path.join(output_dir, "results_output")
        poses_file = os.path.join(output_dir, "camera_poses.txt")
        glb_path = os.path.join(output_dir, "combined.glb")

        if os.path.exists(res_dir) and os.path.exists(poses_file):
            _mod.create_point_cloud(
                npz_folder=res_dir, pose_file=poses_file,
                output_ply=os.path.join(output_dir, "combined.ply"),
                conf_threshold_coef=0.5, sample_ratio=1.0)

        if os.path.exists(glb_path):
            self._load_glb_in_viewer(glb_path, viewer=self.cam_3d_view, cam=True)

            # 新帧显示就绪后，清理上一帧的临时目录
            # （此时摄像头服务器已切换到新目录，旧目录不再被引用）
            prev_tmp = getattr(self, "_cam_prev_tmp_dir", None)
            if prev_tmp and os.path.isdir(prev_tmp):
                shutil.rmtree(prev_tmp, ignore_errors=True)
            self._cam_prev_tmp_dir = os.path.dirname(output_dir)

    def _on_cam_depth_error(self, err):
        self._cam_depth_running = False
        self.cam_depth_label.setText(f"❌ 错误:\n{err[:200]}")
        self.cam_depth_label.setStyleSheet(
            "background-color: #3a1a1a; color: #f88; border-radius: 8px; font-size: 12px;")

    # ----------------------------------------------------------
    def create_status_bar(self):
        sb = self.statusBar()
        sb.setStyleSheet("background: #f8f9fa; color: #333; border-top: 1px solid #e0e0e0; padding: 4px 8px;")
        self.status_label = QLabel("✅ 就绪")
        sb.addWidget(self.status_label)
        self.timer_label = QLabel("⏱ 00:00")
        self.timer_label.setStyleSheet("font-weight: bold; color: #2196F3; margin-right: 10px;")
        sb.addPermanentWidget(self.timer_label)
        sb.addPermanentWidget(QLabel("⚡ GPU: 未使用"))

        # 计时钟（每秒刷新一次）
        self._elapsed = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer_tick)

    # ============================================================
    # 槽
    # ============================================================
    def _on_timer_tick(self):
        self._elapsed += 100
        sec = self._elapsed / 1000.0
        self.timer_label.setText(f"⏱ {sec:.1f}s")

    def _start_timer(self):
        self._elapsed = 0
        self.timer_label.setText("⏱ 0.0s")
        self._timer.start(100)  # 每 100ms 刷新一次

    def _stop_timer(self):
        self._timer.stop()

    def _set_progress(self, pct):
        w = int(self.progress_track.width() * pct / 100)
        self.progress_fill.setFixedWidth(max(2, w))

    def _on_progress(self, msg, pct):
        self.status_label.setText(f"⏳ {msg}")
        self.progress_label.setText(f"状态: {msg} ({pct}%)")
        self._set_progress(pct)

    # ============================================================
    # 统一上传：图片 → 复制文件 | 视频 → 抽帧
    # ============================================================
    def _on_upload(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片或视频", "",
            "Images/Videos (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v)"
        )
        if not paths:
            return

        self.pending_image_dir = os.path.join(
            "workspace", "gui_session",
            datetime.now().strftime("%Y%m%d_%H%M%S"), "images"
        )
        os.makedirs(self.pending_image_dir, exist_ok=True)

        self.upload_btn.setEnabled(False)
        self.upload_bar.setValue(0)
        self.upload_bar.show()

        # 判断文件类型
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
        is_video = any(os.path.splitext(p)[1].lower() in video_exts for p in paths)

        self._upload_emitter = UploadEmitter()
        self._upload_emitter.progress.connect(self._on_upload_progress)
        self._upload_emitter.finished.connect(self._on_upload_finished)
        self._upload_emitter.error.connect(self._on_upload_error)

        if is_video and len(paths) == 1:
            self.status_label.setText("⏳ 正在抽帧...")
            fps = self.fps_slider.value() / 10.0
            runnable = VideoFrameRunnable(
                paths[0], self.pending_image_dir, self._upload_emitter, fps=fps)
        else:
            self.status_label.setText("⏳ 正在上传...")
            runnable = FileCopyRunnable(
                paths, self.pending_image_dir, self._upload_emitter)

        QThreadPool.globalInstance().start(runnable)

    def _on_upload_progress(self, pct):
        self.upload_bar.setValue(pct)
        self.upload_progress_label.setText(f"上传: {pct}%")
        self.status_label.setText(f"⏳ 上传中... {pct}%")

    def _on_upload_finished(self, dest_dir):
        self.status_label.setText("📁 上传成功")
        self.upload_bar.setValue(100)
        self.upload_progress_label.setText("上传: ✅ 完成")
        self.upload_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ 运行推理")
        self.progress_label.setText(f"状态: 就绪 (已选 {dest_dir})")

    def _on_upload_error(self, err):
        self.status_label.setText("❌ 上传失败")
        self.upload_bar.setValue(0)
        self.upload_progress_label.setText("上传: ❌ 失败")
        self.upload_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_label.setText(f"错误: {err}")

    # ============================================================
    # 运行推理
    # ============================================================
    def _on_run_inference(self):
        if not hasattr(self, 'pending_image_dir') or not self.pending_image_dir:
            return

        self.run_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.status_label.setText("⏳ 正在推理...")
        self._start_timer()

        work_dir = os.path.dirname(self.pending_image_dir)
        sample_ratio = self.slider.value() / 100.0
        self._da3_emitter = DA3Emitter()
        self._da3_emitter.progress.connect(self._on_progress)
        self._da3_emitter.finished.connect(self._on_da3_finished)
        self._da3_emitter.error.connect(self._on_da3_error)
        runnable = DA3Runnable(
            self.pending_image_dir, work_dir,
            "da3_streaming/configs/base_config.yaml",
            self._da3_emitter, sample_ratio=sample_ratio)
        QThreadPool.globalInstance().start(runnable)

    def _on_da3_finished(self, output_dir, glb_path):
        self._stop_timer()
        self.status_label.setText("✅ 推理完成")
        self.progress_label.setText("状态: 完成 (100%)")
        self._set_progress(100)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ 运行推理")
        self.upload_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)

        # 初始化深度图查看器
        results_dir = os.path.join(output_dir, "results_output")
        if os.path.exists(results_dir):
            self._init_depth_viewer(results_dir)

        if glb_path and os.path.exists(glb_path):
            self.progress_label.setText(f"状态: 加载 3D 模型中 ({os.path.basename(glb_path)})")
            self._open_in_open3d(os.path.join(output_dir, "combined.ply"))
        else:
            self.progress_label.setText("状态: 完成（无 GLB 文件）")

    def _on_da3_error(self, err):
        self._stop_timer()
        self.status_label.setText("❌ 推理失败")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ 运行推理")
        self.upload_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_label.setText(f"错误:\n{err[:200]}")
        self.clear_btn.setEnabled(True)

    # ============================================================
    # 清空全部，恢复初始状态
    # ============================================================
    def _on_clear_all(self):
        # 等待线程池中的任务完成（已提交的不可取消）
        QThreadPool.globalInstance().clear()
        QThreadPool.globalInstance().waitForDone()

        # 关闭 HTTP 服务器（主流程 + 摄像头两套）
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None
            self._http_port = None
        if hasattr(self, '_cam_http_server') and self._cam_http_server:
            self._cam_http_server.shutdown()
            self._cam_http_server = None
            self._cam_http_port = None

        # 重置上传进度
        self.upload_bar.setValue(0)
        self.upload_progress_label.setText("上传: 等待")

        # 重置推理进度
        self._set_progress(0)
        self.progress_label.setText("推理: 就绪")

        # 重置上传按钮和运行按钮
        self.upload_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.run_btn.setEnabled(False)
        self.run_btn.setText("▶️ 运行推理")
        self.clear_btn.setEnabled(False)

        # 清除 pending 数据
        if hasattr(self, 'pending_image_dir'):
            del self.pending_image_dir

        # 重置 3D 点云查看器
        self._close_embedded_o3d()
        self.pointcloud_label.show()
        self.pointcloud_label.setText("Open3D Viewer\n\nUpload images and run inference to open")

        # 重置深度图查看器
        self.rgb_label.setPixmap(QPixmap())
        self.rgb_label.setText("RGB 图像")
        self.depth_label.setPixmap(QPixmap())
        self.depth_label.setText("深度图像")
        self.depth_combo.clear()
        self.depth_combo.setEnabled(False)
        self.depth_prev_btn.setEnabled(False)
        self.depth_next_btn.setEnabled(False)
        self.depth_results_dir = None
        self.depth_frame_list = []
        self.depth_frame_idx = 0

        # 关闭摄像头
        if hasattr(self, '_cam_cap') and self._cam_cap is not None:
            self._cam_timer.stop()
            self._cam_cap.release()
            self._cam_cap = None
            self.cam_label.setPixmap(QPixmap())
            self.cam_label.setText("摄像头未开启")
            self.cam_toggle_btn.setText("📷 开启摄像头")
            if hasattr(self, '_cam_persistent_worker') and self._cam_persistent_worker:
                self._cam_persistent_worker.stop()
                self._cam_persistent_worker.wait()
                self._cam_persistent_worker = None
            self.cam_depth_label.setPixmap(QPixmap())
            self.cam_depth_label.setText("深度图")
            self.cam_depth_label.setStyleSheet(
                "background-color: #1a1a2e; color: #888; border-radius: 8px; font-size: 14px;")

        # 重置状态栏和计时
        self._stop_timer()
        self._elapsed = 0
        self.timer_label.setText("⏱ 0.0s")
        self.status_label.setText("✅ 就绪")

    def on_slider_changed(self, value):
        self.value_label.setText(f"[ {value}% ]")

    def on_reset_clicked(self):
        self.slider.setValue(90)

    def on_fps_slider_changed(self, value):
        fps = value / 10.0  # 1-100 → 0.1-10.0
        self.fps_label.setText(f"[ {fps:.1f} fps ]")

    def on_fps_reset_clicked(self):
        self.fps_slider.setValue(20)  # 默认 2 fps


# ============================================================
def main():
    app = QApplication(sys.argv)
    window = GradioLikeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
