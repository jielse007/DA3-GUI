import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QSplitter, QTextEdit, QGroupBox, QTabWidget,
    QFrame, QSlider, QCheckBox, QComboBox, QLineEdit,
    QGridLayout, QScrollArea
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer, QTime
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette

import os, glob, shutil, socket, threading, time
import numpy as np
import trimesh
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler


# ============================================================
# 文件复制线程：复制图片并实时报告进度
# ============================================================
class FileCopyWorker(QThread):
    progress = Signal(int)    # 进度百分比
    finished = Signal(str)    # 目标目录
    error    = Signal(str)

    def __init__(self, file_list, dest_dir):
        super().__init__()
        self.file_list = file_list
        self.dest_dir  = dest_dir

    def run(self):
        try:
            total = len(self.file_list)
            for i, f in enumerate(self.file_list):
                shutil.copy(f, os.path.join(self.dest_dir, os.path.basename(f)))
                self.progress.emit(int((i + 1) / total * 100))
                self.msleep(5)
            self.finished.emit(self.dest_dir)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# 视频抽帧线程：用 cv2 读取视频，按帧率抽帧，实时报告进度
# ============================================================
class VideoFrameWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error    = Signal(str)

    def __init__(self, video_path, dest_dir, fps=2, max_width=640):
        super().__init__()
        self.video_path = video_path
        self.dest_dir   = dest_dir
        self.fps        = fps
        self.max_width  = max_width

    def run(self):
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"无法打开视频: {self.video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30
            interval     = max(1, int(video_fps / self.fps))

            saved = 0
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % interval == 0:
                    h, w = frame.shape[:2]
                    if w > self.max_width:
                        scale = self.max_width / w
                        frame = cv2.resize(frame, (self.max_width, int(h * scale)))
                    out_path = os.path.join(
                        self.dest_dir, f"frame_{saved:06d}.png")
                    cv2.imwrite(out_path, frame)
                    saved += 1

                idx += 1
                if total_frames > 0:
                    self.progress.emit(int(idx / total_frames * 100))
                self.msleep(1)

            cap.release()
            self.progress.emit(100)
            self.finished.emit(self.dest_dir)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ============================================================
# 摄像头深度计算线程（复用缓存模型，加速推理）
# ============================================================
class CamDepthWorker(QThread):
    finished = Signal(object)  # 传出 depth numpy array
    error    = Signal(str)

    def __init__(self, frame_bgr, cached_model):
        super().__init__()
        self.frame_bgr   = frame_bgr   # BGR numpy array
        self.cached_model = cached_model  # 预加载的 DepthAnything3

    def run(self):
        try:
            import tempfile, cv2
            # 保存帧为临时文件
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, self.frame_bgr)
                tmp_path = f.name

            prediction = self.cached_model.inference([tmp_path])
            os.unlink(tmp_path)

            depth = np.squeeze(prediction.depth)
            self.finished.emit(depth)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ============================================================
# 后台工作线程：DA3 推理
# ============================================================
class DA3Worker(QThread):
    # ============================================================
    # 定义完成、进度、错误信号
    # ============================================================
    finished = Signal(str, str)
    progress = Signal(str, int)
    error   = Signal(str)

    # ============================================================
    # 接收图片目录、输出目录、配置文件路径
    # ============================================================    
    def __init__(self, image_dir, output_dir, config_path, sample_ratio=1.0, model_cache=None):
        super().__init__()
        self.image_dir    = image_dir
        self.output_dir   = output_dir
        self.config_path  = config_path
        self.sample_ratio = sample_ratio
        self.model_cache  = model_cache  # 外部传入的预加载模型

    def _log(self, msg, pct):
        self.progress.emit(msg, pct)
        self.msleep(10)

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
            da3.run()               # 执行DA3推理（主要耗时操作）
            da3.close()             # 关闭DA3实例，释放资源
            sys.stdout = old_stdout # 恢复标准输出

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

            glb_files = sorted(glob.glob(os.path.join(self.output_dir, "*.glb")))
            glb_path = glb_files[-1] if glb_files else ""
            self.finished.emit(self.output_dir, glb_path)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


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

        self.create_header(main_layout)                     # 调用create_header方法创建头部区域并添加到主布局

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

        from PySide6.QtWebEngineWidgets import QWebEngineView
        self.pointcloud_viewer = QWebEngineView()
        self.pointcloud_viewer.setMinimumHeight(450)
        # 初始显示占位页面
        self.pointcloud_viewer.setHtml("""
            <html><body style="background:#1a1a2e;display:flex;align-items:center;
            justify-content:center;height:100vh;margin:0;font-family:sans-serif">
            <div style="text-align:center;color:#e0e0e0">
            <div style="font-size:48px;margin-bottom:16px">&#127758;</div>
            <div style="font-size:16px">3D点云查看器</div>
            <div style="font-size:12px;color:#888;margin-top:8px">上传图片并运行推理后显示</div>
            </div></body></html>
        """)
        l.addWidget(self.pointcloud_viewer)
        self._http_server = None
        self._http_port = None
        return t

    def _start_http_server(self, serve_dir):
        """启动本地 HTTP 服务器（每次关闭旧服务，确保指向新目录）"""
        # 关闭旧的 HTTP 服务
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None
            self._http_port = None

        # 找空闲端口
        for port in range(8765, 8790):
            try:
                s = socket.socket()
                s.bind(('', port))
                s.close()
                self._http_port = port
                break
            except OSError:
                continue

        if not self._http_port:
            return None

        class CORSHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=serve_dir, **kwargs)
            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                super().end_headers()

        self._http_server = HTTPServer(('127.0.0.1', self._http_port), CORSHandler)
        t = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        t.start()
        return self._http_port

    def _load_glb_in_viewer(self, glb_path):
        """在嵌入的 Babylon.js 查看器中加载 GLB 文件"""
        glb_dir = os.path.dirname(os.path.abspath(glb_path))
        glb_name = os.path.basename(glb_path)

        port = self._start_http_server(glb_dir)
        if not port:
            self.pointcloud_viewer.setHtml(
                "<h3 style='color:red;text-align:center;margin-top:100px'>"
                "无法启动 HTTP 服务器</h3>")
            return

        glb_url = f"http://127.0.0.1:{port}/{glb_name}"

        self.pointcloud_viewer.setHtml(f"""
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

        # 左右分栏：摄像头 | 深度图
        panels = QHBoxLayout()

        # 左：摄像头
        left = QVBoxLayout()
        self.cam_label = QLabel("摄像头未开启")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumHeight(350)
        self.cam_label.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border-radius: 8px; font-size: 14px;")
        left.addWidget(self.cam_label)
        panels.addLayout(left)

        # 右：深度图
        right = QVBoxLayout()
        self.cam_depth_label = QLabel("深度图")
        self.cam_depth_label.setAlignment(Qt.AlignCenter)
        self.cam_depth_label.setMinimumHeight(350)
        self.cam_depth_label.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border-radius: 8px; font-size: 14px;")
        right.addWidget(self.cam_depth_label)
        panels.addLayout(right)

        l.addLayout(panels)

        # 控制栏
        ctrl = QHBoxLayout()
        self.cam_toggle_btn = QPushButton("📷 开启摄像头")
        self.cam_toggle_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; padding: 6px 14px; font-size: 13px; }"
            "QPushButton:hover { background-color: #1976D2; }")
        self.cam_toggle_btn.clicked.connect(self._on_toggle_camera)
        ctrl.addWidget(self.cam_toggle_btn)
        ctrl.addStretch()
        l.addLayout(ctrl)

        self._cam_cap = None
        self._cam_depth_running = False
        self._cam_timer = QTimer()
        self._cam_timer.timeout.connect(self._on_cam_frame)
        return t

    def _on_toggle_camera(self):
        if self._cam_cap is not None:
            self._cam_timer.stop()
            self._cam_cap.release()
            self._cam_cap = None
            self._cam_depth_running = False
            self.cam_label.setPixmap(QPixmap())
            self.cam_label.setText("摄像头未开启")
            self.cam_label.setStyleSheet(
                "background-color: #1a1a2e; color: #888; border-radius: 8px; font-size: 14px;")
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
            self._cam_timer.start(30)

    def _on_cam_frame(self):
        if self._cam_cap is None:
            return
        ret, frame = self._cam_cap.read()
        if not ret:
            return
        # 显示 RGB 画面
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.cam_label.width(), self.cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_label.setPixmap(pix)

        # 自动触发深度计算（复用缓存模型，不再创建临时目录）
        if not self._cam_depth_running and self._cached_model is not None:
            self._cam_depth_running = True
            self._cam_depth_worker = CamDepthWorker(frame, self._cached_model)
            self._cam_depth_worker.finished.connect(self._on_cam_depth_finished)
            self._cam_depth_worker.error.connect(self._on_cam_depth_error)
            self._cam_depth_worker.start()

    def _on_cam_depth_finished(self, depth):
        self._cam_depth_running = False
        from depth_anything_3.utils.visualize import visualize_depth
        d_vis = visualize_depth(depth)
        h, w, _ = d_vis.shape
        qimg = QImage(d_vis.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.cam_depth_label.width(), self.cam_depth_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_depth_label.setPixmap(pix)
        self.cam_depth_label.setText("")

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

        if is_video and len(paths) == 1:
            self.status_label.setText("⏳ 正在抽帧...")
            fps = self.fps_slider.value() / 10.0
            self._copy_worker = VideoFrameWorker(
                paths[0], self.pending_image_dir, fps=fps, max_width=640)
        else:
            self.status_label.setText("⏳ 正在上传...")
            self._copy_worker = FileCopyWorker(paths, self.pending_image_dir)

        self._copy_worker.progress.connect(self._on_upload_progress)
        self._copy_worker.finished.connect(self._on_upload_finished)
        self._copy_worker.error.connect(self._on_upload_error)
        self._copy_worker.start()

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
        sample_ratio = self.slider.value() / 100.0  # 0-100 → 0.0-1.0
        self._worker = DA3Worker(
            self.pending_image_dir, work_dir,
            "da3_streaming/configs/base_config.yaml",
            sample_ratio=sample_ratio,
            model_cache=getattr(self, '_cached_model', None),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_da3_finished)
        self._worker.error.connect(self._on_da3_error)
        self._worker.start()

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
            self._load_glb_in_viewer(glb_path)
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
        # 停止后台线程
        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        if hasattr(self, '_copy_worker') and self._copy_worker.isRunning():
            self._copy_worker.terminate()
            self._copy_worker.wait()

        # 关闭 HTTP 服务器
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None
            self._http_port = None

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
        self.pointcloud_viewer.setHtml("""
            <html><body style="background:#1a1a2e;display:flex;align-items:center;
            justify-content:center;height:100vh;margin:0;font-family:sans-serif">
            <div style="text-align:center;color:#e0e0e0">
            <div style="font-size:48px;margin-bottom:16px">&#127758;</div>
            <div style="font-size:16px">3D点云查看器</div>
            <div style="font-size:12px;color:#888;margin-top:8px">上传图片并运行推理后显示</div>
            </div></body></html>
        """)

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
            self._cam_depth_running = False
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
