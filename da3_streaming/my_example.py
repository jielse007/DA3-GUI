import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QSplitter, QTextEdit, QGroupBox, QTabWidget,
    QFrame, QSlider, QCheckBox, QComboBox, QLineEdit,
    QGridLayout, QScrollArea
)
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette

import os, glob, shutil, socket, threading, time
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
    def __init__(self, image_dir, output_dir, config_path):
        super().__init__()
        self.image_dir   = image_dir
        self.output_dir  = output_dir
        self.config_path = config_path

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

            # 加载 da3_streaming.py（文件名 = 目录名，用 importlib 避免歧义）
            _spec = importlib.util.spec_from_file_location(
                "da3_streaming", os.path.join(_here, "da3_streaming.py")
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)

            DA3_Streaming = _mod.DA3_Streaming

            from loop_utils.config_utils import load_config

            # 关键：移除 da3_streaming/ 防止模型加载时模块冲突导致段错误
            sys.path.remove(_here)

            self._log("加载模型...", 5)
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
                    sample_ratio=1.0,
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

        self.create_status_bar()                            # 调用create_status_bar方法创建状态栏

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

        gl.addWidget(QPushButton("上传视频"))
        self.upload_image_btn = QPushButton("上传图片")
        self.upload_image_btn.setStyleSheet("""
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
        self.upload_image_btn.clicked.connect(self._on_upload_images)
        gl.addWidget(self.upload_image_btn)

        # 上传进度条
        self.upload_progress = QFrame()
        self.upload_progress.setStyleSheet("background-color: #e0e0e0; border-radius: 3px;")
        self.upload_progress.setFixedHeight(6)
        self.upload_progress.hide()
        gl.addWidget(self.upload_progress)
        self.upload_progress_fill = QFrame(self.upload_progress)
        self.upload_progress_fill.setStyleSheet("background-color: #2196F3; border-radius: 3px;")
        self.upload_progress_fill.setFixedHeight(6)
        self.upload_progress_fill.setFixedWidth(0)

        layout.addWidget(g)

        # --- 配置 ---
        cg = QGroupBox("模型配置")
        cl = QGridLayout(cg)
        cl.addWidget(QLabel("模型:"), 0, 0)
        cmb = QComboBox()
        cmb.addItems(["depth_anything_v3_small", "depth_anything_v3_base", "depth_anything_v3_large"])
        cl.addWidget(cmb, 0, 1)
        cl.addWidget(QLabel("设备:"), 1, 0)
        d2 = QComboBox()
        d2.addItems(["cuda:0", "cpu"])
        cl.addWidget(d2, 1, 1)
        layout.addWidget(cg)
        layout.addStretch()

        # --- 操作 ---
        ag = QGroupBox("操作")
        al = QVBoxLayout(ag)
        self.run_btn = QPushButton("▶️ 运行推理")
        self.run_btn.setStyleSheet("padding: 12px; font-size: 14px;")
        self.run_btn.clicked.connect(self._on_run_inference)
        self.run_btn.setEnabled(False)  # 先灰掉，上传图片后才可用
        al.addWidget(self.run_btn)
        row = QHBoxLayout()
        s = QPushButton("⏹️ 停止"); s.setObjectName("danger"); row.addWidget(s)
        c = QPushButton("🗑️ 清空全部"); c.setObjectName("secondary"); row.addWidget(c)
        al.addLayout(row)
        layout.addWidget(ag)

        # --- 进度 ---
        pg = QGroupBox("进度")
        pl = QVBoxLayout(pg)
        self.progress_label = QLabel("状态: 就绪")
        self.progress_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
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
        tabs.addTab(self._tab_camera(), "3DGS")

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
        """启动本地 HTTP 服务器以提供 GLB 文件（Babylon.js 需要 HTTP 加载）"""
        if self._http_server:
            return self._http_port

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
<head><meta charset="utf-8"><title>3D Viewer</title>
<style>
  html, body {{ margin:0; padding:0; overflow:hidden; height:100%; background:#222; }}
  #canvas {{ width:100%; height:100%; outline:none; }}
  #info {{ position:absolute; bottom:10px; right:10px; color:#888;
           font-family:sans-serif; font-size:11px; pointer-events:none; }}
  #loading {{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
              color:#ccc; font-family:sans-serif; font-size:14px; }}
</style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div id="loading">加载中...</div>
  <div id="info">鼠标左键旋转 | 滚轮缩放 | 右键平移</div>
  <script src="https://cdn.babylonjs.com/babylon.js"></script>
  <script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
  <script>
    const canvas = document.getElementById("canvas");
    const engine = new BABYLON.Engine(canvas, true, {{ preserveDrawingBuffer: true, stencil: true }});
    const scene = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color4(0.15, 0.15, 0.2, 1);

    const camera = new BABYLON.ArcRotateCamera("cam", Math.PI/3, Math.PI/3, 5,
                                               BABYLON.Vector3.Zero(), scene);
    camera.attachControl(canvas, true);
    camera.lowerRadiusLimit = 0.0001;   /* 允许缩放到非常近 */
    camera.upperRadiusLimit = 10000;    /* 允许缩放到非常远 */
    camera.minZ = 0.0001;               /* 近裁剪面极小，防止切掉模型 */
    camera.wheelPrecision = 10;         /* 滚轮更灵敏 */

    const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0,1,0), scene);
    light.intensity = 0.8;

    BABYLON.SceneLoader.ImportMesh("", "", "{glb_url}", scene,
      function(meshes) {{
        document.getElementById("loading").style.display = "none";
        if (meshes.length > 0) {{
          const bb = meshes[0].getBoundingInfo().boundingBox;
          const center = bb.center;
          const size = bb.extendSize.length();
          camera.target = center;
          camera.radius = Math.max(size * 2.5, 0.1);
          camera.lowerRadiusLimit = size * 0.001;
          console.log('Model loaded: center', center, 'size', size, 'radius', camera.radius);
        }}
      }},
      function(evt) {{
        if (evt.lengthComputable) {{
          const pct = Math.round(evt.loaded * 100 / evt.total);
          document.getElementById("loading").textContent = "加载中... " + pct + "%";
        }}
      }},
      function() {{
        document.getElementById("loading").textContent = "加载失败";
      }}
    );

    engine.runRenderLoop(() => scene.render());
    window.addEventListener("resize", () => engine.resize());
  </script>
</body>
</html>""")

    def _tab_depth(self):
        t = QWidget()
        l = QVBoxLayout(t)
        f = QFrame()
        f.setStyleSheet("background-color: #2c3e50; border-radius: 8px; border: 2px solid #34495e;")
        f.setMinimumHeight(500)
        hl = QHBoxLayout(f)
        for text in ["RGB图像显示区域", "深度图像显示区域"]:
            lb = QLabel(text)
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet("color: #ecf0f1; font-size: 18px; background: transparent;")
            hl.addWidget(lb)
        l.addWidget(f)
        return t

    def _tab_camera(self):
        t = QWidget()
        l = QVBoxLayout(t)
        f = QFrame()
        f.setStyleSheet("background: white; border-radius: 8px; border: 1px solid #e0e0e0;")
        f.setMinimumHeight(300)
        fl = QVBoxLayout(f)
        headers = ["帧ID", "位置 (x,y,z)", "旋转 (四元数)", "深度中值"]
        hl = QHBoxLayout()
        for h in headers:
            lb = QLabel(h)
            lb.setStyleSheet("font-weight: bold; color: #2c3e50;")
            hl.addWidget(lb)
        fl.addLayout(hl)
        for row in [
            ("001", "(0.12, 0.34, 1.56)", "(0.71, 0.00, 0.71, 0.00)", "1.54m"),
            ("002", "(0.45, 0.12, 1.78)", "(0.50, 0.50, 0.50, 0.50)", "2.34m"),
        ]:
            rw = QWidget()
            rl = QHBoxLayout(rw)
            for v in row:
                lb = QLabel(v); lb.setStyleSheet("font-size: 12px; color: #333;"); rl.addWidget(lb)
            fl.addWidget(rw)
        l.addWidget(f)
        return t

    # ----------------------------------------------------------
    def create_status_bar(self):
        sb = self.statusBar()
        sb.setStyleSheet("background: #f8f9fa; color: #333; border-top: 1px solid #e0e0e0; padding: 4px 8px;")
        self.status_label = QLabel("✅ 就绪")
        sb.addWidget(self.status_label)
        sb.addPermanentWidget(QLabel("⚡ GPU: 未使用"))

    # ============================================================
    # 槽
    # ============================================================
    def _set_progress(self, pct):
        w = int(self.progress_track.width() * pct / 100)
        self.progress_fill.setFixedWidth(max(2, w))

    def _on_progress(self, msg, pct):
        self.status_label.setText(f"⏳ {msg}")
        self.progress_label.setText(f"状态: {msg} ({pct}%)")
        self._set_progress(pct)

    # ============================================================
    # 上传图片：只复制文件，不推理
    # ============================================================
    def _on_upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not files:
            return

        self.pending_image_dir = os.path.join(
            "workspace", "gui_session",
            datetime.now().strftime("%Y%m%d_%H%M%S"), "images"
        )
        os.makedirs(self.pending_image_dir, exist_ok=True)

        self.upload_image_btn.setEnabled(False)
        self.upload_progress.show()
        self.upload_progress_fill.setFixedWidth(0)
        self.status_label.setText("⏳ 正在上传...")

        self._copy_worker = FileCopyWorker(files, self.pending_image_dir)
        self._copy_worker.progress.connect(self._on_upload_progress)
        self._copy_worker.finished.connect(self._on_upload_finished)
        self._copy_worker.error.connect(self._on_upload_error)
        self._copy_worker.start()

    def _on_upload_progress(self, pct):
        w = int(self.upload_progress.width() * pct / 100)
        self.upload_progress_fill.setFixedWidth(max(2, w))
        self.status_label.setText(f"⏳ 上传中... {pct}%")

    def _on_upload_finished(self, dest_dir):
        self.status_label.setText("📁 上传成功")
        self.upload_image_btn.setEnabled(True)
        self.upload_progress.hide()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ 运行推理")
        self.progress_label.setText(f"状态: 就绪 (已选 {dest_dir})")

    def _on_upload_error(self, err):
        self.status_label.setText("❌ 上传失败")
        self.upload_image_btn.setEnabled(True)
        self.upload_progress.hide()
        self.progress_label.setText(f"错误: {err}")

    # ============================================================
    # 运行推理
    # ============================================================
    def _on_run_inference(self):
        if not hasattr(self, 'pending_image_dir') or not self.pending_image_dir:
            return

        self.run_btn.setEnabled(False)
        self.upload_image_btn.setEnabled(False)
        self.status_label.setText("⏳ 正在推理...")

        work_dir = os.path.dirname(self.pending_image_dir)
        self._worker = DA3Worker(
            self.pending_image_dir, work_dir,
            "da3_streaming/configs/base_config.yaml"
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_da3_finished)
        self._worker.error.connect(self._on_da3_error)
        self._worker.start()

    def _on_da3_finished(self, output_dir, glb_path):
        self.status_label.setText("✅ 推理完成")
        self.progress_label.setText("状态: 完成 (100%)")
        self._set_progress(100)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ 运行推理")

        if glb_path and os.path.exists(glb_path):
            self.progress_label.setText(f"状态: 加载 3D 模型中 ({os.path.basename(glb_path)})")
            self._load_glb_in_viewer(glb_path)
        else:
            self.progress_label.setText("状态: 完成（无 GLB 文件）")

    def _on_da3_error(self, err):
        self.status_label.setText("❌ 推理失败")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ 运行推理")
        self.upload_image_btn.setEnabled(True)
        self.progress_label.setText(f"错误:\n{err[:200]}")


# ============================================================
def main():
    app = QApplication(sys.argv)
    window = GradioLikeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
