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

import os, glob, shutil
import trimesh
from datetime import datetime


# ============================================================
# 后台工作线程
# ============================================================
class DA3Worker(QThread):
    finished = Signal(str, str)
    progress = Signal(str, int)
    error   = Signal(str)

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

            # 关键：移除 da3_streaming/ 防止模型加载时模块冲突导致 segfault
            sys.path.remove(_here)

            self._log("加载模型...", 5)
            config = load_config(self.config_path)

            imgs = (
                sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
                + sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
            )
            self._log(f"共 {len(imgs)} 帧", 10)

            da3 = DA3_Streaming(self.image_dir, self.output_dir, config)

            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            self._log("DA3 推理中...", 30)
            da3.run()
            da3.close()
            sys.stdout = old_stdout

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth-Anything-3")
        self.setGeometry(40, 40, 1500, 900)

        self.setStyleSheet("""
            QMainWindow           { background-color: #f5f5f5; }
            QGroupBox             { font-weight: bold; border: 2px solid #d0d0d0;
                                    border-radius: 8px; margin-top: 10px;
                                    padding-top: 10px; background-color: white; }
            QGroupBox::title      { subcontrol-origin: margin; left: 10px;
                                    padding: 0 5px; background-color: white; }
            QPushButton           { background-color: #4CAF50; color: white; border: none;
                                    padding: 8px 16px; border-radius: 4px;
                                    font-weight: bold; font-size: 13px; }
            QPushButton:hover     { background-color: #45a049; }
            QPushButton#danger    { background-color: #f44336; }
            QPushButton#danger:hover { background-color: #da190b; }
            QPushButton#secondary { background-color: #6c757d; }
            QPushButton#secondary:hover { background-color: #5a6268; }
            QTextEdit, QLineEdit  { border: 1px solid #d0d0d0; border-radius: 4px;
                                    padding: 4px; background-color: white; }
            QTabWidget::pane      { border: 1px solid #d0d0d0; border-radius: 4px;
                                    background-color: white; }
            QTabBar::tab          { background-color: #f0f0f0; padding: 8px 16px;
                                    margin-right: 2px; border-top-left-radius: 4px;
                                    border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: white;
                                    border-bottom: 2px solid #4CAF50; }
            QComboBox             { border: 1px solid #d0d0d0; border-radius: 4px;
                                    padding: 4px 8px; background-color: white; }
            QCheckBox             { spacing: 8px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        self.create_header(main_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_left_panel())
        splitter.addWidget(self.create_right_panel())
        splitter.setSizes([300, 1100])
        main_layout.addWidget(splitter)

        self.create_status_bar()

    # ----------------------------------------------------------
    def create_header(self, layout):
        w = QWidget()
        w.setStyleSheet("background-color: white; border-bottom: 2px solid #e0e0e0; padding: 2px 5px;")
        hl = QHBoxLayout(w)
        title = QLabel("🎯 DA3 - Depth Anything V3")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        hl.addWidget(title)
        hl.addStretch()

        btn = ("padding: 2px 10px; font-size: 11px; "
               "QPushButton { background-color: #2196F3; } "
               "QPushButton:hover { background-color: #1976D2; }")
        self.save_btn = QPushButton("💾 保存")
        self.save_btn.setStyleSheet(btn)
        hl.addWidget(self.save_btn)
        self.export_btn = QPushButton("📤 导出")
        self.export_btn.setStyleSheet(btn)
        hl.addWidget(self.export_btn)
        layout.addWidget(w)

    # ----------------------------------------------------------
    def create_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # --- 上传 ---
        g = QGroupBox("上传图片/视频")
        gl = QVBoxLayout(g)

        gl.addWidget(QPushButton("上传视频"))
        self.upload_image_btn = QPushButton("上传图片")
        self.upload_image_btn.setStyleSheet("QPushButton { background-color: #2196F3; }")
        self.upload_image_btn.clicked.connect(self._on_upload_images)
        gl.addWidget(self.upload_image_btn)

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
        run_btn = QPushButton("▶️ 运行推理")
        run_btn.setStyleSheet("padding: 12px; font-size: 14px;")
        al.addWidget(run_btn)
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
        f = QFrame()
        f.setStyleSheet("background-color: #1a1a2e; border-radius: 8px; border: 2px solid #16213e;")
        f.setMinimumHeight(450)
        fl = QVBoxLayout(f)
        self.pointcloud_label = QLabel("🌐 3D点云查看器")
        self.pointcloud_label.setAlignment(Qt.AlignCenter)
        self.pointcloud_label.setWordWrap(True)
        self.pointcloud_label.setStyleSheet("color: #e0e0e0; font-size: 14px; background: transparent;")
        fl.addWidget(self.pointcloud_label)
        l.addWidget(f)
        return t

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

    def _on_upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not files:
            return

        work_dir = os.path.join("workspace", "gui_session",
                                datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(work_dir, "images"), exist_ok=True)
        for f in files:
            shutil.copy(f, os.path.join(work_dir, "images", os.path.basename(f)))

        self.status_label.setText("⏳ 正在推理...")
        self.upload_image_btn.setEnabled(False)

        self._worker = DA3Worker(
            os.path.join(work_dir, "images"),
            work_dir,
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
        self.upload_image_btn.setEnabled(True)

        if glb_path and os.path.exists(glb_path):
            try:
                scene = trimesh.load(glb_path)
                pts = len(scene.vertices) if hasattr(scene, 'vertices') else "未知"
            except Exception:
                pts = "未知"

            self.pointcloud_label.setText(
                f"✅ 点云已生成\n\n"
                f"文件: {os.path.basename(glb_path)}\n"
                f"点数: {pts}\n\n"
                f"正在用 3D 查看器打开..."
            )
            os.startfile(os.path.abspath(glb_path))
        else:
            self.pointcloud_label.setText("⚠️ 推理完成但未找到 .glb 文件")

    def _on_da3_error(self, err):
        self.status_label.setText("❌ 推理失败")
        self.upload_image_btn.setEnabled(True)
        self.pointcloud_label.setText(f"❌ 错误:\n{err}")


# ============================================================
def main():
    app = QApplication(sys.argv)
    window = GradioLikeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
