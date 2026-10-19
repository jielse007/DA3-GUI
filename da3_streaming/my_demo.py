import pybullet as p
import pybullet_data
import time
import sys
import threading
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

class PyBulletThread(QThread):
    """PyBullet仿真线程"""
    update_signal = Signal(float, float)  # 高度, 速度
    ball_stopped_signal = Signal()  # 球体停止信号
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.reset_flag = False
        self.client = None
        self.sphere_body = None
        self.sphere_radius = 0.2
        self.start_height = 3.0
        self.is_ball_stopped = False
        self.bounce_count = 0  # 弹跳计数
        
    def run(self):
        """运行仿真主循环"""
        # 初始化PyBullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 设置重力
        p.setGravity(0, 0, -9.81)
        
        # 加载地面
        plane_id = p.loadURDF("plane.urdf")
        
        # 修改地面的物理属性，使其更有弹性
        p.changeDynamics(plane_id, -1, restitution=0.5, lateralFriction=0.5)

        # 修改地面的物理属性，硬地无法弹跳
        # p.loadURDF("plane.urdf")
        
        # 创建球体
        self.create_sphere()
        
        # 设置球体的物理属性 - 关键：提高恢复系数
        p.changeDynamics(
            self.sphere_body, 
            -1,  # -1 表示所有链接
            restitution=0.85,  # 恢复系数，控制弹跳高度
            lateralFriction=0.3,
            spinningFriction=0.01,
            rollingFriction=0.01
        )
        
        # 给球体初始速度（稍微向下），确保开始下落
        p.resetBaseVelocity(
            self.sphere_body,
            linearVelocity=[0, 0, 0.1]  # 微小向下速度
        )
        
        print("🔬 仿真启动 - 恢复系数: 0.85")
        print("="*50)
        
        current_time = 0.0
        dt = 1/240
        step = 0
        
        while self.running:
            # 检查是否需要重置
            if self.reset_flag:
                self.reset_ball()
                self.reset_flag = False
                self.is_ball_stopped = False
                self.bounce_count = 0
                print("🔄 小球已重置")
            
            p.stepSimulation()
            
            # 获取球体状态
            pos, _ = p.getBasePositionAndOrientation(self.sphere_body)
            vel, _ = p.getBaseVelocity(self.sphere_body)
            
            # 检测弹跳事件（速度方向从负变正）
            if step > 1:
                # 获取之前的速度（需要存储历史）
                if hasattr(self, 'prev_vel'):
                    if self.prev_vel < -0.1 and vel[2] > 0.1:
                        self.bounce_count += 1
                        print(f"💥 第 {self.bounce_count} 次弹跳! 高度: {pos[2]:.3f}m, 速度: {vel[2]:.3f}m/s")
            
            # 保存当前速度供下次比较
            self.prev_vel = vel[2]
            
            # 发送更新信号
            self.update_signal.emit(pos[2], vel[2])
            
            # 检测球体是否完全静止（使用更宽松的条件）
            if pos[2] < self.sphere_radius + 0.02 and abs(vel[2]) < 0.02:
                if not self.is_ball_stopped:
                    self.is_ball_stopped = True
                    self.ball_stopped_signal.emit()
                    print(f"🛑 球体静止! 总弹跳次数: {self.bounce_count}")
            else:
                self.is_ball_stopped = False
            
            current_time += dt
            step += 1
            time.sleep(dt * 0.3)  # 稍微加快速度
        
        p.disconnect()
    
    def create_sphere(self):
        """创建球体"""
        if self.sphere_body is not None:
            p.removeBody(self.sphere_body)
        
        # 创建碰撞形状
        sphere_shape = p.createCollisionShape(
            p.GEOM_SPHERE, 
            radius=self.sphere_radius
        )
        
        # 创建多体
        self.sphere_body = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=sphere_shape,
            basePosition=[0, 0, self.start_height],
            baseOrientation=[0, 0, 0, 1]
        )
    
    def reset_ball(self):
        """重置球体到初始位置"""
        if self.sphere_body is not None:
            # 重置位置
            p.resetBasePositionAndOrientation(
                self.sphere_body,
                [0, 0, self.start_height],
                [0, 0, 0, 1]
            )
            # 重置速度
            p.resetBaseVelocity(
                self.sphere_body,
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0]
            )
            self.bounce_count = 0
    
    def reset_simulation(self):
        """外部调用重置"""
        self.reset_flag = True
    
    def stop(self):
        """停止仿真"""
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # 启动PyBullet仿真线程
        self.sim_thread = PyBulletThread()
        self.sim_thread.update_signal.connect(self.update_display)
        self.sim_thread.ball_stopped_signal.connect(self.on_ball_stopped)
        self.sim_thread.start()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('3D小球弹跳仿真')
        self.setGeometry(100, 100, 500, 300)
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2c3e50, stop:1 #34495e);
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QGroupBox {
                color: white;
                border: 2px solid #5d6d7e;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel('🎮 3D小球弹跳仿真')
        title_label.setStyleSheet('font-size: 24px; font-weight: bold; color: #ecf0f1;')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 信息显示区域
        info_group = QGroupBox('实时状态')
        info_layout = QGridLayout()
        info_group.setLayout(info_layout)
        
        # 高度显示
        self.height_label = QLabel('高度: 0.00 m')
        self.height_label.setStyleSheet('color: #2ecc71; font-size: 18px;')
        info_layout.addWidget(self.height_label, 0, 0)
        
        # 速度显示
        self.velocity_label = QLabel('速度: 0.00 m/s')
        self.velocity_label.setStyleSheet('color: #e74c3c; font-size: 18px;')
        info_layout.addWidget(self.velocity_label, 0, 1)
        
        # 弹跳次数显示
        self.bounce_label = QLabel('弹跳次数: 0')
        self.bounce_label.setStyleSheet('color: #f1c40f; font-size: 18px;')
        info_layout.addWidget(self.bounce_label, 1, 0)
        
        # 状态显示
        self.status_label = QLabel('状态: ⬇️ 下落中')
        self.status_label.setStyleSheet('color: #f1c40f; font-size: 18px;')
        info_layout.addWidget(self.status_label, 1, 1)
        
        layout.addWidget(info_group)
        
        # 控制按钮区域
        button_layout = QHBoxLayout()
        
        # 重置按钮
        self.reset_btn = QPushButton('🔄 重置小球')
        self.reset_btn.clicked.connect(self.reset_simulation)
        self.reset_btn.setMinimumHeight(50)
        button_layout.addWidget(self.reset_btn)
        
        # 退出按钮
        self.exit_btn = QPushButton('❌ 退出')
        self.exit_btn.clicked.connect(self.close)
        self.exit_btn.setMinimumHeight(50)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #922b21;
            }
        """)
        button_layout.addWidget(self.exit_btn)
        
        layout.addLayout(button_layout)
        
        # 操作提示
        tip_label = QLabel('💡 小球会多次弹跳，最终静止。点击「重置小球」重新体验')
        tip_label.setStyleSheet('color: #bdc3c7; font-size: 12px; font-style: italic;')
        tip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tip_label)
        
        # 弹跳状态提示
        self.state_tip = QLabel('🚀 球体开始下落...')
        self.state_tip.setStyleSheet('color: #f39c12; font-size: 13px; font-weight: bold;')
        self.state_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.state_tip)
        
        # 弹跳计数显示
        self.bounce_display = QLabel('')
        self.bounce_display.setStyleSheet('color: #2ecc71; font-size: 14px; font-weight: bold;')
        self.bounce_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.bounce_display)
    
    def update_display(self, height, velocity):
        """更新界面显示"""
        self.height_label.setText(f'高度: {height:.2f} m')
        self.velocity_label.setText(f'速度: {velocity:.2f} m/s')
        
        # 更新状态和按钮
        if height < 0.22 and abs(velocity) < 0.02:
            # 球体静止
            self.status_label.setText('状态: 🛑 已静止')
            self.status_label.setStyleSheet('color: #95a5a6; font-size: 18px;')
            self.reset_btn.setEnabled(True)
            self.reset_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e67e22;
                    color: white;
                    border: none;
                    padding: 12px 30px;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #d35400;
                }
                QPushButton:pressed {
                    background-color: #a04000;
                }
            """)
            self.reset_btn.setText('🔄 重新下落')
            self.state_tip.setText('✅ 小球已静止，点击「重新下落」再次体验')
            self.state_tip.setStyleSheet('color: #2ecc71; font-size: 13px; font-weight: bold;')
            
        elif velocity < -0.1:
            # 下落中
            self.status_label.setText('状态: ⬇️ 下落中')
            self.status_label.setStyleSheet('color: #e74c3c; font-size: 18px;')
            self.reset_btn.setEnabled(True)
            self.reset_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 12px 30px;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #21618c;
                }
            """)
            self.reset_btn.setText('🔄 重置小球')
            self.state_tip.setText('📉 球体下落中...')
            self.state_tip.setStyleSheet('color: #e74c3c; font-size: 13px; font-weight: bold;')
            
        elif velocity > 0.1:
            # 弹跳中
            self.status_label.setText('状态: ⬆️ 弹跳中')
            self.status_label.setStyleSheet('color: #2ecc71; font-size: 18px;')
            self.reset_btn.setEnabled(True)
            self.reset_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 12px 30px;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #21618c;
                }
            """)
            self.reset_btn.setText('🔄 重置小球')
            self.state_tip.setText(f'📈 弹跳中! 速度: {velocity:.1f} m/s')
            self.state_tip.setStyleSheet('color: #2ecc71; font-size: 13px; font-weight: bold;')
        
        # 更新弹跳次数显示（从仿真线程获取）
        if hasattr(self.sim_thread, 'bounce_count'):
            self.bounce_label.setText(f'弹跳次数: {self.sim_thread.bounce_count}')
            if self.sim_thread.bounce_count > 0:
                self.bounce_display.setText(f'🏀 已弹跳 {self.sim_thread.bounce_count} 次')
            else:
                self.bounce_display.setText('')
    
    def on_ball_stopped(self):
        """球体完全停止时的回调"""
        pass  # 已经在 update_display 中处理了
    
    def reset_simulation(self):
        """重置仿真"""
        self.sim_thread.reset_simulation()
        self.status_label.setText('状态: 🔄 重置中...')
        self.status_label.setStyleSheet('color: #f39c12; font-size: 18px;')
        self.state_tip.setText('🔄 重置完成，重新下落')
        self.state_tip.setStyleSheet('color: #f39c12; font-size: 13px; font-weight: bold;')
        self.bounce_display.setText('')
        self.bounce_label.setText('弹跳次数: 0')
        
        # 恢复按钮样式
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.reset_btn.setText('🔄 重置小球')
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        self.sim_thread.stop()
        event.accept()


# ============================================================
# 主程序入口
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon())
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()