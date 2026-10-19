"""
点云地面检测 + PyBullet 物理仿真 + Open3D 实时可视化
修改：随机障碍物点云在物理仿真中会真实阻挡小球滚动
"""

import open3d as o3d
import numpy as np
import pybullet as p
import time


# 倾斜角度（度）
TILT_DEG = 1


# ============================================================
# 1. 生成倾斜地面 + 随机障碍物点云
#    同时返回障碍物的物理信息（中心 + 半径）
# ============================================================
def generate_synthetic_pointcloud(seed=None):
    """
    返回:
        pcd:         完整点云（地面 + 障碍物）
        obstacles:   障碍物列表，每个元素是 dict(center, radius, points)
        n_ground_pts: 地面点的数量（用于着色分段）
    """
    if seed is not None:
        np.random.seed(seed)

    tilt_rad = np.radians(TILT_DEG)

    # ---- 地面点云 ----
    x = np.linspace(-5, 5, 80)
    y = np.linspace(-5, 5, 80)
    xx, yy = np.meshgrid(x, y)
    zz = np.tan(tilt_rad) * xx
    ground = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # ---- 随机障碍物（放在小球滚落路径 x<0 上）----
    obstacles = []
    n_obstacles = 4
    max_attempts = 200
    attempts = 0

    while len(obstacles) < n_obstacles and attempts < max_attempts:
        attempts += 1

        # 在小球滚落路径上随机选位置
        ox = np.random.uniform(-4.5, -1.0)
        oy = np.random.uniform(-1.5, 1.5)
        radius = np.random.uniform(0.35, 0.6)

        # 避免与已有障碍物重叠（否则物理上会卡在一起）
        ok = True
        for ob in obstacles:
            d = np.linalg.norm(
                np.array([ox, oy]) - np.array([ob['center'][0], ob['center'][1]])
            )
            if d < radius + ob['radius'] + 0.3:
                ok = False
                break
        if not ok:
            continue

        # 球心在斜坡上方，底部略嵌入斜坡（防止悬空）
        oz = np.tan(tilt_rad) * ox + radius * 0.8
        center = [ox, oy, oz]

        # 生成球形点云（高斯分布）
        pts = np.random.normal(0, radius * 0.4, (200, 3)) + center

        obstacles.append({
            'center': center,
            'radius': radius,
            'points': pts
        })

    # ---- 合并点云 ----
    all_pts_list = [ground] + [ob['points'] for ob in obstacles]
    all_points = np.vstack(all_pts_list)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    return pcd, obstacles, len(ground)


# ============================================================
# 2. 生成点云
# ============================================================
pcd, obstacles, n_ground_pts = generate_synthetic_pointcloud()

print(f"点云总点数: {len(pcd.points)}")
print(f"地面点数: {n_ground_pts}")
print(f"随机障碍物数量: {len(obstacles)}")
for i, ob in enumerate(obstacles):
    print(f"  障碍物 {i}: 中心={np.round(ob['center'], 2)}, 半径={ob['radius']:.2f}")

# ============================================================
# 3. RANSAC 地面检测
# ============================================================
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.05, ransac_n=3, num_iterations=1000
)
a, b, c, d = plane_model
print(f"平面方程: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

detected_tilt = np.degrees(np.arctan2(np.sqrt(a**2 + b**2), abs(c)))
print(f"检测到平面倾斜角: {detected_tilt:.2f}°")

# 着色：地面绿色，障碍物橙色
colors = np.zeros((len(pcd.points), 3))
colors[:] = [0.5, 0.5, 0.5]
colors[inliers] = [0, 1, 0]
idx = n_ground_pts
for ob in obstacles:
    n_pts = len(ob['points'])
    colors[idx:idx + n_pts] = [0.8, 0.4, 0.1]
    idx += n_pts
pcd.colors = o3d.utility.Vector3dVector(colors)

# ============================================================
# 4. 连接 PyBullet
# ============================================================
p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)

tilt_rad = np.radians(TILT_DEG)

# ---- 倾斜地面 ----
ground_collision = p.createCollisionShape(
    p.GEOM_BOX, halfExtents=[5, 5, 0.05]
)
ground_quat = p.getQuaternionFromEuler([0, -tilt_rad, 0])
normal = np.array([-np.sin(tilt_rad), 0, np.cos(tilt_rad)])
ground_pos = -normal * 0.05

ground_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=ground_collision,
    basePosition=ground_pos,
    baseOrientation=ground_quat
)
p.changeDynamics(ground_id, -1, restitution=0.9)

# ============================================================
# ✅ 关键新增：为每个随机障碍物创建物理碰撞体
# ============================================================
obstacle_ids = []
for ob in obstacles:
    coll = p.createCollisionShape(p.GEOM_SPHERE, radius=ob['radius'])
    body = p.createMultiBody(
        baseMass=0,                       # 静态障碍物
        baseCollisionShapeIndex=coll,
        basePosition=ob['center']
    )
    p.changeDynamics(body, -1, restitution=0.3, lateralFriction=0.5)
    obstacle_ids.append(body)

print(f"✅ 已在 PyBullet 中创建 {len(obstacle_ids)} 个障碍物碰撞体")

# ---- 小球 ----
ball_radius = 0.3
BALL_START = [0.0, 0.0, 4.0]
ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
ball_id = p.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=ball_collision,
    basePosition=BALL_START
)
p.changeDynamics(ball_id, -1, restitution=0.9, lateralFriction=0.3)

# ============================================================
# 5. Open3D 可视化器
# ============================================================
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="倾斜地面 + 随机障碍物（按 R 重置）",
                  width=1200, height=800)

# 降采样点云
pcd_render = pcd.voxel_down_sample(0.03)
vis.add_geometry(pcd_render)

# 地面网格
ground_mesh = o3d.geometry.TriangleMesh.create_box(
    width=10, height=10, depth=0.02
)
ground_mesh.paint_uniform_color([0.3, 0.8, 0.3])
ground_mesh.translate([-5, -5, -0.02])
R = np.array([
    [np.cos(tilt_rad), 0, -np.sin(tilt_rad)],
    [0, 1, 0],
    [np.sin(tilt_rad), 0, np.cos(tilt_rad)]
])
ground_mesh.rotate(R, center=(0, 0, 0))
vis.add_geometry(ground_mesh)

# ✅ 障碍物碰撞体可视化（橙色线框球，与点云叠加，直观看到物理范围）
for ob in obstacles:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=ob['radius'])
    sphere.compute_vertex_normals()
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    lineset.paint_uniform_color([0.9, 0.5, 0.1])
    lineset.translate(ob['center'])
    vis.add_geometry(lineset)

# 小球
ball_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=ball_radius)
ball_mesh.paint_uniform_color([1, 0, 0])
ball_mesh.compute_vertex_normals()
ball_mesh.translate(BALL_START)
vis.add_geometry(ball_mesh)

# 视角
view_control = vis.get_view_control()
view_control.set_lookat([0, 0, 1.0])
view_control.set_up([0, 0, 1])
view_control.set_front([0, -1, 0])
view_control.set_zoom(0.55)

render_option = vis.get_render_option()
render_option.point_size = 1.5
render_option.show_coordinate_frame = False
render_option.light_on = False

# ============================================================
# 6. 键盘回调
# ============================================================
reset_flag = [False]

def on_key_action(vis_obj, action, mods):
    if action == 1:
        reset_flag[0] = True
        print("🔄 检测到 R 键按下，准备重置小球...")

vis.register_key_action_callback(ord("R"), on_key_action)

# ============================================================
# 7. 仿真循环
# ============================================================
print("开始仿真（倾斜地面 + 随机障碍物）...")
print(f"💡 地面倾斜 {TILT_DEG}°，小球会沿斜坡向 -x 方向滚落")
print("💡 障碍物会真实阻挡小球，每次运行位置随机")
print("💡 按 R 键可以让小球重新掉落")

physics_dt = 1/240
render_interval = 1/30
sim_time = 0.0
wall_start = time.time()
last_render_time = 0.0
prev_pos = np.array(BALL_START, dtype=float)
max_sim_time = 20.0
is_static = False
initial_x = BALL_START[0]

while sim_time < max_sim_time:
    wall_elapsed = time.time() - wall_start

    if reset_flag[0]:
        p.resetBasePositionAndOrientation(ball_id, BALL_START, [0, 0, 0, 1])
        p.resetBaseVelocity(ball_id, [0, 0, 0], [0, 0, 0])

        new_pos = np.array(BALL_START, dtype=float)
        ball_mesh.translate(new_pos - prev_pos)
        prev_pos = new_pos.copy()
        vis.update_geometry(ball_mesh)

        sim_time = 0.0
        wall_start = time.time()
        last_render_time = 0.0
        is_static = False
        reset_flag[0] = False

        vis.poll_events()
        vis.update_renderer()
        print("✅ 小球已重置，重新开始掉落！")
        continue

    while sim_time < wall_elapsed:
        p.stepSimulation()
        sim_time += physics_dt

    if wall_elapsed - last_render_time >= render_interval:
        pos, _ = p.getBasePositionAndOrientation(ball_id)
        pos = np.array(pos)
        ball_mesh.translate(pos - prev_pos)
        prev_pos = pos

        vis.update_geometry(ball_mesh)
        vis.poll_events()
        vis.update_renderer()
        last_render_time = wall_elapsed

    if not is_static and int(sim_time * 2) > int((sim_time - physics_dt) * 2):
        pos, _ = p.getBasePositionAndOrientation(ball_id)
        vel, _ = p.getBaseVelocity(ball_id)
        x_shift = pos[0] - initial_x
        print(f"时间 {sim_time:.1f}s | 位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
              f"X位移: {x_shift:+.3f}m")

    if not is_static and sim_time > 3.0:
        pos, _ = p.getBasePositionAndOrientation(ball_id)
        vel, _ = p.getBaseVelocity(ball_id)
        if np.linalg.norm(vel) < 0.05:
            # 判断小球最终是否被某个障碍物挡住
            blocked = False
            for ob in obstacles:
                d = np.linalg.norm(np.array(pos) - np.array(ob['center']))
                if d < ob['radius'] + ball_radius + 0.2:
                    blocked = True
                    break

            print(f"✅ 小球在 {sim_time:.1f}s 后静止")
            print(f"   最终位置: x={pos[0]:.3f}m, y={pos[1]:.3f}m, z={pos[2]:.3f}m")
            if blocked:
                print("   💡 小球被随机障碍物挡住！")
            else:
                print("   💡 小球滚到了斜坡边缘")
            print("💡 按 R 键可以让它重新掉落")
            is_static = True

# ============================================================
# 8. 清理
# ============================================================
vis.destroy_window()
p.disconnect()
print("仿真结束")