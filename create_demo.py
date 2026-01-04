"""
デモ動画生成スクリプト
シミュレーションを10秒間録画してGIFとして保存
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time

# 設定
G = 1.0
SOFTENING = 0.05
DISPLAY_RANGE = 1.5
VELOCITY_ARROW_SCALE = 0.3
FORCE_ARROW_SCALE = 0.15
MASS_MIN = 0.5
MASS_MAX = 2.0
BASE_DT = 0.001
MIN_DT = 0.0001
MAX_DT = 0.01


def compute_accelerations(positions, masses, softening):
    n = len(masses)
    accelerations = np.zeros_like(positions)
    eps2 = softening ** 2
    for i in range(n):
        r_ij = positions - positions[i]
        r2 = np.sum(r_ij ** 2, axis=1) + eps2
        r2[i] = 1.0
        inv_r3 = r2 ** (-1.5)
        inv_r3[i] = 0.0
        acc = G * np.sum(masses[:, np.newaxis] * r_ij * inv_r3[:, np.newaxis], axis=0)
        accelerations[i] = acc
    return accelerations


def compute_forces(positions, masses, softening):
    n = len(masses)
    forces = np.zeros_like(positions)
    eps2 = softening ** 2
    for i in range(n):
        for j in range(n):
            if i != j:
                r_ij = positions[j] - positions[i]
                r2 = np.dot(r_ij, r_ij) + eps2
                force_mag = G * masses[i] * masses[j] / r2
                force_dir = r_ij / np.sqrt(r2)
                forces[i] += force_mag * force_dir
    return forces


def compute_min_distance(positions):
    n = len(positions)
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[j] - positions[i])
            min_dist = min(min_dist, dist)
    return min_dist


def adaptive_timestep(positions):
    min_dist = compute_min_distance(positions)
    factor = min(1.0, min_dist / 0.3)
    dt = BASE_DT * factor
    return max(MIN_DT, min(MAX_DT, dt))


def rk4_step(positions, velocities, masses):
    dt = adaptive_timestep(positions)
    k1_r = velocities
    k1_v = compute_accelerations(positions, masses, SOFTENING)
    k2_r = velocities + 0.5 * dt * k1_v
    k2_v = compute_accelerations(positions + 0.5 * dt * k1_r, masses, SOFTENING)
    k3_r = velocities + 0.5 * dt * k2_v
    k3_v = compute_accelerations(positions + 0.5 * dt * k2_r, masses, SOFTENING)
    k4_r = velocities + dt * k3_v
    k4_v = compute_accelerations(positions + dt * k3_r, masses, SOFTENING)
    new_pos = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_vel = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return new_pos, new_vel, dt


def generate_initial_conditions(n_bodies):
    np.random.seed(42)  # 再現性のための固定シード
    masses = MASS_MIN + np.random.rand(n_bodies) * (MASS_MAX - MASS_MIN)
    positions = np.random.randn(n_bodies, 3) * 0.5
    positions = np.clip(positions, -1.0, 1.0)
    center_of_mass = np.average(positions, axis=0, weights=masses)
    positions -= center_of_mass
    velocities = np.random.randn(n_bodies, 3) * 0.4
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    velocities -= total_momentum / np.sum(masses)
    return positions, velocities, masses


def create_demo_gif():
    """デモGIFを作成"""
    print("🎬 デモ動画生成を開始...")
    
    n_bodies = 3
    positions, velocities, masses = generate_initial_conditions(n_bodies)
    
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    
    fig = plt.figure(figsize=(10, 8), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
    
    ax.set_xlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax.set_ylim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax.set_zlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.tick_params(colors='white')
    ax.set_title('N-Body Problem Simulator', color='white', fontsize=14, fontweight='bold')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # 情報テキスト
    info_text = fig.text(0.02, 0.02, '', color='#00ff88', fontsize=9,
                         fontfamily='monospace', verticalalignment='bottom')
    
    # 物体
    bodies = []
    trails = []
    velocity_arrows = []
    force_arrows = []
    
    for i in range(n_bodies):
        body, = ax.plot([], [], [], 'o', color=colors[i], markersize=12,
                       markeredgecolor='white', markeredgewidth=1.5)
        bodies.append(body)
        trail, = ax.plot([], [], [], '-', color=colors[i], alpha=0.5, linewidth=1.5)
        trails.append(trail)
        arrow, = ax.plot([], [], [], '-', color=colors[i], linewidth=2, alpha=0.8)
        velocity_arrows.append(arrow)
        force, = ax.plot([], [], [], '-', color='#ff4444', linewidth=2, alpha=0.8)
        force_arrows.append(force)
    
    # 状態
    state = {'positions': positions, 'velocities': velocities, 'masses': masses}
    sim_time = [0.0]
    azim = [30]
    trail_history = [[] for _ in range(n_bodies)]
    max_trail = 300
    show_forces = True
    
    total_frames = 300  # 10秒 @ 30fps
    
    def update(frame):
        # シミュレーション進行
        for _ in range(8):
            state['positions'], state['velocities'], dt = rk4_step(
                state['positions'], state['velocities'], state['masses']
            )
            sim_time[0] += dt
        
        # 軌跡更新
        for i in range(n_bodies):
            trail_history[i].append(state['positions'][i].copy())
            if len(trail_history[i]) > max_trail:
                trail_history[i].pop(0)
        
        # 力計算
        forces = compute_forces(state['positions'], state['masses'], SOFTENING)
        
        # 情報表示
        info_text.set_text(f'Time: {sim_time[0]:.1f}  |  Press [F] for forces  [P] to predict')
        
        # 描画更新
        for i in range(n_bodies):
            x, y, z = state['positions'][i]
            vx, vy, vz = state['velocities'][i]
            mass = state['masses'][i]
            
            size = 8 + (mass - MASS_MIN) * 6
            
            bodies[i].set_data([x], [y])
            bodies[i].set_3d_properties([z])
            bodies[i].set_markersize(size)
            
            if trail_history[i]:
                trail_arr = np.array(trail_history[i])
                trails[i].set_data(trail_arr[:, 0], trail_arr[:, 1])
                trails[i].set_3d_properties(trail_arr[:, 2])
            
            # 速度ベクトル
            velocity_arrows[i].set_data([x, x + vx * VELOCITY_ARROW_SCALE], 
                                        [y, y + vy * VELOCITY_ARROW_SCALE])
            velocity_arrows[i].set_3d_properties([z, z + vz * VELOCITY_ARROW_SCALE])
            
            # 力ベクトル
            if show_forces:
                fx, fy, fz = forces[i] * FORCE_ARROW_SCALE
                force_arrows[i].set_data([x, x+fx], [y, y+fy])
                force_arrows[i].set_3d_properties([z, z+fz])
        
        # 視点回転
        azim[0] += 0.6
        ax.view_init(elev=20, azim=azim[0])
        
        # 進捗表示
        if frame % 30 == 0:
            print(f"  進捗: {frame}/{total_frames} ({100*frame/total_frames:.0f}%)")
        
        return bodies + trails + velocity_arrows + force_arrows + [info_text]
    
    print("📹 録画中...")
    anim = FuncAnimation(fig, update, frames=total_frames, blit=False, interval=33)
    
    output_file = 'demo.gif'
    print(f"💾 保存中: {output_file}")
    anim.save(output_file, writer='pillow', fps=30, dpi=80)
    
    print(f"✅ 完了: {output_file}")
    plt.close()
    
    return output_file


if __name__ == "__main__":
    create_demo_gif()
