"""
三体問題シミュレーター 3D版（Three-Body Problem Simulator 3D）
- 自動リスタート機能付き：物体がキューブを出たら新しい初期条件で再スタート

物理モデル: 万有引力の法則（ニュートンの重力法則）
計算手法: 4次ルンゲ＝クッタ法（RK4）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time


# ============================================================
# 物理定数と設定
# ============================================================

G = 1.0
DT = 0.001
T_MAX = 20.0
ANIMATION_INTERVAL = 30
TRAIL_LENGTH = 1500
SOFTENING = 0.05

# 表示範囲（固定スケール）
DISPLAY_RANGE = 1.5


# ============================================================
# 初期条件
# ============================================================

def get_figure8_initial_conditions_3d():
    """8の字解の初期条件"""
    masses = np.array([1.0, 1.0, 1.0])
    x1, y1 = 0.97000436, -0.24308753
    positions = np.array([[ x1,  y1, 0.0], [-x1, -y1, 0.0], [0.0, 0.0, 0.0]])
    vx3, vy3 = -0.93240737, -0.86473146
    velocities = np.array([[vx3/2, vy3/2, 0.0], [vx3/2, vy3/2, 0.0], [-vx3, -vy3, 0.0]])
    return positions, velocities, masses


def get_chaotic_initial_conditions_3d():
    """カオス的な動きを生成するランダム3D初期条件"""
    np.random.seed(int(time.time() * 1000) % (2**32))
    
    masses = np.array([1.0, 1.0 + 0.2 * np.random.randn(), 1.0 + 0.2 * np.random.randn()])
    masses = np.clip(masses, 0.5, 1.5)
    
    positions = np.random.randn(3, 3) * 0.8
    center_of_mass = np.average(positions, axis=0, weights=masses)
    positions -= center_of_mass
    
    velocities = np.random.randn(3, 3) * 0.3
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    velocities -= total_momentum / np.sum(masses)
    
    # 束縛状態を保証
    total_energy = _compute_energy_3d(positions, velocities, masses)
    while total_energy > -0.1:
        velocities *= 0.8
        total_energy = _compute_energy_3d(positions, velocities, masses)
    
    return positions, velocities, masses


def _compute_energy_3d(positions, velocities, masses):
    eps2 = SOFTENING ** 2
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    pe = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r2 = np.dot(positions[j] - positions[i], positions[j] - positions[i])
            pe -= G * masses[i] * masses[j] / np.sqrt(r2 + eps2)
    return ke + pe


# ============================================================
# 物理計算
# ============================================================

def compute_accelerations_3d(positions, masses):
    """加速度計算（Plummerソフトニング）"""
    forces = np.zeros_like(positions)
    eps2 = SOFTENING ** 2
    for i in range(3):
        for j in range(3):
            if i != j:
                r_ij = positions[j] - positions[i]
                r2 = np.dot(r_ij, r_ij)
                denom = (r2 + eps2) ** 1.5
                forces[i] += G * masses[i] * masses[j] * r_ij / denom
    return forces / masses[:, np.newaxis]


def rk4_step(positions, velocities, masses, dt):
    """RK4による1ステップ"""
    k1_r = velocities
    k1_v = compute_accelerations_3d(positions, masses)
    
    k2_r = velocities + 0.5 * dt * k1_v
    k2_v = compute_accelerations_3d(positions + 0.5 * dt * k1_r, masses)
    
    k3_r = velocities + 0.5 * dt * k2_v
    k3_v = compute_accelerations_3d(positions + 0.5 * dt * k2_r, masses)
    
    k4_r = velocities + dt * k3_v
    k4_v = compute_accelerations_3d(positions + dt * k3_r, masses)
    
    new_pos = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_vel = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return new_pos, new_vel


def is_out_of_bounds(positions, bound=DISPLAY_RANGE):
    """物体がキューブ範囲外に出たかチェック"""
    return np.any(np.abs(positions) > bound)


# ============================================================
# リアルタイム3Dアニメーション（自動リスタート機能付き）
# ============================================================

def run_realtime_animation():
    """リアルタイムシミュレーション + 自動リスタート"""
    
    # 初期化
    positions, velocities, masses = get_chaotic_initial_conditions_3d()
    
    # 軌跡保存用
    max_trail = 500
    trail_history = [[] for _ in range(3)]
    
    # カウンター
    generation = [1]  # リスト内でミュータブルに
    sim_time = [0.0]
    
    # カラー設定
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    
    # 3Dプロット設定
    fig = plt.figure(figsize=(12, 10), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
    
    ax.set_xlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax.set_ylim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax.set_zlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax.set_xlabel('X', color='white', fontsize=12)
    ax.set_ylabel('Y', color='white', fontsize=12)
    ax.set_zlabel('Z', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_title('Three-Body Problem 3D\n(Auto-restart when out of bounds)', 
                 color='white', fontsize=14, fontweight='bold')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # 情報パネル
    info_text = fig.text(0.02, 0.02, '', color='#00ff88', fontsize=9,
                         fontfamily='monospace', verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='#0a0a1a', 
                                   edgecolor='#00ff88', alpha=0.9))
    
    # 物体と軌跡
    bodies = []
    trails = []
    for i in range(3):
        body, = ax.plot([], [], [], 'o', color=colors[i], markersize=12,
                        markeredgecolor='white', markeredgewidth=1.5)
        bodies.append(body)
        trail, = ax.plot([], [], [], '-', color=colors[i], alpha=0.5, linewidth=1.5)
        trails.append(trail)
    
    # 状態を保持（クロージャ用）
    state = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'azim': 0
    }
    
    def update(frame):
        nonlocal trail_history
        
        # 複数ステップ進める（速度調整）
        steps_per_frame = 10
        for _ in range(steps_per_frame):
            state['positions'], state['velocities'] = rk4_step(
                state['positions'], state['velocities'], state['masses'], DT
            )
            sim_time[0] += DT
        
        # 境界チェック → 範囲外なら再スタート
        if is_out_of_bounds(state['positions']):
            print(f"🔄 Generation {generation[0]} ended at t={sim_time[0]:.2f} - Restarting...")
            generation[0] += 1
            state['positions'], state['velocities'], state['masses'] = get_chaotic_initial_conditions_3d()
            sim_time[0] = 0.0
            trail_history = [[] for _ in range(3)]
        
        # 軌跡更新
        for i in range(3):
            trail_history[i].append(state['positions'][i].copy())
            if len(trail_history[i]) > max_trail:
                trail_history[i].pop(0)
        
        # 情報テキスト
        energy = _compute_energy_3d(state['positions'], state['velocities'], state['masses'])
        info_lines = [
            f"Generation: {generation[0]}",
            f"Time: {sim_time[0]:.2f}",
            f"Energy: {energy:.4f}",
            ""
        ]
        for i in range(3):
            x, y, z = state['positions'][i]
            info_lines.append(f"Body {i+1}: ({x:+.2f}, {y:+.2f}, {z:+.2f})")
        info_text.set_text('\n'.join(info_lines))
        
        # 描画更新
        for i in range(3):
            x, y, z = state['positions'][i]
            bodies[i].set_data([x], [y])
            bodies[i].set_3d_properties([z])
            
            if trail_history[i]:
                trail_arr = np.array(trail_history[i])
                trails[i].set_data(trail_arr[:, 0], trail_arr[:, 1])
                trails[i].set_3d_properties(trail_arr[:, 2])
        
        # 視点回転
        state['azim'] += 0.3
        ax.view_init(elev=20, azim=state['azim'])
        
        return bodies + trails + [info_text]
    
    anim = FuncAnimation(fig, update, frames=None, blit=False, 
                         interval=ANIMATION_INTERVAL, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Three-Body Problem Simulator【3D版・自動リスタート】")
    print("  物体がキューブ範囲外に出ると自動で新規スタート")
    print("  ウィンドウを閉じると終了")
    print("=" * 60)
    print()
    
    run_realtime_animation()
