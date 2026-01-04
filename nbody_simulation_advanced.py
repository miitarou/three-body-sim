"""
N体問題シミュレーター Advanced Edition

=== 機能一覧 ===
- N体問題への一般化（デフォルト3、設定可能）
- キーボードショートカット（スペース：一時停止、R：リスタート、A：自動回転、Q：終了）
- マウスで視点操作（自動回転オフ時）
- 統計表示（最長Generation、平均生存時間）
- NumPyベクトル化による高速化
- 適応タイムステップ
- 自動リスタート機能

物理モデル: 万有引力の法則 + Plummerソフトニング
計算手法: 4次ルンゲ＝クッタ法（RK4）+ 適応タイムステップ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time


# ============================================================
# デフォルト設定
# ============================================================

DEFAULT_N_BODIES = 3
G = 1.0
BASE_DT = 0.001
MIN_DT = 0.0001
MAX_DT = 0.01
ANIMATION_INTERVAL = 30
SOFTENING = 0.05
DISPLAY_RANGE = 1.5
VELOCITY_ARROW_SCALE = 0.3
MASS_MIN = 0.5
MASS_MAX = 2.0


# ============================================================
# 物理計算（NumPyベクトル化版）
# ============================================================

def compute_accelerations_vectorized(positions, masses, softening):
    """ベクトル化された加速度計算"""
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


def compute_min_distance(positions):
    """最小の物体間距離を計算"""
    n = len(positions)
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[j] - positions[i])
            min_dist = min(min_dist, dist)
    return min_dist


def adaptive_timestep(positions, base_dt, min_dt, max_dt):
    """適応タイムステップ"""
    min_dist = compute_min_distance(positions)
    factor = min(1.0, min_dist / 0.3)
    dt = base_dt * factor
    return max(min_dt, min(max_dt, dt))


def rk4_step_adaptive(positions, velocities, masses, softening, base_dt, min_dt, max_dt):
    """適応タイムステップ付きRK4"""
    dt = adaptive_timestep(positions, base_dt, min_dt, max_dt)
    
    k1_r = velocities
    k1_v = compute_accelerations_vectorized(positions, masses, softening)
    
    k2_r = velocities + 0.5 * dt * k1_v
    k2_v = compute_accelerations_vectorized(positions + 0.5 * dt * k1_r, masses, softening)
    
    k3_r = velocities + 0.5 * dt * k2_v
    k3_v = compute_accelerations_vectorized(positions + 0.5 * dt * k2_r, masses, softening)
    
    k4_r = velocities + dt * k3_v
    k4_v = compute_accelerations_vectorized(positions + dt * k3_r, masses, softening)
    
    new_pos = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_vel = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return new_pos, new_vel, dt


def compute_energy(positions, velocities, masses, softening):
    """全エネルギー計算"""
    n = len(masses)
    eps2 = softening ** 2
    
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    
    pe = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r2 = np.sum((positions[j] - positions[i])**2)
            pe -= G * masses[i] * masses[j] / np.sqrt(r2 + eps2)
    
    return ke + pe


# ============================================================
# 初期条件
# ============================================================

def generate_initial_conditions(n_bodies, mass_min, mass_max):
    """N体のランダム初期条件を生成"""
    np.random.seed(int(time.time() * 1000) % (2**32))
    
    masses = mass_min + np.random.rand(n_bodies) * (mass_max - mass_min)
    
    positions = np.random.randn(n_bodies, 3) * 0.5
    positions = np.clip(positions, -1.0, 1.0)
    
    center_of_mass = np.average(positions, axis=0, weights=masses)
    positions -= center_of_mass
    
    velocities = np.random.randn(n_bodies, 3) * 0.4
    
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    velocities -= total_momentum / np.sum(masses)
    
    energy = compute_energy(positions, velocities, masses, SOFTENING)
    while energy > -0.3:
        velocities *= 0.9
        energy = compute_energy(positions, velocities, masses, SOFTENING)
    
    return positions, velocities, masses


def is_out_of_bounds(positions, bound=DISPLAY_RANGE):
    return np.any(np.abs(positions) > bound)


# ============================================================
# メインシミュレーター（星のみフォーカス）
# ============================================================

def run_advanced_simulation():
    """フル機能版N体シミュレーター"""
    
    n_bodies = DEFAULT_N_BODIES
    softening = SOFTENING
    mass_min = MASS_MIN
    mass_max = MASS_MAX
    
    positions, velocities, masses = generate_initial_conditions(n_bodies, mass_min, mass_max)
    
    paused = [False]
    auto_rotate = [False]
    generation = [1]
    sim_time = [0.0]
    azim = [30]
    zoom = [1.0]  # ズームレベル（1.0=デフォルト）
    
    stats = {
        'max_generation': 1,
        'total_time': 0.0,
        'generation_times': [],
        'current_gen_start': 0.0
    }
    
    max_trail = 400
    trail_history = [[] for _ in range(n_bodies)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_bodies, 10)))[:n_bodies]
    
    # ============================================================
    # プロット設定（星のシミュレーションのみ）
    # ============================================================
    
    fig = plt.figure(figsize=(12, 10), facecolor='#1a1a2e')
    fig.canvas.manager.set_window_title('N-Body Problem Simulator - Advanced Edition')
    
    ax_3d = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
    ax_3d.set_xlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax_3d.set_ylim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax_3d.set_zlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax_3d.set_xlabel('X', color='white')
    ax_3d.set_ylabel('Y', color='white')
    ax_3d.set_zlabel('Z', color='white')
    ax_3d.tick_params(colors='white')
    ax_3d.set_title('N-Body Simulation', color='white', fontsize=14, fontweight='bold')
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('white')
    ax_3d.yaxis.pane.set_edgecolor('white')
    ax_3d.zaxis.pane.set_edgecolor('white')
    
    # 情報パネル（左下）
    info_text = fig.text(0.02, 0.02, '', color='#00ff88', fontsize=9,
                         fontfamily='monospace', verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='#0a0a1a', 
                                   edgecolor='#00ff88', alpha=0.9))
    
    # 操作説明（右下）
    help_text = fig.text(0.98, 0.02, 
                         '[SPACE]=Pause [R]=Restart [A]=Rotate [+/-]=Zoom [Q]=Quit',
                         color='#666666', fontsize=8, fontfamily='monospace',
                         horizontalalignment='right', verticalalignment='bottom')
    
    # 物体
    bodies = []
    trails = []
    velocity_arrows = []
    for i in range(n_bodies):
        body, = ax_3d.plot([], [], [], 'o', color=colors[i], markersize=10,
                          markeredgecolor='white', markeredgewidth=1)
        bodies.append(body)
        trail, = ax_3d.plot([], [], [], '-', color=colors[i], alpha=0.4, linewidth=1)
        trails.append(trail)
        arrow, = ax_3d.plot([], [], [], '-', color=colors[i], linewidth=1.5, alpha=0.7)
        velocity_arrows.append(arrow)
    
    state = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'n_bodies': n_bodies
    }
    
    # ============================================================
    # イベントハンドラ
    # ============================================================
    
    def on_key(event):
        nonlocal trail_history
        
        if event.key == ' ':
            paused[0] = not paused[0]
            print(f"⏯️  {'PAUSED' if paused[0] else 'RUNNING'}")
        
        elif event.key == 'r':
            gen_time = sim_time[0] - stats['current_gen_start']
            if gen_time > 0:
                stats['generation_times'].append(gen_time)
            stats['total_time'] += gen_time
            stats['current_gen_start'] = 0.0
            
            generation[0] += 1
            stats['max_generation'] = max(stats['max_generation'], generation[0])
            
            state['positions'], state['velocities'], state['masses'] = generate_initial_conditions(
                state['n_bodies'], mass_min, mass_max
            )
            sim_time[0] = 0.0
            trail_history = [[] for _ in range(state['n_bodies'])]
            print(f"🔄 Manual restart - Generation {generation[0]}")
        
        elif event.key == 'a':
            auto_rotate[0] = not auto_rotate[0]
            print(f"🔄 Auto-rotate: {'ON' if auto_rotate[0] else 'OFF'}")
        
        elif event.key == 'q':
            print("👋 Exiting...")
            plt.close()
        
        elif event.key in ['+', '=']:
            zoom[0] = max(0.3, zoom[0] * 0.8)  # ズームイン
            update_zoom()
            print(f"🔍 Zoom: {1/zoom[0]:.1f}x")
        
        elif event.key == '-':
            zoom[0] = min(3.0, zoom[0] * 1.25)  # ズームアウト
            update_zoom()
            print(f"🔍 Zoom: {1/zoom[0]:.1f}x")
    
    def on_scroll(event):
        if event.button == 'up':
            zoom[0] = max(0.3, zoom[0] * 0.9)
        else:
            zoom[0] = min(3.0, zoom[0] * 1.1)
        update_zoom()
    
    def update_zoom():
        r = DISPLAY_RANGE * zoom[0]
        ax_3d.set_xlim(-r, r)
        ax_3d.set_ylim(-r, r)
        ax_3d.set_zlim(-r, r)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # ============================================================
    # アニメーション更新
    # ============================================================
    
    def update(frame):
        nonlocal trail_history
        
        if paused[0]:
            return bodies + trails + velocity_arrows + [info_text]
        
        steps_per_frame = 10
        for _ in range(steps_per_frame):
            state['positions'], state['velocities'], dt = rk4_step_adaptive(
                state['positions'], state['velocities'], state['masses'],
                softening, BASE_DT, MIN_DT, MAX_DT
            )
            sim_time[0] += dt
        
        # 境界チェック
        if is_out_of_bounds(state['positions']):
            gen_time = sim_time[0] - stats['current_gen_start']
            if gen_time > 0:
                stats['generation_times'].append(gen_time)
            stats['total_time'] += gen_time
            
            print(f"🔄 Generation {generation[0]} ended at t={sim_time[0]:.2f}")
            generation[0] += 1
            stats['max_generation'] = max(stats['max_generation'], generation[0])
            stats['current_gen_start'] = 0.0
            
            state['positions'], state['velocities'], state['masses'] = generate_initial_conditions(
                state['n_bodies'], mass_min, mass_max
            )
            sim_time[0] = 0.0
            trail_history = [[] for _ in range(state['n_bodies'])]
        
        # 軌跡更新
        for i in range(state['n_bodies']):
            trail_history[i].append(state['positions'][i].copy())
            if len(trail_history[i]) > max_trail:
                trail_history[i].pop(0)
        
        # 計算
        energy = compute_energy(state['positions'], state['velocities'], state['masses'], softening)
        min_dist = compute_min_distance(state['positions'])
        avg_time = np.mean(stats['generation_times']) if stats['generation_times'] else 0
        
        # 情報テキスト（控えめに数字表示）
        info_lines = [
            f"Gen: {generation[0]}  Time: {sim_time[0]:.1f}  Zoom: {1/zoom[0]:.1f}x",
            f"Energy: {energy:.3f}  MinDist: {min_dist:.2f}",
            f"MaxGen: {stats['max_generation']}  AvgLife: {avg_time:.1f}s",
        ]
        info_text.set_text('\n'.join(info_lines))
        
        # 3D描画更新
        for i in range(state['n_bodies']):
            x, y, z = state['positions'][i]
            vx, vy, vz = state['velocities'][i]
            mass = state['masses'][i]
            
            size = 6 + (mass - mass_min) * 6
            
            bodies[i].set_data([x], [y])
            bodies[i].set_3d_properties([z])
            bodies[i].set_markersize(size)
            
            if trail_history[i]:
                trail_arr = np.array(trail_history[i])
                trails[i].set_data(trail_arr[:, 0], trail_arr[:, 1])
                trails[i].set_3d_properties(trail_arr[:, 2])
            
            arrow_end = [x + vx * VELOCITY_ARROW_SCALE, 
                         y + vy * VELOCITY_ARROW_SCALE, 
                         z + vz * VELOCITY_ARROW_SCALE]
            velocity_arrows[i].set_data([x, arrow_end[0]], [y, arrow_end[1]])
            velocity_arrows[i].set_3d_properties([z, arrow_end[2]])
        
        if auto_rotate[0]:
            azim[0] += 0.3
            ax_3d.view_init(elev=20, azim=azim[0])
        
        return bodies + trails + velocity_arrows + [info_text]
    
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
    print("N-Body Problem Simulator【Advanced Edition】")
    print("=" * 60)
    print()
    print("🎮 Controls:")
    print("  [SPACE] = Pause/Resume")
    print("  [R]     = Restart with new conditions")
    print("  [A]     = Toggle auto-rotation")
    print("  [+/-]   = Zoom in/out")
    print("  [Wheel] = Zoom in/out")
    print("  [Q]     = Quit")
    print("  [Mouse] = Drag to rotate view (when auto-rotate is OFF)")
    print()
    
    run_advanced_simulation()
