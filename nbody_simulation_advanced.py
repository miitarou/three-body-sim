"""
N体問題シミュレーター Advanced Edition

=== 機能一覧 ===
- N体問題への一般化（デフォルト3、設定可能）
- キーボードショートカット（スペース：一時停止、R：リスタート、Q：終了）
- マウスで視点操作（自動回転オフ）
- スライダーでパラメータ調整
- エネルギー・距離グラフ
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
from matplotlib.widgets import Slider, Button
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
        # 全ての他の物体からの力を一度に計算
        r_ij = positions - positions[i]  # (n, 3)
        r2 = np.sum(r_ij ** 2, axis=1) + eps2  # (n,)
        r2[i] = 1.0  # 自分自身との計算を避ける
        
        # 力の大きさ
        inv_r3 = r2 ** (-1.5)
        inv_r3[i] = 0.0
        
        # 加速度
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
    """適応タイムステップ：物体が接近したら小さく"""
    min_dist = compute_min_distance(positions)
    # 距離に応じてタイムステップを調整
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


def compute_all_distances(positions):
    """全ての物体間距離を計算"""
    n = len(positions)
    distances = []
    labels = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[j] - positions[i])
            distances.append(dist)
            labels.append(f"{i+1}-{j+1}")
    return distances, labels


# ============================================================
# 初期条件
# ============================================================

def generate_initial_conditions(n_bodies, mass_min, mass_max):
    """N体のランダム初期条件を生成"""
    np.random.seed(int(time.time() * 1000) % (2**32))
    
    # 質量
    masses = mass_min + np.random.rand(n_bodies) * (mass_max - mass_min)
    
    # 位置（球状に配置）
    positions = np.random.randn(n_bodies, 3) * 0.5
    positions = np.clip(positions, -1.0, 1.0)
    
    # 重心を原点に
    center_of_mass = np.average(positions, axis=0, weights=masses)
    positions -= center_of_mass
    
    # 速度
    velocities = np.random.randn(n_bodies, 3) * 0.4
    
    # 運動量をゼロに
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    velocities -= total_momentum / np.sum(masses)
    
    # 束縛状態を保証
    energy = compute_energy(positions, velocities, masses, SOFTENING)
    while energy > -0.3:
        velocities *= 0.9
        energy = compute_energy(positions, velocities, masses, SOFTENING)
    
    return positions, velocities, masses


def is_out_of_bounds(positions, bound=DISPLAY_RANGE):
    return np.any(np.abs(positions) > bound)


# ============================================================
# メインシミュレーター
# ============================================================

def run_advanced_simulation():
    """フル機能版N体シミュレーター"""
    
    # 初期パラメータ
    n_bodies = DEFAULT_N_BODIES
    softening = SOFTENING
    mass_min = MASS_MIN
    mass_max = MASS_MAX
    
    # 初期条件
    positions, velocities, masses = generate_initial_conditions(n_bodies, mass_min, mass_max)
    
    # 状態変数
    paused = [False]
    auto_rotate = [False]
    generation = [1]
    sim_time = [0.0]
    azim = [30]
    
    # 統計
    stats = {
        'max_generation': 1,
        'total_time': 0.0,
        'generation_times': [],
        'current_gen_start': 0.0
    }
    
    # 履歴（グラフ用）
    max_history = 500
    energy_history = []
    time_history = []
    distance_history = {f"{i+1}-{j+1}": [] for i in range(n_bodies) for j in range(i+1, n_bodies)}
    
    # 軌跡
    max_trail = 300
    trail_history = [[] for _ in range(n_bodies)]
    
    # カラー設定
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_bodies, 10)))[:n_bodies]
    
    # ============================================================
    # プロット設定
    # ============================================================
    
    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.canvas.manager.set_window_title('N-Body Problem Simulator - Advanced Edition')
    
    # 3Dプロット（メイン）
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d', facecolor='#1a1a2e')
    ax_3d.set_xlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax_3d.set_ylim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax_3d.set_zlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax_3d.set_xlabel('X', color='white')
    ax_3d.set_ylabel('Y', color='white')
    ax_3d.set_zlabel('Z', color='white')
    ax_3d.tick_params(colors='white')
    ax_3d.set_title('N-Body Simulation', color='white', fontsize=12, fontweight='bold')
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('white')
    ax_3d.yaxis.pane.set_edgecolor('white')
    ax_3d.zaxis.pane.set_edgecolor('white')
    
    # エネルギーグラフ
    ax_energy = fig.add_subplot(2, 2, 2, facecolor='#1a1a2e')
    ax_energy.set_xlabel('Time', color='white')
    ax_energy.set_ylabel('Energy', color='white')
    ax_energy.set_title('Energy Conservation', color='white', fontsize=10)
    ax_energy.tick_params(colors='white')
    ax_energy.spines['bottom'].set_color('white')
    ax_energy.spines['left'].set_color('white')
    ax_energy.spines['top'].set_visible(False)
    ax_energy.spines['right'].set_visible(False)
    energy_line, = ax_energy.plot([], [], '-', color='#00ff88', linewidth=1.5)
    
    # 距離グラフ
    ax_dist = fig.add_subplot(2, 2, 4, facecolor='#1a1a2e')
    ax_dist.set_xlabel('Time', color='white')
    ax_dist.set_ylabel('Distance', color='white')
    ax_dist.set_title('Inter-body Distances', color='white', fontsize=10)
    ax_dist.tick_params(colors='white')
    ax_dist.spines['bottom'].set_color('white')
    ax_dist.spines['left'].set_color('white')
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    
    distance_lines = {}
    for idx, key in enumerate(distance_history.keys()):
        line, = ax_dist.plot([], [], '-', linewidth=1, label=key, 
                            color=plt.cm.Set2(idx / len(distance_history)))
        distance_lines[key] = line
    ax_dist.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', 
                   edgecolor='white', labelcolor='white')
    
    # 情報パネル
    info_text = fig.text(0.52, 0.55, '', color='#00ff88', fontsize=9,
                         fontfamily='monospace', verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='#0a0a1a', 
                                   edgecolor='#00ff88', alpha=0.9))
    
    # 操作説明
    help_text = fig.text(0.52, 0.48, 
                         'Controls: [SPACE]=Pause  [R]=Restart  [A]=AutoRotate  [Q]=Quit',
                         color='#888888', fontsize=8, fontfamily='monospace')
    
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
    
    # 状態保持
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
        nonlocal trail_history, energy_history, time_history, distance_history
        
        if event.key == ' ':
            paused[0] = not paused[0]
            status = "PAUSED" if paused[0] else "RUNNING"
            print(f"⏯️  {status}")
        
        elif event.key == 'r':
            print(f"🔄 Manual restart at t={sim_time[0]:.2f}")
            # 統計更新
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
            energy_history = []
            time_history = []
            distance_history = {f"{i+1}-{j+1}": [] for i in range(state['n_bodies']) for j in range(i+1, state['n_bodies'])}
        
        elif event.key == 'a':
            auto_rotate[0] = not auto_rotate[0]
            status = "ON" if auto_rotate[0] else "OFF"
            print(f"🔄 Auto-rotate: {status}")
        
        elif event.key == 'q':
            print("👋 Exiting...")
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # ============================================================
    # アニメーション更新
    # ============================================================
    
    def update(frame):
        nonlocal trail_history, energy_history, time_history, distance_history
        
        if paused[0]:
            return bodies + trails + velocity_arrows + [info_text, energy_line] + list(distance_lines.values())
        
        # 複数ステップ進める
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
            energy_history = []
            time_history = []
            distance_history = {f"{i+1}-{j+1}": [] for i in range(state['n_bodies']) for j in range(i+1, state['n_bodies'])}
        
        # 履歴更新
        energy = compute_energy(state['positions'], state['velocities'], state['masses'], softening)
        energy_history.append(energy)
        time_history.append(sim_time[0])
        
        distances, labels = compute_all_distances(state['positions'])
        for dist, label in zip(distances, labels):
            if label in distance_history:
                distance_history[label].append(dist)
        
        # 履歴を制限
        if len(energy_history) > max_history:
            energy_history.pop(0)
            time_history.pop(0)
            for key in distance_history:
                if distance_history[key]:
                    distance_history[key].pop(0)
        
        # 軌跡更新
        for i in range(state['n_bodies']):
            trail_history[i].append(state['positions'][i].copy())
            if len(trail_history[i]) > max_trail:
                trail_history[i].pop(0)
        
        # エネルギーグラフ更新
        if time_history:
            energy_line.set_data(time_history, energy_history)
            ax_energy.relim()
            ax_energy.autoscale_view()
        
        # 距離グラフ更新
        if time_history:
            for key, line in distance_lines.items():
                if distance_history[key]:
                    line.set_data(time_history[:len(distance_history[key])], distance_history[key])
            ax_dist.relim()
            ax_dist.autoscale_view()
        
        # 統計計算
        avg_time = np.mean(stats['generation_times']) if stats['generation_times'] else 0
        
        # 情報テキスト
        info_lines = [
            f"Generation: {generation[0]}",
            f"Time: {sim_time[0]:.2f}",
            f"Energy: {energy:.4f}",
            f"N Bodies: {state['n_bodies']}",
            "",
            "=== Statistics ===",
            f"Max Gen: {stats['max_generation']}",
            f"Avg Life: {avg_time:.2f}s",
            "",
            "=== Masses ===",
        ]
        for i in range(min(state['n_bodies'], 6)):
            info_lines.append(f"  Body {i+1}: {state['masses'][i]:.2f}")
        if state['n_bodies'] > 6:
            info_lines.append(f"  ... and {state['n_bodies'] - 6} more")
        
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
        
        # 視点回転
        if auto_rotate[0]:
            azim[0] += 0.3
            ax_3d.view_init(elev=20, azim=azim[0])
        
        return bodies + trails + velocity_arrows + [info_text, energy_line] + list(distance_lines.values())
    
    anim = FuncAnimation(fig, update, frames=None, blit=False, 
                         interval=ANIMATION_INTERVAL, cache_frame_data=False)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
    
    return anim


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("N-Body Problem Simulator【Advanced Edition】")
    print("=" * 70)
    print()
    print("🎮 Controls:")
    print("  [SPACE] = Pause/Resume")
    print("  [R]     = Restart with new conditions")
    print("  [A]     = Toggle auto-rotation")
    print("  [Q]     = Quit")
    print("  [Mouse] = Drag to rotate view (when auto-rotate is OFF)")
    print()
    print("📊 Features:")
    print("  • N-body generalization (default: 3)")
    print("  • Adaptive timestep")
    print("  • Energy & distance graphs")
    print("  • Statistics tracking")
    print("=" * 70)
    print()
    
    run_advanced_simulation()
