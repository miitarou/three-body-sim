"""
N体問題シミュレーター Advanced Edition + Learning Mode

=== 機能一覧 ===
- オートプレイ（起動時に自動で動く）
- 初期条件エディタ（Eキーでパネル表示）
- 力ベクトル表示（Fキーでトグル）
- 予測モード（Pキーで一時停止して予測、Enterで確認）
- キーボード/マウス操作

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
FORCE_ARROW_SCALE = 0.15
MASS_MIN = 0.5
MASS_MAX = 2.0


# ============================================================
# 物理計算
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


def compute_forces(positions, masses, softening):
    """各物体にかかる力を計算（力ベクトル表示用）"""
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


def adaptive_timestep(positions, base_dt, min_dt, max_dt):
    min_dist = compute_min_distance(positions)
    factor = min(1.0, min_dist / 0.3)
    dt = base_dt * factor
    return max(min_dt, min(max_dt, dt))


def rk4_step_adaptive(positions, velocities, masses, softening, base_dt, min_dt, max_dt):
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


def is_out_of_bounds(positions, bound):
    return np.any(np.abs(positions) > bound)


# ============================================================
# メインシミュレーター
# ============================================================

def run_advanced_simulation():
    """フル機能版N体シミュレーター + 教育モード"""
    
    n_bodies = DEFAULT_N_BODIES
    softening = SOFTENING
    mass_min = MASS_MIN
    mass_max = MASS_MAX
    
    positions, velocities, masses = generate_initial_conditions(n_bodies, mass_min, mass_max)
    
    # 状態変数
    paused = [False]
    auto_rotate = [False]
    show_forces = [False]
    show_editor = [False]
    prediction_mode = [False]
    prediction_made = [False]
    user_prediction = [""]
    
    generation = [1]
    sim_time = [0.0]
    azim = [30]
    zoom = [1.0]
    display_range = [DISPLAY_RANGE]
    
    stats = {
        'max_generation': 1,
        'generation_times': [],
    }
    
    max_trail = 400
    trail_history = [[] for _ in range(n_bodies)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_bodies, 10)))[:n_bodies]
    
    # ============================================================
    # プロット設定
    # ============================================================
    
    fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
    fig.canvas.manager.set_window_title('N-Body Problem Simulator - Learning Edition')
    
    # メイン3Dプロット
    ax_3d = fig.add_axes([0.05, 0.1, 0.65, 0.85], projection='3d', facecolor='#1a1a2e')
    ax_3d.set_xlim(-display_range[0], display_range[0])
    ax_3d.set_ylim(-display_range[0], display_range[0])
    ax_3d.set_zlim(-display_range[0], display_range[0])
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
    
    # 操作説明パネル（右側）
    controls_text = fig.text(0.72, 0.95, 
        '🎮 CONTROLS\n'
        '─────────────\n'
        '[SPACE] Pause\n'
        '[R] Restart\n'
        '[A] Auto-rotate\n'
        '[F] Force vectors\n'
        '[E] Editor panel\n'
        '[P] Predict mode\n'
        '[+/-] Zoom\n'
        '[Q] Quit\n'
        '─────────────\n'
        'Drag to rotate\n'
        'Scroll to zoom',
        color='#888888', fontsize=9, fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#0a0a1a', 
                  edgecolor='#444444', alpha=0.9))
    
    # エディタパネル（右側、非表示から開始）
    editor_text = fig.text(0.72, 0.55, '', color='#ffaa00', fontsize=9,
                          fontfamily='monospace', verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='#1a1a0a', 
                                    edgecolor='#ffaa00', alpha=0.9),
                          visible=False)
    
    # 予測モード表示
    prediction_text = fig.text(0.35, 0.95, '', color='#ff6b6b', fontsize=11,
                              fontfamily='monospace', 
                              horizontalalignment='center',
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='#2a1a1a', 
                                        edgecolor='#ff6b6b', alpha=0.9),
                              visible=False)
    
    # 物体
    bodies = []
    trails = []
    velocity_arrows = []
    force_arrows = []
    
    for i in range(n_bodies):
        body, = ax_3d.plot([], [], [], 'o', color=colors[i], markersize=10,
                          markeredgecolor='white', markeredgewidth=1)
        bodies.append(body)
        trail, = ax_3d.plot([], [], [], '-', color=colors[i], alpha=0.4, linewidth=1)
        trails.append(trail)
        arrow, = ax_3d.plot([], [], [], '-', color=colors[i], linewidth=1.5, alpha=0.7)
        velocity_arrows.append(arrow)
        # 力ベクトル（赤系で表示）
        force, = ax_3d.plot([], [], [], '-', color='#ff4444', linewidth=2, alpha=0.8)
        force_arrows.append(force)
    
    state = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'n_bodies': n_bodies
    }
    
    # 力ベクトルのラベル
    force_label = fig.text(0.72, 0.08, '', color='#ff4444', fontsize=8,
                          fontfamily='monospace', visible=False)
    
    # ============================================================
    # イベントハンドラ
    # ============================================================
    
    def on_key(event):
        nonlocal trail_history
        
        if event.key == ' ':
            paused[0] = not paused[0]
            print(f"⏯️  {'PAUSED' if paused[0] else 'RUNNING'}")
        
        elif event.key == 'r':
            generation[0] += 1
            stats['max_generation'] = max(stats['max_generation'], generation[0])
            state['positions'], state['velocities'], state['masses'] = generate_initial_conditions(
                state['n_bodies'], mass_min, mass_max
            )
            sim_time[0] = 0.0
            trail_history = [[] for _ in range(state['n_bodies'])]
            prediction_mode[0] = False
            prediction_text.set_visible(False)
            print(f"🔄 Restart - Generation {generation[0]}")
        
        elif event.key == 'a':
            auto_rotate[0] = not auto_rotate[0]
            print(f"🔄 Auto-rotate: {'ON' if auto_rotate[0] else 'OFF'}")
        
        elif event.key == 'f':
            show_forces[0] = not show_forces[0]
            force_label.set_visible(show_forces[0])
            if show_forces[0]:
                force_label.set_text('🔴 Red arrows = Gravitational force')
            print(f"⚡ Force vectors: {'ON' if show_forces[0] else 'OFF'}")
        
        elif event.key == 'e':
            show_editor[0] = not show_editor[0]
            editor_text.set_visible(show_editor[0])
            if show_editor[0]:
                update_editor_panel()
            print(f"📝 Editor: {'OPEN' if show_editor[0] else 'CLOSED'}")
        
        elif event.key == 'p':
            prediction_mode[0] = not prediction_mode[0]
            if prediction_mode[0]:
                paused[0] = True
                prediction_made[0] = False
                prediction_text.set_text(
                    '🔮 PREDICTION MODE\n'
                    '─────────────────\n'
                    'What will happen next?\n\n'
                    '  Will they...\n'
                    '  • Collide?\n'
                    '  • Escape?\n'
                    '  • Orbit?\n\n'
                    'Press [ENTER] to see!'
                )
                prediction_text.set_visible(True)
                print("🔮 Prediction mode ON - Make your prediction!")
            else:
                prediction_text.set_visible(False)
                print("🔮 Prediction mode OFF")
        
        elif event.key == 'enter' and prediction_mode[0]:
            paused[0] = False
            prediction_made[0] = True
            prediction_text.set_text('▶️ Running...\nWatch what happens!')
        
        elif event.key == 'q':
            print("👋 Exiting...")
            plt.close()
        
        elif event.key in ['+', '=']:
            zoom[0] = max(0.3, zoom[0] * 0.8)
            update_zoom()
        
        elif event.key == '-':
            zoom[0] = min(3.0, zoom[0] * 1.25)
            update_zoom()
        
        # 数字キーで物体数変更
        elif event.key in ['3', '4', '5', '6', '7', '8', '9']:
            new_n = int(event.key)
            if new_n != state['n_bodies']:
                change_n_bodies(new_n)
    
    def on_scroll(event):
        if event.button == 'up':
            zoom[0] = max(0.3, zoom[0] * 0.9)
        else:
            zoom[0] = min(3.0, zoom[0] * 1.1)
        update_zoom()
    
    def update_zoom():
        r = DISPLAY_RANGE * zoom[0]
        display_range[0] = r
        ax_3d.set_xlim(-r, r)
        ax_3d.set_ylim(-r, r)
        ax_3d.set_zlim(-r, r)
    
    def change_n_bodies(new_n):
        nonlocal bodies, trails, velocity_arrows, force_arrows, trail_history, colors
        
        # 既存の描画オブジェクトをクリア（非表示に）
        for body in bodies:
            body.set_data([], [])
            body.set_3d_properties([])
        for trail in trails:
            trail.set_data([], [])
            trail.set_3d_properties([])
        for arrow in velocity_arrows:
            arrow.set_data([], [])
            arrow.set_3d_properties([])
        for force in force_arrows:
            force.set_data([], [])
            force.set_3d_properties([])
        
        # 新しい物体数で初期化
        state['n_bodies'] = new_n
        state['positions'], state['velocities'], state['masses'] = generate_initial_conditions(
            new_n, mass_min, mass_max
        )
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(new_n, 10)))[:new_n]
        
        # 描画オブジェクトを再作成
        bodies.clear()
        trails.clear()
        velocity_arrows.clear()
        force_arrows.clear()
        trail_history = [[] for _ in range(new_n)]
        
        for i in range(new_n):
            body, = ax_3d.plot([], [], [], 'o', color=colors[i], markersize=10,
                              markeredgecolor='white', markeredgewidth=1)
            bodies.append(body)
            trail, = ax_3d.plot([], [], [], '-', color=colors[i], alpha=0.4, linewidth=1)
            trails.append(trail)
            arrow, = ax_3d.plot([], [], [], '-', color=colors[i], linewidth=1.5, alpha=0.7)
            velocity_arrows.append(arrow)
            force, = ax_3d.plot([], [], [], '-', color='#ff4444', linewidth=2, alpha=0.8)
            force_arrows.append(force)
        
        sim_time[0] = 0.0
        generation[0] += 1
        print(f"🔢 Changed to {new_n} bodies - Generation {generation[0]}")
        
        if show_editor[0]:
            update_editor_panel()
    
    def update_editor_panel():
        lines = [
            '📝 EDITOR',
            '─────────────',
            f'N Bodies: {state["n_bodies"]}',
            '(Press 3-9 to change)',
            '',
            '📊 Current masses:',
        ]
        for i in range(min(state['n_bodies'], 6)):
            lines.append(f'  Body {i+1}: {state["masses"][i]:.2f}')
        if state['n_bodies'] > 6:
            lines.append(f'  ... +{state["n_bodies"]-6} more')
        
        lines.extend([
            '',
            '🎯 Tips:',
            '• More bodies = chaos',
            '• Watch the forces!',
            '• Try predicting!',
        ])
        
        editor_text.set_text('\n'.join(lines))
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # ============================================================
    # アニメーション更新
    # ============================================================
    
    def update(frame):
        nonlocal trail_history
        
        if paused[0]:
            # 一時停止中も力ベクトルは更新
            if show_forces[0]:
                forces = compute_forces(state['positions'], state['masses'], softening)
                for i in range(state['n_bodies']):
                    x, y, z = state['positions'][i]
                    fx, fy, fz = forces[i] * FORCE_ARROW_SCALE
                    force_arrows[i].set_data([x, x+fx], [y, y+fy])
                    force_arrows[i].set_3d_properties([z, z+fz])
            return bodies + trails + velocity_arrows + force_arrows + [info_text]
        
        # シミュレーション進行
        steps_per_frame = 10
        for _ in range(steps_per_frame):
            state['positions'], state['velocities'], dt = rk4_step_adaptive(
                state['positions'], state['velocities'], state['masses'],
                softening, BASE_DT, MIN_DT, MAX_DT
            )
            sim_time[0] += dt
        
        # 境界チェック
        if is_out_of_bounds(state['positions'], display_range[0]):
            print(f"🔄 Generation {generation[0]} ended at t={sim_time[0]:.2f}")
            generation[0] += 1
            stats['max_generation'] = max(stats['max_generation'], generation[0])
            
            state['positions'], state['velocities'], state['masses'] = generate_initial_conditions(
                state['n_bodies'], mass_min, mass_max
            )
            sim_time[0] = 0.0
            trail_history = [[] for _ in range(state['n_bodies'])]
            
            if prediction_mode[0]:
                prediction_mode[0] = False
                prediction_text.set_text('💥 They escaped!\nPress [P] to try again')
            
            if show_editor[0]:
                update_editor_panel()
        
        # 軌跡更新
        for i in range(state['n_bodies']):
            trail_history[i].append(state['positions'][i].copy())
            if len(trail_history[i]) > max_trail:
                trail_history[i].pop(0)
        
        # 計算
        energy = compute_energy(state['positions'], state['velocities'], state['masses'], softening)
        min_dist = compute_min_distance(state['positions'])
        
        # 力計算（表示用）
        forces = compute_forces(state['positions'], state['masses'], softening) if show_forces[0] else None
        
        # 情報テキスト
        info_lines = [
            f"Gen: {generation[0]}  Time: {sim_time[0]:.1f}  Zoom: {1/zoom[0]:.1f}x",
            f"Energy: {energy:.3f}  MinDist: {min_dist:.2f}",
            f"Bodies: {state['n_bodies']}  MaxGen: {stats['max_generation']}",
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
            
            # 速度ベクトル
            arrow_end = [x + vx * VELOCITY_ARROW_SCALE, 
                         y + vy * VELOCITY_ARROW_SCALE, 
                         z + vz * VELOCITY_ARROW_SCALE]
            velocity_arrows[i].set_data([x, arrow_end[0]], [y, arrow_end[1]])
            velocity_arrows[i].set_3d_properties([z, arrow_end[2]])
            
            # 力ベクトル
            if show_forces[0] and forces is not None:
                fx, fy, fz = forces[i] * FORCE_ARROW_SCALE
                force_arrows[i].set_data([x, x+fx], [y, y+fy])
                force_arrows[i].set_3d_properties([z, z+fz])
            else:
                force_arrows[i].set_data([], [])
                force_arrows[i].set_3d_properties([])
        
        if auto_rotate[0]:
            azim[0] += 0.3
            ax_3d.view_init(elev=20, azim=azim[0])
        
        return bodies + trails + velocity_arrows + force_arrows + [info_text]
    
    anim = FuncAnimation(fig, update, frames=None, blit=False, 
                         interval=ANIMATION_INTERVAL, cache_frame_data=False)
    
    plt.show()
    
    return anim


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("N-Body Problem Simulator【Learning Edition】")
    print("=" * 65)
    print()
    print("🎬 The simulation starts automatically!")
    print("   Watch the stars dance, then explore with these controls:")
    print()
    print("🎮 Basic Controls:")
    print("  [SPACE] = Pause/Resume")
    print("  [R]     = Restart with new conditions")
    print("  [A]     = Toggle auto-rotation")
    print("  [Q]     = Quit")
    print()
    print("📚 Learning Features:")
    print("  [F]     = Show force vectors (see gravity in action!)")
    print("  [E]     = Open editor panel")
    print("  [P]     = Prediction mode (guess what happens next)")
    print("  [3-9]   = Change number of bodies")
    print()
    print("🔍 View Controls:")
    print("  [+/-]   = Zoom in/out")
    print("  [Wheel] = Zoom in/out")
    print("  [Drag]  = Rotate view")
    print("=" * 65)
    print()
    
    run_advanced_simulation()
