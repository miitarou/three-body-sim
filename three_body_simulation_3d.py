"""
三体問題シミュレーター 3D版（Three-Body Problem Simulator 3D）

物理モデル: 万有引力の法則（ニュートンの重力法則）
計算手法: 4次ルンゲ＝クッタ法（RK4）
初期条件: 3D空間でのランダム配置 または 8の字解（2D平面埋め込み）

参考文献:
- Chenciner & Montgomery (2000): "A remarkable periodic solution of the three-body problem"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 物理定数と設定
# ============================================================

# 万有引力定数 G（シミュレーション単位系では G = 1 と正規化）
G = 1.0

# タイムステップ
DT = 0.001

# シミュレーション総時間
T_MAX = 20.0

# アニメーション更新間隔（ミリ秒）
ANIMATION_INTERVAL = 30

# 軌跡の長さ
TRAIL_LENGTH = 1500

# Plummerソフトニング長
SOFTENING = 0.05


# ============================================================
# 8の字解の初期条件（3D空間のXY平面に埋め込み）
# ============================================================

def get_figure8_initial_conditions_3d():
    """
    8の字解の初期条件を3D空間で返す（XY平面に埋め込み）
    """
    masses = np.array([1.0, 1.0, 1.0])
    
    x1 = 0.97000436
    y1 = -0.24308753
    
    positions = np.array([
        [ x1,  y1, 0.0],
        [-x1, -y1, 0.0],
        [ 0.0, 0.0, 0.0]
    ])
    
    vx3 = -0.93240737
    vy3 = -0.86473146
    
    velocities = np.array([
        [ vx3/2,  vy3/2, 0.0],
        [ vx3/2,  vy3/2, 0.0],
        [-vx3,   -vy3,   0.0]
    ])
    
    return positions, velocities, masses


def get_chaotic_initial_conditions_3d(seed=None):
    """
    カオス的な動きを生成するランダム3D初期条件を返す
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        import time
        np.random.seed(int(time.time() * 1000) % (2**32))
    
    # 質量（少しばらつきを持たせる）
    masses = np.array([1.0, 1.0 + 0.2 * np.random.randn(), 
                       1.0 + 0.2 * np.random.randn()])
    masses = np.clip(masses, 0.5, 1.5)
    
    # ランダムな初期位置（3D球殻上に配置）
    positions = np.random.randn(3, 3) * 0.8
    
    # 重心を原点に移動
    center_of_mass = np.average(positions, axis=0, weights=masses)
    positions -= center_of_mass
    
    # ランダムな初期速度
    velocities = np.random.randn(3, 3) * 0.3
    
    # 総運動量をゼロに調整
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    velocities -= total_momentum / np.sum(masses)
    
    # エネルギーチェックと調整
    total_energy = _compute_energy_3d(positions, velocities, masses)
    while total_energy > -0.1:
        velocities *= 0.8
        total_energy = _compute_energy_3d(positions, velocities, masses)
    
    return positions, velocities, masses


def _compute_energy_3d(positions, velocities, masses):
    """初期条件チェック用のエネルギー計算"""
    n = len(masses)
    eps2 = SOFTENING ** 2
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    pe = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r_vec = positions[j] - positions[i]
            r2 = np.dot(r_vec, r_vec)
            pe -= G * masses[i] * masses[j] / np.sqrt(r2 + eps2)
    return ke + pe


# ============================================================
# 万有引力の法則（3Dベクトル形式 with Plummerソフトニング）
# ============================================================

def compute_gravitational_forces_3d(positions, masses):
    """
    Plummerソフトニングを使用した万有引力計算（3D）
    """
    n = len(masses)
    forces = np.zeros_like(positions)
    eps2 = SOFTENING ** 2
    
    for i in range(n):
        for j in range(n):
            if i != j:
                r_ij = positions[j] - positions[i]
                r2 = np.dot(r_ij, r_ij)
                denom = (r2 + eps2) ** 1.5
                force_vec = G * masses[i] * masses[j] * r_ij / denom
                forces[i] += force_vec
    
    return forces


def compute_accelerations_3d(positions, masses):
    """各物体の加速度を計算"""
    forces = compute_gravitational_forces_3d(positions, masses)
    accelerations = forces / masses[:, np.newaxis]
    return accelerations


# ============================================================
# 4次ルンゲ＝クッタ法（RK4）
# ============================================================

def rk4_step_3d(positions, velocities, masses, dt):
    """4次ルンゲ＝クッタ法による1ステップの時間発展（3D）"""
    k1_r = velocities
    k1_v = compute_accelerations_3d(positions, masses)
    
    r2 = positions + 0.5 * dt * k1_r
    v2 = velocities + 0.5 * dt * k1_v
    k2_r = v2
    k2_v = compute_accelerations_3d(r2, masses)
    
    r3 = positions + 0.5 * dt * k2_r
    v3 = velocities + 0.5 * dt * k2_v
    k3_r = v3
    k3_v = compute_accelerations_3d(r3, masses)
    
    r4 = positions + dt * k3_r
    v4 = velocities + dt * k3_v
    k4_r = v4
    k4_v = compute_accelerations_3d(r4, masses)
    
    new_positions = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_velocities = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return new_positions, new_velocities


# ============================================================
# エネルギー計算
# ============================================================

def compute_total_energy_3d(positions, velocities, masses):
    """系の全エネルギーを計算（3D）"""
    n = len(masses)
    eps2 = SOFTENING ** 2
    
    kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    
    potential_energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r_vec = positions[j] - positions[i]
            r2 = np.dot(r_vec, r_vec)
            potential_energy -= G * masses[i] * masses[j] / np.sqrt(r2 + eps2)
    
    return kinetic_energy + potential_energy


# ============================================================
# シミュレーション実行
# ============================================================

def run_simulation_3d(mode='figure8', dt=DT, t_max=T_MAX, seed=None):
    """3Dシミュレーションを実行"""
    if mode == 'chaos':
        positions, velocities, masses = get_chaotic_initial_conditions_3d(seed)
        print("🌀 カオスモード: ランダム3D初期条件")
    else:
        positions, velocities, masses = get_figure8_initial_conditions_3d()
        print("♾️  8の字解モード: XY平面での周期軌道")
    
    n_steps = int(t_max / dt)
    history = np.zeros((n_steps, 3, 3))  # 3次元座標
    energies = np.zeros(n_steps)
    times = np.zeros(n_steps)
    
    print("シミュレーション開始...")
    print(f"  タイムステップ: {dt}")
    print(f"  総時間: {t_max}")
    print(f"  総ステップ数: {n_steps}")
    
    for step in range(n_steps):
        history[step] = positions.copy()
        energies[step] = compute_total_energy_3d(positions, velocities, masses)
        times[step] = step * dt
        
        positions, velocities = rk4_step_3d(positions, velocities, masses, dt)
        
        if step % (n_steps // 10) == 0:
            progress = 100 * step / n_steps
            print(f"  進捗: {progress:.0f}%")
    
    print("シミュレーション完了!")
    
    energy_drift = abs(energies[-1] - energies[0]) / abs(energies[0]) * 100
    print(f"  初期エネルギー: {energies[0]:.6f}")
    print(f"  最終エネルギー: {energies[-1]:.6f}")
    print(f"  エネルギードリフト: {energy_drift:.4f}%")
    
    return history, energies, times


# ============================================================
# 3Dアニメーション可視化
# ============================================================

def create_animation_3d(history, times, energies=None, save_file=None, title='Figure-8 Solution'):
    """3D軌跡付きアニメーションを作成"""
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    
    # 固定スケール（ダイナミックな動きを見せるため）
    FIXED_RANGE = 1.5
    
    # 3Dプロット設定
    fig = plt.figure(figsize=(12, 10), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
    
    ax.set_xlim(-FIXED_RANGE, FIXED_RANGE)
    ax.set_ylim(-FIXED_RANGE, FIXED_RANGE)
    ax.set_zlim(-FIXED_RANGE, FIXED_RANGE)
    ax.set_xlabel('X', color='white', fontsize=12)
    ax.set_ylabel('Y', color='white', fontsize=12)
    ax.set_zlabel('Z', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_title(f'Three-Body Problem 3D\n({title})', 
                 color='white', fontsize=14, fontweight='bold')
    
    # 背景色設定
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
                        markeredgecolor='white', markeredgewidth=1.5,
                        label=f'Body {i+1}')
        bodies.append(body)
        
        trail, = ax.plot([], [], [], '-', color=colors[i], alpha=0.6, linewidth=1.5)
        trails.append(trail)
    
    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white',
              labelcolor='white', fontsize=10)
    
    # サンプリング
    sample_rate = max(1, len(history) // 500)
    sampled_history = history[::sample_rate]
    sampled_times = times[::sample_rate]
    sampled_energies = energies[::sample_rate] if energies is not None else None
    initial_energy = energies[0] if energies is not None else 0
    
    trail_frames = TRAIL_LENGTH // sample_rate
    
    def update(frame):
        trail_start = max(0, frame - trail_frames)
        
        info_lines = [f"Time: {sampled_times[frame]:.2f}"]
        
        if sampled_energies is not None:
            current_energy = sampled_energies[frame]
            drift = abs(current_energy - initial_energy) / abs(initial_energy) * 100
            info_lines.append(f"Energy: {current_energy:.4f}")
            info_lines.append(f"Drift: {drift:.4f}%")
        info_lines.append("")
        
        for i, (body, trail) in enumerate(zip(bodies, trails)):
            x, y, z = sampled_history[frame, i]
            body.set_data([x], [y])
            body.set_3d_properties([z])
            
            trail_x = sampled_history[trail_start:frame+1, i, 0]
            trail_y = sampled_history[trail_start:frame+1, i, 1]
            trail_z = sampled_history[trail_start:frame+1, i, 2]
            trail.set_data(trail_x, trail_y)
            trail.set_3d_properties(trail_z)
            
            info_lines.append(f"Body {i+1}: ({x:+.2f}, {y:+.2f}, {z:+.2f})")
        
        info_text.set_text('\n'.join(info_lines))
        
        # ゆっくり回転（視点を変える）
        ax.view_init(elev=20, azim=frame * 0.3)
        
        return bodies + trails + [info_text]
    
    anim = FuncAnimation(
        fig, update, frames=len(sampled_history),
        blit=False, interval=ANIMATION_INTERVAL
    )
    
    if save_file:
        print(f"アニメーション保存中: {save_file}")
        anim.save(save_file, writer='pillow', fps=20)
        print("保存完了!")
    
    plt.tight_layout()
    plt.show()
    
    return anim


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Three-Body Problem Simulator【3D版】")
    print("  Physics: Newton's Law of Universal Gravitation")
    print("  Integration: 4th-order Runge-Kutta (RK4)")
    print("=" * 60)
    print()
    print("モードを選択してください:")
    print("  1: ♾️  8の字解（XY平面での周期軌道）")
    print("  2: 🌀 カオスモード（3D空間で予測不能な動き）")
    print()
    
    choice = input("選択 (1 または 2): ").strip()
    
    if choice == '2':
        mode = 'chaos'
        print()
        print("🌀 カオスモードを選択しました")
        print("3D空間でランダムな初期条件から始まります。")
    else:
        mode = 'figure8'
        print()
        print("♾️  8の字解モードを選択しました")
        print("XY平面で美しい周期軌道を描きます。")
    
    print()
    
    history, energies, times = run_simulation_3d(mode=mode)
    
    print()
    print("3Dアニメーション表示を開始...")
    print("（マウスで視点を回転できます。ウィンドウを閉じると終了）")
    
    title = 'Chaotic Motion 3D' if mode == 'chaos' else 'Figure-8 Solution'
    anim = create_animation_3d(history, times, energies=energies, title=title)
