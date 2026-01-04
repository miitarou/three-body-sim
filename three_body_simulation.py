"""
三体問題シミュレーター（Three-Body Problem Simulator）

物理モデル: 万有引力の法則（ニュートンの重力法則）
計算手法: 4次ルンゲ＝クッタ法（RK4）
初期条件: 8の字解（Figure-8 solution）

参考文献:
- Chenciner & Montgomery (2000): "A remarkable periodic solution of the three-body problem"
- 8の字解の初期条件: arXiv:math/0011268
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


# ============================================================
# 物理定数と設定
# ============================================================

# 万有引力定数 G（シミュレーション単位系では G = 1 と正規化）
G = 1.0

# タイムステップ（小さいほど精度向上、ただし計算コスト増加）
DT = 0.001

# シミュレーション総時間
T_MAX = 20.0

# アニメーション更新間隔（ミリ秒）
ANIMATION_INTERVAL = 20

# 軌跡の長さ（何ステップ分を表示するか）
TRAIL_LENGTH = 1500


# ============================================================
# 8の字解の初期条件
# Chenciner & Montgomery (2000) による有名な周期解
# 質量は全て等しい m1 = m2 = m3 = 1
# ============================================================

def get_figure8_initial_conditions():
    """
    8の字解の初期条件を返す
    
    Returns:
        positions: 形状 (3, 2) の位置配列 [x, y]
        velocities: 形状 (3, 2) の速度配列 [vx, vy]
        masses: 形状 (3,) の質量配列
    """
    # 質量（全て等しい）
    masses = np.array([1.0, 1.0, 1.0])
    
    # 初期位置（Chenciner-Montgomery solution）
    # 物体1は原点から右寄り、物体2は左寄り、物体3は原点
    x1 = 0.97000436
    y1 = -0.24308753
    
    positions = np.array([
        [ x1,  y1],     # 物体1
        [-x1, -y1],     # 物体2（物体1の点対称）
        [ 0.0, 0.0]     # 物体3（原点）
    ])
    
    # 初期速度
    # 物体3の速度、物体1と2は逆向きで半分
    vx3 = -0.93240737
    vy3 = -0.86473146
    
    velocities = np.array([
        [ vx3/2,  vy3/2],   # 物体1
        [ vx3/2,  vy3/2],   # 物体2
        [-vx3,   -vy3   ]   # 物体3
    ])
    
    return positions, velocities, masses


def get_chaotic_initial_conditions(seed=None):
    """
    カオス的な動きを生成するランダム初期条件を返す
    
    物体が飛び去らないよう、束縛状態（負の全エネルギー）を保証する。
    重心が原点に固定され、運動量がゼロになるよう調整。
    
    Args:
        seed: 乱数シード（再現性のため、Noneなら毎回異なる）
    
    Returns:
        positions: 形状 (3, 2) の位置配列 [x, y]
        velocities: 形状 (3, 2) の速度配列 [vx, vy]
        masses: 形状 (3,) の質量配列
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        # 毎回異なる軌道を生成するため、現在時刻でシードを初期化
        import time
        np.random.seed(int(time.time() * 1000) % (2**32))
    
    # 質量（少しばらつきを持たせる）
    masses = np.array([1.0, 1.0 + 0.2 * np.random.randn(), 
                       1.0 + 0.2 * np.random.randn()])
    masses = np.clip(masses, 0.5, 1.5)  # 質量は0.5〜1.5の範囲
    
    # ランダムな初期位置（物体同士が十分離れるように配置）
    # 三角形配置をベースに、少しランダムにズラす
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3]) + np.random.randn(3) * 0.3
    radius = 0.8 + np.random.rand() * 0.4  # 0.8-1.2の範囲
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # ランダムな操動を加える
    positions += np.random.randn(3, 2) * 0.2
    
    # 重心を原点に移動
    center_of_mass = np.average(positions, axis=0, weights=masses)
    positions -= center_of_mass
    
    # ランダムな初期速度（小さめに設定して束縛状態を保証）
    velocities = np.random.randn(3, 2) * 0.3
    
    # 総運動量をゼロに調整（運動量保存のため）
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    velocities -= total_momentum / np.sum(masses)
    
    # エネルギーをチェックし、必要なら速度を調整して束縛状態を保証
    total_energy = _compute_energy_for_ic(positions, velocities, masses)
    
    # エネルギーが正（非束縛）の場合、速度を減らす
    while total_energy > -0.1:
        velocities *= 0.8
        total_energy = _compute_energy_for_ic(positions, velocities, masses)
    
    return positions, velocities, masses


def _compute_energy_for_ic(positions, velocities, masses):
    """初期条件チェック用のエネルギー計算（内部関数）"""
    n = len(masses)
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    pe = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[j] - positions[i])
            if r > 1e-10:
                pe -= G * masses[i] * masses[j] / r
    return ke + pe


# ============================================================
# 万有引力の法則（ベクトル形式）
# ============================================================

def compute_gravitational_forces(positions, masses):
    """
    万有引力の法則に基づき、各物体に働く力を計算
    
    Plummerソフトニングを使用: F = G * m_i * m_j / (r^2 + eps^2)^(3/2) * r
    これにより近接時の数値的特異性を回避しながら、遠方では通常の重力に収束
    
    Args:
        positions: 形状 (N, 2) の位置配列
        masses: 形状 (N,) の質量配列
    
    Returns:
        forces: 形状 (N, 2) の力配列
    """
    n = len(masses)
    forces = np.zeros_like(positions)
    softening = 0.05  # Plummerソフトニング長
    eps2 = softening ** 2
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # 物体jから物体iへの相対位置ベクトル
                r_ij = positions[j] - positions[i]
                r2 = np.dot(r_ij, r_ij)  # |r|^2
                
                # Plummerソフトニング力: F = G*m*M*r / (r^2 + eps^2)^(3/2)
                denom = (r2 + eps2) ** 1.5
                force_vec = G * masses[i] * masses[j] * r_ij / denom
                
                forces[i] += force_vec
    
    return forces



def compute_accelerations(positions, masses):
    """
    各物体の加速度を計算（F = ma より a = F/m）
    
    Args:
        positions: 形状 (N, 2) の位置配列
        masses: 形状 (N,) の質量配列
    
    Returns:
        accelerations: 形状 (N, 2) の加速度配列
    """
    forces = compute_gravitational_forces(positions, masses)
    accelerations = forces / masses[:, np.newaxis]
    return accelerations


# ============================================================
# 4次ルンゲ＝クッタ法（RK4）
# ============================================================

def rk4_step(positions, velocities, masses, dt):
    """
    4次ルンゲ＝クッタ法による1ステップの時間発展
    
    運動方程式:
        dr/dt = v
        dv/dt = a(r)
    
    RK4の4つの傾き（k1, k2, k3, k4）を計算し、加重平均で更新
    
    Args:
        positions: 現在の位置 (N, 2)
        velocities: 現在の速度 (N, 2)
        masses: 質量 (N,)
        dt: タイムステップ
    
    Returns:
        new_positions: 更新後の位置 (N, 2)
        new_velocities: 更新後の速度 (N, 2)
    """
    # k1: 現在の状態での傾き
    k1_r = velocities
    k1_v = compute_accelerations(positions, masses)
    
    # k2: 中間点1での傾き
    r2 = positions + 0.5 * dt * k1_r
    v2 = velocities + 0.5 * dt * k1_v
    k2_r = v2
    k2_v = compute_accelerations(r2, masses)
    
    # k3: 中間点2での傾き
    r3 = positions + 0.5 * dt * k2_r
    v3 = velocities + 0.5 * dt * k2_v
    k3_r = v3
    k3_v = compute_accelerations(r3, masses)
    
    # k4: 終端での傾き
    r4 = positions + dt * k3_r
    v4 = velocities + dt * k3_v
    k4_r = v4
    k4_v = compute_accelerations(r4, masses)
    
    # 加重平均による更新（RK4公式）
    new_positions = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_velocities = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return new_positions, new_velocities


# ============================================================
# エネルギー計算（シミュレーション精度の検証用）
# ============================================================

def compute_total_energy(positions, velocities, masses):
    """
    系の全エネルギー（運動エネルギー + ポテンシャルエネルギー）を計算
    エネルギー保存則により、この値は時間発展でほぼ一定であるべき
    
    Args:
        positions: 位置 (N, 2)
        velocities: 速度 (N, 2)
        masses: 質量 (N,)
    
    Returns:
        total_energy: スカラー値
    """
    n = len(masses)
    
    # 運動エネルギー: KE = Σ 0.5 * m * v^2
    kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    
    # Plummerポテンシャルエネルギー: PE = -Σ G * m_i * m_j / sqrt(r^2 + eps^2)
    potential_energy = 0.0
    softening = 0.05  # Plummerソフトニング長（力の計算と同じ値）
    eps2 = softening ** 2
    for i in range(n):
        for j in range(i+1, n):
            r_vec = positions[j] - positions[i]
            r2 = np.dot(r_vec, r_vec)
            potential_energy -= G * masses[i] * masses[j] / np.sqrt(r2 + eps2)
    
    return kinetic_energy + potential_energy


# ============================================================
# シミュレーション実行
# ============================================================

def run_simulation(mode='figure8', dt=DT, t_max=T_MAX, seed=None):
    """
    シミュレーションを実行し、全時刻の状態を記録
    
    Args:
        mode: 'figure8'(安定・8の字解) または 'chaos'(カオス的)
        dt: タイムステップ
        t_max: シミュレーション総時間
        seed: 乱数シード（chaosモードのみ、再現性のため）
    
    Returns:
        history: 各時刻の位置を格納した配列 (n_steps, 3, 2)
        energies: 各時刻のエネルギー (n_steps,)
        times: 時刻配列 (n_steps,)
    """
    # 初期条件の選択
    if mode == 'chaos':
        positions, velocities, masses = get_chaotic_initial_conditions(seed)
        print("🌀 カオスモード: ランダム初期条件")
    else:
        positions, velocities, masses = get_figure8_initial_conditions()
        print("♾️  8の字解モード: 安定周期軌道")
    
    # 記録用配列
    n_steps = int(t_max / dt)
    history = np.zeros((n_steps, 3, 2))
    energies = np.zeros(n_steps)
    times = np.zeros(n_steps)
    
    print("シミュレーション開始...")
    print(f"  タイムステップ: {dt}")
    print(f"  総時間: {t_max}")
    print(f"  総ステップ数: {n_steps}")
    
    # 時間発展ループ
    for step in range(n_steps):
        # 現在の状態を記録
        history[step] = positions.copy()
        energies[step] = compute_total_energy(positions, velocities, masses)
        times[step] = step * dt
        
        # RK4で1ステップ進める
        positions, velocities = rk4_step(positions, velocities, masses, dt)
        
        # 進捗表示
        if step % (n_steps // 10) == 0:
            progress = 100 * step / n_steps
            print(f"  進捗: {progress:.0f}%")
    
    print("シミュレーション完了!")
    
    # エネルギー保存の確認
    energy_drift = abs(energies[-1] - energies[0]) / abs(energies[0]) * 100
    print(f"  初期エネルギー: {energies[0]:.6f}")
    print(f"  最終エネルギー: {energies[-1]:.6f}")
    print(f"  エネルギードリフト: {energy_drift:.4f}%")
    
    return history, energies, times



# ============================================================
# アニメーション可視化
# ============================================================

def create_animation(history, times, energies=None, save_file=None, title='Figure-8 Solution'):
    """
    軌跡付きアニメーションを作成（リアルタイム情報表示付き）
    
    Args:
        history: 位置履歴 (n_steps, 3, 2)
        times: 時刻配列 (n_steps,)
        energies: エネルギー履歴 (n_steps,) - オプション
        save_file: 保存ファイル名（Noneなら保存しない）
        title: 表示タイトル
    """
    # カラー設定（鮮やかな配色）
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']  # 赤、青緑、黄
    
    # 表示範囲を動的に計算（カオスモードは広がる可能性がある）
    max_range = max(np.abs(history).max() * 1.2, 1.5)
    
    # プロット設定（情報パネル用に少し広めに）
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect('equal')
    ax.set_xlabel('X', color='white', fontsize=12)
    ax.set_ylabel('Y', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # タイトル
    ax.set_title(f'Three-Body Problem Simulation\n({title})', 
                 color='white', fontsize=14, fontweight='bold')
    
    # 情報パネル（左下に配置）
    info_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, 
                        color='#00ff88', fontsize=9, verticalalignment='bottom',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='#0a0a1a', 
                                  edgecolor='#00ff88', alpha=0.9))
    
    # 物体のプロット要素
    bodies = []
    trails = []
    
    for i in range(3):
        # 物体（大きな点）
        body, = ax.plot([], [], 'o', color=colors[i], markersize=15, 
                        markeredgecolor='white', markeredgewidth=1.5,
                        label=f'Body {i+1}')
        bodies.append(body)
        
        # 軌跡
        trail, = ax.plot([], [], '-', color=colors[i], alpha=0.6, linewidth=2)
        trails.append(trail)
    
    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white',
              labelcolor='white', fontsize=10)
    
    # サンプリング（アニメーション用に間引く）
    sample_rate = max(1, len(history) // 1000)
    sampled_history = history[::sample_rate]
    sampled_times = times[::sample_rate]
    sampled_energies = energies[::sample_rate] if energies is not None else None
    
    # 初期エネルギー（ドリフト計算用）
    initial_energy = energies[0] if energies is not None else 0
    
    # 軌跡の長さをサンプリング後の値に調整
    trail_frames = TRAIL_LENGTH // sample_rate
    
    def init():
        """アニメーション初期化"""
        for body, trail in zip(bodies, trails):
            body.set_data([], [])
            trail.set_data([], [])
        info_text.set_text('')
        return bodies + trails + [info_text]
    
    def update(frame):
        """アニメーション更新"""
        # 軌跡の開始フレーム
        trail_start = max(0, frame - trail_frames)
        
        # 情報テキストを構築
        info_lines = [f"Time: {sampled_times[frame]:.2f}"]
        
        # エネルギー情報
        if sampled_energies is not None:
            current_energy = sampled_energies[frame]
            drift = abs(current_energy - initial_energy) / abs(initial_energy) * 100
            info_lines.append(f"Energy: {current_energy:.4f}")
            info_lines.append(f"Drift: {drift:.4f}%")
        
        info_lines.append("")  # 空行
        
        for i, (body, trail) in enumerate(zip(bodies, trails)):
            # 物体の現在位置
            x, y = sampled_history[frame, i]
            body.set_data([x], [y])
            
            # 軌跡
            trail_x = sampled_history[trail_start:frame+1, i, 0]
            trail_y = sampled_history[trail_start:frame+1, i, 1]
            trail.set_data(trail_x, trail_y)
            
            # 各物体の位置情報
            info_lines.append(f"Body {i+1}: ({x:+.3f}, {y:+.3f})")
        
        info_text.set_text('\n'.join(info_lines))
        
        return bodies + trails + [info_text]
    
    # アニメーション作成
    anim = FuncAnimation(
        fig, update, frames=len(sampled_history),
        init_func=init, blit=True, interval=ANIMATION_INTERVAL
    )
    
    # 保存（オプション）
    if save_file:
        print(f"アニメーション保存中: {save_file}")
        anim.save(save_file, writer='pillow', fps=30)
        print("保存完了!")
    
    plt.tight_layout()
    plt.show()
    
    return anim



# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Three-Body Problem Simulator")
    print("  Physics: Newton's Law of Universal Gravitation")
    print("  Integration: 4th-order Runge-Kutta (RK4)")
    print("=" * 60)
    print()
    print("モードを選択してください:")
    print("  1: ♾️  8の字解（安定した美しい軌道）")
    print("  2: 🌀 カオスモード（予測不能な動き）")
    print()
    
    choice = input("選択 (1 または 2): ").strip()
    
    if choice == '2':
        mode = 'chaos'
        print()
        print("🌀 カオスモードを選択しました")
        print("初期条件がランダムなので、毎回異なる軌道が見られます。")
        print("短期的には正確ですが、長期予測は不可能（カオス）です。")
    else:
        mode = 'figure8'
        print()
        print("♾️  8の字解モードを選択しました")
        print("数学的に証明された美しい周期軌道です。")
    
    print()
    
    # シミュレーション実行
    history, energies, times = run_simulation(mode=mode)
    
    print()
    print("アニメーション表示を開始...")
    print("（ウィンドウを閉じると終了します）")
    
    # アニメーション表示
    title = 'Chaotic Motion' if mode == 'chaos' else 'Figure-8 Solution'
    anim = create_animation(history, times, energies=energies, title=title)

