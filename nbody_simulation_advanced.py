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

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import time


# ============================================================
# 設定クラス
# ============================================================

@dataclass
class SimulationConfig:
    """シミュレーション設定"""
    n_bodies: int = 3
    g: float = 1.0
    base_dt: float = 0.001
    min_dt: float = 0.0001
    max_dt: float = 0.01
    softening: float = 0.05
    display_range: float = 1.5
    mass_min: float = 0.5
    mass_max: float = 2.0
    animation_interval: int = 30
    velocity_arrow_scale: float = 0.3
    force_arrow_scale: float = 0.15
    max_trail: int = 400
    steps_per_frame: int = 10

    def validate(self) -> None:
        """設定のバリデーション"""
        validate_parameters(
            self.n_bodies,
            mass_min=self.mass_min,
            mass_max=self.mass_max,
            softening=self.softening
        )


# デフォルト設定（後方互換性のため）
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
# 周期解カタログ
# ============================================================

PERIODIC_SOLUTIONS = [
    {
        "name": "Figure-8 Classic",
        "label": "[1/8] Figure-8 Classic",
        "description": "Chenciner-Montgomery (2000)",
        "positions": np.array([
            [0.97000436, -0.24308753, 0.0],
            [-0.97000436, 0.24308753, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.466203685, 0.43236573, 0.0],
            [0.466203685, 0.43236573, 0.0],
            [-0.93240737, -0.86473146, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
    {
        "name": "Figure-8 (I.2.A)",
        "label": "[2/8] Figure-8 (I.2.A)",
        "description": "Suvakov-Dmitrasinovic (2013)",
        "positions": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.306893, 0.125507, 0.0],
            [0.306893, 0.125507, 0.0],
            [-0.613786, -0.251014, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
    {
        "name": "Butterfly I",
        "label": "[3/8] Butterfly I",
        "description": "Suvakov-Dmitrasinovic I.8.A",
        "positions": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.412103, 0.283384, 0.0],
            [0.412103, 0.283384, 0.0],
            [-0.824206, -0.566768, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
    {
        "name": "Lagrange Triangle",
        "label": "[4/8] Lagrange Triangle",
        "description": "Lagrange (1772)",
        "positions": np.array([
            [0.0, 1.0, 0.0],
            [np.sqrt(3)/2, -0.5, 0.0],
            [-np.sqrt(3)/2, -0.5, 0.0]
        ]),
        "velocities": np.array([
            [0.5, 0.0, 0.0],
            [-0.25, -np.sqrt(3)/4, 0.0],
            [-0.25, np.sqrt(3)/4, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
    {
        "name": "Moth I",
        "label": "[5/8] Moth I",
        "description": "Suvakov-Dmitrasinovic I.B.1",
        "positions": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.46444, 0.39606, 0.0],
            [0.46444, 0.39606, 0.0],
            [-0.92888, -0.79212, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
    {
        "name": "Yin-Yang Ia",
        "label": "[6/8] Yin-Yang Ia",
        "description": "Suvakov-Dmitrasinovic II.C.2a",
        "positions": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.51394, 0.30474, 0.0],
            [0.51394, 0.30474, 0.0],
            [-1.02788, -0.60948, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
    {
        "name": "Yin-Yang Ib",
        "label": "[7/8] Yin-Yang Ib",
        "description": "Suvakov-Dmitrasinovic II.C.2b",
        "positions": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.28270, 0.32721, 0.0],
            [0.28270, 0.32721, 0.0],
            [-0.56540, -0.65442, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
    {
        "name": "Yin-Yang II",
        "label": "[8/8] Yin-Yang II",
        "description": "Suvakov-Dmitrasinovic II.C.3a",
        "positions": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.41682, 0.33033, 0.0],
            [0.41682, 0.33033, 0.0],
            [-0.83364, -0.66066, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
]


# ============================================================
# バリデーション
# ============================================================

def validate_parameters(
    n_bodies: int,
    masses: Optional[np.ndarray] = None,
    softening: Optional[float] = None,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None
) -> None:
    """パラメータのバリデーション"""
    if n_bodies < 2:
        raise ValueError(f"物体数は2以上である必要があります: {n_bodies}")
    if n_bodies > 20:
        raise ValueError(f"物体数が多すぎます（パフォーマンス警告）: {n_bodies}")
    if masses is not None and np.any(masses <= 0):
        raise ValueError("質量は正の値である必要があります")
    if softening is not None and softening <= 0:
        raise ValueError(f"ソフトニングは正の値である必要があります: {softening}")
    if mass_min is not None and mass_max is not None:
        if mass_min <= 0 or mass_max <= 0:
            raise ValueError("質量範囲は正の値である必要があります")
        if mass_min > mass_max:
            raise ValueError("mass_min は mass_max 以下である必要があります")


# ============================================================
# 物理計算（純粋関数）
# ============================================================

def compute_accelerations_vectorized(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    g: float = G
) -> np.ndarray:
    """完全ベクトル化された加速度計算（ループなし・高速化版）"""
    n = len(masses)
    eps2 = softening ** 2
    
    # 全ペア間の差分ベクトルを一括計算
    # r_ij[i, j] = positions[j] - positions[i]
    r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    
    # 距離の二乗 + ソフトニング
    r2 = np.sum(r_ij ** 2, axis=2) + eps2
    np.fill_diagonal(r2, 1.0)  # 自己相互作用を避ける
    
    # 1/r³ 計算
    inv_r3 = r2 ** (-1.5)
    np.fill_diagonal(inv_r3, 0.0)
    
    # 加速度 = G * Σ(m_j * r_ij / |r_ij|³)
    accelerations = g * np.sum(
        masses[np.newaxis, :, np.newaxis] * r_ij * inv_r3[:, :, np.newaxis],
        axis=1
    )
    
    return accelerations


def compute_forces(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    g: float = G
) -> np.ndarray:
    """各物体にかかる力を計算（力ベクトル表示用）"""
    n = len(masses)
    forces = np.zeros_like(positions)
    eps2 = softening ** 2
    
    for i in range(n):
        for j in range(n):
            if i != j:
                r_ij = positions[j] - positions[i]
                r2 = np.dot(r_ij, r_ij) + eps2
                force_mag = g * masses[i] * masses[j] / r2
                force_dir = r_ij / np.sqrt(r2)
                forces[i] += force_mag * force_dir
    
    return forces


def compute_min_distance(positions: np.ndarray) -> float:
    """最小距離を計算"""
    n = len(positions)
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[j] - positions[i])
            min_dist = min(min_dist, dist)
    return min_dist


def adaptive_timestep(
    positions: np.ndarray,
    base_dt: float,
    min_dt: float,
    max_dt: float
) -> float:
    """適応タイムステップを計算"""
    min_dist = compute_min_distance(positions)
    factor = min(1.0, min_dist / 0.3)
    dt = base_dt * factor
    return max(min_dt, min(max_dt, dt))


def rk4_step_adaptive(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float,
    base_dt: float,
    min_dt: float,
    max_dt: float,
    g: float = G
) -> Tuple[np.ndarray, np.ndarray, float]:
    """適応タイムステップ付きRK4積分"""
    dt = adaptive_timestep(positions, base_dt, min_dt, max_dt)
    
    k1_r = velocities
    k1_v = compute_accelerations_vectorized(positions, masses, softening, g)
    
    k2_r = velocities + 0.5 * dt * k1_v
    k2_v = compute_accelerations_vectorized(positions + 0.5 * dt * k1_r, masses, softening, g)
    
    k3_r = velocities + 0.5 * dt * k2_v
    k3_v = compute_accelerations_vectorized(positions + 0.5 * dt * k2_r, masses, softening, g)
    
    k4_r = velocities + dt * k3_v
    k4_v = compute_accelerations_vectorized(positions + dt * k3_r, masses, softening, g)
    
    new_pos = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_vel = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return new_pos, new_vel, dt


def compute_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float,
    g: float = G
) -> float:
    """全エネルギーを計算"""
    n = len(masses)
    eps2 = softening ** 2
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    pe = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r2 = np.sum((positions[j] - positions[i])**2)
            pe -= g * masses[i] * masses[j] / np.sqrt(r2 + eps2)
    return ke + pe


def is_out_of_bounds(positions: np.ndarray, bound: float) -> bool:
    """境界外かどうかを判定"""
    return np.any(np.abs(positions) > bound)


# ============================================================
# 初期条件生成
# ============================================================

def generate_initial_conditions(
    n_bodies: int,
    mass_min: float,
    mass_max: float,
    softening: float = SOFTENING,
    g: float = G
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """初期条件生成（バリデーション付き）"""
    validate_parameters(n_bodies, mass_min=mass_min, mass_max=mass_max)
    
    np.random.seed(int(time.time() * 1000) % (2**32))
    
    masses = mass_min + np.random.rand(n_bodies) * (mass_max - mass_min)
    positions = np.random.randn(n_bodies, 3) * 0.5
    positions = np.clip(positions, -1.0, 1.0)
    center_of_mass = np.average(positions, axis=0, weights=masses)
    positions -= center_of_mass
    
    velocities = np.random.randn(n_bodies, 3) * 0.4
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    velocities -= total_momentum / np.sum(masses)
    
    energy = compute_energy(positions, velocities, masses, softening, g)
    while energy > -0.3:
        velocities *= 0.9
        energy = compute_energy(positions, velocities, masses, softening, g)
    
    return positions, velocities, masses


# ============================================================
# シミュレーション状態
# ============================================================

@dataclass
class SimulationState:
    """シミュレーション状態"""
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    n_bodies: int
    generation: int = 1
    sim_time: float = 0.0
    max_generation: int = 1
    
    # UIの状態
    paused: bool = False
    auto_rotate: bool = False
    show_forces: bool = False
    show_editor: bool = False
    prediction_mode: bool = False
    prediction_made: bool = False
    
    # 周期解モード
    periodic_mode: bool = False
    periodic_index: int = 0
    periodic_name: str = ""
    
    # 視点
    azim: float = 30.0
    zoom: float = 1.0
    
    # 軌跡
    trail_history: List[List[np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trail_history:
            self.trail_history = [[] for _ in range(self.n_bodies)]


# ============================================================
# シミュレーター クラス
# ============================================================

class NBodySimulator:
    """N体シミュレーター"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.config.validate()
        self.state = self._create_initial_state()
    
    def _create_initial_state(self) -> SimulationState:
        """初期状態を作成"""
        positions, velocities, masses = generate_initial_conditions(
            self.config.n_bodies,
            self.config.mass_min,
            self.config.mass_max,
            self.config.softening,
            self.config.g
        )
        return SimulationState(
            positions=positions,
            velocities=velocities,
            masses=masses,
            n_bodies=self.config.n_bodies
        )
    
    def step(self, steps: int = 1) -> float:
        """シミュレーションをn ステップ進める"""
        total_dt = 0.0
        for _ in range(steps):
            self.state.positions, self.state.velocities, dt = rk4_step_adaptive(
                self.state.positions,
                self.state.velocities,
                self.state.masses,
                self.config.softening,
                self.config.base_dt,
                self.config.min_dt,
                self.config.max_dt,
                self.config.g
            )
            self.state.sim_time += dt
            total_dt += dt
        return total_dt
    
    def restart(self) -> None:
        """新しい初期条件でリスタート"""
        self.state.generation += 1
        self.state.max_generation = max(self.state.max_generation, self.state.generation)
        self.state.positions, self.state.velocities, self.state.masses = generate_initial_conditions(
            self.state.n_bodies,
            self.config.mass_min,
            self.config.mass_max,
            self.config.softening,
            self.config.g
        )
        self.state.sim_time = 0.0
        self.state.trail_history = [[] for _ in range(self.state.n_bodies)]
        self.state.prediction_mode = False
    
    def change_n_bodies(self, new_n: int) -> None:
        """物体数を変更"""
        if new_n == self.state.n_bodies:
            return
        
        validate_parameters(new_n, mass_min=self.config.mass_min, mass_max=self.config.mass_max)
        
        self.state.n_bodies = new_n
        self.state.positions, self.state.velocities, self.state.masses = generate_initial_conditions(
            new_n,
            self.config.mass_min,
            self.config.mass_max,
            self.config.softening,
            self.config.g
        )
        self.state.sim_time = 0.0
        self.state.trail_history = [[] for _ in range(new_n)]
        self.state.generation += 1
        print(f"🔢 Changed to {new_n} bodies - Generation {self.state.generation}")
    
    def update_trails(self) -> None:
        """軌跡を更新"""
        for i in range(self.state.n_bodies):
            self.state.trail_history[i].append(self.state.positions[i].copy())
            if len(self.state.trail_history[i]) > self.config.max_trail:
                self.state.trail_history[i].pop(0)
    
    def is_out_of_bounds(self) -> bool:
        """物体が範囲外かどうか"""
        return is_out_of_bounds(self.state.positions, self.config.display_range * self.state.zoom)
    
    def get_energy(self) -> float:
        """現在の全エネルギー"""
        return compute_energy(
            self.state.positions,
            self.state.velocities,
            self.state.masses,
            self.config.softening,
            self.config.g
        )
    
    def get_min_distance(self) -> float:
        """現在の最小距離"""
        return compute_min_distance(self.state.positions)
    
    def get_forces(self) -> np.ndarray:
        """力ベクトルを取得"""
        return compute_forces(
            self.state.positions,
            self.state.masses,
            self.config.softening,
            self.config.g
        )
    
    def toggle_periodic_mode(self) -> None:
        """周期解モードのトグル/次の解へ切り替え"""
        if not self.state.periodic_mode:
            # 周期解モードに入る
            self.state.periodic_mode = True
            self.state.periodic_index = 0
            self._apply_periodic_solution(0)
        else:
            # 次の解へ
            self.state.periodic_index = (self.state.periodic_index + 1) % len(PERIODIC_SOLUTIONS)
            if self.state.periodic_index == 0:
                # 一周したら通常モードに戻る
                self.state.periodic_mode = False
                self.state.periodic_name = ""
                self.restart()
                print("🔄 周期解モード終了 → 通常モードへ")
            else:
                self._apply_periodic_solution(self.state.periodic_index)
    
    def _apply_periodic_solution(self, index: int) -> None:
        """周期解を適用"""
        solution = PERIODIC_SOLUTIONS[index]
        self.state.positions = solution["positions"].copy()
        self.state.velocities = solution["velocities"].copy()
        self.state.masses = solution["masses"].copy()
        self.state.n_bodies = 3
        self.state.periodic_name = solution['label']
        self.state.sim_time = 0.0
        self.state.trail_history = [[] for _ in range(3)]
        self.state.generation += 1
        print(f"* {solution['label']} - {solution['description']}")
    
    def reload_periodic_solution(self) -> None:
        """現在の周期解をリロード（Rキー用）"""
        if self.state.periodic_mode:
            self._apply_periodic_solution(self.state.periodic_index)
        else:
            self.restart()
    
    def run(self) -> None:
        """GUIを起動して実行"""
        run_simulation_gui(self)


# ============================================================
# GUI / アニメーション
# ============================================================

def run_simulation_gui(simulator: NBodySimulator) -> FuncAnimation:
    """シミュレーションGUIを実行"""
    
    config = simulator.config
    state = simulator.state
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(state.n_bodies, 10)))[:state.n_bodies]
    
    # プロット設定
    fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
    fig.canvas.manager.set_window_title('N-Body Problem Simulator - Learning Edition')
    
    # メイン3Dプロット
    display_range = config.display_range
    ax_3d = fig.add_axes([0.05, 0.1, 0.65, 0.85], projection='3d', facecolor='#1a1a2e')
    ax_3d.set_xlim(-display_range, display_range)
    ax_3d.set_ylim(-display_range, display_range)
    ax_3d.set_zlim(-display_range, display_range)
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
    
    # 情報パネル
    info_text = fig.text(0.02, 0.02, '', color='#00ff88', fontsize=9,
                         fontfamily='monospace', verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='#0a0a1a', 
                                   edgecolor='#00ff88', alpha=0.9))
    
    # 操作説明パネル
    controls_text = fig.text(0.72, 0.95, 
        '🎮 CONTROLS\n'
        '─────────────\n'
        '[SPACE] Pause\n'
        '[R] Restart\n'
        '[A] Auto-rotate\n'
        '[F] Force vectors\n'
        '[E] Editor panel\n'
        '[P] Predict mode\n'
        '[M] Periodic sols\n'
        '[+/-] Zoom\n'
        '[Q] Quit\n'
        '─────────────\n'
        'Drag to rotate\n'
        'Scroll to zoom',
        color='#888888', fontsize=9, fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#0a0a1a', 
                  edgecolor='#444444', alpha=0.9))
    
    # エディタパネル
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
    
    # 周期解名表示
    periodic_text = fig.text(0.35, 0.92, '', color='#00ccff', fontsize=12,
                            fontfamily='monospace', fontweight='bold',
                            horizontalalignment='center',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='#0a1a2a', 
                                      edgecolor='#00ccff', alpha=0.9),
                            visible=False)
    
    def update_periodic_display() -> None:
        """周期解名の表示を更新"""
        if state.periodic_mode and state.periodic_name:
            periodic_text.set_text(f"{state.periodic_name}\n[M] next solution")
            periodic_text.set_visible(True)
        else:
            periodic_text.set_visible(False)
    
    # 描画オブジェクト
    bodies: List = []
    trails: List = []
    velocity_arrows: List = []
    force_arrows: List = []
    
    def create_plot_objects(n: int) -> None:
        nonlocal bodies, trails, velocity_arrows, force_arrows, colors
        
        # 既存のプロットオブジェクトをAxesから削除
        for body in bodies:
            body.remove()
        for trail in trails:
            trail.remove()
        for arrow in velocity_arrows:
            arrow.remove()
        for force in force_arrows:
            force.remove()
        
        bodies.clear()
        trails.clear()
        velocity_arrows.clear()
        force_arrows.clear()
        colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))[:n]
        
        for i in range(n):
            body, = ax_3d.plot([], [], [], 'o', color=colors[i], markersize=10,
                              markeredgecolor='white', markeredgewidth=1)
            bodies.append(body)
            trail, = ax_3d.plot([], [], [], '-', color=colors[i], alpha=0.4, linewidth=1)
            trails.append(trail)
            arrow, = ax_3d.plot([], [], [], '-', color=colors[i], linewidth=1.5, alpha=0.7)
            velocity_arrows.append(arrow)
            force, = ax_3d.plot([], [], [], '-', color='#ff4444', linewidth=2, alpha=0.8)
            force_arrows.append(force)
    
    create_plot_objects(state.n_bodies)
    
    force_label = fig.text(0.72, 0.08, '', color='#ff4444', fontsize=8,
                          fontfamily='monospace', visible=False)
    
    def update_editor_panel() -> None:
        lines = [
            '📝 EDITOR',
            '─────────────',
            f'N Bodies: {state.n_bodies}',
            '(Press 3-9 to change)',
            '',
            '📊 Current masses:',
        ]
        for i in range(min(state.n_bodies, 6)):
            lines.append(f'  Body {i+1}: {state.masses[i]:.2f}')
        if state.n_bodies > 6:
            lines.append(f'  ... +{state.n_bodies-6} more')
        
        lines.extend([
            '',
            '🎯 Tips:',
            '• More bodies = chaos',
            '• Watch the forces!',
            '• Try predicting!',
        ])
        
        editor_text.set_text('\n'.join(lines))
    
    def update_zoom() -> None:
        r = config.display_range * state.zoom
        ax_3d.set_xlim(-r, r)
        ax_3d.set_ylim(-r, r)
        ax_3d.set_zlim(-r, r)
    
    def on_key(event) -> None:
        if event.key == ' ':
            state.paused = not state.paused
            print(f"  {'PAUSED' if state.paused else 'RUNNING'}")
        
        elif event.key == 'r':
            # 周期解モードなら現在の解をリロード、通常ならランダム再生成
            simulator.reload_periodic_solution()
            create_plot_objects(state.n_bodies)
            prediction_text.set_visible(False)
            if state.periodic_mode:
                print(f"Reload: {state.periodic_name}")
            else:
                print(f"Restart - Generation {state.generation}")
        
        elif event.key == 'a':
            state.auto_rotate = not state.auto_rotate
            print(f"🔄 Auto-rotate: {'ON' if state.auto_rotate else 'OFF'}")
        
        elif event.key == 'f':
            state.show_forces = not state.show_forces
            force_label.set_visible(state.show_forces)
            if state.show_forces:
                force_label.set_text('🔴 Red arrows = Gravitational force')
            print(f"⚡ Force vectors: {'ON' if state.show_forces else 'OFF'}")
        
        elif event.key == 'e':
            state.show_editor = not state.show_editor
            editor_text.set_visible(state.show_editor)
            if state.show_editor:
                update_editor_panel()
            print(f"📝 Editor: {'OPEN' if state.show_editor else 'CLOSED'}")
        
        elif event.key == 'p':
            state.prediction_mode = not state.prediction_mode
            if state.prediction_mode:
                state.paused = True
                state.prediction_made = False
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
        
        elif event.key == 'enter' and state.prediction_mode:
            state.paused = False
            state.prediction_made = True
            prediction_text.set_text('▶️ Running...\nWatch what happens!')
        
        elif event.key == 'q':
            print("👋 Exiting...")
            plt.close()
        
        elif event.key == 'm':
            # 周期解モードのトグル/次の解へ
            simulator.toggle_periodic_mode()
            create_plot_objects(state.n_bodies)
            update_periodic_display()
            if state.show_editor:
                update_editor_panel()
        
        elif event.key in ['+', '=']:
            state.zoom = max(0.3, state.zoom * 0.8)
            update_zoom()
        
        elif event.key == '-':
            state.zoom = min(3.0, state.zoom * 1.25)
            update_zoom()
        
        elif event.key in ['3', '4', '5', '6', '7', '8', '9']:
            new_n = int(event.key)
            if new_n != state.n_bodies:
                # 周期解モードを終了
                state.periodic_mode = False
                state.periodic_name = ""
                update_periodic_display()
                simulator.change_n_bodies(new_n)
                create_plot_objects(state.n_bodies)
                if state.show_editor:
                    update_editor_panel()
    
    def on_scroll(event) -> None:
        if event.button == 'up':
            state.zoom = max(0.3, state.zoom * 0.9)
        else:
            state.zoom = min(3.0, state.zoom * 1.1)
        update_zoom()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    def update(frame: int) -> List:
        if state.paused:
            if state.show_forces:
                forces = simulator.get_forces()
                for i in range(state.n_bodies):
                    x, y, z = state.positions[i]
                    fx, fy, fz = forces[i] * config.force_arrow_scale
                    force_arrows[i].set_data([x, x+fx], [y, y+fy])
                    force_arrows[i].set_3d_properties([z, z+fz])
            return bodies + trails + velocity_arrows + force_arrows + [info_text]
        
        # シミュレーション進行
        simulator.step(config.steps_per_frame)
        
        # 境界チェック（周期解モードでは無効化 - 数値ドリフトの観察のため）
        if not state.periodic_mode and simulator.is_out_of_bounds():
            print(f"Generation {state.generation} ended at t={state.sim_time:.2f}")
            simulator.restart()
            create_plot_objects(state.n_bodies)
            
            if state.prediction_mode:
                state.prediction_mode = False
                prediction_text.set_text('They escaped!\nPress [P] to try again')
            
            if state.show_editor:
                update_editor_panel()
        
        # 軌跡更新
        simulator.update_trails()
        
        # 計算
        energy = simulator.get_energy()
        min_dist = simulator.get_min_distance()
        
        # 力計算
        forces = simulator.get_forces() if state.show_forces else None
        
        # 情報テキスト
        info_lines = [
            f"Gen: {state.generation}  Time: {state.sim_time:.1f}  Zoom: {1/state.zoom:.1f}x",
            f"Energy: {energy:.3f}  MinDist: {min_dist:.2f}",
            f"Bodies: {state.n_bodies}  MaxGen: {state.max_generation}",
        ]
        info_text.set_text('\n'.join(info_lines))
        
        # 3D描画更新
        for i in range(state.n_bodies):
            x, y, z = state.positions[i]
            vx, vy, vz = state.velocities[i]
            mass = state.masses[i]
            
            size = 6 + (mass - config.mass_min) * 6
            
            bodies[i].set_data([x], [y])
            bodies[i].set_3d_properties([z])
            bodies[i].set_markersize(size)
            
            if state.trail_history[i]:
                trail_arr = np.array(state.trail_history[i])
                trails[i].set_data(trail_arr[:, 0], trail_arr[:, 1])
                trails[i].set_3d_properties(trail_arr[:, 2])
            
            # 速度ベクトル
            arrow_end = [x + vx * config.velocity_arrow_scale, 
                         y + vy * config.velocity_arrow_scale, 
                         z + vz * config.velocity_arrow_scale]
            velocity_arrows[i].set_data([x, arrow_end[0]], [y, arrow_end[1]])
            velocity_arrows[i].set_3d_properties([z, arrow_end[2]])
            
            # 力ベクトル
            if state.show_forces and forces is not None:
                fx, fy, fz = forces[i] * config.force_arrow_scale
                force_arrows[i].set_data([x, x+fx], [y, y+fy])
                force_arrows[i].set_3d_properties([z, z+fz])
            else:
                force_arrows[i].set_data([], [])
                force_arrows[i].set_3d_properties([])
        
        if state.auto_rotate:
            state.azim += 0.3
            ax_3d.view_init(elev=20, azim=state.azim)
        
        return bodies + trails + velocity_arrows + force_arrows + [info_text]
    
    anim = FuncAnimation(fig, update, frames=None, blit=False, 
                         interval=config.animation_interval, cache_frame_data=False)
    
    plt.show()
    
    return anim


# ============================================================
# 後方互換性のための関数
# ============================================================

def run_advanced_simulation() -> FuncAnimation:
    """フル機能版N体シミュレーター + 教育モード（後方互換性）"""
    simulator = NBodySimulator()
    return run_simulation_gui(simulator)


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
    
    # 方法1: 新しいクラスベースのAPI
    simulator = NBodySimulator()
    simulator.run()
    
    # 方法2: 後方互換性のある関数
    # run_advanced_simulation()
