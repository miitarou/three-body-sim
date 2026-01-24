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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
    # Plummerソフトニング: F = Gm1m2 / (r^2 + ε^2)^(3/2)
    # 通常モード: 極端な接近時の数値発散（1/r^2 → ∞）を防止するために必要
    softening: float = 0.05
    # 周期解モード: 周期解は純粋なニュートン力学（ε=0）で発見されたものであるため、
    # ソフトニングが大きいと軌道が理論値からずれる。より純粋な1/r^2に近づけるため極小化。
    softening_periodic: float = 0.001
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
    # ⭐ おすすめ 1: 数学史上最も有名な三体周期解
    {
        "name": "Figure-8 Classic",
        "label": "[1/10] Figure-8 Classic",
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
    # ⭐ おすすめ 2: 歴史的価値最高（1772年発見）
    {
        "name": "Lagrange Triangle",
        "label": "[2/10] Lagrange Triangle",
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
    # ⭐ おすすめ 3: 美しい蝶の軌道
    {
        "name": "Butterfly I",
        "label": "[3/10] Butterfly I",
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
        "name": "Figure-8 (I.2.A)",
        "label": "[4/10] Figure-8 (I.2.A)",
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
        "name": "Moth I",
        "label": "[5/10] Moth I",
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
        "label": "[6/10] Yin-Yang Ia",
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
        "label": "[7/10] Yin-Yang Ib",
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
        "label": "[8/10] Yin-Yang II",
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
    {
        "name": "Yin-Yang III",
        "label": "[9/10] Yin-Yang III",
        "description": "Suvakov-Dmitrasinovic III.9.A",
        "positions": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.513150, 0.289437, 0.0],
            [0.513150, 0.289437, 0.0],
            [-1.02630, -0.578874, 0.0]
        ]),
        "masses": np.array([1.0, 1.0, 1.0])
    },
    {
        "name": "Yarn",
        "label": "[10/10] Yarn",
        "description": "Suvakov-Dmitrasinovic III.13.A",
        "positions": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
        "velocities": np.array([
            [0.416444, 0.336397, 0.0],
            [0.416444, 0.336397, 0.0],
            [-0.832888, -0.672794, 0.0]
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


import mojo_backend

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
    
    # Mojoバックエンドを使用（利用可能な場合）
    # バックエンド側でNumPyフォールバックも持っているが、
    # ここではエンジンを取得して委譲する
    engine = mojo_backend.get_engine()
    
    new_pos, new_vel = engine.rk4_step(
        positions, velocities, masses, softening, dt, g
    )
    
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
    
    # 予測クイズ
    quiz_active: bool = False
    quiz_answer: int = 0  # 0=未設定, 1=衝突, 2=逃亡, 3=安定
    quiz_user_choice: int = 0
    quiz_correct: int = 0  # 正解数
    quiz_total: int = 0  # 総回答数
    
    # ゴーストモード（カオス可視化）
    ghost_mode: bool = False
    ghost_positions: Optional[np.ndarray] = None
    ghost_velocities: Optional[np.ndarray] = None
    ghost_trail_history: List[List[np.ndarray]] = field(default_factory=list)
    
    # 周期解モード
    periodic_mode: bool = False
    periodic_index: int = 0
    periodic_name: str = ""
    
    # 視点
    azim: float = 30.0
    zoom: float = 1.0
    
    # 軌跡
    trail_history: List[List[np.ndarray]] = field(default_factory=list)
    
    # 初期条件履歴（巻き戻し用）
    history_buffer: List[dict] = field(default_factory=list)
    history_max_size: int = 10

    def __post_init__(self) -> None:
        if not self.trail_history:
            self.trail_history = [[] for _ in range(self.n_bodies)]
        if not self.ghost_trail_history:
            self.ghost_trail_history = [[] for _ in range(self.n_bodies)]


# ============================================================
# シミュレーター クラス
# ============================================================

class NBodySimulator:
    """N体シミュレーター"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.config.validate()
        self.state = self._create_initial_state()
        # 最初の状態を履歴に保存
        self.save_to_history()
    
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
    
    def get_effective_softening(self) -> float:
        """現在のモードに応じたソフトニング値を返す"""
        if self.state.periodic_mode:
            return self.config.softening_periodic
        return self.config.softening
    
    def step(self, steps: int = 1) -> float:
        """シミュレーションをn ステップ進める"""
        softening = self.get_effective_softening()
        total_dt = 0.0
        for _ in range(steps):
            self.state.positions, self.state.velocities, dt = rk4_step_adaptive(
                self.state.positions,
                self.state.velocities,
                self.state.masses,
                softening,
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
        
        # 履歴に自動保存
        self.save_to_history()
    
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
            self.get_effective_softening(),
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
            self.get_effective_softening(),
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
    
    def predict_future(self, real_seconds: float = 2.5) -> int:
        """
        指定実時間後の状態をシャドウ計算で予測
        
        Args:
            real_seconds: 予測する実時間（秒）
        
        Returns:
            1: 衝突（急接近）
            2: 逃亡（境界外に出る）
            3: 安定軌道（どちらでもない）
        """
        # 実時間をシミュレーション時間に変換
        # 1フレーム = 30ms, 10ステップ/フレーム, dt=0.001
        # → 1秒 ≈ 0.33 シミュレーション時間
        sim_time_target = real_seconds * 0.33
        
        # 現在の状態をコピー
        pos = self.state.positions.copy()
        vel = self.state.velocities.copy()
        masses = self.state.masses.copy()
        initial_pos = pos.copy()
        
        elapsed = 0.0
        # 衝突判定: ソフトニングの1.5倍以下なら「衝突」（非常に近い接近）
        collision_threshold = self.get_effective_softening() * 1.5
        # 逃亡判定: 表示範囲の1.5倍を超えたら逃亡
        escape_bound = self.config.display_range * 1.5
        
        min_distance_found = float('inf')
        max_displacement = 0.0
        
        while elapsed < sim_time_target:
            dt = self.config.base_dt
            pos, vel, _ = rk4_step_adaptive(
                pos, vel, masses,
                self.get_effective_softening(),
                dt, dt, dt,
                self.config.g
            )
            elapsed += dt
            
            # 最小距離を追跡
            min_dist = compute_min_distance(pos)
            min_distance_found = min(min_distance_found, min_dist)
            
            # 最大変位を追跡
            displacement = np.max(np.linalg.norm(pos - initial_pos, axis=1))
            max_displacement = max(max_displacement, displacement)
        
        # 判定（優先順位: 逃亡 > 衝突 > 安定）
        if max_displacement > escape_bound:
            print(f"  [Debug] Prediction: ESCAPE (displacement={max_displacement:.2f})")
            return 2  # 逃亡
        elif min_distance_found < collision_threshold:
            print(f"  [Debug] Prediction: COLLISION (min_dist={min_distance_found:.3f})")
            return 1  # 衝突
        else:
            print(f"  [Debug] Prediction: STABLE (min_dist={min_distance_found:.3f}, disp={max_displacement:.2f})")
            return 3  # 安定軌道

    
    def start_quiz(self) -> None:
        """予測クイズを開始"""
        self.state.quiz_active = True
        self.state.quiz_user_choice = 0
        self.state.quiz_answer = self.predict_future(2.5)
        self.state.paused = True
    
    def answer_quiz(self, choice: int) -> bool:
        """
        クイズに回答
        
        Args:
            choice: 1=衝突, 2=逃亡, 3=安定
        
        Returns:
            正解ならTrue
        """
        self.state.quiz_user_choice = choice
        self.state.quiz_total += 1
        correct = (choice == self.state.quiz_answer)
        if correct:
            self.state.quiz_correct += 1
        return correct
    
    def toggle_ghost_mode(self) -> None:
        """ゴーストモードをトグル"""
        self.state.ghost_mode = not self.state.ghost_mode
        if self.state.ghost_mode:
            # 現在の状態を少しだけずらしてゴーストを初期化
            perturbation = 0.001  # 非常に小さな摂動
            self.state.ghost_positions = self.state.positions.copy() + \
                np.random.randn(*self.state.positions.shape) * perturbation
            self.state.ghost_velocities = self.state.velocities.copy()
            self.state.ghost_trail_history = [[] for _ in range(self.state.n_bodies)]
            print("[G] Ghost mode ON - Watch the chaos unfold!")
        else:
            self.state.ghost_positions = None
            self.state.ghost_velocities = None
            self.state.ghost_trail_history = [[] for _ in range(self.state.n_bodies)]
            print("[G] Ghost mode OFF")
    
    def step_ghost(self, dt: float) -> None:
        """ゴーストのシミュレーションを1ステップ進める"""
        if not self.state.ghost_mode or self.state.ghost_positions is None:
            return
        
        self.state.ghost_positions, self.state.ghost_velocities, _ = rk4_step_adaptive(
            self.state.ghost_positions,
            self.state.ghost_velocities,
            self.state.masses,
            self.get_effective_softening(),
            dt, self.config.min_dt, self.config.max_dt,
            self.config.g
        )
        
        # ゴーストの軌跡を更新
        for i in range(self.state.n_bodies):
            self.state.ghost_trail_history[i].append(self.state.ghost_positions[i].copy())
            if len(self.state.ghost_trail_history[i]) > self.config.max_trail:
                self.state.ghost_trail_history[i].pop(0)
    
    def save_to_history(self) -> None:
        """現在の初期条件を履歴に保存"""
        snapshot = {
            'positions': self.state.positions.copy(),
            'velocities': self.state.velocities.copy(),
            'masses': self.state.masses.copy(),
            'n_bodies': self.state.n_bodies,
            'generation': self.state.generation
        }
        self.state.history_buffer.append(snapshot)
        # 最大サイズを超えたら古いものを削除
        if len(self.state.history_buffer) > self.state.history_max_size:
            self.state.history_buffer.pop(0)
    
    def rewind(self) -> bool:
        """直前のGenerationに巻き戻す（B key用）"""
        if len(self.state.history_buffer) < 2:
            print("[B] No history to rewind to")
            return False
        
        # 現在の状態を削除して1つ前に戻る
        self.state.history_buffer.pop()
        prev = self.state.history_buffer[-1]
        
        self.state.positions = prev['positions'].copy()
        self.state.velocities = prev['velocities'].copy()
        self.state.masses = prev['masses'].copy()
        self.state.n_bodies = prev['n_bodies']
        self.state.sim_time = 0.0
        self.state.trail_history = [[] for _ in range(self.state.n_bodies)]
        
        print(f"[B] Rewound to Generation {prev['generation']}")
        return True
    
    def export_json(self, filepath: Optional[str] = None) -> str:
        """初期条件をJSONファイルにエクスポート（S key用）"""
        import json
        from datetime import datetime
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'orbit_{timestamp}.json'
        
        data = {
            'positions': self.state.positions.tolist(),
            'velocities': self.state.velocities.tolist(),
            'masses': self.state.masses.tolist(),
            'n_bodies': self.state.n_bodies,
            'generation': self.state.generation,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[S] Exported to {filepath}")
        return filepath
    
    def import_json(self, filepath: str) -> bool:
        """JSONファイルから初期条件をインポート（L key用）"""
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.state.positions = np.array(data['positions'])
            self.state.velocities = np.array(data['velocities'])
            self.state.masses = np.array(data['masses'])
            self.state.n_bodies = data['n_bodies']
            self.state.sim_time = 0.0
            self.state.trail_history = [[] for _ in range(self.state.n_bodies)]
            self.state.generation += 1
            
            # 履歴に保存
            self.save_to_history()
            
            print(f"[L] Imported from {filepath}")
            return True
        except Exception as e:
            print(f"[L] Import failed: {e}")
            return False
    
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
        '== CONTROLS ==\n'
        '-------------\n'
        '[SPACE] Pause\n'
        '[R] Restart\n'
        '[A] Auto-rotate\n'
        '[F] Force vectors\n'
        '[G] Ghost mode\n'
        '[E] Editor panel\n'
        '[P] Predict mode\n'
        '[M] Periodic sols\n'
        '[+/-] Zoom\n'
        '[Q] Quit\n'
        '-------------\n'
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
    ghost_bodies: List = []
    ghost_trails: List = []
    glow_objects: List = []  # 星の輝きエフェクト
    
    def create_plot_objects(n: int) -> None:
        nonlocal bodies, trails, velocity_arrows, force_arrows, ghost_bodies, ghost_trails, glow_objects, colors
        
        # 既存のプロットオブジェクトをAxesから削除
        for body in bodies:
            body.remove()
        for trail_segments in trails:
            for segment in trail_segments:
                segment.remove()
        for arrow in velocity_arrows:
            arrow.remove()
        for force in force_arrows:
            force.remove()
        for ghost in ghost_bodies:
            ghost.remove()
        for ghost_t in ghost_trails:
            ghost_t.remove()
        for glow in glow_objects:
            glow.remove()
        
        bodies.clear()
        trails.clear()
        velocity_arrows.clear()
        force_arrows.clear()
        ghost_bodies.clear()
        ghost_trails.clear()
        glow_objects.clear()
        colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))[:n]
        
        # 軌跡のセグメント数（グラデーション効果用）
        n_segments = 10
        
        for i in range(n):
            body, = ax_3d.plot([], [], [], 'o', color=colors[i], markersize=10,
                              markeredgecolor='white', markeredgewidth=1)
            bodies.append(body)
            
            # グロー（輝き）エフェクト - 大きめの半透明の円を背後に
            glow, = ax_3d.plot([], [], [], 'o', color=colors[i], markersize=20,
                               alpha=0.2, markeredgewidth=0)
            glow_objects.append(glow)
            
            # 軌跡をセグメントに分割（直近=太く濃く、過去=細く淡く）
            trail_segments = []
            for seg in range(n_segments):
                # seg=0が最も古い、seg=n_segments-1が最も新しい
                alpha = 0.1 + 0.6 * (seg / (n_segments - 1))  # 0.1 → 0.7
                lw = 0.5 + 2.0 * (seg / (n_segments - 1))     # 0.5 → 2.5
                line, = ax_3d.plot([], [], [], '-', color=colors[i], alpha=alpha, linewidth=lw)
                trail_segments.append(line)
            trails.append(trail_segments)
            arrow, = ax_3d.plot([], [], [], '-', color=colors[i], linewidth=1.5, alpha=0.7)
            velocity_arrows.append(arrow)
            force, = ax_3d.plot([], [], [], '-', color='#ff4444', linewidth=2, alpha=0.8)
            force_arrows.append(force)
            # ゴースト（半透明・点線）
            ghost_body, = ax_3d.plot([], [], [], 'o', color=colors[i], markersize=8,
                                    alpha=0.6, markeredgecolor='white', markeredgewidth=0.5)
            ghost_bodies.append(ghost_body)
            ghost_trail, = ax_3d.plot([], [], [], '--', color=colors[i], alpha=0.4, linewidth=1.5)
            ghost_trails.append(ghost_trail)
    
    create_plot_objects(state.n_bodies)
    
    force_label = fig.text(0.72, 0.08, '', color='#ff4444', fontsize=8,
                          fontfamily='monospace', visible=False)
    
    # ゴーストモード表示
    ghost_label = fig.text(0.02, 0.95, '[G] GHOST MODE ON\nChaos visualization', 
                          color='#00ffaa', fontsize=10, fontweight='bold',
                          fontfamily='monospace', verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='#0a2a1a', 
                                    edgecolor='#00ffaa', alpha=0.9),
                          visible=False)
    
    def update_editor_panel() -> None:
        lines = [
            '== EDITOR ==',
            '-------------',
            f'N Bodies: {state.n_bodies}',
            '(Press 3-9 to change)',
            '',
            '-- Masses --',
        ]
        for i in range(min(state.n_bodies, 6)):
            lines.append(f'  Body {i+1}: {state.masses[i]:.2f}')
        if state.n_bodies > 6:
            lines.append(f'  ... +{state.n_bodies-6} more')
        
        lines.extend([
            '',
            '-- Tips --',
            '* More bodies = chaos',
            '* Watch the forces!',
            '* Try predicting!',
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
            
            # ゴーストもリセット（ONのままなら新しい位置で再初期化）
            if state.ghost_mode:
                simulator.toggle_ghost_mode()  # OFF
                simulator.toggle_ghost_mode()  # ON（新しい位置で）
            
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
            if not state.quiz_active:
                # クイズ開始
                simulator.start_quiz()
                prediction_text.set_text(
                    '[?] PREDICTION QUIZ\n'
                    '-----------------\n'
                    'In 2.5 seconds...\n'
                    'What will happen?\n\n'
                    'Press a key:\n'
                    '  [1] Collision\n'
                    '  [2] Escape\n'
                    '  [3] Stable orbit\n\n'
                    f'Score: {state.quiz_correct}/{state.quiz_total}'
                )
                prediction_text.set_visible(True)
                print("[?] Quiz started - What will happen in 2.5 seconds?")
            else:
                # クイズキャンセル
                state.quiz_active = False
                state.paused = False
                prediction_text.set_visible(False)
                print("[?] Quiz cancelled")
        
        elif event.key in ['1', '2', '3'] and state.quiz_active:
            choice = int(event.key)
            correct = simulator.answer_quiz(choice)
            
            answer_names = {1: 'Collision', 2: 'Escape', 3: 'Stable'}
            your_answer = answer_names[choice]
            correct_answer = answer_names[state.quiz_answer]
            
            if correct:
                result_text = (
                    '[O] CORRECT!\n'
                    '-----------------\n'
                    f'You said: {your_answer}\n'
                    f'Answer:   {correct_answer}\n\n'
                    f'Score: {state.quiz_correct}/{state.quiz_total}\n\n'
                    'Press [ENTER] to watch!'
                )
                print(f"[O] Correct! {your_answer}")
            else:
                result_text = (
                    '[X] WRONG!\n'
                    '-----------------\n'
                    f'You said: {your_answer}\n'
                    f'Answer:   {correct_answer}\n\n'
                    f'Score: {state.quiz_correct}/{state.quiz_total}\n\n'
                    'Press [ENTER] to watch!'
                )
                print(f"[X] Wrong! Correct was {correct_answer}")
            
            prediction_text.set_text(result_text)
        
        elif event.key == 'enter' and state.quiz_active:
            state.paused = False
            state.quiz_active = False
            prediction_text.set_text('[>] Running...\nWatch what happens!')
        
        elif event.key == 'g':
            simulator.toggle_ghost_mode()
            ghost_label.set_visible(state.ghost_mode)
        
        elif event.key == 'q':
            print("[Q] Exiting...")
            plt.close()
        
        elif event.key == 'm':
            # 周期解モードのトグル/次の解へ
            simulator.toggle_periodic_mode()
            create_plot_objects(state.n_bodies)
            update_periodic_display()
            
            # ゴーストもリセット
            if state.ghost_mode:
                simulator.toggle_ghost_mode()  # OFF
                simulator.toggle_ghost_mode()  # ON（新しい位置で）
            
            if state.show_editor:
                update_editor_panel()
        
        elif event.key == 'b':
            # 巻き戻し
            if simulator.rewind():
                create_plot_objects(state.n_bodies)
                if state.ghost_mode:
                    simulator.toggle_ghost_mode()  # OFF
                    simulator.toggle_ghost_mode()  # ON（新しい位置で）
        
        elif event.key == 's':
            # エクスポート
            filepath = simulator.export_json()
            prediction_text.set_text(f'[S] Saved to:\\n{filepath}')
            prediction_text.set_visible(True)
        
        elif event.key == 'l':
            # インポート（ファイル選択ダイアログ）
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                title='Select orbit file',
                filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
            )
            root.destroy()
            if filepath:
                if simulator.import_json(filepath):
                    create_plot_objects(state.n_bodies)
                    prediction_text.set_text(f'[L] Loaded from:\\n{filepath}')
                    prediction_text.set_visible(True)
                    if state.ghost_mode:
                        simulator.toggle_ghost_mode()  # OFF
                        simulator.toggle_ghost_mode()  # ON
        
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
            # trailsはネストリストなのでflatten
            flat_trails = [seg for segs in trails for seg in segs]
            return bodies + flat_trails + velocity_arrows + force_arrows + glow_objects + [info_text]
        
        # シミュレーション進行
        simulator.step(config.steps_per_frame)
        
        # ゴーストのシミュレーション進行
        if state.ghost_mode:
            for _ in range(config.steps_per_frame):
                simulator.step_ghost(config.base_dt)
        
        # 境界チェック（周期解モードでは無効化 - 数値ドリフトの観察のため）
        if not state.periodic_mode and simulator.is_out_of_bounds():
            print(f"Generation {state.generation} ended at t={state.sim_time:.2f}")
            simulator.restart()
            create_plot_objects(state.n_bodies)
            
            # ゴーストモードをオフにする（新しい初期条件で再スタート）
            if state.ghost_mode:
                simulator.toggle_ghost_mode()
            
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
            
            # グロー（輝き）エフェクトの更新
            if i < len(glow_objects):
                glow_objects[i].set_data([x], [y])
                glow_objects[i].set_3d_properties([z])
                glow_objects[i].set_markersize(size * 2.5)  # 天体より大きく
            
            if state.trail_history[i]:
                trail_arr = np.array(state.trail_history[i])
                n_points = len(trail_arr)
                n_segments = len(trails[i])
                
                # 軌跡をセグメントに分割して描画
                for seg_idx, segment_line in enumerate(trails[i]):
                    # 各セグメントの開始・終了インデックス
                    start = int(seg_idx * n_points / n_segments)
                    end = int((seg_idx + 1) * n_points / n_segments)
                    if end <= start:
                        end = start + 1
                    if end > n_points:
                        end = n_points
                    
                    if start < n_points:
                        seg_data = trail_arr[start:end]
                        if len(seg_data) > 0:
                            segment_line.set_data(seg_data[:, 0], seg_data[:, 1])
                            segment_line.set_3d_properties(seg_data[:, 2])
                        else:
                            segment_line.set_data([], [])
                            segment_line.set_3d_properties([])
                    else:
                        segment_line.set_data([], [])
                        segment_line.set_3d_properties([])
            
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
            
            # ゴースト描画
            if state.ghost_mode and state.ghost_positions is not None:
                gx, gy, gz = state.ghost_positions[i]
                ghost_bodies[i].set_data([gx], [gy])
                ghost_bodies[i].set_3d_properties([gz])
                ghost_bodies[i].set_visible(True)
                
                if state.ghost_trail_history[i]:
                    ghost_arr = np.array(state.ghost_trail_history[i])
                    ghost_trails[i].set_data(ghost_arr[:, 0], ghost_arr[:, 1])
                    ghost_trails[i].set_3d_properties(ghost_arr[:, 2])
                    ghost_trails[i].set_visible(True)
            else:
                ghost_bodies[i].set_visible(False)
                ghost_trails[i].set_visible(False)
        
        if state.auto_rotate:
            state.azim += 0.3
            ax_3d.view_init(elev=20, azim=state.azim)
        
        # trailsはネストリストなのでflatten
        flat_trails = [seg for segs in trails for seg in segs]
        return bodies + flat_trails + velocity_arrows + force_arrows + ghost_bodies + ghost_trails + glow_objects + [info_text]
    
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
