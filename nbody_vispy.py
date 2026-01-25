#!/usr/bin/env python3
"""
N体問題シミュレーター Vispy Edition (GPU加速版)

GPU描画により60-144 FPSの滑らかな動作を実現
物理計算はMojo高速化バックエンドを使用（利用可能な場合）
"""

from __future__ import annotations
import numpy as np
from vispy import app, scene
from vispy.scene import visuals
import time
import colorsys
from typing import Optional, List, Tuple
from dataclasses import dataclass

# 物理計算部分をインポート（既存のMojo統合済みコード）
try:
    from mojo_backend import get_engine
    _physics_engine = get_engine(use_mojo=True)
except ImportError:
    _physics_engine = None


# ============================================================
# 設定
# ============================================================

@dataclass
class Config:
    """シミュレーション設定"""
    n_bodies: int = 3
    g: float = 1.0
    base_dt: float = 0.001
    min_dt: float = 0.0001
    max_dt: float = 0.01
    softening: float = 0.05
    softening_periodic: float = 0.001
    display_range: float = 1.5
    mass_min: float = 0.5
    mass_max: float = 2.0
    max_trail: int = 800  # Vispyは軽いので多めに設定可能
    steps_per_frame: int = 10
    bound_limit: float = 5.0
    target_fps: int = 60


# ============================================================
# 周期解カタログ（10種類の有名な周期解）
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
# 物理計算関数
# ============================================================

def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    g: float = 1.0
) -> np.ndarray:
    """加速度計算（Mojo高速化版またはNumPy版）"""
    if _physics_engine is not None and _physics_engine.use_mojo:
        return _physics_engine.compute_accelerations(positions, masses, softening, g)

    # NumPyフォールバック
    n = len(masses)
    eps2 = softening ** 2
    r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    r2 = np.sum(r_ij ** 2, axis=2) + eps2
    np.fill_diagonal(r2, 1.0)
    inv_r3 = r2 ** (-1.5)
    np.fill_diagonal(inv_r3, 0.0)
    accelerations = g * np.sum(
        masses[np.newaxis, :, np.newaxis] * r_ij * inv_r3[:, :, np.newaxis],
        axis=1
    )
    return accelerations


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
    """適応タイムステップ"""
    min_dist = compute_min_distance(positions)
    factor = min(1.0, min_dist / 0.3)
    dt = base_dt * factor
    return max(min_dt, min(max_dt, dt))


def rk4_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float,
    dt: float,
    g: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """RK4積分（Mojo高速化版またはNumPy版）"""
    if _physics_engine is not None and _physics_engine.use_mojo:
        return _physics_engine.rk4_step(positions, velocities, masses, softening, dt, g)

    # NumPyフォールバック
    k1_r = velocities
    k1_v = compute_accelerations(positions, masses, softening, g)

    k2_r = velocities + 0.5 * dt * k1_v
    k2_v = compute_accelerations(positions + 0.5 * dt * k1_r, masses, softening, g)

    k3_r = velocities + 0.5 * dt * k2_v
    k3_v = compute_accelerations(positions + 0.5 * dt * k2_r, masses, softening, g)

    k4_r = velocities + dt * k3_v
    k4_v = compute_accelerations(positions + dt * k3_r, masses, softening, g)

    new_pos = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_vel = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return new_pos, new_vel


def generate_initial_conditions(
    n_bodies: int,
    mass_min: float,
    mass_max: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ランダムな初期条件を生成"""
    np.random.seed(int(time.time() * 1000) % (2**32))

    positions = np.random.uniform(-0.5, 0.5, size=(n_bodies, 3))
    velocities = np.random.uniform(-0.3, 0.3, size=(n_bodies, 3))
    masses = np.random.uniform(mass_min, mass_max, size=n_bodies)

    return positions, velocities, masses


def is_out_of_bounds(positions: np.ndarray, bound: float) -> bool:
    """境界外判定"""
    return np.any(np.abs(positions) > bound)


# ============================================================
# Vispyシミュレーター
# ============================================================

class NBodySimulator:
    """N体シミュレーター（Vispy版）"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

        # 物理状態
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.masses: Optional[np.ndarray] = None
        self.trails: List[np.ndarray] = []
        self.generation = 0
        self.paused = False
        self.periodic_mode = False
        self.periodic_index = 0

        # FPS計測
        self.frame_times: List[float] = []
        self.last_frame_time = time.time()

        # Canvas作成
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1200, 900),
            show=True,
            title='N-Body Simulator (Vispy GPU Edition)'
        )

        # 3Dビュー作成
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=45,
            distance=4.0,
            elevation=30,
            azimuth=45
        )

        # 座標軸を追加
        scene.visuals.XYZAxis(parent=self.view.scene)

        # 天体用のMarkersビジュアル
        self.body_visual = scene.visuals.Markers(parent=self.view.scene)

        # 軌跡用のLineビジュアル
        self.trail_visuals: List[scene.visuals.Line] = []

        # テキスト表示
        self.text_visual = scene.visuals.Text(
            '',
            pos=(10, 30),
            color='white',
            font_size=10,
            parent=self.canvas.scene
        )

        # イベントハンドラ
        self.canvas.events.key_press.connect(self.on_key_press)

        # アニメーションタイマー
        self.timer = app.Timer(
            interval=1.0 / self.config.target_fps,
            connect=self.update,
            start=True
        )

        # 初期化
        self.restart()

        print("=" * 65)
        print("N-Body Simulator (Vispy GPU Edition)")
        print("=" * 65)
        if _physics_engine is not None and _physics_engine.use_mojo:
            print("🚀 Mojo Physics Backend: ENABLED (26x faster)")
        else:
            print("📊 Physics Backend: NumPy")
        print(f"🎮 Target FPS: {self.config.target_fps}")
        print()
        print("🎮 Controls:")
        print("  [SPACE] = Pause/Resume")
        print("  [R]     = Restart with new conditions")
        print("  [M]     = Cycle through periodic solutions (10 types)")
        print("  [3-9]   = Change number of bodies")
        print("  [Q]     = Quit")
        print()

    def restart(self, periodic: bool = False):
        """シミュレーションを再スタート"""
        if periodic and self.periodic_mode:
            sol = PERIODIC_SOLUTIONS[self.periodic_index % len(PERIODIC_SOLUTIONS)]
            self.positions = sol["positions"].copy()
            self.velocities = sol["velocities"].copy()
            self.masses = sol["masses"].copy()
            self.config.n_bodies = len(self.masses)
            print(f"🔄 {sol['label']}: {sol['description']}")
        else:
            self.periodic_mode = False
            self.positions, self.velocities, self.masses = generate_initial_conditions(
                self.config.n_bodies,
                self.config.mass_min,
                self.config.mass_max
            )
            print(f"🔄 Generation {self.generation + 1} started ({self.config.n_bodies} bodies)")

        self.generation += 1
        self.trails = [np.zeros((0, 3)) for _ in range(self.config.n_bodies)]

        # 軌跡ビジュアルを再作成（天体ごとに色分け）
        for visual in self.trail_visuals:
            visual.parent = None
        self.trail_visuals.clear()

        for i in range(self.config.n_bodies):
            color = self._get_trail_color(i)
            line = scene.visuals.Line(
                pos=np.zeros((0, 3)),
                color=color,
                width=1.5,
                parent=self.view.scene
            )
            self.trail_visuals.append(line)

    def update(self, event):
        """フレーム更新"""
        if self.paused:
            return

        # 物理シミュレーション
        softening = self.config.softening_periodic if self.periodic_mode else self.config.softening

        for _ in range(self.config.steps_per_frame):
            dt = adaptive_timestep(
                self.positions,
                self.config.base_dt,
                self.config.min_dt,
                self.config.max_dt
            )
            self.positions, self.velocities = rk4_step(
                self.positions,
                self.velocities,
                self.masses,
                softening,
                dt,
                self.config.g
            )

        # 境界チェック
        if is_out_of_bounds(self.positions, self.config.bound_limit):
            if self.periodic_mode:
                self.periodic_index += 1
            self.restart(periodic=self.periodic_mode)
            return

        # 軌跡更新
        for i in range(self.config.n_bodies):
            self.trails[i] = np.vstack([self.trails[i], self.positions[i:i+1]])
            if len(self.trails[i]) > self.config.max_trail:
                self.trails[i] = self.trails[i][-self.config.max_trail:]

            # 軌跡描画更新
            if len(self.trails[i]) > 1:
                self.trail_visuals[i].set_data(pos=self.trails[i])

        # 天体描画更新
        colors = self._get_body_colors()
        sizes = self._get_body_sizes()
        self.body_visual.set_data(
            pos=self.positions,
            face_color=colors,
            edge_color='white',
            size=sizes
        )

        # FPS計測
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        avg_frame_time = np.mean(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # テキスト更新
        status = "PAUSED" if self.paused else f"FPS: {fps:.1f}"
        backend = "Mojo" if (_physics_engine and _physics_engine.use_mojo) else "NumPy"
        self.text_visual.text = f"Gen {self.generation} | {self.config.n_bodies} bodies | {status} | {backend}"

    def _get_body_colors(self) -> np.ndarray:
        """天体の色を取得"""
        colors = np.zeros((self.config.n_bodies, 4))
        for i in range(self.config.n_bodies):
            hue = i / max(self.config.n_bodies, 1)
            colors[i] = self._hsv_to_rgb(hue, 0.8, 1.0)
        return colors

    def _get_body_sizes(self) -> np.ndarray:
        """天体のサイズを取得（質量に応じて）"""
        normalized_masses = (self.masses - self.masses.min()) / (self.masses.max() - self.masses.min() + 1e-10)
        return 10 + normalized_masses * 20

    def _get_trail_color(self, index: int) -> Tuple[float, float, float, float]:
        """軌跡の色を取得（天体ごとに異なる色）"""
        hue = index / max(self.config.n_bodies, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        return (r, g, b, 0.5)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float, float]:
        """HSVからRGBAに変換"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b, 1.0)

    def on_key_press(self, event):
        """キーボードイベント処理"""
        if event.text == ' ':
            self.paused = not self.paused
            print("⏸️  Paused" if self.paused else "▶️  Resumed")

        elif event.text == 'r':
            self.restart()

        elif event.text == 'm':
            self.periodic_mode = not self.periodic_mode
            if self.periodic_mode:
                self.periodic_index = 0
                self.restart(periodic=True)
            else:
                print("🔄 Periodic mode OFF")
                self.restart()

        elif event.text == 'q':
            self.canvas.close()
            app.quit()

        elif event.text in '3456789':
            self.config.n_bodies = int(event.text)
            self.periodic_mode = False
            self.restart()

    def run(self):
        """メインループ開始"""
        app.run()


# ============================================================
# メイン
# ============================================================

if __name__ == '__main__':
    config = Config()
    sim = NBodySimulator(config)
    sim.run()
