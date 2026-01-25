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
    force_arrow_scale: float = 0.15


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


def compute_forces(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    g: float = 1.0
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
        self.show_forces = False
        self.auto_rotate = False
        self.rotation_angle = 0.0

        # ゴーストモード（カオス可視化）
        self.ghost_mode = False
        self.ghost_positions: Optional[np.ndarray] = None
        self.ghost_velocities: Optional[np.ndarray] = None
        self.ghost_trails: List[np.ndarray] = []

        # 予測クイズ
        self.quiz_active = False
        self.quiz_answer = 0  # 0=未設定, 1=衝突, 2=逃亡, 3=安定
        self.quiz_correct = 0
        self.quiz_total = 0

        # エディタパネル
        self.show_editor = False

        # ズーム
        self.zoom = 1.0

        # 履歴バッファ（巻き戻し用）
        self.history_buffer: List[dict] = []
        self.max_history = 10

        # FPS計測
        self.frame_times: List[float] = []
        self.last_frame_time = time.time()

        # Canvas作成
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1200, 900),
            show=True,
            title='N-Body Simulator (Vispy GPU Edition)',
            bgcolor='#0a0a1a',  # 深い宇宙色（ほぼ黒だがわずかに青み）
            resizable=True
        )

        # macOSのネイティブフルスクリーンボタンでクラッシュするため無効化
        try:
            if hasattr(self.canvas.native, 'setCollectionBehavior'):
                # PyQt5/PySide6のウィンドウ
                from PySide6.QtCore import Qt
                # フルスクリーンボタンを非表示にしてクラッシュを防ぐ
                # 代わりにFキーで疑似フルスクリーンを使用
                pass
        except Exception:
            pass

        # 3Dビュー作成
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=45,
            distance=4.0,
            elevation=30,
            azimuth=45
        )

        # 背景の星（宇宙っぽい雰囲気）を追加
        self._create_background_stars()

        # グリッド線と軸を追加（単色、Matplotlib風）
        self._create_grid_and_axes()

        # 天体用のMarkersビジュアル
        self.body_visual = scene.visuals.Markers(parent=self.view.scene)

        # ゴースト天体用のMarkersビジュアル
        self.ghost_body_visual = scene.visuals.Markers(parent=self.view.scene)
        self.ghost_body_visual.visible = False

        # 軌跡用のLineビジュアル
        self.trail_visuals: List[scene.visuals.Line] = []

        # ゴースト軌跡用のLineビジュアル
        self.ghost_trail_visuals: List[scene.visuals.Line] = []

        # 力ベクトル用のLineビジュアル
        self.force_visuals: List[scene.visuals.Line] = []

        # テキスト表示
        self.text_visual = scene.visuals.Text(
            '',
            pos=(10, 30),
            color='white',
            font_size=10,
            parent=self.canvas.scene
        )

        # 予測クイズ用テキスト
        self.quiz_visual = scene.visuals.Text(
            '',
            pos=(self.canvas.size[0] // 2, 100),
            color=(1, 0.4, 0.4),
            font_size=14,
            anchor_x='center',
            parent=self.canvas.scene
        )
        self.quiz_visual.visible = False

        # エディタパネル用テキスト
        self.editor_visual = scene.visuals.Text(
            '',
            pos=(self.canvas.size[0] - 20, 200),
            color=(1, 0.7, 0),
            font_size=9,
            anchor_x='right',
            parent=self.canvas.scene
        )
        self.editor_visual.visible = False

        # イベントハンドラ
        self.canvas.events.key_press.connect(self.on_key_press)

        # macOSフルスクリーンボタンでのクラッシュを防ぐ
        fullscreen_disabled = False
        try:
            import platform
            if platform.system() == 'Darwin' and hasattr(self.canvas, 'native'):
                native = self.canvas.native
                if native is not None:
                    try:
                        # PySide6/PyQt6でフルスクリーンボタンを無効化
                        from PySide6.QtCore import Qt

                        # 現在のフラグを取得
                        current_flags = native.windowFlags()

                        # フルスクリーンボタンヒントを除外
                        # WindowType部分は保持し、ヒントのみ変更
                        new_flags = current_flags & ~Qt.WindowFullscreenButtonHint

                        # フラグを設定（ウィンドウを一時的に隠して再表示）
                        was_visible = native.isVisible()
                        native.setWindowFlags(new_flags)
                        if was_visible:
                            native.show()

                        fullscreen_disabled = True
                        print("[✓] Disabled macOS fullscreen button")
                    except Exception as e:
                        print(f"[!] Could not disable fullscreen button: {e}")
        except Exception as e:
            print(f"[!] Warning: {e}")

        if not fullscreen_disabled:
            print("[!] WARNING: macOS fullscreen button may cause crashes!")
            print("    Please resize window manually instead of using fullscreen.")

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
        print("  [A]     = Auto-rotate camera")
        print("  [F]     = Show force vectors")
        print("  [G]     = Ghost mode (chaos visualization)")
        print("  [E]     = Editor panel (show body info)")
        print("  [P]     = Prediction quiz (guess what happens next)")
        print("  [M]     = Cycle through periodic solutions (10 types)")
        print("  [B]     = Rewind to previous generation")
        print("  [S]     = Save current state to JSON")
        print("  [L]     = Load state from JSON")
        print("  [+/-]   = Zoom in/out")
        print("  [3-9]   = Change number of bodies")
        print("  [Q]     = Quit")
        print()
        print("⚠️  Note: Do not use the macOS fullscreen button (green button)")
        print("    It may cause crashes. Use window resize instead.")
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
        self.ghost_trails = [np.zeros((0, 3)) for _ in range(self.config.n_bodies)]

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

        # ゴースト軌跡ビジュアルを再作成
        for visual in self.ghost_trail_visuals:
            visual.parent = None
        self.ghost_trail_visuals.clear()

        for i in range(self.config.n_bodies):
            color = self._get_trail_color(i)
            # ゴーストは半透明・点線風
            ghost_color = (color[0], color[1], color[2], 0.4)
            line = scene.visuals.Line(
                pos=np.zeros((0, 3)),
                color=ghost_color,
                width=1.0,
                parent=self.view.scene
            )
            line.visible = False
            self.ghost_trail_visuals.append(line)

        # 力矢印ビジュアルを再作成
        for visual in self.force_visuals:
            visual.parent = None
        self.force_visuals.clear()

        for _ in range(self.config.n_bodies):
            arrow = scene.visuals.Line(
                pos=np.zeros((0, 3)),
                color=(1.0, 0.3, 0.3, 0.9),
                width=3.0,
                parent=self.view.scene
            )
            arrow.visible = False
            self.force_visuals.append(arrow)

        # ゴーストモードがONなら初期化
        if self.ghost_mode:
            self._initialize_ghost()

        # 履歴に保存
        self.save_to_history()

    def save_to_history(self):
        """現在の初期条件を履歴に保存"""
        snapshot = {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'masses': self.masses.copy(),
            'n_bodies': self.config.n_bodies,
            'generation': self.generation,
            'periodic_mode': self.periodic_mode,
            'periodic_index': self.periodic_index if self.periodic_mode else 0
        }
        self.history_buffer.append(snapshot)

        # 履歴が上限を超えたら古いものを削除
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

    def rewind(self) -> bool:
        """直前のGenerationに巻き戻す"""
        if len(self.history_buffer) < 2:
            print("[B] No history to rewind to")
            return False

        # 現在の状態を削除して1つ前に戻る
        self.history_buffer.pop()
        snapshot = self.history_buffer[-1]

        # 状態を復元
        self.positions = snapshot['positions'].copy()
        self.velocities = snapshot['velocities'].copy()
        self.masses = snapshot['masses'].copy()
        self.config.n_bodies = snapshot['n_bodies']
        self.generation = snapshot['generation']
        self.periodic_mode = snapshot['periodic_mode']
        self.periodic_index = snapshot['periodic_index']

        # 軌跡とゴーストをリセット
        self.trails = [np.zeros((0, 3)) for _ in range(self.config.n_bodies)]
        self.ghost_trails = [np.zeros((0, 3)) for _ in range(self.config.n_bodies)]

        # ビジュアルを再作成
        self._recreate_visuals()

        print(f"[B] Rewound to Generation {self.generation}")
        return True

    def _recreate_visuals(self):
        """ビジュアルオブジェクトを再作成"""
        # 軌跡ビジュアル再作成
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

        # ゴースト軌跡ビジュアル再作成
        for visual in self.ghost_trail_visuals:
            visual.parent = None
        self.ghost_trail_visuals.clear()

        for i in range(self.config.n_bodies):
            color = self._get_trail_color(i)
            ghost_color = (color[0], color[1], color[2], 0.4)
            line = scene.visuals.Line(
                pos=np.zeros((0, 3)),
                color=ghost_color,
                width=1.0,
                parent=self.view.scene
            )
            line.visible = False
            self.ghost_trail_visuals.append(line)

        # 力矢印ビジュアル再作成
        for visual in self.force_visuals:
            visual.parent = None
        self.force_visuals.clear()

        for _ in range(self.config.n_bodies):
            arrow = scene.visuals.Line(
                pos=np.zeros((0, 3)),
                color=(1.0, 0.3, 0.3, 0.9),
                width=3.0,
                parent=self.view.scene
            )
            arrow.visible = False
            self.force_visuals.append(arrow)

        # ゴーストモードがONなら再初期化
        if self.ghost_mode:
            self._initialize_ghost()

    def export_json(self, filepath: Optional[str] = None) -> str:
        """初期条件をJSONファイルにエクスポート"""
        import json
        from datetime import datetime

        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'nbody_save_{timestamp}.json'

        data = {
            'positions': self.positions.tolist(),
            'velocities': self.velocities.tolist(),
            'masses': self.masses.tolist(),
            'n_bodies': self.config.n_bodies,
            'generation': self.generation,
            'exported_at': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[S] Saved to {filepath}")
        return filepath

    def import_json(self, filepath: str) -> bool:
        """JSONファイルから初期条件をインポート"""
        import json

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.positions = np.array(data['positions'])
            self.velocities = np.array(data['velocities'])
            self.masses = np.array(data['masses'])
            self.config.n_bodies = data['n_bodies']
            self.generation = data.get('generation', 1)

            # 軌跡とゴーストをリセット
            self.trails = [np.zeros((0, 3)) for _ in range(self.config.n_bodies)]
            self.ghost_trails = [np.zeros((0, 3)) for _ in range(self.config.n_bodies)]

            # ビジュアルを再作成
            self._recreate_visuals()

            # 履歴に保存
            self.save_to_history()

            print(f"[L] Loaded from {filepath}")
            return True

        except Exception as e:
            print(f"[L] Error loading file: {e}")
            return False

    def _create_background_stars(self):
        """背景の星を作成（宇宙っぽい雰囲気）"""
        # ランダムに星を配置（遠くに大きな球面上）
        n_stars = 200
        radius = 15.0  # シミュレーション空間より遠く

        # 球面上にランダム配置
        phi = np.random.uniform(0, 2 * np.pi, n_stars)
        theta = np.random.uniform(0, np.pi, n_stars)

        star_positions = np.zeros((n_stars, 3))
        star_positions[:, 0] = radius * np.sin(theta) * np.cos(phi)
        star_positions[:, 1] = radius * np.sin(theta) * np.sin(phi)
        star_positions[:, 2] = radius * np.cos(theta)

        # 星の明るさをランダムに
        star_brightness = np.random.uniform(0.3, 1.0, n_stars)
        star_colors = np.zeros((n_stars, 4))
        star_colors[:, 0] = star_brightness  # R
        star_colors[:, 1] = star_brightness  # G
        star_colors[:, 2] = star_brightness  # B
        star_colors[:, 3] = 0.8  # Alpha

        # サイズもランダムに
        star_sizes = np.random.uniform(1.5, 4.0, n_stars)

        self.background_stars = scene.visuals.Markers(
            pos=star_positions,
            face_color=star_colors,
            edge_color=None,
            size=star_sizes,
            parent=self.view.scene
        )

    def _create_grid_and_axes(self):
        """グリッド線と軸を作成（Matplotlib風）"""
        r = self.config.display_range

        # XY平面のグリッド（底面）- 太くして見やすく
        n_lines = 5
        step = 2 * r / n_lines
        for i in range(n_lines + 1):
            pos = -r + i * step
            # X方向の線
            scene.visuals.Line(
                pos=np.array([[pos, -r, -r], [pos, r, -r]]),
                color=(0.5, 0.5, 0.5, 0.4),
                width=1.5,
                parent=self.view.scene
            )
            # Y方向の線
            scene.visuals.Line(
                pos=np.array([[-r, pos, -r], [r, pos, -r]]),
                color=(0.5, 0.5, 0.5, 0.4),
                width=1.5,
                parent=self.view.scene
            )

        # YZ平面のグリッド（左側の面）
        for i in range(n_lines + 1):
            pos = -r + i * step
            # Y方向の線
            scene.visuals.Line(
                pos=np.array([[-r, pos, -r], [-r, pos, r]]),
                color=(0.5, 0.5, 0.5, 0.25),
                width=1.5,
                parent=self.view.scene
            )
            # Z方向の線
            scene.visuals.Line(
                pos=np.array([[-r, -r, pos], [-r, r, pos]]),
                color=(0.5, 0.5, 0.5, 0.25),
                width=1.5,
                parent=self.view.scene
            )

        # XZ平面のグリッド（後ろ側の面）
        for i in range(n_lines + 1):
            pos = -r + i * step
            # X方向の線
            scene.visuals.Line(
                pos=np.array([[pos, -r, -r], [pos, -r, r]]),
                color=(0.5, 0.5, 0.5, 0.25),
                width=1.5,
                parent=self.view.scene
            )
            # Z方向の線
            scene.visuals.Line(
                pos=np.array([[-r, -r, pos], [r, -r, pos]]),
                color=(0.5, 0.5, 0.5, 0.25),
                width=1.5,
                parent=self.view.scene
            )

        # 座標軸（X, Y, Z）- 明るい白で統一
        axis_color = (0.95, 0.95, 0.95, 1.0)

        # X軸
        scene.visuals.Line(
            pos=np.array([[-r, -r, -r], [r, -r, -r]]),
            color=axis_color,
            width=1.5,
            parent=self.view.scene
        )

        # Y軸
        scene.visuals.Line(
            pos=np.array([[-r, -r, -r], [-r, r, -r]]),
            color=axis_color,
            width=1.5,
            parent=self.view.scene
        )

        # Z軸
        scene.visuals.Line(
            pos=np.array([[-r, -r, -r], [-r, -r, r]]),
            color=axis_color,
            width=1.5,
            parent=self.view.scene
        )

        # 軸ラベル（X, Y, Z）- より明るく大きく
        scene.visuals.Text(
            'X',
            pos=(r + 0.25, -r, -r),
            color=(1, 1, 1),
            font_size=16,
            bold=True,
            parent=self.view.scene
        )
        scene.visuals.Text(
            'Y',
            pos=(-r, r + 0.25, -r),
            color=(1, 1, 1),
            font_size=16,
            bold=True,
            parent=self.view.scene
        )
        scene.visuals.Text(
            'Z',
            pos=(-r, -r, r + 0.25),
            color=(1, 1, 1),
            font_size=16,
            bold=True,
            parent=self.view.scene
        )

        # 目盛り数字（主要な点のみ）- より明るく
        ticks = [-r, 0, r]
        tick_labels = [f'{-self.config.display_range:.1f}', '0', f'{self.config.display_range:.1f}']

        for tick, label in zip(ticks, tick_labels):
            # X軸の目盛り
            scene.visuals.Text(
                label,
                pos=(tick, -r - 0.3, -r),
                color=(1.0, 1.0, 1.0),
                font_size=20,
                bold=True,
                parent=self.view.scene
            )
            # Y軸の目盛り
            scene.visuals.Text(
                label,
                pos=(-r - 0.3, tick, -r),
                color=(1.0, 1.0, 1.0),
                font_size=20,
                bold=True,
                parent=self.view.scene
            )
            # Z軸の目盛り
            scene.visuals.Text(
                label,
                pos=(-r - 0.3, -r, tick),
                color=(1.0, 1.0, 1.0),
                font_size=20,
                bold=True,
                parent=self.view.scene
            )

    def _initialize_ghost(self):
        """ゴーストを初期化（わずかにずらした初期条件）"""
        perturbation = 0.001
        self.ghost_positions = self.positions.copy() + \
            np.random.randn(*self.positions.shape) * perturbation
        self.ghost_velocities = self.velocities.copy()
        self.ghost_trails = [np.zeros((0, 3)) for _ in range(self.config.n_bodies)]

    def predict_future(self, seconds: float) -> int:
        """未来を予測（1=衝突, 2=逃亡, 3=安定）"""
        # 現在の状態をコピー
        sim_pos = self.positions.copy()
        sim_vel = self.velocities.copy()
        softening = self.config.softening_periodic if self.periodic_mode else self.config.softening

        initial_pos = sim_pos.copy()
        time = 0.0
        min_distance_found = float('inf')
        max_displacement = 0.0
        collision_threshold = 0.1
        escape_bound = 2.0

        while time < seconds:
            dt = adaptive_timestep(sim_pos, self.config.base_dt, self.config.min_dt, self.config.max_dt)
            sim_pos, sim_vel = rk4_step(sim_pos, sim_vel, self.masses, softening, dt, self.config.g)
            time += dt

            # 最小距離チェック
            min_dist = compute_min_distance(sim_pos)
            min_distance_found = min(min_distance_found, min_dist)

            # 最大変位チェック
            displacement = np.max(np.linalg.norm(sim_pos - initial_pos, axis=1))
            max_displacement = max(max_displacement, displacement)

        if max_displacement > escape_bound:
            return 2  # 逃亡
        elif min_distance_found < collision_threshold:
            return 1  # 衝突
        else:
            return 3  # 安定

    def update(self, event):
        """フレーム更新"""
        if self.paused or self.quiz_active:
            return

        # 物理シミュレーション（0.7倍速）
        softening = self.config.softening_periodic if self.periodic_mode else self.config.softening

        for _ in range(self.config.steps_per_frame):
            dt = adaptive_timestep(
                self.positions,
                self.config.base_dt,
                self.config.min_dt,
                self.config.max_dt
            )
            # シミュレーション速度を0.7倍に調整
            dt *= 0.7
            self.positions, self.velocities = rk4_step(
                self.positions,
                self.velocities,
                self.masses,
                softening,
                dt,
                self.config.g
            )

            # ゴーストのシミュレーション
            if self.ghost_mode and self.ghost_positions is not None:
                ghost_dt = adaptive_timestep(
                    self.ghost_positions,
                    self.config.base_dt,
                    self.config.min_dt,
                    self.config.max_dt
                )
                # シミュレーション速度を0.7倍に調整
                ghost_dt *= 0.7
                self.ghost_positions, self.ghost_velocities = rk4_step(
                    self.ghost_positions,
                    self.ghost_velocities,
                    self.masses,
                    softening,
                    ghost_dt,
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

        # ゴースト描画更新
        if self.ghost_mode and self.ghost_positions is not None:
            # ゴースト軌跡更新
            for i in range(self.config.n_bodies):
                self.ghost_trails[i] = np.vstack([self.ghost_trails[i], self.ghost_positions[i:i+1]])
                if len(self.ghost_trails[i]) > self.config.max_trail:
                    self.ghost_trails[i] = self.ghost_trails[i][-self.config.max_trail:]

                if len(self.ghost_trails[i]) > 1:
                    self.ghost_trail_visuals[i].set_data(pos=self.ghost_trails[i])
                    self.ghost_trail_visuals[i].visible = True

            # ゴースト天体描画
            ghost_colors = colors.copy()
            ghost_colors[:, 3] = 0.5  # 半透明
            self.ghost_body_visual.set_data(
                pos=self.ghost_positions,
                face_color=ghost_colors,
                edge_color=(1, 1, 1, 0.5),
                size=sizes * 0.8
            )
            self.ghost_body_visual.visible = True
        else:
            self.ghost_body_visual.visible = False
            for visual in self.ghost_trail_visuals:
                visual.visible = False

        # 力ベクトル更新
        if self.show_forces:
            softening = self.config.softening_periodic if self.periodic_mode else self.config.softening
            forces = compute_forces(self.positions, self.masses, softening, self.config.g)
            for i in range(self.config.n_bodies):
                start = self.positions[i]
                end = start + forces[i] * self.config.force_arrow_scale
                arrow_line = np.array([start, end])
                self.force_visuals[i].set_data(pos=arrow_line)
                self.force_visuals[i].visible = True
        else:
            for visual in self.force_visuals:
                visual.visible = False

        # 自動回転（0.7倍速）
        if self.auto_rotate and not self.paused:
            self.rotation_angle += 0.35
            self.view.camera.azimuth = self.rotation_angle

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

        # エディタパネル更新
        if self.show_editor:
            self._update_editor_panel()

    def _update_editor_panel(self):
        """エディタパネルの内容を更新"""
        lines = ['== EDITOR ==', '-------------']
        for i in range(self.config.n_bodies):
            lines.append(f'Body {i}:')
            lines.append(f'  m={self.masses[i]:.2f}')
            lines.append(f'  pos=({self.positions[i,0]:.2f},{self.positions[i,1]:.2f},{self.positions[i,2]:.2f})')
            lines.append(f'  vel=({self.velocities[i,0]:.2f},{self.velocities[i,1]:.2f},{self.velocities[i,2]:.2f})')
        self.editor_visual.text = '\n'.join(lines)

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
        try:
            self._handle_key_press(event)
        except Exception as e:
            print(f"[!] Error in key handler: {e}")
            import traceback
            traceback.print_exc()

    def _handle_key_press(self, event):
        """キーボードイベント処理の実装"""
        if event.text == ' ':
            self.paused = not self.paused
            print("⏸️  Paused" if self.paused else "▶️  Resumed")

        elif event.text == 'r':
            self.restart()

        elif event.text == 'a':
            self.auto_rotate = not self.auto_rotate
            print(f"🔄 Auto-rotate: {'ON' if self.auto_rotate else 'OFF'}")

        elif event.text == 'f':
            self.show_forces = not self.show_forces
            print(f"⚡ Force vectors: {'ON' if self.show_forces else 'OFF'}")

        elif event.text == 'g':
            self.ghost_mode = not self.ghost_mode
            if self.ghost_mode:
                self._initialize_ghost()
                print("[G] Ghost mode ON - Watch the chaos unfold!")
            else:
                self.ghost_positions = None
                self.ghost_velocities = None
                self.ghost_body_visual.visible = False
                for visual in self.ghost_trail_visuals:
                    visual.visible = False
                print("[G] Ghost mode OFF")

        elif event.text == 'm':
            self.periodic_mode = not self.periodic_mode
            if self.periodic_mode:
                self.periodic_index = 0
                self.restart(periodic=True)
            else:
                print("🔄 Periodic mode OFF")
                self.restart()

        elif event.text == 'p':
            if not self.quiz_active:
                # クイズ開始
                self.quiz_active = True
                self.quiz_answer = self.predict_future(2.5)
                quiz_text = (
                    '[?] PREDICTION QUIZ\n'
                    '-----------------\n'
                    'In 2.5 seconds...\n'
                    'What will happen?\n\n'
                    'Press a key:\n'
                    '  [1] Collision\n'
                    '  [2] Escape\n'
                    '  [3] Stable orbit\n\n'
                    f'Score: {self.quiz_correct}/{self.quiz_total}'
                )
                self.quiz_visual.text = quiz_text
                self.quiz_visual.visible = True
                print("[?] Quiz started - What will happen in 2.5 seconds?")
            else:
                # クイズキャンセル
                self.quiz_active = False
                self.quiz_visual.visible = False
                print("[?] Quiz cancelled")

        elif event.text in '123' and self.quiz_active:
            choice = int(event.text)
            self.quiz_total += 1
            correct = (choice == self.quiz_answer)
            if correct:
                self.quiz_correct += 1

            answer_names = {1: 'Collision', 2: 'Escape', 3: 'Stable'}
            your_answer = answer_names[choice]
            correct_answer = answer_names[self.quiz_answer]

            if correct:
                result_text = (
                    '[O] CORRECT!\n'
                    '-----------------\n'
                    f'You said: {your_answer}\n'
                    f'Answer:   {correct_answer}\n\n'
                    f'Score: {self.quiz_correct}/{self.quiz_total}\n\n'
                    'Press [ENTER] to watch!'
                )
                print(f"[O] Correct! {your_answer}")
            else:
                result_text = (
                    '[X] WRONG!\n'
                    '-----------------\n'
                    f'You said: {your_answer}\n'
                    f'Answer:   {correct_answer}\n\n'
                    f'Score: {self.quiz_correct}/{self.quiz_total}\n\n'
                    'Press [ENTER] to watch!'
                )
                print(f"[X] Wrong. It was {correct_answer}, not {your_answer}")

            self.quiz_visual.text = result_text

        elif event.key.name == 'Enter' and self.quiz_active:
            self.quiz_active = False
            self.quiz_visual.text = '[>] Running...\nWatch what happens!'
            # 3秒後に非表示にするタイマーを設定
            self.quiz_hide_timer = self.canvas.app.Timer(interval=3.0, connect=self._hide_quiz_text, iterations=1, start=True)
            print("[>] Running simulation...")

        elif event.text == 'e':
            self.show_editor = not self.show_editor
            self.editor_visual.visible = self.show_editor
            if self.show_editor:
                self._update_editor_panel()
            print(f"📝 Editor: {'OPEN' if self.show_editor else 'CLOSED'}")

        elif event.text in ['+', '=']:
            self.zoom = max(0.2, self.zoom - 0.1)
            self.view.camera.distance = 4.0 / self.zoom
            print(f"🔍 Zoom: {self.zoom:.1f}x")

        elif event.text == '-':
            self.zoom = min(3.0, self.zoom + 0.1)
            self.view.camera.distance = 4.0 / self.zoom
            print(f"🔍 Zoom: {self.zoom:.1f}x")

        elif event.text == 'b':
            # 巻き戻し
            if self.rewind():
                # ゴーストモードがONなら再初期化
                if self.ghost_mode:
                    self._initialize_ghost()

        elif event.text == 's':
            # 保存
            filepath = self.export_json()
            self.quiz_visual.text = f'[S] Saved to:\n{filepath}'
            self.quiz_visual.visible = True
            # 3秒後に非表示
            self.canvas.app.Timer(interval=3.0, connect=self._hide_quiz_text, iterations=1, start=True)

        elif event.text == 'l':
            # 読込（ファイル選択ダイアログ）
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                # ウィンドウを最前面に
                root.attributes('-topmost', True)
                root.update()

                filepath = filedialog.askopenfilename(
                    title='Load Initial Conditions',
                    filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
                )

                # tkinterウィンドウを完全にクリーンアップ
                root.attributes('-topmost', False)
                root.update()
                root.quit()
                root.destroy()

                if filepath:
                    if self.import_json(filepath):
                        self.quiz_visual.text = f'[L] Loaded from:\n{filepath}'
                        self.quiz_visual.visible = True
                        # 3秒後に非表示
                        self.canvas.app.Timer(interval=3.0, connect=self._hide_quiz_text, iterations=1, start=True)

                        # ゴーストモードがONなら再初期化
                        if self.ghost_mode:
                            self._initialize_ghost()
            except Exception as e:
                print(f"[L] Error loading file: {e}")

        elif event.text == 'q':
            # 安全に終了（canvasのcloseでアプリも終了する）
            try:
                self.canvas.close()
            except Exception as e:
                print(f"[Q] Warning during shutdown: {e}")
                import sys
                sys.exit(0)

        elif event.text in '3456789':
            self.config.n_bodies = int(event.text)
            self.periodic_mode = False
            self.restart()

    def _hide_quiz_text(self, event):
        """クイズテキストを非表示にする"""
        self.quiz_visual.visible = False

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
