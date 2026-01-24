#!/usr/bin/env python3
"""
Mojo バックエンド統合モジュール

MojoコンパイラとPythonの橋渡しを行う。
Mojoが利用可能な場合は高速計算を使用し、
利用不可能な場合はNumPyにフォールバック。
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


# Mojoバイナリのパス
MOJO_BINARY_PATH = Path(__file__).parent / "mojo_physics"


def is_mojo_available() -> bool:
    """Mojoバックエンドが利用可能かチェック"""
    return MOJO_BINARY_PATH.exists() and os.access(MOJO_BINARY_PATH, os.X_OK)


def get_backend_info() -> dict:
    """バックエンド情報を取得"""
    return {
        "mojo_available": is_mojo_available(),
        "mojo_path": str(MOJO_BINARY_PATH) if is_mojo_available() else None,
        "numpy_version": np.__version__,
    }


class MojoPhysicsEngine:
    """
    Mojo物理エンジンのPythonラッパー

    シミュレーションの計算部分をMojoに委譲することで
    大幅な高速化を実現。
    """

    def __init__(self, use_mojo: bool = True):
        """
        Args:
            use_mojo: Mojoバックエンドを使用するかどうか
                      Falseの場合はNumPyフォールバック
        """
        self.use_mojo = use_mojo and is_mojo_available()
        self.process = None
        
        if self.use_mojo:
            print("🚀 Mojo physics backend enabled (IPC Mode)")
            self._start_mojo_process()
        else:
            print("📊 Using NumPy backend")

    def _start_mojo_process(self):
        """Mojoプロセスを起動"""
        try:
            cmd = [str(MOJO_BINARY_PATH), "ipc"]
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            # Wait for READY signal
            ready = self.process.stdout.readline().strip()
            if ready != "READY":
                print(f"⚠️ Mojo process failed to start: {ready}")
                self.use_mojo = False
                self.process = None
        except Exception as e:
            print(f"⚠️ Failed to start Mojo process: {e}")
            self.use_mojo = False

    def __del__(self):
        """デストラクタ: プロセス終了"""
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.terminate()
            except:
                pass

    def compute_accelerations(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        softening: float,
        g: float = 1.0
    ) -> np.ndarray:
        """加速度を計算"""
        # 加速度計算だけのAPIはMojo側には用意していない（RK4全体を委譲するため）
        # 必要ならNumPy実装を使うか、APIを追加する
        return self._compute_accelerations_numpy(positions, masses, softening, g)

    def _compute_accelerations_numpy(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        softening: float,
        g: float
    ) -> np.ndarray:
        """NumPy版加速度計算"""
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

    def rk4_step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        softening: float,
        dt: float,
        g: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """RK4積分ステップ"""
        if self.use_mojo and self.process:
            try:
                return self._communicate_with_mojo(positions, velocities, masses, softening, dt, g)
            except Exception as e:
                print(f"⚠️ Mojo IPC error: {e}. Falling back to NumPy.")
                self.use_mojo = False
                # Fallthrough to NumPy
        
        # NumPy fallback
        k1_r = velocities
        k1_v = self.compute_accelerations(positions, masses, softening, g)

        k2_r = velocities + 0.5 * dt * k1_v
        k2_v = self.compute_accelerations(positions + 0.5 * dt * k1_r, masses, softening, g)

        k3_r = velocities + 0.5 * dt * k2_v
        k3_v = self.compute_accelerations(positions + 0.5 * dt * k2_r, masses, softening, g)

        k4_r = velocities + dt * k3_v
        k4_v = self.compute_accelerations(positions + dt * k3_r, masses, softening, g)

        new_pos = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        new_vel = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return new_pos, new_vel

    def _communicate_with_mojo(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        softening: float,
        dt: float,
        g: float,
        steps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mojoプロセスと通信"""
        if self.process.poll() is not None:
             # Process died, restart?
             self._start_mojo_process()
             if not self.use_mojo:
                 raise RuntimeError("Mojo process died")

        n = len(masses)
        
        # Protocol:
        # N DT SOFTENING G STEPS
        header = f"{n} {dt} {softening} {g} {steps}\n"
        self.process.stdin.write(header)
        
        # MASSES
        masses_str = " ".join(map(str, masses)) + "\n"
        self.process.stdin.write(masses_str)
        
        # POSITIONS: x y z x y z ...
        pos_flat = positions.flatten()
        pos_str = " ".join(map(str, pos_flat)) + "\n"
        self.process.stdin.write(pos_str)
        
        # VELOCITIES
        vel_flat = velocities.flatten()
        vel_str = " ".join(map(str, vel_flat)) + "\n"
        self.process.stdin.write(vel_str)
        
        self.process.stdin.flush()
        
        # Read response
        # POS Output
        new_pos_line = self.process.stdout.readline().strip()
        if not new_pos_line:
             raise RuntimeError("Empty response from Mojo (pos)")
        new_pos_vals = list(map(float, new_pos_line.split()))
        new_pos = np.array(new_pos_vals).reshape((n, 3))
        
        # VEL Output
        new_vel_line = self.process.stdout.readline().strip()
        if not new_vel_line:
             raise RuntimeError("Empty response from Mojo (vel)")
        new_vel_vals = list(map(float, new_vel_line.split()))
        new_vel = np.array(new_vel_vals).reshape((n, 3))
        
        return new_pos, new_vel

    def run_batch_steps(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        softening: float,
        dt: float,
        steps: int,
        g: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        複数ステップを一括実行
        """
        if self.use_mojo and self.process:
            try:
                new_pos, new_vel = self._communicate_with_mojo(
                    positions, velocities, masses, softening, dt, g, steps=steps
                )
                return new_pos, new_vel, dt * steps
            except Exception as e:
                print(f"⚠️ Mojo IPC error (batch): {e}. Falling back to NumPy.")
                self.use_mojo = False

        total_dt = 0.0

        for _ in range(steps):
            positions, velocities = self.rk4_step(
                positions, velocities, masses, softening, dt, g
            )
            total_dt += dt

        return positions, velocities, total_dt


# グローバルエンジンインスタンス
_engine: Optional[MojoPhysicsEngine] = None


def get_engine(use_mojo: bool = True) -> MojoPhysicsEngine:
    """物理エンジンのシングルトンインスタンスを取得"""
    global _engine
    if _engine is None:
        _engine = MojoPhysicsEngine(use_mojo=use_mojo)
    return _engine


def reset_engine():
    """エンジンをリセット"""
    global _engine
    _engine = None


# 便利関数（既存コードとの互換性のため）
def compute_accelerations_fast(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    g: float = 1.0
) -> np.ndarray:
    """高速加速度計算（Mojoバックエンド使用時）"""
    return get_engine().compute_accelerations(positions, masses, softening, g)


def rk4_step_fast(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float,
    dt: float,
    g: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """高速RK4ステップ"""
    return get_engine().rk4_step(positions, velocities, masses, softening, dt, g)


if __name__ == "__main__":
    print("Mojo Backend Info:")
    info = get_backend_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
