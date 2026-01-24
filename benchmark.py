#!/usr/bin/env python3
"""
N体シミュレーション ベンチマーク比較

Python (NumPy) vs Mojo の速度比較
"""

import time
import numpy as np
import subprocess
import sys

# NumPy版の物理計算（nbody_simulation_advanced.pyから抽出）
G = 1.0


def compute_accelerations_numpy(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    g: float = G
) -> np.ndarray:
    """NumPyベクトル化版の加速度計算"""
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


def rk4_step_numpy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float,
    dt: float,
    g: float = G
) -> tuple:
    """NumPy版RK4積分"""
    k1_r = velocities
    k1_v = compute_accelerations_numpy(positions, masses, softening, g)

    k2_r = velocities + 0.5 * dt * k1_v
    k2_v = compute_accelerations_numpy(positions + 0.5 * dt * k1_r, masses, softening, g)

    k3_r = velocities + 0.5 * dt * k2_v
    k3_v = compute_accelerations_numpy(positions + 0.5 * dt * k2_r, masses, softening, g)

    k4_r = velocities + dt * k3_v
    k4_v = compute_accelerations_numpy(positions + dt * k3_r, masses, softening, g)

    new_pos = positions + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_vel = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return new_pos, new_vel


def compute_energy_numpy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float,
    g: float = G
) -> float:
    """エネルギー計算"""
    n = len(masses)
    eps2 = softening ** 2
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    pe = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r2 = np.sum((positions[j] - positions[i])**2)
            pe -= g * masses[i] * masses[j] / np.sqrt(r2 + eps2)
    return ke + pe


def benchmark_python(steps: int = 10000) -> tuple:
    """Python/NumPy版ベンチマーク"""
    # Figure-8 初期条件
    positions = np.array([
        [0.97000436, -0.24308753, 0.0],
        [-0.97000436, 0.24308753, 0.0],
        [0.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.466203685, 0.43236573, 0.0],
        [0.466203685, 0.43236573, 0.0],
        [-0.93240737, -0.86473146, 0.0]
    ])
    masses = np.array([1.0, 1.0, 1.0])

    softening = 0.001
    dt = 0.001

    start = time.perf_counter()
    for _ in range(steps):
        positions, velocities = rk4_step_numpy(positions, velocities, masses, softening, dt)
    elapsed = time.perf_counter() - start

    energy = compute_energy_numpy(positions, velocities, masses, softening)

    return elapsed, energy, positions[0]


def benchmark_mojo(steps: int = 10000) -> tuple:
    """Mojo版ベンチマーク（サブプロセス経由）"""
    start = time.perf_counter()
    result = subprocess.run(
        ['./mojo_physics'],
        capture_output=True,
        text=True,
        cwd='/Users/miitarou/three-body-sim'
    )
    elapsed = time.perf_counter() - start

    # 出力からエネルギーと位置を抽出
    lines = result.stdout.strip().split('\n')
    energy = None
    pos = None
    for line in lines:
        if 'Final energy:' in line:
            energy = float(line.split(':')[1].strip())
        if 'Final position[0]:' in line:
            parts = line.split(':')[1].strip().split()
            pos = [float(p) for p in parts]

    return elapsed, energy, pos


def main():
    print("=" * 60)
    print("N-Body Simulation Benchmark")
    print("Python (NumPy) vs Mojo")
    print("=" * 60)
    print()

    steps = 10000
    print(f"Running {steps} simulation steps...")
    print()

    # Python ベンチマーク
    print("[Python/NumPy]")
    py_time, py_energy, py_pos = benchmark_python(steps)
    print(f"  Time:     {py_time:.4f} sec")
    print(f"  Energy:   {py_energy:.6f}")
    print(f"  Pos[0]:   {py_pos}")
    print()

    # Mojo ベンチマーク
    print("[Mojo]")
    try:
        mojo_time, mojo_energy, mojo_pos = benchmark_mojo(steps)
        print(f"  Time:     {mojo_time:.4f} sec")
        print(f"  Energy:   {mojo_energy:.6f}")
        print(f"  Pos[0]:   {mojo_pos}")
        print()

        # 比較
        print("-" * 60)
        speedup = py_time / mojo_time
        print(f"Speedup: {speedup:.1f}x faster with Mojo")
        print(f"Energy diff: {abs(py_energy - mojo_energy):.2e}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  (Mojo binary not found or failed to run)")

    print("=" * 60)


if __name__ == "__main__":
    main()
