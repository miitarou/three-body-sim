"""
三体問題シミュレーターのユニットテスト
実行: python -m pytest test_nbody.py -v
または: python test_nbody.py
"""
import pytest
import numpy as np
import sys

# テスト対象モジュールのインポート
from nbody_simulation_advanced import (
    compute_accelerations_vectorized,
    compute_forces,
    compute_energy,
    generate_initial_conditions,
    rk4_step_adaptive,
    validate_parameters,
    SOFTENING, G, BASE_DT, MIN_DT, MAX_DT
)


class TestParameterValidation:
    """パラメータ検証のテスト"""
    
    def test_valid_parameters(self):
        """正常なパラメータでエラーが発生しないことを確認"""
        validate_parameters(3, mass_min=0.5, mass_max=2.0)
        validate_parameters(5)
        validate_parameters(10, softening=0.1)
    
    def test_invalid_n_bodies_too_small(self):
        """物体数が2未満でエラーが発生することを確認"""
        with pytest.raises(ValueError, match="物体数は2以上"):
            validate_parameters(1)
        with pytest.raises(ValueError, match="物体数は2以上"):
            validate_parameters(0)
    
    def test_invalid_n_bodies_too_large(self):
        """物体数が20を超えるとエラーが発生することを確認"""
        with pytest.raises(ValueError, match="物体数が多すぎます"):
            validate_parameters(25)
    
    def test_invalid_mass_range(self):
        """無効な質量範囲でエラーが発生することを確認"""
        with pytest.raises(ValueError, match="質量範囲は正の値"):
            validate_parameters(3, mass_min=-1.0, mass_max=2.0)
        with pytest.raises(ValueError, match="mass_min は mass_max 以下"):
            validate_parameters(3, mass_min=3.0, mass_max=1.0)
    
    def test_invalid_softening(self):
        """無効なソフトニングでエラーが発生することを確認"""
        with pytest.raises(ValueError, match="ソフトニングは正の値"):
            validate_parameters(3, softening=0)
        with pytest.raises(ValueError, match="ソフトニングは正の値"):
            validate_parameters(3, softening=-0.1)


class TestPhysicsCalculations:
    """物理計算の正確性テスト"""
    
    def test_two_body_acceleration_direction(self):
        """2体問題：加速度が正しい方向を向くことを確認"""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        masses = np.array([1.0, 1.0])
        
        acc = compute_accelerations_vectorized(positions, masses, SOFTENING)
        
        # 物体0は+x方向へ、物体1は-x方向へ加速されるはず
        assert acc[0, 0] > 0, "物体0はx正方向へ加速されるべき"
        assert acc[1, 0] < 0, "物体1はx負方向へ加速されるべき"
        # y, z成分はほぼ0
        assert np.abs(acc[0, 1]) < 1e-10
        assert np.abs(acc[0, 2]) < 1e-10
    
    def test_three_body_symmetric(self):
        """3体問題：対称配置での力の対称性を確認"""
        # 正三角形配置
        positions = np.array([
            [0.0, 1.0, 0.0],
            [np.sqrt(3)/2, -0.5, 0.0],
            [-np.sqrt(3)/2, -0.5, 0.0]
        ])
        masses = np.array([1.0, 1.0, 1.0])
        
        acc = compute_accelerations_vectorized(positions, masses, SOFTENING)
        
        # 加速度の大きさはすべて等しいはず
        magnitudes = np.linalg.norm(acc, axis=1)
        assert np.allclose(magnitudes[0], magnitudes[1], rtol=1e-5)
        assert np.allclose(magnitudes[1], magnitudes[2], rtol=1e-5)
    
    def test_forces_vs_accelerations(self):
        """力 = 質量 × 加速度 の関係を確認"""
        positions, velocities, masses = generate_initial_conditions(3, 0.5, 2.0)
        
        forces = compute_forces(positions, masses, SOFTENING)
        accelerations = compute_accelerations_vectorized(positions, masses, SOFTENING)
        
        # F = m * a
        for i in range(3):
            expected_acc = forces[i] / masses[i]
            assert np.allclose(accelerations[i], expected_acc, rtol=1e-5)
    
    def test_energy_conservation(self):
        """エネルギー保存則のテスト（短期間）"""
        np.random.seed(42)  # 再現性のため
        positions = np.array([
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0]
        ], dtype=float)
        velocities = np.array([
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.1, 0.0, 0.0]
        ], dtype=float)
        masses = np.array([1.0, 1.0, 1.0])
        
        initial_energy = compute_energy(positions, velocities, masses, SOFTENING)
        
        # 50ステップ進める
        for _ in range(50):
            positions, velocities, _ = rk4_step_adaptive(
                positions, velocities, masses, SOFTENING, BASE_DT, MIN_DT, MAX_DT
            )
        
        final_energy = compute_energy(positions, velocities, masses, SOFTENING)
        
        # エネルギー変化が1%以内であることを確認
        relative_change = abs((final_energy - initial_energy) / initial_energy)
        assert relative_change < 0.01, f"エネルギー変化が大きすぎます: {relative_change*100:.2f}%"


class TestInitialConditions:
    """初期条件生成のテスト"""
    
    def test_momentum_conservation(self):
        """初期運動量がゼロであることを確認"""
        positions, velocities, masses = generate_initial_conditions(5, 0.5, 2.0)
        
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        
        assert np.allclose(total_momentum, 0, atol=1e-10), \
            f"初期運動量がゼロではありません: {total_momentum}"
    
    def test_center_of_mass(self):
        """重心が原点にあることを確認"""
        positions, velocities, masses = generate_initial_conditions(4, 0.5, 2.0)
        
        center_of_mass = np.average(positions, axis=0, weights=masses)
        
        assert np.allclose(center_of_mass, 0, atol=1e-10), \
            f"重心が原点にありません: {center_of_mass}"
    
    def test_bound_system(self):
        """束縛系（負のエネルギー）であることを確認"""
        positions, velocities, masses = generate_initial_conditions(3, 0.5, 2.0)
        
        energy = compute_energy(positions, velocities, masses, SOFTENING)
        
        assert energy < 0, f"系が束縛されていません（E={energy} >= 0）"


class TestEdgeCases:
    """エッジケースのテスト"""
    
    def test_generate_with_invalid_n_bodies(self):
        """無効な物体数で初期条件生成がエラーになることを確認"""
        with pytest.raises(ValueError):
            generate_initial_conditions(0, 0.5, 2.0)
        with pytest.raises(ValueError):
            generate_initial_conditions(1, 0.5, 2.0)
    
    def test_generate_with_invalid_mass(self):
        """無効な質量範囲でエラーが発生することを確認"""
        with pytest.raises(ValueError):
            generate_initial_conditions(3, -1.0, 2.0)
        with pytest.raises(ValueError):
            generate_initial_conditions(3, 2.0, 1.0)  # min > max


class TestRK4Integration:
    """RK4積分のテスト"""
    
    def test_rk4_returns_correct_shape(self):
        """RK4が正しい形状の配列を返すことを確認"""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        velocities = np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0], [0.1, 0.0, 0.0]])
        masses = np.array([1.0, 1.0, 1.0])
        
        new_pos, new_vel, dt = rk4_step_adaptive(
            positions, velocities, masses, SOFTENING, BASE_DT, MIN_DT, MAX_DT
        )
        
        assert new_pos.shape == positions.shape
        assert new_vel.shape == velocities.shape
        assert isinstance(dt, float)
        assert dt > 0
    
    def test_rk4_changes_positions(self):
        """RK4によって位置が変化することを確認"""
        positions = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        velocities = np.array([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0], [0.1, 0.0, 0.0]])
        masses = np.array([1.0, 1.0, 1.0])
        
        new_pos, new_vel, _ = rk4_step_adaptive(
            positions, velocities, masses, SOFTENING, BASE_DT, MIN_DT, MAX_DT
        )
        
        # 位置が変化していることを確認
        assert not np.allclose(new_pos, positions)
    
    def test_rk4_velocity_changes_due_to_gravity(self):
        """重力により速度が変化することを確認"""
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        velocities = np.zeros((2, 3))  # 初期速度ゼロ
        masses = np.array([1.0, 1.0])
        
        new_pos, new_vel, _ = rk4_step_adaptive(
            positions, velocities, masses, SOFTENING, BASE_DT, MIN_DT, MAX_DT
        )
        
        # 速度が変化していることを確認（重力により引き合う）
        assert not np.allclose(new_vel, velocities)
        # 物体0は+x方向へ、物体1は-x方向へ加速
        assert new_vel[0, 0] > 0
        assert new_vel[1, 0] < 0


class TestAdaptiveTimestep:
    """適応タイムステップのテスト"""
    
    def test_timestep_decreases_when_close(self):
        """物体が近いとタイムステップが小さくなることを確認"""
        from nbody_simulation_advanced import adaptive_timestep
        
        # 遠い配置
        far_positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        dt_far = adaptive_timestep(far_positions, BASE_DT, MIN_DT, MAX_DT)
        
        # 近い配置
        close_positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        dt_close = adaptive_timestep(close_positions, BASE_DT, MIN_DT, MAX_DT)
        
        assert dt_close <= dt_far, "近い物体ではタイムステップが小さくなるべき"
    
    def test_timestep_within_bounds(self):
        """タイムステップが指定範囲内に収まることを確認"""
        from nbody_simulation_advanced import adaptive_timestep
        
        # 非常に近い配置
        very_close = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        dt = adaptive_timestep(very_close, BASE_DT, MIN_DT, MAX_DT)
        
        assert dt >= MIN_DT, f"タイムステップが最小値を下回っています: {dt}"
        assert dt <= MAX_DT, f"タイムステップが最大値を超えています: {dt}"


class TestBoundaryChecking:
    """境界チェックのテスト"""
    
    def test_in_bounds(self):
        """範囲内の物体が正しく判定されることを確認"""
        from nbody_simulation_advanced import is_out_of_bounds
        
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        assert not is_out_of_bounds(positions, 1.0)
    
    def test_out_of_bounds(self):
        """範囲外の物体が正しく判定されることを確認"""
        from nbody_simulation_advanced import is_out_of_bounds
        
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        assert is_out_of_bounds(positions, 1.0)
    
    def test_boundary_edge_case(self):
        """境界上の物体の判定"""
        from nbody_simulation_advanced import is_out_of_bounds
        
        positions = np.array([[1.0, 0.0, 0.0]])  # 境界上
        assert not is_out_of_bounds(positions, 1.0)  # 境界上は範囲内
        
        positions = np.array([[1.0001, 0.0, 0.0]])  # 境界をわずかに超える
        assert is_out_of_bounds(positions, 1.0)


class TestNumericalStability:
    """数値安定性のテスト"""
    
    def test_no_nan_or_inf(self):
        """計算結果にNaNやInfが含まれないことを確認"""
        positions, velocities, masses = generate_initial_conditions(5, 0.5, 2.0)
        
        # 100ステップ実行
        for _ in range(100):
            positions, velocities, _ = rk4_step_adaptive(
                positions, velocities, masses, SOFTENING, BASE_DT, MIN_DT, MAX_DT
            )
            
            assert not np.any(np.isnan(positions)), "NaNが検出されました"
            assert not np.any(np.isinf(positions)), "Infが検出されました"
            assert not np.any(np.isnan(velocities)), "NaNが検出されました"
            assert not np.any(np.isinf(velocities)), "Infが検出されました"
    
    def test_softening_prevents_divergence(self):
        """ソフトニングが発散を防ぐことを確認"""
        # 非常に近い2物体
        positions = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        masses = np.array([10.0, 10.0])  # 大きな質量
        
        # ソフトニングありで計算
        acc = compute_accelerations_vectorized(positions, masses, SOFTENING)
        
        assert not np.any(np.isnan(acc)), "ソフトニングがあってもNaNが発生"
        assert not np.any(np.isinf(acc)), "ソフトニングがあってもInfが発生"
        # 加速度が有限の値であることを確認
        assert np.all(np.abs(acc) < 1e6), "加速度が異常に大きい"


class TestMinDistance:
    """最小距離計算のテスト"""
    
    def test_min_distance_two_bodies(self):
        """2体間の最小距離が正しく計算されることを確認"""
        from nbody_simulation_advanced import compute_min_distance
        
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])  # 距離5
        min_dist = compute_min_distance(positions)
        
        assert np.isclose(min_dist, 5.0)
    
    def test_min_distance_three_bodies(self):
        """3体間の最小距離が正しく計算されることを確認"""
        from nbody_simulation_advanced import compute_min_distance
        
        positions = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # 物体0との距離: 10
            [0.0, 2.0, 0.0]   # 物体0との距離: 2 ← 最小
        ])
        min_dist = compute_min_distance(positions)
        
        assert np.isclose(min_dist, 2.0)
    
    # pytest がインストールされている場合
    try:
        sys.exit(pytest.main([__file__, "-v"]))
    except SystemExit:
        pass
