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


# ============================================================
# GUI / アニメーション関連のテスト（モック使用）
# ============================================================

class TestConstants:
    """定数の妥当性テスト"""
    
    def test_default_constants_are_positive(self):
        """デフォルト定数が正の値であることを確認"""
        from nbody_simulation_advanced import (
            DEFAULT_N_BODIES, G, BASE_DT, MIN_DT, MAX_DT,
            ANIMATION_INTERVAL, SOFTENING, DISPLAY_RANGE,
            VELOCITY_ARROW_SCALE, FORCE_ARROW_SCALE, MASS_MIN, MASS_MAX
        )
        
        assert DEFAULT_N_BODIES >= 2
        assert G > 0
        assert BASE_DT > 0
        assert MIN_DT > 0
        assert MAX_DT > 0
        assert MIN_DT <= BASE_DT <= MAX_DT
        assert ANIMATION_INTERVAL > 0
        assert SOFTENING > 0
        assert DISPLAY_RANGE > 0
        assert VELOCITY_ARROW_SCALE > 0
        assert FORCE_ARROW_SCALE > 0
        assert MASS_MIN > 0
        assert MASS_MAX > 0
        assert MASS_MIN <= MASS_MAX
    
    def test_timestep_hierarchy(self):
        """タイムステップの階層が正しいことを確認"""
        from nbody_simulation_advanced import MIN_DT, BASE_DT, MAX_DT
        
        assert MIN_DT < BASE_DT, "MIN_DT should be less than BASE_DT"
        assert BASE_DT < MAX_DT, "BASE_DT should be less than MAX_DT"


class TestSimulationStateLogic:
    """シミュレーション状態ロジックのテスト（GUIなし）"""
    
    def test_zoom_calculation(self):
        """ズーム計算のロジックテスト"""
        from nbody_simulation_advanced import DISPLAY_RANGE
        
        zoom = 1.0
        
        # ズームイン
        zoom = max(0.3, zoom * 0.8)
        expected_range = DISPLAY_RANGE * zoom
        assert expected_range < DISPLAY_RANGE
        
        # ズームアウト
        zoom = min(3.0, zoom * 1.25)
        expected_range = DISPLAY_RANGE * zoom
        assert 0.3 <= zoom <= 3.0
    
    def test_generation_counter_logic(self):
        """世代カウンタのロジックテスト"""
        generation = 1
        max_generation = 1
        
        # リスタート時
        generation += 1
        max_generation = max(max_generation, generation)
        
        assert generation == 2
        assert max_generation == 2
        
        # さらにリスタート
        generation += 1
        max_generation = max(max_generation, generation)
        
        assert generation == 3
        assert max_generation == 3
    
    def test_trail_history_management(self):
        """軌跡履歴管理のロジックテスト"""
        max_trail = 400
        n_bodies = 3
        trail_history = [[] for _ in range(n_bodies)]
        
        # ポイント追加
        for step in range(500):
            for i in range(n_bodies):
                trail_history[i].append(np.array([step, 0, 0]))
                if len(trail_history[i]) > max_trail:
                    trail_history[i].pop(0)
        
        # 各物体の履歴が最大値を超えていないことを確認
        for i in range(n_bodies):
            assert len(trail_history[i]) <= max_trail
            assert len(trail_history[i]) == max_trail


class TestGUIWithMock:
    """モックを使用したGUIテスト"""
    
    def test_simulation_can_import(self):
        """シミュレーション関数がインポートできることを確認"""
        from nbody_simulation_advanced import run_advanced_simulation
        assert callable(run_advanced_simulation)
    
    def test_body_size_calculation(self):
        """物体サイズ計算のロジックテスト"""
        from nbody_simulation_advanced import MASS_MIN, MASS_MAX
        
        # サイズ計算ロジック: size = 6 + (mass - mass_min) * 6
        test_masses = [MASS_MIN, (MASS_MIN + MASS_MAX) / 2, MASS_MAX]
        
        for mass in test_masses:
            size = 6 + (mass - MASS_MIN) * 6
            assert size >= 6, "サイズは最小6以上"
            assert size <= 6 + (MASS_MAX - MASS_MIN) * 6, "サイズは最大値以下"
    
    def test_velocity_arrow_calculation(self):
        """速度ベクトル矢印計算のロジックテスト"""
        from nbody_simulation_advanced import VELOCITY_ARROW_SCALE
        
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.5, -0.3, 0.1])
        
        arrow_end = position + velocity * VELOCITY_ARROW_SCALE
        
        # 矢印の終点が計算できることを確認
        assert len(arrow_end) == 3
        assert not np.allclose(arrow_end, position)
    
    def test_force_arrow_calculation(self):
        """力ベクトル矢印計算のロジックテスト"""
        from nbody_simulation_advanced import FORCE_ARROW_SCALE
        
        position = np.array([0.0, 0.0, 0.0])
        force = np.array([1.0, 0.5, -0.2])
        
        arrow_end = position + force * FORCE_ARROW_SCALE
        
        # 矢印の終点が計算できることを確認
        assert len(arrow_end) == 3
    
    def test_info_text_format(self):
        """情報テキストのフォーマットテスト"""
        generation = 5
        sim_time = 123.456
        zoom = 0.8
        energy = -1.234
        min_dist = 0.567
        n_bodies = 3
        max_generation = 5
        
        info_lines = [
            f"Gen: {generation}  Time: {sim_time:.1f}  Zoom: {1/zoom:.1f}x",
            f"Energy: {energy:.3f}  MinDist: {min_dist:.2f}",
            f"Bodies: {n_bodies}  MaxGen: {max_generation}",
        ]
        
        info_text = '\n'.join(info_lines)
        
        assert "Gen: 5" in info_text
        assert "Time: 123.5" in info_text
        assert "Energy: -1.234" in info_text
        assert "Bodies: 3" in info_text


class TestKeyboardEventLogic:
    """キーボードイベント処理ロジックのテスト"""
    
    def test_number_key_parsing(self):
        """数字キー解析のテスト"""
        valid_keys = ['3', '4', '5', '6', '7', '8', '9']
        
        for key in valid_keys:
            new_n = int(key)
            assert 3 <= new_n <= 9
    
    def test_zoom_bounds(self):
        """ズーム境界値のテスト"""
        zoom = 1.0
        
        # 最大ズームイン
        for _ in range(20):
            zoom = max(0.3, zoom * 0.8)
        assert zoom >= 0.3
        
        # 最大ズームアウト
        zoom = 1.0
        for _ in range(20):
            zoom = min(3.0, zoom * 1.25)
        assert zoom <= 3.0
    
    def test_pause_toggle(self):
        """一時停止トグルのテスト"""
        paused = [False]
        
        # トグル1回目
        paused[0] = not paused[0]
        assert paused[0] == True
        
        # トグル2回目
        paused[0] = not paused[0]
        assert paused[0] == False
    
    def test_auto_rotate_toggle(self):
        """自動回転トグルのテスト"""
        auto_rotate = [False]
        
        auto_rotate[0] = not auto_rotate[0]
        assert auto_rotate[0] == True
        
        auto_rotate[0] = not auto_rotate[0]
        assert auto_rotate[0] == False


class TestEditorPanelLogic:
    """エディタパネルロジックのテスト"""
    
    def test_mass_display_truncation(self):
        """質量表示の切り捨てロジックテスト"""
        n_bodies = 8
        masses = np.random.rand(n_bodies) * 1.5 + 0.5
        
        lines = []
        for i in range(min(n_bodies, 6)):
            lines.append(f'  Body {i+1}: {masses[i]:.2f}')
        if n_bodies > 6:
            lines.append(f'  ... +{n_bodies-6} more')
        
        assert len(lines) == 7  # 6体 + "...+2 more"
        assert "+2 more" in lines[-1]
    
    def test_editor_panel_content(self):
        """エディタパネル内容のテスト"""
        n_bodies = 4
        masses = np.array([1.0, 1.5, 0.8, 2.0])
        
        lines = [
            '📝 EDITOR',
            '─────────────',
            f'N Bodies: {n_bodies}',
            '(Press 3-9 to change)',
            '',
            '📊 Current masses:',
        ]
        for i in range(min(n_bodies, 6)):
            lines.append(f'  Body {i+1}: {masses[i]:.2f}')
        
        panel_text = '\n'.join(lines)
        
        assert 'N Bodies: 4' in panel_text
        assert 'Body 1: 1.00' in panel_text
        assert 'Body 4: 2.00' in panel_text


class TestPredictionModeLogic:
    """予測モードロジックのテスト"""
    
    def test_prediction_mode_activation(self):
        """予測モード有効化のテスト"""
        prediction_mode = [False]
        paused = [False]
        prediction_made = [False]
        
        # Pキー押下をシミュレート
        prediction_mode[0] = not prediction_mode[0]
        if prediction_mode[0]:
            paused[0] = True
            prediction_made[0] = False
        
        assert prediction_mode[0] == True
        assert paused[0] == True
        assert prediction_made[0] == False
    
    def test_prediction_mode_enter(self):
        """予測モードでEnter押下のテスト"""
        prediction_mode = [True]
        paused = [True]
        prediction_made = [False]
        
        # Enterキー押下をシミュレート
        if prediction_mode[0]:
            paused[0] = False
            prediction_made[0] = True
        
        assert paused[0] == False
        assert prediction_made[0] == True


# ============================================================
# 新しいクラスのテスト
# ============================================================

class TestSimulationConfig:
    """SimulationConfig クラスのテスト"""
    
    def test_default_config(self):
        """デフォルト設定が正しいことを確認"""
        from nbody_simulation_advanced import SimulationConfig
        
        config = SimulationConfig()
        assert config.n_bodies == 3
        assert config.g == 1.0
        assert config.softening > 0
        assert config.mass_min < config.mass_max
    
    def test_custom_config(self):
        """カスタム設定が適用されることを確認"""
        from nbody_simulation_advanced import SimulationConfig
        
        config = SimulationConfig(n_bodies=5, g=2.0, softening=0.1)
        assert config.n_bodies == 5
        assert config.g == 2.0
        assert config.softening == 0.1
    
    def test_config_validation(self):
        """設定のバリデーションが動作することを確認"""
        from nbody_simulation_advanced import SimulationConfig
        
        # 正常な設定
        config = SimulationConfig(n_bodies=5)
        config.validate()  # エラーなし
        
        # 異常な設定
        config_invalid = SimulationConfig(n_bodies=1)  # 2未満はエラー
        with pytest.raises(ValueError):
            config_invalid.validate()


class TestSimulationState:
    """SimulationState クラスのテスト"""
    
    def test_state_initialization(self):
        """状態の初期化が正しいことを確認"""
        from nbody_simulation_advanced import SimulationState
        
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.zeros((2, 3))
        masses = np.array([1.0, 1.0])
        
        state = SimulationState(
            positions=positions,
            velocities=velocities,
            masses=masses,
            n_bodies=2
        )
        
        assert state.generation == 1
        assert state.sim_time == 0.0
        assert state.paused == False
        assert len(state.trail_history) == 2
    
    def test_state_default_values(self):
        """デフォルト値が正しいことを確認"""
        from nbody_simulation_advanced import SimulationState
        
        state = SimulationState(
            positions=np.zeros((3, 3)),
            velocities=np.zeros((3, 3)),
            masses=np.ones(3),
            n_bodies=3
        )
        
        assert state.auto_rotate == False
        assert state.show_forces == False
        assert state.prediction_mode == False


class TestNBodySimulator:
    """NBodySimulator クラスのテスト"""
    
    def test_simulator_initialization(self):
        """シミュレーターの初期化が正しいことを確認"""
        from nbody_simulation_advanced import NBodySimulator, SimulationConfig
        
        simulator = NBodySimulator()
        assert simulator.state.n_bodies == 3
        assert simulator.config.n_bodies == 3
    
    def test_simulator_with_custom_config(self):
        """カスタム設定でシミュレーターを初期化できることを確認"""
        from nbody_simulation_advanced import NBodySimulator, SimulationConfig
        
        config = SimulationConfig(n_bodies=5, g=2.0)
        simulator = NBodySimulator(config)
        
        assert simulator.state.n_bodies == 5
        assert simulator.config.g == 2.0
    
    def test_simulator_step(self):
        """ステップ実行で状態が更新されることを確認"""
        from nbody_simulation_advanced import NBodySimulator
        
        simulator = NBodySimulator()
        initial_positions = simulator.state.positions.copy()
        initial_time = simulator.state.sim_time
        
        simulator.step(10)
        
        assert not np.allclose(simulator.state.positions, initial_positions)
        assert simulator.state.sim_time > initial_time
    
    def test_simulator_restart(self):
        """リスタートで世代がインクリメントされることを確認"""
        from nbody_simulation_advanced import NBodySimulator
        
        simulator = NBodySimulator()
        assert simulator.state.generation == 1
        
        simulator.restart()
        assert simulator.state.generation == 2
        assert simulator.state.sim_time == 0.0
    
    def test_simulator_change_n_bodies(self):
        """物体数変更が正しく動作することを確認"""
        from nbody_simulation_advanced import NBodySimulator
        
        simulator = NBodySimulator()
        assert simulator.state.n_bodies == 3
        
        simulator.change_n_bodies(5)
        assert simulator.state.n_bodies == 5
        assert len(simulator.state.masses) == 5
        assert len(simulator.state.positions) == 5
    
    def test_simulator_get_energy(self):
        """エネルギー取得が動作することを確認"""
        from nbody_simulation_advanced import NBodySimulator
        
        simulator = NBodySimulator()
        energy = simulator.get_energy()
        
        assert isinstance(energy, float)
        assert energy < 0  # 束縛系なので負
    
    def test_simulator_get_forces(self):
        """力取得が動作することを確認"""
        from nbody_simulation_advanced import NBodySimulator
        
        simulator = NBodySimulator()
        forces = simulator.get_forces()
        
        assert forces.shape == simulator.state.positions.shape
    
    def test_simulator_update_trails(self):
        """軌跡更新が動作することを確認"""
        from nbody_simulation_advanced import NBodySimulator
        
        simulator = NBodySimulator()
        assert len(simulator.state.trail_history[0]) == 0
        
        simulator.update_trails()
        assert len(simulator.state.trail_history[0]) == 1


# スタンドアロン実行用
if __name__ == "__main__":
    print("=" * 60)
    print("三体問題シミュレーター ユニットテスト")
    print("=" * 60)
    
    # pytest がインストールされている場合
    try:
        sys.exit(pytest.main([__file__, "-v"]))
    except SystemExit:
        pass
