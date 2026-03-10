using UnityEngine;

/// <summary>
/// N体問題の物理計算エンジン
/// Python版 compute_accelerations_vectorized, rk4_step_adaptive 等を移植
/// </summary>
public static class NBodyPhysics
{
    // 万有引力定数
    public const float G = 1.0f;
    // ソフトニング長（衝突回避）
    public const float SOFTENING = 0.05f;
    // タイムステップ設定
    public const float BASE_DT = 0.001f;
    public const float MIN_DT = 0.0001f;
    public const float MAX_DT = 0.01f;

    /// <summary>
    /// 全天体間の重力加速度を計算
    /// Python版 compute_accelerations_vectorized の移植
    /// </summary>
    public static Vector3[] ComputeAccelerations(Vector3[] positions, float[] masses, float softening = SOFTENING)
    {
        int n = positions.Length;
        Vector3[] accelerations = new Vector3[n];
        float eps2 = softening * softening;

        for (int i = 0; i < n; i++)
        {
            Vector3 acc = Vector3.zero;
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;

                // r_ij = positions[j] - positions[i]
                Vector3 rij = positions[j] - positions[i];
                // |r_ij|² + ε²（ソフトニング付き）
                float r2 = rij.sqrMagnitude + eps2;
                // 1 / |r_ij|³
                float invR3 = 1.0f / (r2 * Mathf.Sqrt(r2));
                // a_i += G * m_j * r_ij / |r_ij|³
                acc += G * masses[j] * rij * invR3;
            }
            accelerations[i] = acc;
        }

        return accelerations;
    }

    /// <summary>
    /// 最小天体間距離を計算（適応タイムステップ用）
    /// </summary>
    public static float ComputeMinDistance(Vector3[] positions)
    {
        float minDist = float.MaxValue;
        int n = positions.Length;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                float dist = (positions[j] - positions[i]).magnitude;
                if (dist < minDist) minDist = dist;
            }
        }

        return minDist;
    }

    /// <summary>
    /// 適応タイムステップを計算
    /// 天体間距離が近いほど細かいステップにする
    /// </summary>
    public static float AdaptiveTimestep(Vector3[] positions)
    {
        float minDist = ComputeMinDistance(positions);
        float dt = BASE_DT * Mathf.Clamp(minDist, 0.1f, 10.0f);
        return Mathf.Clamp(dt, MIN_DT, MAX_DT);
    }

    /// <summary>
    /// RK4積分（4次ルンゲ=クッタ法）＋適応タイムステップ
    /// Python版 rk4_step_adaptive の移植
    /// </summary>
    public static float RK4StepAdaptive(
        ref Vector3[] positions,
        ref Vector3[] velocities,
        float[] masses,
        float softening = SOFTENING)
    {
        int n = positions.Length;
        float dt = AdaptiveTimestep(positions);

        // k1
        Vector3[] k1r = (Vector3[])velocities.Clone();
        Vector3[] k1v = ComputeAccelerations(positions, masses, softening);

        // k2 用の中間位置・速度
        Vector3[] tempPos = new Vector3[n];
        Vector3[] tempVel = new Vector3[n];

        for (int i = 0; i < n; i++)
        {
            tempPos[i] = positions[i] + 0.5f * dt * k1r[i];
            tempVel[i] = velocities[i] + 0.5f * dt * k1v[i];
        }
        Vector3[] k2r = tempVel;
        Vector3[] k2v = ComputeAccelerations(tempPos, masses, softening);

        // k3
        for (int i = 0; i < n; i++)
        {
            tempPos[i] = positions[i] + 0.5f * dt * k2r[i];
            tempVel[i] = velocities[i] + 0.5f * dt * k2v[i];
        }
        Vector3[] k3r = tempVel;
        Vector3[] k3v = ComputeAccelerations(tempPos, masses, softening);

        // k4
        for (int i = 0; i < n; i++)
        {
            tempPos[i] = positions[i] + dt * k3r[i];
            tempVel[i] = velocities[i] + dt * k3v[i];
        }
        Vector3[] k4r = tempVel;
        Vector3[] k4v = ComputeAccelerations(tempPos, masses, softening);

        // 最終更新: y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
        for (int i = 0; i < n; i++)
        {
            positions[i] += (dt / 6.0f) * (k1r[i] + 2f * k2r[i] + 2f * k3r[i] + k4r[i]);
            velocities[i] += (dt / 6.0f) * (k1v[i] + 2f * k2v[i] + 2f * k3v[i] + k4v[i]);
        }

        return dt;
    }

    /// <summary>
    /// 全エネルギー（運動エネルギー + ポテンシャルエネルギー）を計算
    /// エネルギー保存のチェック用
    /// </summary>
    public static float ComputeEnergy(Vector3[] positions, Vector3[] velocities, float[] masses, float softening = SOFTENING)
    {
        int n = positions.Length;
        float kinetic = 0f;
        float potential = 0f;

        for (int i = 0; i < n; i++)
        {
            // 運動エネルギー: (1/2) * m * v²
            kinetic += 0.5f * masses[i] * velocities[i].sqrMagnitude;

            // ポテンシャルエネルギー: -G * m_i * m_j / r_ij
            for (int j = i + 1; j < n; j++)
            {
                float r = Mathf.Sqrt((positions[j] - positions[i]).sqrMagnitude + softening * softening);
                potential -= G * masses[i] * masses[j] / r;
            }
        }

        return kinetic + potential;
    }

    /// <summary>
    /// 境界外判定（天体が飛び出したかチェック）
    /// </summary>
    public static bool IsOutOfBounds(Vector3[] positions, float bound = 10.0f)
    {
        for (int i = 0; i < positions.Length; i++)
        {
            if (positions[i].magnitude > bound) return true;
        }
        return false;
    }
}
