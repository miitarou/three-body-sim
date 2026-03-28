using UnityEngine;

/// <summary>
/// N体シミュレーションの管理クラス
/// 初期条件生成、シミュレーション進行、天体オブジェクトの管理を行う
/// </summary>
public class SimulationManager : MonoBehaviour
{
    [Header("シミュレーション設定")]
    [Range(3, 9)]
    public int nBodies = 3;
    public float massMin = 0.5f;
    public float massMax = 2.0f;
    public float simTimeBudgetPerFrame = 0.003f;  // 1フレームで進めるシミュレーション時間（UIスライダーで変更可能）
    public int maxStepsPerFrame = 500;            // 安全上限
    public float outOfBoundsDist = 10.0f;

    [Header("プレハブ")]
    public GameObject celestialBodyPrefab;

    // シミュレーション状態
    private Vector3[] positions;
    private Vector3[] velocities;
    private float[] masses;
    private int generation = 1;
    private float simTime = 0f;
    private bool paused = false;

    // 天体GameObjectの参照
    private GameObject[] bodyObjects;
    private CelestialBody[] celestialBodies;

    // UI通知用イベント
    public System.Action<int, float> OnStateChanged;

    // 天体の色パレット（暖色系グラデーション）
    private static readonly Color[] bodyColors = new Color[]
    {
        new Color(1.0f, 0.4f, 0.2f),    // オレンジレッド
        new Color(0.3f, 0.7f, 1.0f),    // スカイブルー
        new Color(1.0f, 0.85f, 0.2f),   // ゴールド
        new Color(0.6f, 0.3f, 1.0f),    // パープル
        new Color(0.2f, 1.0f, 0.6f),    // エメラルド
        new Color(1.0f, 0.5f, 0.7f),    // ピンク
        new Color(0.4f, 0.9f, 0.9f),    // シアン
        new Color(1.0f, 0.6f, 0.1f),    // アンバー
        new Color(0.8f, 0.4f, 0.8f),    // マゼンタ
    };

    void Start()
    {
        Initialize();
    }

    void Update()
    {
        if (paused) return;

        // 固定時間バジェット方式:
        // 毎フレーム同じ量のシミュレーション時間を進める
        // dtが小さい時（天体が近い）→ 多くのステップを踏む
        // dtが大きい時（天体が遠い）→ 少ないステップで済む
        // → 視覚的に常に滑らかな一定速度の動き
        float accumulated = 0f;
        int step = 0;

        while (accumulated < simTimeBudgetPerFrame && step < maxStepsPerFrame)
        {
            float dt = NBodyPhysics.RK4StepAdaptive(ref positions, ref velocities, masses);
            simTime += dt;
            accumulated += dt;
            step++;
        }

        // 天体の位置を更新
        UpdateBodyPositions();

        // 境界外チェック → 自動リスタート
        if (NBodyPhysics.IsOutOfBounds(positions, outOfBoundsDist))
        {
            Restart();
        }

        // 天体収束（衝突）チェック → 自動リスタート
        float minDist = NBodyPhysics.ComputeMinDistance(positions);
        if (minDist < 0.02f)
        {
            Restart();
        }

        // UI通知
        OnStateChanged?.Invoke(generation, simTime);
    }

    /// <summary>
    /// シミュレーション初期化
    /// </summary>
    public void Initialize()
    {
        // 既存の天体を削除
        ClearBodies();

        // 初期条件を生成（Python版 generate_initial_conditions の移植）
        GenerateInitialConditions();

        // 天体オブジェクトを生成
        CreateBodyObjects();

        generation = 1;
        simTime = 0f;

        OnStateChanged?.Invoke(generation, simTime);
    }

    /// <summary>
    /// 新しい初期条件でリスタート
    /// </summary>
    public void Restart()
    {
        generation++;

        // 既存の天体を削除
        ClearBodies();

        // 新しい初期条件を生成
        GenerateInitialConditions();

        // 天体を再生成
        CreateBodyObjects();

        simTime = 0f;

        OnStateChanged?.Invoke(generation, simTime);
    }

    /// <summary>
    /// 物体数を変更してリスタート
    /// </summary>
    public void SetBodyCount(int count)
    {
        nBodies = Mathf.Clamp(count, 3, 9);
        generation = 0;
        Restart();
    }

    /// <summary>
    /// 一時停止/再開
    /// </summary>
    public void TogglePause()
    {
        paused = !paused;
    }

    public bool IsPaused => paused;

    /// <summary>
    /// 初期条件を生成
    /// Python版: generate_initial_conditions の移植
    /// 重心を原点に、全運動量をゼロに調整する
    /// </summary>
    private void GenerateInitialConditions()
    {
        positions = new Vector3[nBodies];
        velocities = new Vector3[nBodies];
        masses = new float[nBodies];

        // 質量をランダム生成
        for (int i = 0; i < nBodies; i++)
        {
            masses[i] = Random.Range(massMin, massMax);
        }

        // 位置: ランダムに配置（±0.5範囲、クリップ±1.0）
        float totalMass = 0f;
        Vector3 centerOfMass = Vector3.zero;
        for (int i = 0; i < nBodies; i++)
        {
            positions[i] = new Vector3(
                Mathf.Clamp(RandomGaussian() * 0.5f, -1f, 1f),
                Mathf.Clamp(RandomGaussian() * 0.5f, -1f, 1f),
                Mathf.Clamp(RandomGaussian() * 0.5f, -1f, 1f)
            );
            totalMass += masses[i];
            centerOfMass += masses[i] * positions[i];
        }
        // 重心を原点に移動
        centerOfMass /= totalMass;
        for (int i = 0; i < nBodies; i++)
        {
            positions[i] -= centerOfMass;
        }

        // 速度: ランダム生成し、全運動量をゼロに調整
        Vector3 totalMomentum = Vector3.zero;
        for (int i = 0; i < nBodies; i++)
        {
            velocities[i] = new Vector3(
                RandomGaussian() * 0.4f,
                RandomGaussian() * 0.4f,
                RandomGaussian() * 0.4f
            );
            totalMomentum += masses[i] * velocities[i];
        }
        for (int i = 0; i < nBodies; i++)
        {
            velocities[i] -= totalMomentum / totalMass;
        }

        // エネルギー調整: 束縛状態にする（E < -0.3）
        float energy = NBodyPhysics.ComputeEnergy(positions, velocities, masses);
        int safetyCounter = 0;
        while (energy > -0.3f && safetyCounter < 100)
        {
            for (int i = 0; i < nBodies; i++)
            {
                velocities[i] *= 0.9f;
            }
            energy = NBodyPhysics.ComputeEnergy(positions, velocities, masses);
            safetyCounter++;
        }
    }

    /// <summary>
    /// 天体GameObjectを生成
    /// </summary>
    private void CreateBodyObjects()
    {
        bodyObjects = new GameObject[nBodies];
        celestialBodies = new CelestialBody[nBodies];

        for (int i = 0; i < nBodies; i++)
        {
            GameObject obj;
            if (celestialBodyPrefab != null)
            {
                obj = Instantiate(celestialBodyPrefab, positions[i], Quaternion.identity);
            }
            else
            {
                // プレハブがない場合はSphereを動的生成
                obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                obj.transform.position = positions[i];

                // Colliderは不要（物理エンジンは使わない）
                Destroy(obj.GetComponent<Collider>());

                // TrailRendererを追加
                var trail = obj.AddComponent<TrailRenderer>();
                trail.time = 5.0f;
                trail.startWidth = 0.05f;
                trail.endWidth = 0.01f;
                trail.material = new Material(Shader.Find("Sprites/Default"));

                // CelestialBodyを追加
                obj.AddComponent<CelestialBody>();
            }

            obj.name = $"Body_{i}";

            // 質量に応じたサイズ
            float scale = Mathf.Lerp(0.08f, 0.2f, (masses[i] - massMin) / (massMax - massMin));
            obj.transform.localScale = Vector3.one * scale;

            // CelestialBodyの初期化
            var body = obj.GetComponent<CelestialBody>();
            if (body == null) body = obj.AddComponent<CelestialBody>();

            Color color = bodyColors[i % bodyColors.Length];
            body.Initialize(color, masses[i]);

            bodyObjects[i] = obj;
            celestialBodies[i] = body;
        }
    }

    /// <summary>
    /// 天体の位置を更新
    /// </summary>
    private void UpdateBodyPositions()
    {
        if (bodyObjects == null) return;

        for (int i = 0; i < nBodies; i++)
        {
            if (bodyObjects[i] != null)
            {
                bodyObjects[i].transform.position = positions[i];
            }
        }
    }

    /// <summary>
    /// 天体を全削除
    /// </summary>
    private void ClearBodies()
    {
        if (bodyObjects != null)
        {
            for (int i = 0; i < bodyObjects.Length; i++)
            {
                if (bodyObjects[i] != null)
                {
                    Destroy(bodyObjects[i]);
                }
            }
        }
        bodyObjects = null;
        celestialBodies = null;
    }

    /// <summary>
    /// Box-Muller法によるガウス分布乱数の生成
    /// Python版の np.random.randn に相当
    /// </summary>
    private float RandomGaussian()
    {
        float u1 = Random.Range(0.0001f, 1f);
        float u2 = Random.Range(0.0001f, 1f);
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }
}
