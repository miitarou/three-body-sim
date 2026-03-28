using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 最小限のUI管理
/// リスタートボタン、物体数スライダー、速度スライダー、情報表示
/// </summary>
public class UIManager : MonoBehaviour
{
    [Header("参照")]
    public SimulationManager simManager;

    [Header("UIオブジェクト")]
    public Text generationText;
    public Text bodyCountText;
    public Button restartButton;
    public Slider bodyCountSlider;

    // UIが見つからない場合のIMGUI フォールバック
    private bool useIMGUI = false;
    private int sliderBodyCount = 3;

    // 速度スライダー: 対数スケール（0〜1 → 実際の速度に変換）
    // 0.0 = 最遅 (0.0005), 0.5 = デフォルト (0.003), 1.0 = 最速 (0.05)
    private float speedSliderValue = 0.35f;
    private const float SPEED_MIN_LOG = -3.3f;   // log10(0.0005)
    private const float SPEED_MAX_LOG = -1.3f;   // log10(0.05)

    void Start()
    {
        // UIコンポーネントがアタッチされていない場合はIMGUIにフォールバック
        if (restartButton == null || bodyCountSlider == null)
        {
            useIMGUI = true;
            if (simManager != null)
            {
                sliderBodyCount = simManager.nBodies;
                // 現在の速度からスライダー位置を逆算
                speedSliderValue = SpeedToSlider(simManager.simTimeBudgetPerFrame);
            }
            return;
        }

        // ボタンイベント登録
        restartButton.onClick.AddListener(OnRestartClicked);

        // スライダー設定
        bodyCountSlider.minValue = 3;
        bodyCountSlider.maxValue = 7;
        bodyCountSlider.wholeNumbers = true;
        bodyCountSlider.value = simManager != null ? simManager.nBodies : 3;
        bodyCountSlider.onValueChanged.AddListener(OnBodyCountChanged);

        // 状態変更通知を購読
        if (simManager != null)
        {
            simManager.OnStateChanged += UpdateUI;
        }
    }

    void OnDestroy()
    {
        if (simManager != null)
        {
            simManager.OnStateChanged -= UpdateUI;
        }
    }

    /// <summary>
    /// UI表示の更新
    /// </summary>
    private void UpdateUI(int generation, float simTime)
    {
        if (generationText != null)
        {
            generationText.text = $"Generation: {generation}";
        }
        if (bodyCountText != null)
        {
            bodyCountText.text = $"Bodies: {(simManager != null ? simManager.nBodies : 3)}";
        }
    }

    private void OnRestartClicked()
    {
        if (simManager != null)
        {
            simManager.Restart();
        }
    }

    private void OnBodyCountChanged(float value)
    {
        if (simManager != null)
        {
            simManager.SetBodyCount((int)value);
        }
    }

    /// <summary>
    /// 対数スケール変換: スライダー値(0〜1) → シミュレーション速度
    /// </summary>
    private float SliderToSpeed(float t)
    {
        float logVal = Mathf.Lerp(SPEED_MIN_LOG, SPEED_MAX_LOG, t);
        return Mathf.Pow(10f, logVal);
    }

    /// <summary>
    /// 逆変換: シミュレーション速度 → スライダー値(0〜1)
    /// </summary>
    private float SpeedToSlider(float speed)
    {
        float logVal = Mathf.Log10(Mathf.Max(speed, 0.0001f));
        return Mathf.InverseLerp(SPEED_MIN_LOG, SPEED_MAX_LOG, logVal);
    }

    /// <summary>
    /// 速度の表示ラベル（x0.1〜x10 的な相対表記）
    /// </summary>
    private string SpeedLabel(float speed)
    {
        float defaultSpeed = 0.003f;
        float ratio = speed / defaultSpeed;
        if (ratio < 0.1f) return $"x{ratio:F2}";
        if (ratio < 1f) return $"x{ratio:F1}";
        return $"x{ratio:F1}";
    }

    /// <summary>
    /// IMGUI フォールバック
    /// </summary>
    void OnGUI()
    {
        if (!useIMGUI) return;
        if (simManager == null) return;

        // 半透明の暗い背景スタイル
        GUIStyle boxStyle = new GUIStyle(GUI.skin.box);
        boxStyle.normal.background = MakeTexture(1, 1, new Color(0, 0, 0, 0.6f));

        GUIStyle labelStyle = new GUIStyle(GUI.skin.label);
        labelStyle.fontSize = Mathf.Max(16, Screen.height / 40);
        labelStyle.normal.textColor = Color.white;
        labelStyle.fontStyle = FontStyle.Bold;

        GUIStyle buttonStyle = new GUIStyle(GUI.skin.button);
        buttonStyle.fontSize = Mathf.Max(14, Screen.height / 45);

        GUIStyle infoStyle = new GUIStyle(GUI.skin.label);
        infoStyle.fontSize = Mathf.Max(12, Screen.height / 55);
        infoStyle.normal.textColor = new Color(0.8f, 0.8f, 0.8f);

        float padding = 20f;
        float panelWidth = Screen.width * 0.35f;
        float panelHeight = Screen.height * 0.35f;

        // 左上パネル
        GUILayout.BeginArea(new Rect(padding, padding, panelWidth, panelHeight), boxStyle);
        GUILayout.Space(8);

        // タイトル
        GUILayout.Label("三体シミュレーター", labelStyle);
        GUILayout.Space(4);

        // 情報表示
        GUILayout.Label($"{simManager.nBodies}体  |  Generation #{GetGeneration()}", infoStyle);
        GUILayout.Space(8);

        // リスタートボタン
        if (GUILayout.Button("🔄 リスタート", buttonStyle, GUILayout.Height(Screen.height * 0.035f)))
        {
            simManager.Restart();
        }
        GUILayout.Space(6);

        // 物体数スライダー
        GUILayout.Label($"物体数: {sliderBodyCount}", infoStyle);
        float newCount = GUILayout.HorizontalSlider(sliderBodyCount, 3, 7);
        int rounded = Mathf.RoundToInt(newCount);
        if (rounded != sliderBodyCount)
        {
            sliderBodyCount = rounded;
            simManager.SetBodyCount(sliderBodyCount);
        }
        GUILayout.Space(6);

        // 速度スライダー（対数スケール）
        float currentSpeed = SliderToSpeed(speedSliderValue);
        GUILayout.Label($"速度: {SpeedLabel(currentSpeed)}", infoStyle);
        float newSpeedSlider = GUILayout.HorizontalSlider(speedSliderValue, 0f, 1f);
        if (Mathf.Abs(newSpeedSlider - speedSliderValue) > 0.001f)
        {
            speedSliderValue = newSpeedSlider;
            simManager.simTimeBudgetPerFrame = SliderToSpeed(speedSliderValue);
        }

        GUILayout.EndArea();
    }

    private int GetGeneration()
    {
        var field = typeof(SimulationManager).GetField("generation",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (field != null)
        {
            return (int)field.GetValue(simManager);
        }
        return 1;
    }

    /// <summary>
    /// 単色テクスチャを生成（GUI背景用）
    /// </summary>
    private Texture2D MakeTexture(int width, int height, Color color)
    {
        Color[] pix = new Color[width * height];
        for (int i = 0; i < pix.Length; i++) pix[i] = color;
        Texture2D result = new Texture2D(width, height);
        result.SetPixels(pix);
        result.Apply();
        return result;
    }
}
