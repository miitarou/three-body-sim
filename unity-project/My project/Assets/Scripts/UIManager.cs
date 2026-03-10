using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 最小限のUI管理
/// リスタートボタン、物体数スライダー、情報表示
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

    void Start()
    {
        // UIコンポーネントがアタッチされていない場合はIMGUIにフォールバック
        if (restartButton == null || bodyCountSlider == null)
        {
            useIMGUI = true;
            if (simManager != null)
            {
                sliderBodyCount = simManager.nBodies;
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
    /// IMGUI フォールバック（Canvas UIが設定されていない場合に使用）
    /// シンプルなボタンとスライダーを画面に表示
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

        float padding = 20f;
        float panelWidth = Screen.width * 0.35f;
        float panelHeight = Screen.height * 0.25f;

        // 左上パネル
        GUILayout.BeginArea(new Rect(padding, padding, panelWidth, panelHeight), boxStyle);
        GUILayout.Space(8);

        // タイトル
        GUILayout.Label("三体シミュレーター", labelStyle);
        GUILayout.Space(4);

        // 情報表示
        GUIStyle infoStyle = new GUIStyle(GUI.skin.label);
        infoStyle.fontSize = Mathf.Max(12, Screen.height / 55);
        infoStyle.normal.textColor = new Color(0.8f, 0.8f, 0.8f);

        GUILayout.Label($"Generation: {simManager.nBodies}体  |  #{GetGeneration()}", infoStyle);
        GUILayout.Space(8);

        // リスタートボタン
        if (GUILayout.Button("🔄 リスタート", buttonStyle, GUILayout.Height(Screen.height * 0.04f)))
        {
            simManager.Restart();
        }
        GUILayout.Space(4);

        // 物体数スライダー
        GUILayout.Label($"物体数: {sliderBodyCount}", infoStyle);
        float newCount = GUILayout.HorizontalSlider(sliderBodyCount, 3, 7);
        int rounded = Mathf.RoundToInt(newCount);
        if (rounded != sliderBodyCount)
        {
            sliderBodyCount = rounded;
            simManager.SetBodyCount(sliderBodyCount);
        }

        GUILayout.EndArea();
    }

    private int GetGeneration()
    {
        // SimulationManagerのgeneration フィールドにアクセス（パブリックにする必要がある場合は調整）
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
