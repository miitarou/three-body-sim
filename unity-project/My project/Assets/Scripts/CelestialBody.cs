using UnityEngine;

/// <summary>
/// 天体の見た目を管理するコンポーネント
/// 発光マテリアル、色の設定を担当
/// </summary>
public class CelestialBody : MonoBehaviour
{
    private Color bodyColor;
    private float mass;
    private Renderer bodyRenderer;
    private TrailRenderer trailRenderer;
    private Material bodyMaterial;

    /// <summary>
    /// 天体の初期化（色と質量を設定）
    /// </summary>
    public void Initialize(Color color, float mass)
    {
        this.bodyColor = color;
        this.mass = mass;

        bodyRenderer = GetComponent<Renderer>();
        trailRenderer = GetComponent<TrailRenderer>();

        SetupMaterial();
        SetupTrail();
    }

    /// <summary>
    /// 発光マテリアルの設定
    /// </summary>
    private void SetupMaterial()
    {
        if (bodyRenderer == null) return;

        // URP Litシェーダーを使用（見つからない場合はStandard）
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");

        bodyMaterial = new Material(shader);
        bodyMaterial.color = bodyColor;

        // 発光（Emission）を有効化
        bodyMaterial.EnableKeyword("_EMISSION");
        bodyMaterial.SetColor("_EmissionColor", bodyColor * 2.0f);

        bodyRenderer.material = bodyMaterial;
    }

    /// <summary>
    /// トレイル（軌道の軌跡）の設定
    /// </summary>
    private void SetupTrail()
    {
        if (trailRenderer == null) return;

        trailRenderer.time = 5.0f;
        trailRenderer.startWidth = 0.04f;
        trailRenderer.endWidth = 0.005f;

        // トレイル色: 同じ色でフェードアウト
        Gradient gradient = new Gradient();
        gradient.SetKeys(
            new GradientColorKey[]
            {
                new GradientColorKey(bodyColor, 0.0f),
                new GradientColorKey(bodyColor * 0.5f, 1.0f)
            },
            new GradientAlphaKey[]
            {
                new GradientAlphaKey(0.8f, 0.0f),
                new GradientAlphaKey(0.0f, 1.0f)
            }
        );
        trailRenderer.colorGradient = gradient;

        // トレイル用マテリアル
        Material trailMat = new Material(Shader.Find("Sprites/Default"));
        trailRenderer.material = trailMat;
    }

    void OnDestroy()
    {
        // マテリアルのクリーンアップ
        if (bodyMaterial != null) Destroy(bodyMaterial);
    }
}
