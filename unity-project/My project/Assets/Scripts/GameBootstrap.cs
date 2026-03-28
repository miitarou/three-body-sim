using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// ゲーム開始時に自動でシーンを構築するブートストラップ
/// シーンに何もなくても、Playするだけで自動セットアップ
/// [RuntimeInitializeOnLoadMethod] で自動実行される
/// </summary>
public static class GameBootstrap
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void AutoSetupScene()
    {
        Debug.Log("🚀 三体シミュレーター: 自動セットアップ開始...");

        // SimulationManagerが既にあれば何もしない
        if (Object.FindAnyObjectByType<SimulationManager>() != null)
        {
            Debug.Log("✅ SimulationManager は既にセットアップ済みです");
            return;
        }

        // --- 1. カメラの設定 ---
        Camera mainCam = Camera.main;
        if (mainCam == null)
        {
            GameObject camObj = new GameObject("Main Camera");
            mainCam = camObj.AddComponent<Camera>();
            camObj.tag = "MainCamera";
            camObj.AddComponent<AudioListener>();
        }

        // カメラをPerspectiveに変更（2Dテンプレートで作られた場合Orthographic）
        mainCam.orthographic = false;
        mainCam.fieldOfView = 60f;
        mainCam.nearClipPlane = 0.1f;
        mainCam.farClipPlane = 100f;

        // 背景色を宇宙っぽい暗い色に
        mainCam.clearFlags = CameraClearFlags.SolidColor;
        mainCam.backgroundColor = new Color(0.02f, 0.02f, 0.05f);

        // CameraControllerを追加
        if (mainCam.GetComponent<CameraController>() == null)
        {
            mainCam.gameObject.AddComponent<CameraController>();
        }

        // URP Volume がある場合、Bloom を有効に
        SetupPostProcessing();

        // --- 2. ライトの設定 ---
        Light[] lights = Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
        bool hasDirectional = false;
        foreach (var l in lights)
        {
            if (l.type == LightType.Directional)
            {
                hasDirectional = true;
                break;
            }
        }
        if (!hasDirectional)
        {
            GameObject lightObj = new GameObject("Directional Light");
            Light light = lightObj.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = new Color(0.6f, 0.6f, 0.8f);
            light.intensity = 0.3f;
            lightObj.transform.rotation = Quaternion.Euler(50, -30, 0);
        }

        // 環境光を暗めに
        RenderSettings.ambientLight = new Color(0.1f, 0.1f, 0.15f);
        RenderSettings.ambientMode = AmbientMode.Flat;

        // --- 3. SimulationManager を作成 ---
        GameObject simObj = new GameObject("SimulationManager");
        SimulationManager simManager = simObj.AddComponent<SimulationManager>();

        // --- 4. UIManager を追加 ---
        UIManager uiManager = simObj.AddComponent<UIManager>();
        uiManager.simManager = simManager;

        Debug.Log("✅ 三体シミュレーター: セットアップ完了！天体が表示されるはずです。");
    }

    /// <summary>
    /// URPのポストプロセシング（Bloom等）を設定
    /// </summary>
    static void SetupPostProcessing()
    {
        // URP Volumeがあれば、Bloomを有効にしたい
        // ランタイムでのVolume操作は複雑なのでスキップ
        // （Editorで後から追加設定可能）
    }
}
