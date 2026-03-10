using UnityEngine;
using UnityEditor;

/// <summary>
/// Unityエディタメニューからシーンを自動構築するスクリプト
/// メニュー: Tools > Setup Three Body Scene
/// </summary>
public class SceneSetup : EditorWindow
{
    [MenuItem("Tools/Setup Three Body Scene")]
    public static void SetupScene()
    {
        // --- 1. カメラの設定 ---
        Camera mainCam = Camera.main;
        if (mainCam == null)
        {
            GameObject camObj = new GameObject("Main Camera");
            mainCam = camObj.AddComponent<Camera>();
            camObj.tag = "MainCamera";
        }

        // 背景色を宇宙っぽい暗い色に
        mainCam.clearFlags = CameraClearFlags.SolidColor;
        mainCam.backgroundColor = new Color(0.02f, 0.02f, 0.05f);

        // CameraControllerをアタッチ
        if (mainCam.GetComponent<CameraController>() == null)
        {
            mainCam.gameObject.AddComponent<CameraController>();
        }

        // --- 2. ディレクショナルライト ---
        Light[] lights = Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
        if (lights.Length == 0)
        {
            GameObject lightObj = new GameObject("Directional Light");
            Light light = lightObj.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = new Color(0.8f, 0.8f, 1.0f);
            light.intensity = 0.5f;
            lightObj.transform.rotation = Quaternion.Euler(50, -30, 0);
        }

        // --- 3. SimulationManager ---
        GameObject simObj = GameObject.Find("SimulationManager");
        if (simObj == null)
        {
            simObj = new GameObject("SimulationManager");
        }
        SimulationManager simManager = simObj.GetComponent<SimulationManager>();
        if (simManager == null)
        {
            simManager = simObj.AddComponent<SimulationManager>();
        }

        // --- 4. UIManager ---
        UIManager uiManager = simObj.GetComponent<UIManager>();
        if (uiManager == null)
        {
            uiManager = simObj.AddComponent<UIManager>();
        }
        uiManager.simManager = simManager;

        // --- 5. 環境光の設定 ---
        RenderSettings.ambientLight = new Color(0.15f, 0.15f, 0.2f);

        Debug.Log("✅ Three Body Scene のセットアップが完了しました！Playボタンで実行してください。");
        EditorUtility.DisplayDialog(
            "セットアップ完了",
            "Three Body Scene のセットアップが完了しました！\n\nPlayボタンを押してシミュレーションを開始してください。",
            "OK"
        );
    }
}
