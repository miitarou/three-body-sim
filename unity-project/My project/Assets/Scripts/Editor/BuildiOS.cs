using UnityEngine;
using UnityEditor;
using UnityEditor.Build.Reporting;

/// <summary>
/// バッチモードからiOSビルドを実行するためのEditorスクリプト
/// コマンドライン: Unity -batchmode -quit -executeMethod BuildiOS.Build
/// </summary>
public class BuildiOS
{
    [MenuItem("Tools/Build iOS")]
    public static void Build()
    {
        // ビルド設定
        string buildPath = "../ios-build";

        // 出力ディレクトリを作成
        if (!System.IO.Directory.Exists(buildPath))
        {
            System.IO.Directory.CreateDirectory(buildPath);
        }

        // ビルドに含めるシーン
        string[] scenes = new string[] { "Assets/Scenes/SampleScene.unity" };

        // PlayerSettings（バッチモードでも念のため設定）
        PlayerSettings.iOS.targetOSVersionString = "15.0";
        PlayerSettings.iOS.appleEnableAutomaticSigning = true;

        // ビルド実行
        BuildPlayerOptions buildOptions = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = buildPath,
            target = BuildTarget.iOS,
            options = BuildOptions.None
        };

        Debug.Log("🔨 iOSビルド開始...");
        BuildReport report = BuildPipeline.BuildPlayer(buildOptions);
        BuildSummary summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"✅ iOSビルド成功！出力先: {buildPath}");
            Debug.Log($"   ビルド時間: {summary.totalTime}");
            Debug.Log($"   ファイルサイズ: {summary.totalSize / 1024 / 1024} MB");
        }
        else
        {
            Debug.LogError($"❌ iOSビルド失敗: {summary.result}");
            foreach (var step in report.steps)
            {
                foreach (var msg in step.messages)
                {
                    if (msg.type == LogType.Error || msg.type == LogType.Warning)
                    {
                        Debug.LogError($"  {msg.content}");
                    }
                }
            }
        }
    }
}
