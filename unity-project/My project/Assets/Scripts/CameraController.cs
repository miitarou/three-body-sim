using UnityEngine;

/// <summary>
/// カメラコントローラー
/// タッチ操作での3D視点回転＆ピンチズーム、自動回転
/// </summary>
public class CameraController : MonoBehaviour
{
    [Header("回転設定")]
    public float rotationSpeed = 0.3f;
    public float autoRotateSpeed = 15f;
    public bool autoRotate = true;

    [Header("ズーム設定")]
    public float zoomSpeed = 0.5f;
    public float minDistance = 1.5f;
    public float maxDistance = 15f;

    [Header("初期位置")]
    public float initialDistance = 5f;
    public float initialElevation = 30f;

    // 内部状態
    private float azimuth = 0f;
    private float elevation = 30f;
    private float distance;
    private Vector3 lookTarget = Vector3.zero;

    // タッチ入力
    private Vector2 lastTouchPos;
    private bool isDragging = false;

    void Start()
    {
        distance = initialDistance;
        elevation = initialElevation;
        UpdateCameraPosition();
    }

    void Update()
    {
        HandleInput();

        if (autoRotate && !isDragging)
        {
            azimuth += autoRotateSpeed * Time.deltaTime;
        }

        UpdateCameraPosition();
    }

    /// <summary>
    /// タッチ/マウス入力の処理
    /// </summary>
    private void HandleInput()
    {
        // --- タッチ入力（iOS） ---
        if (Input.touchCount == 1)
        {
            Touch touch = Input.GetTouch(0);

            if (touch.phase == TouchPhase.Began)
            {
                isDragging = true;
                lastTouchPos = touch.position;
                autoRotate = false;
            }
            else if (touch.phase == TouchPhase.Moved)
            {
                Vector2 delta = touch.position - lastTouchPos;
                azimuth += delta.x * rotationSpeed;
                elevation -= delta.y * rotationSpeed;
                elevation = Mathf.Clamp(elevation, -80f, 80f);
                lastTouchPos = touch.position;
            }
            else if (touch.phase == TouchPhase.Ended || touch.phase == TouchPhase.Canceled)
            {
                isDragging = false;
            }
        }
        else if (Input.touchCount == 2)
        {
            // ピンチズーム
            Touch t0 = Input.GetTouch(0);
            Touch t1 = Input.GetTouch(1);

            float prevDist = (t0.position - t0.deltaPosition - (t1.position - t1.deltaPosition)).magnitude;
            float currDist = (t0.position - t1.position).magnitude;

            float diff = currDist - prevDist;
            distance -= diff * zoomSpeed * 0.01f;
            distance = Mathf.Clamp(distance, minDistance, maxDistance);
        }
        else
        {
            isDragging = false;
        }

        // --- マウス入力（エディタ/デスクトップ） ---
#if UNITY_EDITOR || UNITY_STANDALONE
        if (Input.GetMouseButtonDown(0))
        {
            isDragging = true;
            lastTouchPos = Input.mousePosition;
            autoRotate = false;
        }
        else if (Input.GetMouseButton(0))
        {
            Vector2 delta = (Vector2)Input.mousePosition - lastTouchPos;
            azimuth += delta.x * rotationSpeed;
            elevation -= delta.y * rotationSpeed;
            elevation = Mathf.Clamp(elevation, -80f, 80f);
            lastTouchPos = Input.mousePosition;
        }
        else if (Input.GetMouseButtonUp(0))
        {
            isDragging = false;
        }

        // マウスホイールズーム
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.01f)
        {
            distance -= scroll * zoomSpeed * 5f;
            distance = Mathf.Clamp(distance, minDistance, maxDistance);
        }
#endif
    }

    /// <summary>
    /// 極座標からカメラ位置を更新
    /// </summary>
    private void UpdateCameraPosition()
    {
        float azimRad = azimuth * Mathf.Deg2Rad;
        float elevRad = elevation * Mathf.Deg2Rad;

        Vector3 offset = new Vector3(
            distance * Mathf.Cos(elevRad) * Mathf.Sin(azimRad),
            distance * Mathf.Sin(elevRad),
            distance * Mathf.Cos(elevRad) * Mathf.Cos(azimRad)
        );

        transform.position = lookTarget + offset;
        transform.LookAt(lookTarget);
    }

    /// <summary>
    /// 自動回転のオン/オフ切り替え
    /// </summary>
    public void ToggleAutoRotate()
    {
        autoRotate = !autoRotate;
    }
}
