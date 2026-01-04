# 三体問題シミュレーター（tensor-andromeda）

## プロジェクト概要

三体問題（Three-Body Problem）のシミュレーターアプリケーション。3つの物体が万有引力で相互作用する様子を可視化する。

## 技術スタック

- **言語**: Python 3.8+
- **数値計算**: NumPy
- **可視化**: Matplotlib (FuncAnimation)
- **積分法**: 4次ルンゲ＝クッタ法（RK4）

## MVP範囲

1. ✅ 2D平面での三体シミュレーション
2. ✅ 8の字解（Figure-8 solution）の初期条件
3. ✅ リアルタイムアニメーション表示
4. ✅ 軌跡（Trail）の表示
5. ✅ エネルギー保存の検証

## 現在のステップ

- [x] 物理モデル実装（万有引力の法則）
- [x] RK4積分器の実装
- [x] 可視化機能の実装
- [ ] 動作確認テスト
- [ ] パラメータ調整機能追加（オプション）

## ファイル構成

```
tensor-andromeda/
├── antigravity.md          # このファイル（プロジェクト憲法）
├── three_body_simulation.py # メインシミュレーター
└── requirements.txt        # 依存パッケージ
```

## 実行方法

```bash
# 依存パッケージのインストール
pip install numpy matplotlib

# シミュレーション実行
python three_body_simulation.py
```

## 調整可能なパラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `DT` | 0.001 | タイムステップ（小さいほど精度向上） |
| `T_MAX` | 20.0 | シミュレーション総時間 |
| `TRAIL_LENGTH` | 1500 | 軌跡の長さ（ステップ数） |

## 参考文献

- Chenciner & Montgomery (2000): "A remarkable periodic solution of the three-body problem"
- 8の字解の初期条件: arXiv:math/0011268
