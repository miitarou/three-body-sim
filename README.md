# 三体問題シミュレーター / N-Body Problem Simulator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Tests](https://github.com/miitarou/three-body-sim/actions/workflows/test.yml/badge.svg)](https://github.com/miitarou/three-body-sim/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

N個の天体が万有引力で相互作用する様子をリアルタイムでシミュレーション・可視化するプログラムです。

![Demo](demo.gif)

## 🚀 機能

- **N体シミュレーション**: 3体〜9体まで自由に変更可能
- **3D可視化**: あらゆる角度から観察
- **自動リスタート**: 物体が範囲外に出たら新規スタート
- **教育モード**: 力ベクトル表示、予測モードなど

---

## 📦 インストール

```bash
git clone https://github.com/miitarou/three-body-sim.git
cd three-body-sim
python3 -m venv venv

# Linux/macOS:
source venv/bin/activate

# Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

# Windows (cmd):
# venv\Scripts\activate.bat

pip install numpy matplotlib
```

## ▶️ 実行

```bash
# Learning Edition（推奨）
python nbody_simulation_advanced.py

# 2D版（8の字解など）
python three_body_simulation.py
```

---

## 🎮 操作方法

### 基本操作

| キー | 機能 |
|------|------|
| **SPACE** | 一時停止/再開 |
| **R** | 新しい初期条件でリスタート |
| **A** | 自動回転のオン/オフ |
| **+/-** | ズームイン/アウト |
| **マウスホイール** | ズームイン/アウト |
| **ドラッグ** | 視点回転（自動回転オフ時） |
| **Q** | 終了 |

### 教育機能

| キー | 機能 | 学習効果 |
|------|------|---------|
| **F** | 力ベクトル表示 | 重力の方向と大きさを赤い矢印で可視化 |
| **E** | エディタパネル | 物体数や質量を確認 |
| **P** | 予測モード | 一時停止して「次に何が起きる？」を予測 |
| **M** | 周期解モード | 有名な周期軌道を順次表示 |
| **3〜9** | 物体数変更 | 3体から9体まで自由に変更 |

### 周期解カタログ（Mキー）

**M**キーを押すと、数学的に発見された有名な三体周期解を順番に体験できます。

| # | 名前 | 発見者 | 特徴 |
|---|------|--------|------|
| 1 | Figure-8 Classic | Chenciner-Montgomery (2000) | 3体が8の字を描く有名な解 |
| 2 | Figure-8 (I.2.A) | Šuvakov-Dmitrašinović (2013) | 8の字解のバリエーション |
| 3 | Butterfly I | Šuvakov-Dmitrašinović (2013) | 蝶のような対称的な軌道 |
| 4 | Lagrange Triangle | Lagrange (1772) | 正三角形を保ったまま回転 |
| 5 | Moth I | Šuvakov-Dmitrašinović (2013) | 蛾のような複雑な軌道 |
| 6 | Yin-Yang Ia | Šuvakov-Dmitrašinović (2013) | 陰陽のような対称軌道 |
| 7 | Yin-Yang Ib | Šuvakov-Dmitrašinović (2013) | Yin-Yangの別バリエーション |
| 8 | Yin-Yang II | Šuvakov-Dmitrašinović (2013) | より複雑なYin-Yang軌道 |

> **Note**: 周期解は理論上は永続しますが、数値計算の誤差が蓄積するため、時間が経つと軌道がずれることがあります。これは「カオス系の数値シミュレーションの限界」を体験できる教育的な現象です。

---

## 📚 教育的な使い方

### 1. まず眺める（オートプレイ体験）

起動すると自動的にシミュレーションが始まります。まずは星が踊る様子を楽しんでください。毎回異なる初期条件で新しい軌道が生成されます。

### 2. 力ベクトルを観察する

**F**キーを押すと、各物体にかかる重力が赤い矢印で表示されます。

- 矢印の**方向** = 重力が引っ張る方向
- 矢印の**長さ** = 重力の強さ
- 物体が近づくと矢印が長くなる様子を観察できます

### 3. 予測してみる

**P**キーで予測モードに入ります。

1. シミュレーションが一時停止
2. 「次に何が起きる？」を考える
3. **Enter**で再開して答え合わせ

これにより「カオス」の本質を体験できます。

### 4. 物体を増やす

**5**や**7**キーで物体数を増やすと、より複雑なカオス的挙動を観察できます。3体でも予測困難ですが、物体が増えるほど更に複雑になります。

---

## 🔬 物理学的背景

### 万有引力の法則

```
F = G × m₁ × m₂ / r²
```

| 記号 | 意味 |
|------|------|
| **F** | 2つの物体間に働く引力 [N] |
| **G** | 万有引力定数 (6.674×10⁻¹¹ N⋅m²/kg²) |
| **m₁, m₂** | 各物体の質量 [kg] |
| **r** | 2つの物体間の距離 [m] |

各物体は他のすべての物体から引力を受けます。

### 三体問題とカオス

三体問題は「一般解がない」ことで有名です。これは：

1. **初期値鋭敏性**: わずかな違いが長期的に全く異なる結果を生む
2. **長期予測不可能**: 短期は正確、長期は原理的に不可能
3. **決定論的**: ランダムではない（同じ条件なら同じ結果）

### 数値計算の工夫

| 手法 | 目的 |
|------|------|
| **RK4積分法** | 高精度な時間発展計算 |
| **適応タイムステップ** | 接近時は細かく、離れている時は粗く |
| **Plummerソフトニング** | 極端な接近時の数値発散を防止 |

---

## 🎯 学習ポイント

このシミュレーターで学べること：

| トピック | 学習内容 |
|---------|---------|
| **力と運動** | 力が物体の運動をどう変えるか |
| **エネルギー保存** | 運動E + ポテンシャルE = 一定 |
| **カオス理論** | 初期条件の微小な違いが結果を大きく変える |
| **N体問題** | 3体以上になると解析解が存在しない |
| **数値シミュレーション** | コンピュータで物理をどう再現するか |

---

## 📁 ファイル構成

```
three-body-sim/
├── nbody_simulation_advanced.py  # Learning Edition（推奨）
├── three_body_simulation.py      # 2D版（8の字解デモ）
├── demo.gif                      # デモ動画
└── README.md                     # このファイル
```

---

## ⚙️ カスタマイズ

コード内の定数を変更することで挙動を調整できます：

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `DEFAULT_N_BODIES` | 3 | 初期物体数 |
| `G` | 1.0 | 万有引力定数 |
| `SOFTENING` | 0.05 | 衝突回避のソフトニング長 |
| `MASS_MIN/MAX` | 0.5/2.0 | 質量のランダム範囲 |

---

## 📖 参考文献

- Chenciner & Montgomery (2000): "A remarkable periodic solution of the three-body problem"
- arXiv:math/0011268

---

## 📜 ライセンス

MIT License

---

## 🙏 謝辞

- 8の字解の発見者 Chenciner & Montgomery
- Matplotlib / NumPy 開発チーム
