# 三体問題シミュレーター / Three-Body Problem Simulator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

3つの天体が万有引力で相互作用する様子をリアルタイムでシミュレーション・可視化するプログラムです。

![Three-Body Simulation](https://raw.githubusercontent.com/miitarou/three-body-sim/main/preview.gif)

## 🚀 機能

- **2D/3Dシミュレーション**: 平面と立体の両方に対応
- **8の字解**: 数学的に証明された美しい周期軌道
- **カオスモード**: 毎回異なるランダム初期条件
- **自動リスタート**: 物体が範囲外に出たら新規スタート
- **リアルタイム情報表示**: エネルギー、位置、世代カウンター
- **視点自動回転**: 3D版で全方向から観察可能

---

## 📦 インストール

```bash
git clone https://github.com/miitarou/three-body-sim.git
cd three-body-sim
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

## ▶️ 実行

```bash
# 2D版
python three_body_simulation.py

# 3D版（自動リスタート機能付き）
python three_body_simulation_3d.py
```

---

## 🔬 数学的基盤

### 万有引力の法則（ベクトル形式）

```
F_ij = G × m_i × m_j / |r_ij|² × r̂_ij
```

- `F_ij`: 物体iが物体jから受ける力
- `G`: 万有引力定数（シミュレーションでは G=1 に正規化）
- `r_ij`: 相対位置ベクトル

### N体問題の運動方程式

```
m_i × d²r_i/dt² = Σ G × m_i × m_j / |r_j - r_i|³ × (r_j - r_i)
```

3つの物体が互いに引き合う力を計算し、加速度→速度→位置を更新します。

---

## 🧮 数値積分法：4次ルンゲ＝クッタ法（RK4）

運動方程式（2階ODE）を1階の連立ODEに変換：

```
dr/dt = v
dv/dt = a(r)
```

RK4の更新則（4次精度）：

```python
k1_r = v
k1_v = a(r)

k2_r = v + dt/2 × k1_v
k2_v = a(r + dt/2 × k1_r)

k3_r = v + dt/2 × k2_v
k3_v = a(r + dt/2 × k2_r)

k4_r = v + dt × k3_v
k4_v = a(r + dt × k3_r)

r_new = r + dt/6 × (k1_r + 2×k2_r + 2×k3_r + k4_r)
v_new = v + dt/6 × (k1_v + 2×k2_v + 2×k3_v + k4_v)
```

**なぜRK4？**
- オイラー法（1次）→ エネルギーが発散
- RK4（4次）→ 長時間シミュレーションでもエネルギー保存が良好

---

## 🛡️ Plummerソフトニング

物体が極端に接近すると `1/r²` が発散して数値計算がクラッシュします。
これを防ぐために **Plummerソフトニング** を使用：

通常のポテンシャル：
```
φ = -G × m1 × m2 / r
```

ソフトニング後：
```
φ = -G × m1 × m2 / √(r² + ε²)
```

- `ε = 0.05`（ソフトニング長）
- r → 0 でも有限値
- r >> ε では元のポテンシャルに収束
- 天体力学シミュレーションで標準的な手法

---

## 🎯 8の字解（Figure-8 Solution）

2000年にChenciner & Montgomeryが発見した、三体問題の周期解。

**初期条件**（Chenciner-Montgomery solution）：

```python
# 位置
x1, y1 = 0.97000436, -0.24308753
positions = [[x1, y1], [-x1, -y1], [0, 0]]

# 速度
vx3, vy3 = -0.93240737, -0.86473146
velocities = [[vx3/2, vy3/2], [vx3/2, vy3/2], [-vx3, -vy3]]
```

3つの物体が同じ「8の字」軌道を追いかけるように周回し続けます。

**参考文献**:
- Chenciner & Montgomery (2000): "A remarkable periodic solution of the three-body problem"
- arXiv:math/0011268

---

## 🌀 カオスとは

三体問題は**カオス的**です。これは：

1. **初期値鋭敏性**: わずかな初期条件の違いが、長期的に全く異なる結果を生む
2. **長期予測不可能**: 短期的には正確だが、長期予測は原理的に不可能
3. **決定論的**: ランダムではなく、同じ初期条件なら同じ結果

シミュレーションで「カオスモード」を選ぶと、この性質を体験できます。

---

## 💻 プログラミングモデル

### 状態空間

各物体の状態を位置+速度の6次元ベクトルで表現：

```python
state = [x, y, z, vx, vy, vz]
```

3物体 × 6次元 = **18次元の状態空間**

### リアルタイムシミュレーション

```
┌─────────────────────────────────────┐
│  FuncAnimation イベントループ       │
│  ┌───────────────────────────────┐  │
│  │  update(frame)               │  │
│  │  ├─ RK4で物理計算             │  │
│  │  ├─ 境界チェック（範囲外?）    │  │
│  │  ├─ 描画更新                  │  │
│  │  └─ 30ms待機                  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 📁 ファイル構成

```
three-body-sim/
├── three_body_simulation.py     # 2D版シミュレーター
├── three_body_simulation_3d.py  # 3D版（自動リスタート機能付き）
├── requirements.txt             # 依存パッケージ
├── README.md                    # このファイル
└── antigravity.md               # プロジェクト憲法
```

---

## ⚙️ パラメータ調整

`three_body_simulation_3d.py` 内の定数を変更することで挙動を調整できます：

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `G` | 1.0 | 万有引力定数 |
| `DT` | 0.001 | タイムステップ（小さいほど精度↑） |
| `T_MAX` | 20.0 | シミュレーション総時間 |
| `SOFTENING` | 0.05 | Plummerソフトニング長 |
| `DISPLAY_RANGE` | 1.5 | 表示範囲（固定スケール） |

---

## 📚 関連する学問分野

| 分野 | 適用箇所 |
|------|---------|
| **古典力学** | 運動方程式、エネルギー保存則 |
| **天体力学** | N体問題、Plummerポテンシャル |
| **数値解析** | RK4、離散化誤差、安定性 |
| **力学系理論** | カオス、初期値鋭敏性 |
| **計算物理学** | シミュレーション手法全般 |

---

## 📜 ライセンス

MIT License

---

## 🙏 謝辞

- Chenciner & Montgomery の8の字解の発見
- Matplotlib開発チーム
- NumPy開発チーム
