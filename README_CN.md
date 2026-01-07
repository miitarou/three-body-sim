# N体问题模拟器

**🌐 语言: [日本語](README.md) | [English](README_EN.md) | 中文**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Tests](https://github.com/miitarou/three-body-sim/actions/workflows/test.yml/badge.svg)](https://github.com/miitarou/three-body-sim/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个实时模拟和可视化N个天体在万有引力作用下相互作用的程序。

![Demo](demo.gif)

## 🚀 功能

- **N体模拟**: 可自由切换3到9个天体
- **3D可视化**: 从任意角度观察
- **自动重启**: 当天体飞出边界时自动重新开始
- **教育模式**: 力向量显示、预测模式等

---

## 📦 安装

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

## ▶️ 运行

```bash
python nbody_simulation_advanced.py
```

---

## 🎮 操作方法

### 基本操作

| 按键 | 功能 |
|------|------|
| **空格键** | 暂停/继续 |
| **R** | 使用新的初始条件重新开始 |
| **A** | 开启/关闭自动旋转 |
| **+/-** | 放大/缩小 |
| **鼠标滚轮** | 放大/缩小 |
| **拖拽** | 旋转视角（自动旋转关闭时） |
| **Q** | 退出 |

### 教育功能

| 按键 | 功能 | 学习效果 |
|------|------|----------|
| **F** | 力向量显示 | 用红色箭头可视化重力的方向和大小 |
| **G** | 幽灵模式 | 可视化对初始条件的敏感性（混沌的本质） |
| **E** | 编辑器面板 | 查看天体数量和质量 |
| **P** | 预测竞猜 | 预测2.5秒后的状态并竞争得分 |
| **M** | 周期解模式 | 循环展示著名的周期轨道 |
| **B** | 回退 | 返回上一个Generation |
| **S** | 保存 | 将当前初始条件导出为JSON文件 |
| **L** | 加载 | 从JSON文件重现轨道 |
| **3-9** | 改变天体数量 | 自由切换3到9个天体 |

### 周期解目录（M键）

按 **M** 键体验数学上发现的著名三体周期解。

| # | 名称 | 发现者 | 特点 |
|---|------|--------|------|
| 1 ⭐ | **8字形经典解** | Chenciner-Montgomery (2000) | 数学史上最著名的三体周期解 |
| 2 ⭐ | **拉格朗日三角形** | Lagrange (1772) | 保持等边三角形旋转（历史价值最高） |
| 3 ⭐ | **蝴蝶 I** | Šuvakov-Dmitrašinović (2013) | 美丽的蝴蝶状轨道 |
| 4 | 8字形 (I.2.A) | Šuvakov-Dmitrašinović (2013) | 8字形的变体 |
| 5 | 飞蛾 I | Šuvakov-Dmitrašinović (2013) | 复杂的飞蛾状轨道 |
| 6-10 | 阴阳系列 | Šuvakov-Dmitrašinović (2013) | 各种对称轨道 |

> **注意**: 周期解理论上可以永久持续，但由于数值误差的累积，轨道会随时间偏离。这展示了混沌系统数值模拟的局限性。

---

## 📚 教育用途

### 1. 观看（自动播放体验）

模拟会自动开始。享受观看天体跳舞的过程。每次运行都会生成新的初始条件。

### 2. 观察力向量

按 **F** 键将重力显示为红色箭头。

- 箭头**方向** = 重力拉动的方向
- 箭头**长度** = 重力的强度
- 注意观察天体接近时箭头变长

### 3. 参加预测竞猜

按 **P** 键进入预测竞猜。

1. 模拟暂停
2. 预测 **2.5秒后** 会发生什么：
   - **1** = 碰撞（天体相互接近）
   - **2** = 逃逸（天体飞出）
   - **3** = 稳定轨道（两者都不）
3. 显示正确/错误，记录得分
4. 按 **Enter** 继续并验证

体验预测混沌系统有多困难。

### 5. 幽灵模式 - 体验混沌

按 **G** 键显示“幽灵”天体。

1. 幽灵与真实天体的位置仅相差 **0.001**
2. 初始时重叠，但随时间推移会 **完全分离**
3. 体验“微小的初始差异导致截然不同的结果”

### 4. 增加天体

按 **5** 或 **7** 键增加天体数量，观察更复杂的混沌行为。

---

## 🔬 物理背景

### 万有引力定律

```
F = G × m₁ × m₂ / r²
```

| 符号 | 含义 |
|------|------|
| **F** | 两个物体之间的引力 [N] |
| **G** | 万有引力常数 (6.674×10⁻¹¹ N⋅m²/kg²) |
| **m₁, m₂** | 各物体的质量 [kg] |
| **r** | 两物体之间的距离 [m] |

### 三体问题与混沌

三体问题以没有一般解析解而闻名：

1. **对初始条件敏感**: 微小差异导致截然不同的结果
2. **长期不可预测**: 短期准确，长期原则上不可能
3. **确定性**: 非随机（相同条件 = 相同结果）

### 数值方法

| 方法 | 目的 |
|------|------|
| **RK4积分法** | 高精度时间演化 |
| **自适应时间步长** | 接近时步长更细 |
| **Plummer软化** | 防止近距离时数值发散 |

---

## 🎯 学习要点

| 主题 | 内容 |
|------|------|
| **力与运动** | 力如何改变物体的运动 |
| **能量守恒** | 动能 + 势能 = 常数 |
| **混沌理论** | 初始条件的微小差异 → 结果的巨大变化 |
| **N体问题** | 3个以上天体没有解析解 |
| **数值模拟** | 计算机如何再现物理 |

---

## 📁 文件结构

```
three-body-sim/
├── nbody_simulation_advanced.py  # 主模拟器
├── test_nbody.py                 # 测试套件
├── demo.gif                      # 演示动画
└── README.md                     # 文档
```

---

## ⚙️ 自定义

修改代码中的常量以调整行为：

| 常量 | 默认值 | 描述 |
|------|--------|------|
| `DEFAULT_N_BODIES` | 3 | 初始天体数量 |
| `G` | 1.0 | 万有引力常数 |
| `SOFTENING` | 0.05 | 碰撞避免软化长度 |
| `MASS_MIN/MAX` | 0.5/2.0 | 随机质量范围 |

---

## 📖 参考文献

- Chenciner & Montgomery (2000): "A remarkable periodic solution of the three-body problem"
- Šuvakov & Dmitrašinović (2013): "Three Classes of Newtonian Three-Body Planar Periodic Orbits"
- arXiv:math/0011268, arXiv:1303.0181

---

## 📜 许可证

MIT License

---

## 🙏 致谢

- **约瑟夫-路易斯·拉格朗日**: 1772年发现了等边三角形解
- **Milovan Šuvakov & Veljko Dmitrašinović**: 2013年发现了13种新的周期解
- Chenciner & Montgomery - 8字形解的发现者
- Matplotlib / NumPy 开发团队
