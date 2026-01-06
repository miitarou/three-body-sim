# N-Body Problem Simulator

**🌐 Language: [日本語](README.md) | English | [中文](README_CN.md)**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Tests](https://github.com/miitarou/three-body-sim/actions/workflows/test.yml/badge.svg)](https://github.com/miitarou/three-body-sim/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time simulation and visualization program for N celestial bodies interacting through gravitational forces.

![Demo](demo.gif)

## 🚀 Features

- **N-body Simulation**: Freely switch between 3 to 9 bodies
- **3D Visualization**: Observe from any angle
- **Auto-restart**: Automatically restart when bodies escape the boundary
- **Educational Mode**: Force vector display, prediction mode, and more

---

## 📦 Installation

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

## ▶️ Run

```bash
python nbody_simulation_advanced.py
```

---

## 🎮 Controls

### Basic Controls

| Key | Function |
|-----|----------|
| **SPACE** | Pause/Resume |
| **R** | Restart with new initial conditions |
| **A** | Toggle auto-rotation |
| **+/-** | Zoom in/out |
| **Mouse wheel** | Zoom in/out |
| **Drag** | Rotate view (when auto-rotation is off) |
| **Q** | Quit |

### Educational Features

| Key | Function | Learning Effect |
|-----|----------|-----------------|
| **F** | Force vectors | Visualize gravity direction and magnitude with red arrows |
| **E** | Editor panel | View body count and masses |
| **P** | Prediction mode | Pause and predict "what happens next?" |
| **M** | Periodic solutions | Cycle through famous periodic orbits |
| **3-9** | Change body count | Freely switch between 3 to 9 bodies |

### Periodic Solution Catalog (M key)

Press **M** to experience mathematically discovered famous three-body periodic solutions.

| # | Name | Discoverer | Feature |
|---|------|------------|---------|
| 1 ⭐ | **Figure-8 Classic** | Chenciner-Montgomery (2000) | Most famous three-body periodic solution |
| 2 ⭐ | **Lagrange Triangle** | Lagrange (1772) | Rotates while maintaining equilateral triangle |
| 3 ⭐ | **Butterfly I** | Šuvakov-Dmitrašinović (2013) | Beautiful butterfly-like orbit |
| 4 | Figure-8 (I.2.A) | Šuvakov-Dmitrašinović (2013) | Figure-8 variation |
| 5 | Moth I | Šuvakov-Dmitrašinović (2013) | Complex moth-like orbit |
| 6-10 | Yin-Yang series | Šuvakov-Dmitrašinović (2013) | Various symmetric orbits |

> **Note**: Periodic solutions theoretically persist forever, but numerical errors accumulate over time, causing orbits to deviate. This demonstrates the limitations of numerical simulation in chaotic systems.

---

## 📚 Educational Usage

### 1. Watch (Auto-play Experience)

The simulation starts automatically. Enjoy watching the celestial bodies dance. Every run generates new initial conditions.

### 2. Observe Force Vectors

Press **F** to display gravitational forces as red arrows.

- Arrow **direction** = direction of gravitational pull
- Arrow **length** = strength of gravity
- Watch arrows grow longer as bodies approach

### 3. Make Predictions

Press **P** to enter prediction mode.

1. Simulation pauses
2. Think about "what happens next?"
3. Press **Enter** to resume and verify

Experience the essence of chaos theory.

### 4. Add More Bodies

Press **5** or **7** to increase body count for more complex chaotic behavior.

---

## 🔬 Physics Background

### Law of Universal Gravitation

```
F = G × m₁ × m₂ / r²
```

| Symbol | Meaning |
|--------|---------|
| **F** | Gravitational force between two bodies [N] |
| **G** | Gravitational constant (6.674×10⁻¹¹ N⋅m²/kg²) |
| **m₁, m₂** | Mass of each body [kg] |
| **r** | Distance between bodies [m] |

### Three-Body Problem and Chaos

The three-body problem is famous for having no general analytical solution:

1. **Sensitivity to initial conditions**: Tiny differences lead to vastly different outcomes
2. **Long-term unpredictability**: Short-term accuracy, long-term impossibility
3. **Deterministic**: Not random (same conditions = same result)

### Numerical Methods

| Method | Purpose |
|--------|---------|
| **RK4 Integration** | High-precision time evolution |
| **Adaptive Timestep** | Finer steps for close approaches |
| **Plummer Softening** | Prevents numerical divergence at close ranges |

---

## 🎯 Learning Objectives

| Topic | Content |
|-------|---------|
| **Force and Motion** | How forces change object motion |
| **Energy Conservation** | Kinetic E + Potential E = constant |
| **Chaos Theory** | Small initial differences → large outcome changes |
| **N-body Problem** | No analytical solution for 3+ bodies |
| **Numerical Simulation** | How computers reproduce physics |

---

## 📁 File Structure

```
three-body-sim/
├── nbody_simulation_advanced.py  # Main simulator
├── test_nbody.py                 # Test suite
├── demo.gif                      # Demo animation
└── README.md                     # Documentation
```

---

## ⚙️ Customization

Modify constants in the code to adjust behavior:

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_N_BODIES` | 3 | Initial body count |
| `G` | 1.0 | Gravitational constant |
| `SOFTENING` | 0.05 | Collision avoidance softening length |
| `MASS_MIN/MAX` | 0.5/2.0 | Random mass range |

---

## 📖 References

- Chenciner & Montgomery (2000): "A remarkable periodic solution of the three-body problem"
- Šuvakov & Dmitrašinović (2013): "Three Classes of Newtonian Three-Body Planar Periodic Orbits"
- arXiv:math/0011268, arXiv:1303.0181

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- **Joseph-Louis Lagrange**: Discovered the equilateral triangle solution in 1772
- **Milovan Šuvakov & Veljko Dmitrašinović**: Discovered 13 new periodic solutions in 2013
- Chenciner & Montgomery for the Figure-8 solution
- Matplotlib / NumPy development teams
