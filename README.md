# 3-DOF Spherical Manipulator (R-R-P) — Kinematics & Workspace Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3D%20Graphics-11557c.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced kinematics simulation and workspace modeling environment for a 3-degree-of-freedom spherical robotic arm (Revolute-Revolute-Prismatic: R-R-P). Built in Python with NumPy and Matplotlib, this toolkit implements standard Denavit–Hartenberg (DH) parameterization, forward kinematics, 3D polygonal volumetric rendering, real-time interactive sliders, automated trajectory sweeps, and reachability boundary analysis.

---

## 🧭 System Architecture & Kinematic Model

The manipulator follows an **R-R-P** mechanical topology:
1. **Joint 1 (Base Turntable):** Revolute joint rotating azimuthally about the inertial vertical axis $Z_0$.
2. **Joint 2 (Shoulder Pitch / Elevation):** Revolute joint elevating the arm pitch angle about axis $Z_1$.
3. **Joint 3 (Prismatic Extender):** Prismatic joint displacing the end-effector radially along the pointing vector $Z_2$.

```
           [Z0] (Base Axis)
             |
       +-----+-----+ (Turntable: theta1)
             |  d1 = 0.35m
       +-----+-----+ (Shoulder: theta2)
              \
               \  d3 in [0.20m, 0.90m] (Prismatic Extension)
                \
                 [End-Effector / Gripper Tool]
```

### Denavit–Hartenberg (DH) Parameter Convention

The kinematics are formulated via the Standard Denavit–Hartenberg transformation:

$$A_i = \text{Rot}_z(\theta_i) \cdot \text{Trans}_z(d_i) \cdot \text{Trans}_x(a_i) \cdot \text{Rot}_x(\alpha_i)$$

| Frame $i$ | Link Length $a_i$ | Link Twist $\alpha_i$ | Link Offset $d_i$ | Joint Angle $\theta_i$ | Motion Type |
|:---------:|:-----------------:|:--------------------:|:-----------------:|:----------------------:|:-----------:|
|     1     |         0         |      $-\pi / 2$      |     $d_1 = 0.35$ m|       $\theta_1$       |  Revolute   |
|     2     |         0         |      $+\pi / 2$      |         0         |       $\theta_2$       |  Revolute   |
|     3     |         0         |          0           |       $d_3$       |           0            |  Prismatic  |

### Analytical Transformation Matrices

- **Base to Shoulder ($A_1$):**
  $$A_1 = \begin{bmatrix}
  \cos\theta_1 & 0 & -\sin\theta_1 & 0 \\
  \sin\theta_1 & 0 &  \cos\theta_1 & 0 \\
  0            & -1 & 0            & d_1 \\
  0            & 0  & 0            & 1
  \end{bmatrix}$$

- **Shoulder to Extension Axis ($A_2$):**
  $$A_2 = \begin{bmatrix}
  \cos\theta_2 & 0 & \sin\theta_2 & 0 \\
  \sin\theta_2 & 0 & -\cos\theta_2 & 0 \\
  0            & 1 & 0            & 0 \\
  0            & 0 & 0            & 1
  \end{bmatrix}$$

- **Extension Axis to Tool Center Point ($A_3$):**
  $$A_3 = \begin{bmatrix}
  1 & 0 & 0 & 0 \\
  0 & 1 & 0 & 0 \\
  0 & 0 & 1 & d_3 \\
  0 & 0 & 0 & 1
  \end{bmatrix}$$

- **Cumulative Transformation ($T_0^3 = A_1 A_2 A_3$):**
  $$T_0^3 = \begin{bmatrix}
  \cos\theta_1 \cos\theta_2 & -\sin\theta_1 & \cos\theta_1 \sin\theta_2 & d_3 \cos\theta_1 \sin\theta_2 \\
  \sin\theta_1 \cos\theta_2 &  \cos\theta_1 & \sin\theta_1 \sin\theta_2 & d_3 \sin\theta_1 \sin\theta_2 \\
  -\sin\theta_2              &  0            & \cos\theta_2              & d_1 + d_3 \cos\theta_2        \\
  0                          &  0            & 0                         & 1
  \end{bmatrix}$$

### Closed-Form Cartesian Coordinates

The tool position $[X, Y, Z]^T$ in the global base reference frame is:

$$\begin{cases}
X = d_3 \cdot \cos(\theta_1) \cdot \sin(\theta_2) \\
Y = d_3 \cdot \sin(\theta_1) \cdot \sin(\theta_2) \\
Z = d_1 + d_3 \cdot \cos(\theta_2)
\end{cases}$$

---

## 🌐 Workspace Geometry & Limits

- **Azimuth Rotation ($\theta_1$):** $-180^\circ \le \theta_1 \le +180^\circ$ ($360^\circ$ continuous sweep)
- **Elevation / Pitch ($\theta_2$):** $-75^\circ \le \theta_2 \le +75^\circ$
- **Extension Stroke ($d_3$):** $0.20\text{ m} \le d_3 \le 0.90\text{ m}$ (Stroke $= 0.70$ m)
- **Reachable Volume:** A spherical shell sector centered at $(0, 0, d_1)$ bounded between inner radius $r_{\min} = 0.20$ m and outer radius $r_{\max} = 0.90$ m.

---

## ✨ Features

- **Polygonal 3D Robot Assembly:** Procedural rendering with `mpl_toolkits.mplot3d.art3d.Poly3DCollection`, including:
  - Base mounting plate and rotating turntable indicator
  - Structural support column and rounded protective shoulder dome
  - Fixed outer sleeve guide housing and inner telescoping rod
  - Dual-finger parallel gripper end-effector
- **Interactive Dual-Mode Viewer:**
  - **Manual Control:** Responsive GUI sliders for $\theta_1$, $\theta_2$, and $d_3$ with live digital Cartesian readouts.
  - **Automatic Trajectory Demo:** Smooth sinusoidal trajectory execution showcasing boundary exploration with Play/Pause button controls.
- **Custom Dimension Builder:** Interactive terminal prompt allowing dynamic reconfiguration of link heights, housing lengths, and joint travel boundaries.
- **Trajectory Dataset:** Includes `spherical_workspace_trajectory.json` containing thousands of precomputed boundary points for validation and cross-platform verification.
- **Standalone Build:** PyInstaller specification (`spherical_robot_modern.spec`) to bundle the Python environment into an independent Windows executable.

---

## 📁 Repository Contents

```text
spherical-robot-workspace/
├── spherical_robot_modern.py      # Core kinematics, 3D visualization, and interactive UI
├── spherical_workspace_trajectory.json # Precomputed spherical workspace boundary dataset
├── spherical_robot_modern.spec    # PyInstaller standalone executable specification
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # Technical documentation
```

---

## 🚀 Installation & Running

### 1. Prerequisites
- Python 3.8 or higher.

### 2. Setup
```bash
git clone https://github.com/Omgiri01/spherical-robot-workspace.git
cd spherical-robot-workspace
pip install -r requirements.txt
```

### 3. Launching the Application
```bash
python spherical_robot_modern.py
```

Upon launching, choose between:
1. **Default Dimensions** or **Custom Dimensions & Limits**
2. **Automatic demo motion** or **Manual control with sliders**

---

## 🔗 Companion Real-Time 3D Simulation

For the interactive Unity 3D engine simulation with GPU particle trail tracing, touch controls, and pre-compiled Windows/Android binaries, visit:
👉 **[spherical-robot-unity](https://github.com/Omgiri01/spherical-robot-unity)**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
