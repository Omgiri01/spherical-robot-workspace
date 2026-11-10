# 3-DOF Spherical Robot Workspace & Kinematics Simulator

A Python simulation package for a 3-DOF Spherical Manipulator (Revolute-Revolute-Prismatic: R-R-P). It models forward kinematics using the Standard Denavit–Hartenberg (DH) convention, renders full 3D robot geometry in Matplotlib, and provides interactive sliders as well as trajectory-driven workspace analysis.

---

## Kinematic Model & DH Parameters

The manipulator consists of three joints:
1. **Base Rotation ($\theta_1$):** Revolute joint around the global vertical axis.
2. **Elevation ($\theta_2$):** Revolute joint controlling the pitch angle.
3. **Radial Extension ($d_3$):** Prismatic joint extending along the rotated arm vector.

### Standard DH Table

| Joint $i$ | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ | Description |
|---|---|---|---|---|---|
| 1 | 0 | $-\pi/2$ | $d_1 = 0.35$ m | $\theta_1$ | Base rotation |
| 2 | 0 | $+\pi/2$ | 0 | $\theta_2$ | Elevation / Pitch |
| 3 | 0 | 0 | $d_3$ | 0 | Telescoping arm extension |

### Transformation Matrix

The homogeneous transformation matrix from base $\{0\}$ to tool $\{3\}$ is:

$$T_0^3 = \begin{bmatrix}
\cos\theta_1 \cos\theta_2 & -\sin\theta_1 & \cos\theta_1 \sin\theta_2 & d_3 \cos\theta_1 \sin\theta_2 \\
\sin\theta_1 \cos\theta_2 &  \cos\theta_1 & \sin\theta_1 \sin\theta_2 & d_3 \sin\theta_1 \sin\theta_2 \\
-\sin\theta_2              &  0            & \cos\theta_2              & d_1 + d_3 \cos\theta_2        \\
0                          &  0            & 0                         & 1
\end{bmatrix}$$

---

## Features

- **Kinematics Engine:** Evaluates forward kinematics and link coordinate transforms.
- **3D Polygonal Mesh Visualizer:** Renders the turntable, vertical column, shoulder housing, telescoping beam, and two-finger parallel gripper.
- **Interactive UI:** Real-time sliders for $\theta_1 \in [-180^\circ, 180^\circ]$, $\theta_2 \in [-75^\circ, 75^\circ]$, and $d_3 \in [0.20, 0.90]$ meters.
- **Trajectory Sweep Dataset:** Contains precomputed trajectory sweeps in `spherical_workspace_trajectory.json` mapping out the reachable workspace shell.
- **Companion Unity Simulation:** Real-time mobile/desktop Unity simulation available in [spherical-robot-unity](https://github.com/Omgiri01/spherical-robot-unity).

---

## Installation & Usage

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Simulator

```bash
python spherical_robot_modern.py
```

Follow the terminal prompts:
- Choose configuration: default dimensions or custom link dimensions.
- Choose mode:
  - **1) Automatic demo motion:** plays continuous periodic trajectory sweeps.
  - **2) Manual control with sliders:** manipulate joint sliders in real time.

---

## File Structure

- `spherical_robot_modern.py`: Main kinematics modeling, DH computation, and 3D visualization.
- `spherical_workspace_trajectory.json`: Sampled workspace boundary trajectory data.
- `spherical_robot_modern.spec`: PyInstaller standalone executable build specification.
- `requirements.txt`: Python package dependencies.
