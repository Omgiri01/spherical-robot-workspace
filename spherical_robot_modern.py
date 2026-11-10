from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class RobotDimensions:
    """
    Geometric dimensions of the spherical robot.

    All units are in meters.
    """

    # Vertical distance from base frame {0} to the spherical joint center.
    base_height: float = 0.35
    # Visual: fixed prismatic "housing" length (outer sleeve).
    # The prismatic joint total length is d3; we draw it as:
    #   housing_length (fixed) + (d3 - housing_length) sliding extension.
    prismatic_housing_length: float = 0.25

    # Visual dimensions (meters) to make the model look like the blueprint
    base_plate_size: Tuple[float, float, float] = (0.80, 0.60, 0.10)  # (x, y, z)
    turntable_radius: float = 0.24
    turntable_thickness: float = 0.06
    column_radius: float = 0.07

    # Upper cover / housing (the tall rounded body in your sketch)
    cover_size: Tuple[float, float, float] = (0.38, 0.26, 0.40)  # (length x, depth y, height z)
    cover_top_radius: float = 0.13  # rounded top radius (half-cylinder)
    shoulder_dome_radius: float = 0.11

    housing_thickness: Tuple[float, float] = (0.055, 0.036)  # (width, height)
    slider_thickness: Tuple[float, float] = (0.042, 0.028)

    gripper_palm_length: float = 0.06
    gripper_palm_thickness: Tuple[float, float] = (0.05, 0.03)
    gripper_finger_length: float = 0.08
    gripper_finger_thickness: Tuple[float, float] = (0.015, 0.012)
    gripper_finger_gap: float = 0.035


@dataclass(frozen=True)
class JointLimits:
    """
    Joint limits of the 3-DOF spherical manipulator.

    theta1, theta2 are in degrees, d3 in meters.
    """

    theta1_deg: Tuple[float, float] = (-180.0, 180.0)
    theta2_deg: Tuple[float, float] = (-75.0, 75.0)
    d3: Tuple[float, float] = (0.20, 0.90)


class SphericalRobot:
    """
    Kinematic model of a 3-DOF spherical manipulator (R-R-P) using
    the standard Denavit–Hartenberg (DH) algorithm.

    DH table (standard DH; A_i = Rotz(theta_i) Transz(d_i) Transx(a_i) Rotx(alpha_i)):

        i   a_i   alpha_i     d_i      theta_i
        1   0     -pi/2       d1       theta1
        2   0     +pi/2       0        theta2
        3   0      0          d3       0

    where:
        theta1 -> base rotation
        theta2 -> elevation
        d3     -> radial extension along the rotated z-axis
        d1     -> base_height
    """

    def __init__(self, dims: RobotDimensions | None = None, limits: JointLimits | None = None) -> None:
        self.dims = dims or RobotDimensions()
        self.limits = limits or JointLimits()

    # ------------------------------------------------------------------
    # Core kinematics
    # ------------------------------------------------------------------
    @staticmethod
    def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def T_0_3(self, theta1: float, theta2: float, d3: float) -> np.ndarray:
        """
        Homogeneous transform from base frame {0} to end-effector frame {3}.

        Parameters
        ----------
        theta1 : float
            Base joint angle in radians.
        theta2 : float
            Second joint angle in radians, exactly as defined in
            your original notebook / MATLAB code for the spherical arm.
        d3 : float
            Prismatic extension along the rotated z-axis (meters).
        """
        d1 = self.dims.base_height

        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)

        # This matches the analytic T_0^3 used in spherical_robot_workspace.py
        # and your handwritten derivation.
        return np.array(
            [
                [c1 * c2, -s1, c1 * s2, d3 * c1 * s2],
                [s1 * c2, c1, s1 * s2, d3 * s1 * s2],
                [-s2, 0.0, c2, d3 * c2 + d1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def forward_kinematics(self, theta1: float, theta2: float, d3: float):
        """
        Compute joint and end-effector positions for the current configuration.

        Geometry note:
        - DH describes a single prismatic link of total length d3.
        - For visualization only, we split that link into:
            p1 -> p2  fixed "housing" (outer sleeve)
            p2 -> p3  sliding extension

        Returns
        -------
        p0 : (3,) ndarray
            Base position.
        p1 : (3,) ndarray
            Spherical joint center (top of the base column).
        p2 : (3,) ndarray
            End of the fixed prismatic housing (visual only).
        p3 : (3,) ndarray
            End-effector position.
        T : (4, 4) ndarray
            Homogeneous transform T_0^3.
        """
        T = self.T_0_3(theta1, theta2, d3)

        # Base origin
        p0 = np.array([0.0, 0.0, 0.0])

        # Top of the base column (spherical joint center)
        p1 = np.array([0.0, 0.0, self.dims.base_height])

        # Prismatic axis is z3 (and z2); use the end-effector z-axis direction
        z_dir = _unit(T[0:3, 2])

        # Clamp d3 to the configured limits for "physical" behavior
        d3_clamped = float(np.clip(d3, self.limits.d3[0], self.limits.d3[1]))
        housing_len = float(np.clip(self.dims.prismatic_housing_length, 0.0, d3_clamped))
        slider_len = max(d3_clamped - housing_len, 0.0)

        p2 = p1 + housing_len * z_dir
        p3 = p2 + slider_len * z_dir

        return p0, p1, p2, p3, T

    # ------------------------------------------------------------------
    # Workspace utilities
    # ------------------------------------------------------------------
    def sample_workspace(self, n1: int = 60, n2: int = 30, n3: int = 15):
        """
        Sample the joint space within the configured limits and
        return the corresponding end-effector points.
        """
        th1_vals = np.deg2rad(
            np.linspace(self.limits.theta1_deg[0], self.limits.theta1_deg[1], n1)
        )
        th2_vals = np.deg2rad(
            np.linspace(self.limits.theta2_deg[0], self.limits.theta2_deg[1], n2)
        )
        d3_vals = np.linspace(self.limits.d3[0], self.limits.d3[1], n3)

        xs, ys, zs = [], [], []

        for th1 in th1_vals:
            for th2 in th2_vals:
                for d3 in d3_vals:
                    _, _, _, p3, _ = self.forward_kinematics(th1, th2, d3)
                    xs.append(p3[0])
                    ys.append(p3[1])
                    zs.append(p3[2])

        return np.array(xs), np.array(ys), np.array(zs)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _basis_from_z(z_dir: np.ndarray) -> np.ndarray:
    """
    Create an orthonormal basis with z-axis aligned to z_dir.

    Returns a 3x3 rotation matrix with columns [x, y, z].
    """
    z = _unit(z_dir)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    x = _unit(np.cross(up, z))
    y = _unit(np.cross(z, x))
    return np.column_stack([x, y, z])


def _rotz(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _hemisphere_surface(center: np.ndarray, radius: float, n_u: int = 22, n_v: int = 16):
    """
    Hemisphere surface oriented "up" (+z), returned in world coordinates.
    """
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, 0.5 * np.pi, n_v)  # 0..pi/2 (upper hemisphere)
    uu, vv = np.meshgrid(u, v)

    x = radius * np.cos(uu) * np.sin(vv)
    y = radius * np.sin(uu) * np.sin(vv)
    z = radius * np.cos(vv)

    X = center[0] + x
    Y = center[1] + y
    Z = center[2] + z
    return X, Y, Z


def _half_cylinder_surface(center: np.ndarray, R: np.ndarray, radius: float, length: float, n_theta: int = 20, n_x: int = 16):
    """
    Half-cylinder "cap" oriented along local x-axis.
    Local coords:
      x in [-L/2, +L/2]
      y = r cos(theta), z = r sin(theta) for theta in [0..pi] (upper half)
    """
    xs = np.linspace(-0.5 * length, 0.5 * length, n_x)
    th = np.linspace(0.0, np.pi, n_theta)
    XX, TH = np.meshgrid(xs, th)
    YY = radius * np.cos(TH)
    ZZ = radius * np.sin(TH)

    local = np.stack([XX, YY, ZZ], axis=-1)  # (n_theta,n_x,3)
    world = center + local @ R.T
    return world[..., 0], world[..., 1], world[..., 2]


def _faces_from_grid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """
    Convert a (m,n) surface grid into quad faces for Poly3DCollection.
    """
    faces = []
    m, n = X.shape
    for i in range(m - 1):
        for j in range(n - 1):
            faces.append(
                [
                    np.array([X[i, j], Y[i, j], Z[i, j]]),
                    np.array([X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]),
                    np.array([X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]]),
                    np.array([X[i + 1, j], Y[i + 1, j], Z[i + 1, j]]),
                ]
            )
    return faces


def _box_faces_from_corners(corners: np.ndarray):
    """
    corners: (8,3) using the local-corner order:
      0 (-x,-y,-z), 1 (+x,-y,-z), 2 (-x,+y,-z), 3 (+x,+y,-z),
      4 (-x,-y,+z), 5 (+x,-y,+z), 6 (-x,+y,+z), 7 (+x,+y,+z)
    """
    idx = [
        [0, 1, 3, 2],  # -z face
        [4, 5, 7, 6],  # +z face
        [0, 1, 5, 4],  # -y face
        [2, 3, 7, 6],  # +y face
        [0, 2, 6, 4],  # -x face
        [1, 3, 7, 5],  # +x face
    ]
    return [[corners[i] for i in face] for face in idx]


def _prism_faces(center: np.ndarray, R: np.ndarray, width: float, height: float, length: float):
    hx, hy, hz = 0.5 * width, 0.5 * height, 0.5 * length
    local = np.array(
        [
            [-hx, -hy, -hz],
            [+hx, -hy, -hz],
            [-hx, +hy, -hz],
            [+hx, +hy, -hz],
            [-hx, -hy, +hz],
            [+hx, -hy, +hz],
            [-hx, +hy, +hz],
            [+hx, +hy, +hz],
        ],
        dtype=float,
    )
    corners = center + (local @ R.T)
    return _box_faces_from_corners(corners)


def _prism_between(p_start: np.ndarray, p_end: np.ndarray, width: float, height: float):
    axis = p_end - p_start
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        length = 1e-9
        axis = np.array([0.0, 0.0, 1.0])
    R = _basis_from_z(axis)
    center = 0.5 * (p_start + p_end)
    return _prism_faces(center=center, R=R, width=width, height=height, length=length), R


def _add_poly(ax, faces, color: str, alpha: float = 0.9, edgecolor: str = "k", linewidth: float = 0.5):
    poly = Poly3DCollection(faces, facecolors=color, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(poly)
    return poly


def plot_workspace(robot: SphericalRobot) -> None:
    """
    Static 3D plot of the robot's workspace.
    """
    xs, ys, zs = robot.sample_workspace()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=5, alpha=0.35)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Spherical Robot Workspace (modern model)")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def interactive_viewer(robot: SphericalRobot) -> None:
    """
    Interactive 3D viewer with sliders for theta1, theta2 and d3.
    """
    # Initial joint values (center of each range)
    th1_deg_init = 0.5 * (robot.limits.theta1_deg[0] + robot.limits.theta1_deg[1])
    th2_deg_init = 0.5 * (robot.limits.theta2_deg[0] + robot.limits.theta2_deg[1])
    d3_init = 0.5 * (robot.limits.d3[0] + robot.limits.d3[1])

    th1 = np.deg2rad(th1_deg_init)
    th2 = np.deg2rad(th2_deg_init)
    d3 = d3_init

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.1, bottom=0.3)

    p0, p1, p2, p3, T0_3 = robot.forward_kinematics(th1, th2, d3)

    # ----------------------------
    # Static base geometry (looks like the blueprint base + turntable + column)
    # ----------------------------
    base_x, base_y, base_z = robot.dims.base_plate_size
    base_center = np.array([0.0, 0.0, -0.5 * base_z])
    base_R = np.eye(3)
    base_faces = _prism_faces(base_center, base_R, width=base_x, height=base_y, length=base_z)
    _add_poly(ax, base_faces, color="#c9c9c9", alpha=0.9, edgecolor="#555555", linewidth=0.6)

    # Turntable as a thin prism (approximate cylinder look)
    tt_center = np.array([0.0, 0.0, 0.5 * robot.dims.turntable_thickness])
    tt_faces = _prism_faces(
        center=tt_center,
        R=np.eye(3),
        width=2.0 * robot.dims.turntable_radius,
        height=2.0 * robot.dims.turntable_radius,
        length=robot.dims.turntable_thickness,
    )
    _add_poly(ax, tt_faces, color="#9e9e9e", alpha=0.95, edgecolor="#444444", linewidth=0.6)

    # Turntable direction indicator (so theta1 rotation is visible)
    (tt_indicator,) = ax.plot(
        [0.0, robot.dims.turntable_radius],
        [0.0, 0.0],
        [robot.dims.turntable_thickness, robot.dims.turntable_thickness],
        color="#2c3e50",
        linewidth=2.5,
        alpha=0.9,
        label="Base yaw (θ1)",
    )

    # Column (to the spherical joint center)
    col_center = np.array([0.0, 0.0, 0.5 * robot.dims.base_height])
    col_faces = _prism_faces(
        center=col_center,
        R=np.eye(3),
        width=2.0 * robot.dims.column_radius,
        height=2.0 * robot.dims.column_radius,
        length=robot.dims.base_height,
    )
    _add_poly(ax, col_faces, color="#b0b0b0", alpha=0.9, edgecolor="#555555", linewidth=0.6)

    # Upper cover / housing (rounded top), rotates with theta1 like the sketch
    cov_L, cov_D, cov_H = robot.dims.cover_size
    cov_r = robot.dims.cover_top_radius
    cov_base_z = robot.dims.turntable_thickness
    cov_body_h = max(cov_H - cov_r, 0.05)

    def cover_geometry(theta1_rad: float):
        Rz = _rotz(theta1_rad)
        body_center = np.array([0.0, 0.0, cov_base_z + 0.5 * cov_body_h])
        body_faces = _prism_faces(center=body_center, R=Rz, width=cov_L, height=cov_D, length=cov_body_h)

        cap_center = np.array([0.0, 0.0, cov_base_z + cov_body_h])
        Xc, Yc, Zc = _half_cylinder_surface(center=cap_center, R=Rz, radius=cov_r, length=cov_L)
        return body_faces, (Xc, Yc, Zc)

    cover_body_faces, (Xc, Yc, Zc) = cover_geometry(th1)
    cover_poly = _add_poly(ax, cover_body_faces, color="#d0d0d0", alpha=0.65, edgecolor="#666666", linewidth=0.5)
    cover_cap_faces = _faces_from_grid(Xc, Yc, Zc)
    cover_cap = _add_poly(ax, cover_cap_faces, color="#d0d0d0", alpha=0.65, edgecolor="#666666", linewidth=0.2)

    # Shoulder dome at the spherical joint center (p1)
    Xd, Yd, Zd = _hemisphere_surface(center=p1, radius=robot.dims.shoulder_dome_radius)
    dome_faces = _faces_from_grid(Xd, Yd, Zd)
    dome = _add_poly(ax, dome_faces, color="#bbbbbb", alpha=0.9, edgecolor="#666666", linewidth=0.15)

    # ----------------------------
    # Dynamic geometry (DH-correct prismatic link: housing + sliding extension + gripper)
    # ----------------------------
    housing_w, housing_h = robot.dims.housing_thickness
    slider_w, slider_h = robot.dims.slider_thickness

    housing_faces, _ = _prism_between(p1, p2, width=housing_w, height=housing_h)
    housing_poly = _add_poly(ax, housing_faces, color="#2ecc71", alpha=0.85)

    slider_faces, _ = _prism_between(p2, p3, width=slider_w, height=slider_h)
    slider_poly = _add_poly(ax, slider_faces, color="#8e44ad", alpha=0.85)

    # Gripper: palm + 2 fingers
    z_dir = T0_3[0:3, 2]
    R_ee = _basis_from_z(z_dir)
    x_dir = R_ee[:, 0]
    y_dir = R_ee[:, 1]

    palm_len = robot.dims.gripper_palm_length
    palm_w, palm_h = robot.dims.gripper_palm_thickness
    palm_center = p3 + _unit(z_dir) * (0.5 * palm_len)
    palm_faces = _prism_faces(center=palm_center, R=R_ee, width=palm_w, height=palm_h, length=palm_len)
    palm_poly = _add_poly(ax, palm_faces, color="#e74c3c", alpha=0.9)

    finger_len = robot.dims.gripper_finger_length
    finger_w, finger_h = robot.dims.gripper_finger_thickness
    gap = robot.dims.gripper_finger_gap

    finger_base = p3 + _unit(z_dir) * (palm_len + 0.5 * finger_len)
    left_center = finger_base + 0.5 * gap * x_dir + 0.5 * palm_h * y_dir
    right_center = finger_base - 0.5 * gap * x_dir + 0.5 * palm_h * y_dir

    left_faces = _prism_faces(center=left_center, R=R_ee, width=finger_w, height=finger_h, length=finger_len)
    right_faces = _prism_faces(center=right_center, R=R_ee, width=finger_w, height=finger_h, length=finger_len)
    left_poly = _add_poly(ax, left_faces, color="#d35400", alpha=0.95)
    right_poly = _add_poly(ax, right_faces, color="#d35400", alpha=0.95)

    # Optional skeleton lines (helps understanding while still looking "real")
    (skel_line,) = ax.plot(
        [p0[0], p1[0], p2[0], p3[0]],
        [p0[1], p1[1], p2[1], p3[1]],
        [p0[2], p1[2], p2[2], p3[2]],
        color="#1f1f1f",
        linewidth=1.2,
        alpha=0.35,
        label="Kinematic chain",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Interactive Spherical Manipulator (modern model)")
    # Viewing volume based on maximum reach
    max_reach = (
        robot.dims.base_height
        + robot.limits.d3[1]
        + robot.dims.gripper_palm_length
        + robot.dims.gripper_finger_length
    )
    ax.set_xlim(-max_reach, max_reach)
    ax.set_ylim(-max_reach, max_reach)
    ax.set_zlim(0.0, max_reach * 1.2)
    ax.set_box_aspect([1, 1, 1])

    # Dynamic end-effector path (highlights where the end-effector has been)
    path_x = [p3[0]]
    path_y = [p3[1]]
    path_z = [p3[2]]
    path_scatter = ax.scatter(
        path_x,
        path_y,
        path_z,
        s=18,
        alpha=0.9,
        c="#00bcd4",  # light blue, shows only the explored workspace
        edgecolors="none",
        label="End-effector workspace",
    )

    ax.legend(loc="upper right")

    # Slider axes
    ax_th1 = fig.add_axes([0.1, 0.2, 0.8, 0.03])
    ax_th2 = fig.add_axes([0.1, 0.15, 0.8, 0.03])
    ax_d3 = fig.add_axes([0.1, 0.1, 0.8, 0.03])

    s_th1 = Slider(
        ax_th1,
        "theta1 [deg]",
        robot.limits.theta1_deg[0],
        robot.limits.theta1_deg[1],
        valinit=th1_deg_init,
    )
    s_th2 = Slider(
        ax_th2,
        "theta2 elev [deg]",
        robot.limits.theta2_deg[0],
        robot.limits.theta2_deg[1],
        valinit=th2_deg_init,
    )
    s_d3 = Slider(
        ax_d3,
        "d3 [m]",
        robot.limits.d3[0],
        robot.limits.d3[1],
        valinit=d3_init,
    )

    def update(_):
        th1_rad = np.deg2rad(s_th1.val)
        th2_rad = np.deg2rad(s_th2.val)
        d3_val = s_d3.val

        p0_new, p1_new, p2_new, p3_new, T = robot.forward_kinematics(th1_rad, th2_rad, d3_val)

        # Update turntable indicator (shows theta1 rotation)
        tip = _rotz(th1_rad) @ np.array([robot.dims.turntable_radius, 0.0, 0.0])
        tt_indicator.set_data([0.0, tip[0]], [0.0, tip[1]])
        tt_indicator.set_3d_properties([robot.dims.turntable_thickness, robot.dims.turntable_thickness])

        # Update cover geometry (rotate with theta1)
        cover_faces_new, (Xc_new, Yc_new, Zc_new) = cover_geometry(th1_rad)
        cover_poly.set_verts(cover_faces_new)
        nonlocal_vars["cover_cap"].set_verts(_faces_from_grid(Xc_new, Yc_new, Zc_new))

        # Update prismatic link prisms
        housing_faces_new, _ = _prism_between(p1_new, p2_new, width=housing_w, height=housing_h)
        housing_poly.set_verts(housing_faces_new)

        slider_faces_new, _ = _prism_between(p2_new, p3_new, width=slider_w, height=slider_h)
        slider_poly.set_verts(slider_faces_new)

        # Update gripper (orientation from end-effector z-axis)
        z_dir_new = T[0:3, 2]
        R_ee_new = _basis_from_z(z_dir_new)
        x_new = R_ee_new[:, 0]
        y_new = R_ee_new[:, 1]

        palm_center_new = p3_new + _unit(z_dir_new) * (0.5 * palm_len)
        palm_faces_new = _prism_faces(center=palm_center_new, R=R_ee_new, width=palm_w, height=palm_h, length=palm_len)
        palm_poly.set_verts(palm_faces_new)

        finger_base_new = p3_new + _unit(z_dir_new) * (palm_len + 0.5 * finger_len)
        left_center_new = finger_base_new + 0.5 * gap * x_new + 0.5 * palm_h * y_new
        right_center_new = finger_base_new - 0.5 * gap * x_new + 0.5 * palm_h * y_new

        left_faces_new = _prism_faces(center=left_center_new, R=R_ee_new, width=finger_w, height=finger_h, length=finger_len)
        right_faces_new = _prism_faces(center=right_center_new, R=R_ee_new, width=finger_w, height=finger_h, length=finger_len)
        left_poly.set_verts(left_faces_new)
        right_poly.set_verts(right_faces_new)

        # Update skeleton line
        skel_line.set_data([p0_new[0], p1_new[0], p2_new[0], p3_new[0]], [p0_new[1], p1_new[1], p2_new[1], p3_new[1]])
        skel_line.set_3d_properties([p0_new[2], p1_new[2], p2_new[2], p3_new[2]])

        # Update shoulder dome position (stays at p1; orientation not needed)
        Xd_new, Yd_new, Zd_new = _hemisphere_surface(center=p1_new, radius=robot.dims.shoulder_dome_radius)
        nonlocal_vars["dome"].set_verts(_faces_from_grid(Xd_new, Yd_new, Zd_new))

        # Append current end-effector position to the path and update scatter
        path_x.append(p3_new[0])
        path_y.append(p3_new[1])
        path_z.append(p3_new[2])
        # Keep path reasonably short
        max_points = 1500
        if len(path_x) > max_points:
            del path_x[0 : len(path_x) - max_points]
            del path_y[0 : len(path_y) - max_points]
            del path_z[0 : len(path_z) - max_points]
        path_scatter._offsets3d = (path_x, path_y, path_z)

        fig.canvas.draw_idle()

    nonlocal_vars = {"cover_cap": cover_cap, "dome": dome}

    s_th1.on_changed(update)
    s_th2.on_changed(update)
    s_d3.on_changed(update)

    plt.show()


def auto_demo_viewer(robot: SphericalRobot) -> None:
    """
    Automatic motion demo: the robot moves itself through a simple
    periodic trajectory without sliders.
    """
    # Start from the center of each joint range
    th1_deg_init = 0.0
    th2_deg_init = 0.0
    d3_init = 0.5 * (robot.limits.d3[0] + robot.limits.d3[1])

    th1 = np.deg2rad(th1_deg_init)
    th2 = np.deg2rad(th2_deg_init)
    d3 = d3_init

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Leave space at the bottom for Pause/Resume buttons
    plt.subplots_adjust(left=0.08, bottom=0.18, right=0.98, top=0.96)

    p0, p1, p2, p3, T0_3 = robot.forward_kinematics(th1, th2, d3)

    # Static base geometry
    base_x, base_y, base_z = robot.dims.base_plate_size
    base_center = np.array([0.0, 0.0, -0.5 * base_z])
    base_R = np.eye(3)
    base_faces = _prism_faces(base_center, base_R, width=base_x, height=base_y, length=base_z)
    _add_poly(ax, base_faces, color="#c9c9c9", alpha=0.9, edgecolor="#555555", linewidth=0.6)

    tt_center = np.array([0.0, 0.0, 0.5 * robot.dims.turntable_thickness])
    tt_faces = _prism_faces(
        center=tt_center,
        R=np.eye(3),
        width=2.0 * robot.dims.turntable_radius,
        height=2.0 * robot.dims.turntable_radius,
        length=robot.dims.turntable_thickness,
    )
    _add_poly(ax, tt_faces, color="#9e9e9e", alpha=0.95, edgecolor="#444444", linewidth=0.6)

    (tt_indicator,) = ax.plot(
        [0.0, robot.dims.turntable_radius],
        [0.0, 0.0],
        [robot.dims.turntable_thickness, robot.dims.turntable_thickness],
        color="#2c3e50",
        linewidth=2.5,
        alpha=0.9,
        label="Base yaw (θ1)",
    )

    col_center = np.array([0.0, 0.0, 0.5 * robot.dims.base_height])
    col_faces = _prism_faces(
        center=col_center,
        R=np.eye(3),
        width=2.0 * robot.dims.column_radius,
        height=2.0 * robot.dims.column_radius,
        length=robot.dims.base_height,
    )
    _add_poly(ax, col_faces, color="#b0b0b0", alpha=0.9, edgecolor="#555555", linewidth=0.6)

    # Cover and dome (same as interactive_viewer)
    cov_L, cov_D, cov_H = robot.dims.cover_size
    cov_r = robot.dims.cover_top_radius
    cov_base_z = robot.dims.turntable_thickness
    cov_body_h = max(cov_H - cov_r, 0.05)

    def cover_geometry(theta1_rad: float):
        Rz = _rotz(theta1_rad)
        body_center = np.array([0.0, 0.0, cov_base_z + 0.5 * cov_body_h])
        body_faces = _prism_faces(center=body_center, R=Rz, width=cov_L, height=cov_D, length=cov_body_h)
        cap_center = np.array([0.0, 0.0, cov_base_z + cov_body_h])
        Xc, Yc, Zc = _half_cylinder_surface(center=cap_center, R=Rz, radius=cov_r, length=cov_L)
        return body_faces, (Xc, Yc, Zc)

    cover_body_faces, (Xc, Yc, Zc) = cover_geometry(th1)
    cover_poly = _add_poly(ax, cover_body_faces, color="#d0d0d0", alpha=0.65, edgecolor="#666666", linewidth=0.5)
    cover_cap_faces = _faces_from_grid(Xc, Yc, Zc)
    cover_cap = _add_poly(ax, cover_cap_faces, color="#d0d0d0", alpha=0.65, edgecolor="#666666", linewidth=0.2)

    Xd, Yd, Zd = _hemisphere_surface(center=p1, radius=robot.dims.shoulder_dome_radius)
    dome_faces = _faces_from_grid(Xd, Yd, Zd)
    dome = _add_poly(ax, dome_faces, color="#bbbbbb", alpha=0.9, edgecolor="#666666", linewidth=0.15)

    housing_w, housing_h = robot.dims.housing_thickness
    slider_w, slider_h = robot.dims.slider_thickness

    housing_faces, _ = _prism_between(p1, p2, width=housing_w, height=housing_h)
    housing_poly = _add_poly(ax, housing_faces, color="#2ecc71", alpha=0.85)

    slider_faces, _ = _prism_between(p2, p3, width=slider_w, height=slider_h)
    slider_poly = _add_poly(ax, slider_faces, color="#8e44ad", alpha=0.85)

    z_dir = T0_3[0:3, 2]
    R_ee = _basis_from_z(z_dir)
    x_dir = R_ee[:, 0]
    y_dir = R_ee[:, 1]

    palm_len = robot.dims.gripper_palm_length
    palm_w, palm_h = robot.dims.gripper_palm_thickness
    palm_center = p3 + _unit(z_dir) * (0.5 * palm_len)
    palm_faces = _prism_faces(center=palm_center, R=R_ee, width=palm_w, height=palm_h, length=palm_len)
    palm_poly = _add_poly(ax, palm_faces, color="#e74c3c", alpha=0.9)

    finger_len = robot.dims.gripper_finger_length
    finger_w, finger_h = robot.dims.gripper_finger_thickness
    gap = robot.dims.gripper_finger_gap

    finger_base = p3 + _unit(z_dir) * (palm_len + 0.5 * finger_len)
    left_center = finger_base + 0.5 * gap * x_dir + 0.5 * palm_h * y_dir
    right_center = finger_base - 0.5 * gap * x_dir + 0.5 * palm_h * y_dir

    left_faces = _prism_faces(center=left_center, R=R_ee, width=finger_w, height=finger_h, length=finger_len)
    right_faces = _prism_faces(center=right_center, R=R_ee, width=finger_w, height=finger_h, length=finger_len)
    left_poly = _add_poly(ax, left_faces, color="#d35400", alpha=0.95)
    right_poly = _add_poly(ax, right_faces, color="#d35400", alpha=0.95)

    (skel_line,) = ax.plot(
        [p0[0], p1[0], p2[0], p3[0]],
        [p0[1], p1[1], p2[1], p3[1]],
        [p0[2], p1[2], p2[2], p3[2]],
        color="#1f1f1f",
        linewidth=1.2,
        alpha=0.35,
        label="Kinematic chain",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Automatic Spherical Manipulator Demo")
    max_reach = (
        robot.dims.base_height
        + robot.limits.d3[1]
        + robot.dims.gripper_palm_length
        + robot.dims.gripper_finger_length
    )
    ax.set_xlim(-max_reach, max_reach)
    ax.set_ylim(-max_reach, max_reach)
    ax.set_zlim(0.0, max_reach * 1.2)
    ax.set_box_aspect([1, 1, 1])

    # End-effector path in automatic mode (only path, no static cloud)
    path_x = [p3[0]]
    path_y = [p3[1]]
    path_z = [p3[2]]
    path_scatter = ax.scatter(
        path_x,
        path_y,
        path_z,
        s=18,
        alpha=0.9,
        c="#00bcd4",
        edgecolors="none",
        label="End-effector workspace",
    )

    ax.legend(loc="upper right")

    nonlocal_vars = {"cover_cap": cover_cap, "dome": dome}

    # Simple pause / resume for automatic modes: key or buttons
    paused = False

    def toggle_pause(_event=None):
        nonlocal paused
        paused = not paused

    def resume(_event=None):
        nonlocal paused
        paused = False

    def on_key(event):
        if event.key in ("p", " "):
            toggle_pause()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Buttons: Pause/Continue at the bottom
    ax_pause = fig.add_axes([0.25, 0.05, 0.18, 0.06])
    ax_resume = fig.add_axes([0.55, 0.05, 0.18, 0.06])
    btn_pause = Button(ax_pause, "Pause", color="#eeeeee", hovercolor="#dddddd")
    btn_resume = Button(ax_resume, "Continue", color="#eeeeee", hovercolor="#dddddd")
    btn_pause.on_clicked(toggle_pause)
    btn_resume.on_clicked(resume)

    def update_state(th1_rad: float, th2_rad: float, d3_val: float) -> None:
        p0_new, p1_new, p2_new, p3_new, T = robot.forward_kinematics(th1_rad, th2_rad, d3_val)

        # Turntable indicator
        tip = _rotz(th1_rad) @ np.array([robot.dims.turntable_radius, 0.0, 0.0])
        tt_indicator.set_data([0.0, tip[0]], [0.0, tip[1]])
        tt_indicator.set_3d_properties([robot.dims.turntable_thickness, robot.dims.turntable_thickness])

        # Cover
        cover_faces_new, (Xc_new, Yc_new, Zc_new) = cover_geometry(th1_rad)
        cover_poly.set_verts(cover_faces_new)
        nonlocal_vars["cover_cap"].set_verts(_faces_from_grid(Xc_new, Yc_new, Zc_new))

        # Prismatic link prisms
        housing_faces_new, _ = _prism_between(p1_new, p2_new, width=housing_w, height=housing_h)
        housing_poly.set_verts(housing_faces_new)

        slider_faces_new, _ = _prism_between(p2_new, p3_new, width=slider_w, height=slider_h)
        slider_poly.set_verts(slider_faces_new)

        # Gripper
        z_dir_new = T[0:3, 2]
        R_ee_new = _basis_from_z(z_dir_new)
        x_new = R_ee_new[:, 0]
        y_new = R_ee_new[:, 1]

        palm_center_new = p3_new + _unit(z_dir_new) * (0.5 * palm_len)
        palm_faces_new = _prism_faces(center=palm_center_new, R=R_ee_new, width=palm_w, height=palm_h, length=palm_len)
        palm_poly.set_verts(palm_faces_new)

        finger_base_new = p3_new + _unit(z_dir_new) * (palm_len + 0.5 * finger_len)
        left_center_new = finger_base_new + 0.5 * gap * x_new + 0.5 * palm_h * y_new
        right_center_new = finger_base_new - 0.5 * gap * x_new + 0.5 * palm_h * y_new

        left_faces_new = _prism_faces(center=left_center_new, R=R_ee_new, width=finger_w, height=finger_h, length=finger_len)
        right_faces_new = _prism_faces(center=right_center_new, R=R_ee_new, width=finger_w, height=finger_h, length=finger_len)
        left_poly.set_verts(left_faces_new)
        right_poly.set_verts(right_faces_new)

        skel_line.set_data(
            [p0_new[0], p1_new[0], p2_new[0], p3_new[0]],
            [p0_new[1], p1_new[1], p2_new[1], p3_new[1]],
        )
        skel_line.set_3d_properties([p0_new[2], p1_new[2], p2_new[2], p3_new[2]])

        Xd_new, Yd_new, Zd_new = _hemisphere_surface(center=p1_new, radius=robot.dims.shoulder_dome_radius)
        nonlocal_vars["dome"].set_verts(_faces_from_grid(Xd_new, Yd_new, Zd_new))

        # Update path (limit number of points to keep UI responsive)
        path_x.append(p3_new[0])
        path_y.append(p3_new[1])
        path_z.append(p3_new[2])
        max_points = 1500
        if len(path_x) > max_points:
            del path_x[0 : len(path_x) - max_points]
            del path_y[0 : len(path_y) - max_points]
            del path_z[0 : len(path_z) - max_points]
        path_scatter._offsets3d = (path_x, path_y, path_z)

        fig.canvas.draw_idle()

    print("\nAutomatic motion pattern:")
    print("  1) Spherical shell (θ1 & θ2, fixed d3)  [nice clear sphere]")
    print("  2) 3-DOF workspace exploration (all joints move)")
    print("  3) Single-joint motion (choose θ1, θ2 or d3)")
    pattern = input("Enter 1, 2 or 3 [default 1]: ").strip()

    # ----------------------------
    # Pattern 1: clear spherical shell
    # ----------------------------
    if pattern == "1" or pattern == "":
        # Moderate sampling for smooth performance
        n_th1 = 80   # longitude samples
        n_th2 = 40   # latitude samples

        th1_vals = np.deg2rad(np.linspace(robot.limits.theta1_deg[0], robot.limits.theta1_deg[1], n_th1))
        th2_vals = np.deg2rad(np.linspace(robot.limits.theta2_deg[0], robot.limits.theta2_deg[1], n_th2))
        d3_shell = robot.limits.d3[1]

        path_states = []
        dir_th1 = 1
        for th2_val in th2_vals:
            th1_iter = th1_vals if dir_th1 == 1 else th1_vals[::-1]
            for th1_val in th1_iter:
                path_states.append((th1_val, th2_val, d3_shell))
            dir_th1 *= -1

        dt = 0.03
        idx = 0
        total = len(path_states)

        while plt.fignum_exists(fig.number):
            if paused:
                plt.pause(dt)
                continue
            th1_rad, th2_rad, d3_val = path_states[idx]
            update_state(th1_rad, th2_rad, d3_val)
            plt.pause(dt)
            idx = (idx + 1) % total

        return

    # ----------------------------
    # Patterns 2 and 3: joint-space exploration with bouncing
    # ----------------------------
    th1_min = np.deg2rad(robot.limits.theta1_deg[0])
    th1_max = np.deg2rad(robot.limits.theta1_deg[1])
    th2_min = np.deg2rad(robot.limits.theta2_deg[0])
    th2_max = np.deg2rad(robot.limits.theta2_deg[1])

    d3_min = robot.limits.d3[0]
    d3_max = robot.limits.d3[1]

    # Start near the center of each range
    th1 = 0.5 * (th1_min + th1_max)
    th2 = 0.5 * (th2_min + th2_max)
    d3_val = 0.5 * (d3_min + d3_max)

    # Base velocities (will be adjusted for pattern 3)
    w1 = 0.7
    w2 = 0.9
    v3 = 0.2

    if pattern == "3":
        print("\nSingle-joint automatic motion:")
        print("  1) Move θ1 only (base yaw)")
        print("  2) Move θ2 only (shoulder elevation)")
        print("  3) Move d3 only (prismatic extension)")
        j_choice = input("Enter 1, 2 or 3 [default 1]: ").strip()

        if j_choice == "2":
            w1 = 0.0
            v3 = 0.0
        elif j_choice == "3":
            w1 = 0.0
            w2 = 0.0
        else:
            # default: θ1 only
            w2 = 0.0
            v3 = 0.0

    dt = 0.03

    while plt.fignum_exists(fig.number):
        if paused:
            plt.pause(dt)
            continue
        # Update joints
        th1 += w1 * dt
        th2 += w2 * dt
        d3_val += v3 * dt

        # Reflect at limits so motion stays bounded but continuous
        if th1 < th1_min or th1 > th1_max:
            w1 *= -1.0
            th1 = float(np.clip(th1, th1_min, th1_max))

        if th2 < th2_min or th2 > th2_max:
            w2 *= -1.0
            th2 = float(np.clip(th2, th2_min, th2_max))

        if d3_val < d3_min or d3_val > d3_max:
            v3 *= -1.0
            d3_val = float(np.clip(d3_val, d3_min, d3_max))

        update_state(th1, th2, d3_val)
        plt.pause(dt)


def _build_custom_robot_from_input() -> SphericalRobot:
    """
    Create a SphericalRobot from simple user inputs.

    All inputs are optional; pressing Enter keeps the default value.
    """
    default_dims = RobotDimensions()
    default_limits = JointLimits()

    def ask_float(prompt: str, default: float) -> float:
        text = input(f"{prompt} [{default}]: ").strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            print("  Invalid number, keeping default.")
            return default

    print("\nConfigure robot dimensions (meters). Press Enter to keep defaults.")
    base_height = ask_float("Base height d1", default_dims.base_height)
    housing_len = ask_float("Prismatic housing length (visual)", default_dims.prismatic_housing_length)

    print("\nConfigure joint limits.")
    th1_min = ask_float("theta1 min [deg]", default_limits.theta1_deg[0])
    th1_max = ask_float("theta1 max [deg]", default_limits.theta1_deg[1])
    th2_min = ask_float("theta2 min [deg]", default_limits.theta2_deg[0])
    th2_max = ask_float("theta2 max [deg]", default_limits.theta2_deg[1])
    d3_min = ask_float("d3 min [m]", default_limits.d3[0])
    d3_max = ask_float("d3 max [m]", default_limits.d3[1])

    dims = RobotDimensions(
        base_height=base_height,
        prismatic_housing_length=housing_len,
        base_plate_size=default_dims.base_plate_size,
        turntable_radius=default_dims.turntable_radius,
        turntable_thickness=default_dims.turntable_thickness,
        column_radius=default_dims.column_radius,
        cover_size=default_dims.cover_size,
        cover_top_radius=default_dims.cover_top_radius,
        shoulder_dome_radius=default_dims.shoulder_dome_radius,
        housing_thickness=default_dims.housing_thickness,
        slider_thickness=default_dims.slider_thickness,
        gripper_palm_length=default_dims.gripper_palm_length,
        gripper_palm_thickness=default_dims.gripper_palm_thickness,
        gripper_finger_length=default_dims.gripper_finger_length,
        gripper_finger_thickness=default_dims.gripper_finger_thickness,
        gripper_finger_gap=default_dims.gripper_finger_gap,
    )

    limits = JointLimits(
        theta1_deg=(th1_min, th1_max),
        theta2_deg=(th2_min, th2_max),
        d3=(d3_min, d3_max),
    )

    return SphericalRobot(dims=dims, limits=limits)


def main() -> None:
    print("Robot configuration:")
    print("  1) Use default dimensions")
    print("  2) Customize dimensions and joint limits")
    cfg_choice = input("Enter 1 or 2 [default 1]: ").strip()

    if cfg_choice == "2":
        robot = _build_custom_robot_from_input()
    else:
        robot = SphericalRobot()

    print("\nSelect mode:")
    print("  1) Automatic demo motion")
    print("  2) Manual control with sliders")
    choice = input("Enter 1 or 2 [default 2]: ").strip()

    if choice == "1":
        auto_demo_viewer(robot)
    else:
        interactive_viewer(robot)


if __name__ == "__main__":
    main()

