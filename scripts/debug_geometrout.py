#!/usr/bin/env python3
"""
Hands-on debug playground for geometrout.

Usage examples:
  python scripts/debug_geometrout.py --all
  python scripts/debug_geometrout.py --section so3 --pause
  python scripts/debug_geometrout.py --section se3 --section pointcloud --seed 0

Tips:
- Run with --pause to stop at key checkpoints via breakpoint().
- In VS Code, launch this file in Debug mode and inspect local variables.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List

import numpy as np

from geometrout.pointcloud import project, transform as transform_pointcloud
from geometrout.primitive import Cuboid, Cylinder, Sphere
from geometrout.transform import SE3, SO3


def pause_here(enabled: bool, title: str) -> None:
    if not enabled:
        return
    print(f"\n[PAUSE] {title}")
    breakpoint()


def make_demo_primitives() -> tuple[Sphere, Cuboid, Cylinder]:
    sphere = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0)
    cuboid = Cuboid(
        center=np.array([0.5, 0.0, 0.0]),
        dims=np.array([1.0, 2.0, 0.5]),
        quaternion=SO3.from_rpy(0.0, np.pi / 6, 0.0).q,
    )
    cylinder = Cylinder(
        center=np.array([-0.5, 0.0, 0.0]),
        radius=0.4,
        height=1.2,
        quaternion=SO3.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 8).q,
    )
    return sphere, cuboid, cylinder


def section_so3(pause: bool) -> None:
    print("\n=== [SO3] Rotation fundamentals ===")
    r0 = SO3.unit()
    r1 = SO3.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)

    print("r0 quaternion (wxyz):", r0.wxyz)
    print("r1 quaternion (wxyz):", np.round(r1.wxyz, 6).tolist())
    print("r1 matrix:\n", np.round(r1.matrix, 6))
    pause_here(pause, "SO3 created: inspect r0/r1")

    r_mid = SO3.interpolate(r0, r1, 0.5)
    print("interpolate(r0, r1, t=0.5) radians:", float(r_mid.radians))
    print("interpolate(r0, r1, t=0.5) rpy:", np.round(r_mid.rpy, 6))
    pause_here(pause, "SO3 interpolation complete")

    mat = r1.matrix
    r_from_mat = SO3.from_matrix(mat)
    print("from_matrix -> quaternion (wxyz):", np.round(r_from_mat.wxyz, 6).tolist())
    print("inverse consistency ||R*R^-1 - I||:", float(np.linalg.norm((r1 * r1.inverse).matrix - np.eye(3))))


def section_se3(pause: bool) -> None:
    print("\n=== [SE3] Pose composition and inversion ===")
    p0 = SE3(np.array([1.0, 0.0, 0.0]), SO3.unit().q)
    p1 = SE3(np.array([0.0, 2.0, 0.0]), SO3.from_rpy(0.0, 0.0, np.pi / 2).q)

    p01 = p0 * p1
    p10 = p1 * p0

    print("p0 matrix:\n", np.round(p0.matrix, 6))
    print("p1 matrix:\n", np.round(p1.matrix, 6))
    print("p0 * p1 xyz:", np.round(p01.xyz, 6))
    print("p1 * p0 xyz:", np.round(p10.xyz, 6))
    pause_here(pause, "SE3 composition ready")

    eye_approx = p01.matrix @ p01.inverse.matrix
    print("||T*T^-1 - I||:", float(np.linalg.norm(eye_approx - np.eye(4))))

    p_half = SE3.interpolate(p0, p1, 0.5)
    print("interpolate(p0, p1, 0.5) xyz:", np.round(p_half.xyz, 6))
    print("interpolate(p0, p1, 0.5) quat(wxyz):", np.round(p_half.quaternion, 6).tolist())


def section_pointcloud(pause: bool) -> None:
    print("\n=== [PointCloud] Project and transform ===")
    pose = SE3(np.array([0.5, -0.2, 0.3]), SO3.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 4).q)

    point = np.array([1.0, 0.0, 0.0])
    point_full = project(pose.matrix, point)
    point_rot = project(pose.matrix, point, rotate_only=True)

    print("point:", point)
    print("project full transform:", np.round(point_full, 6))
    print("project rotation only:", np.round(point_rot, 6))

    pc = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pc_out = transform_pointcloud(pc.copy(), pose.matrix, in_place=False)
    print("pc transformed (first 2 rows):\n", np.round(pc_out[:2], 6))
    pause_here(pause, "Point cloud transformed")


def section_primitives(pause: bool) -> None:
    print("\n=== [Primitives] SDF and sampling ===")
    sphere, cuboid, cylinder = make_demo_primitives()

    query = np.array([0.2, 0.1, 0.0])
    print("query point:", query)
    print("sphere sdf:", float(sphere.sdf(query)))
    print("cuboid sdf:", float(cuboid.sdf(query)))
    print("cylinder sdf:", float(cylinder.sdf(query)))

    surf = sphere.sample_surface(8)
    vol = cuboid.sample_volume(8)
    cyl = cylinder.sample_surface(8)
    print("sphere.sample_surface shape:", surf.shape)
    print("cuboid.sample_volume shape:", vol.shape)
    print("cylinder.sample_surface shape:", cyl.shape)
    pause_here(pause, "Primitive sampling done")


def section_pybullet(mode: str, seconds: float) -> None:
    print("\n=== [PyBullet] Primitive visualization ===")
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        raise RuntimeError(
            "PyBullet is not installed. Run: conda run -n robot pip install pybullet"
        ) from exc

    if mode not in {"gui", "direct"}:
        raise ValueError("pybullet mode must be one of {'gui', 'direct'}")

    sphere, cuboid, cylinder = make_demo_primitives()

    connection_mode = p.GUI if mode == "gui" else p.DIRECT
    cid = p.connect(connection_mode)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # Sphere
        sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=float(sphere.radius))
        sphere_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=float(sphere.radius),
            rgbaColor=[0.9, 0.2, 0.2, 0.75],
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=sphere_shape,
            baseVisualShapeIndex=sphere_visual,
            basePosition=sphere.center.tolist(),
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )

        # Cuboid
        half_extents = (cuboid.dims / 2.0).tolist()
        cuboid_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        cuboid_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.2, 0.4, 0.9, 0.65],
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=cuboid_shape,
            baseVisualShapeIndex=cuboid_visual,
            basePosition=cuboid.center.tolist(),
            baseOrientation=cuboid.pose.so3.xyzw,
        )

        # Cylinder (aligned with local z-axis in pybullet)
        cylinder_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=float(cylinder.radius),
            height=float(cylinder.height),
        )
        cylinder_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=float(cylinder.radius),
            length=float(cylinder.height),
            rgbaColor=[0.2, 0.8, 0.4, 0.65],
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=cylinder_shape,
            baseVisualShapeIndex=cylinder_visual,
            basePosition=cylinder.center.tolist(),
            baseOrientation=cylinder.pose.so3.xyzw,
        )

        if mode == "gui":
            p.resetDebugVisualizerCamera(
                cameraDistance=4.0,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=[0.0, 0.0, 0.0],
            )
            print(f"GUI opened. Running for {seconds:.1f}s...")
        else:
            print(f"DIRECT mode running for {seconds:.1f}s (no GUI window).")

        start = time.time()
        while time.time() - start < seconds:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        if p.isConnected(cid):
            p.disconnect(cid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug playground for geometrout")
    parser.add_argument(
        "--section",
        action="append",
        choices=["so3", "se3", "pointcloud", "primitives", "pybullet"],
        help="Run only selected section(s). Can be repeated.",
    )
    parser.add_argument("--all", action="store_true", help="Run all sections.")
    parser.add_argument("--pause", action="store_true", help="Trigger breakpoint() at checkpoints.")
    parser.add_argument(
        "--pybullet-mode",
        choices=["gui", "direct"],
        default="gui",
        help="PyBullet backend. Use 'direct' on headless servers.",
    )
    parser.add_argument(
        "--pybullet-seconds",
        type=float,
        default=20.0,
        help="How long to keep PyBullet simulation running.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def resolve_sections(selected: List[str] | None, run_all: bool) -> Iterable[str]:
    default = ["so3", "se3", "pointcloud", "primitives", "pybullet"]
    if run_all or not selected:
        return default
    return selected


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    sections = resolve_sections(args.section, args.all)
    handlers = {
        "so3": lambda: section_so3(args.pause),
        "se3": lambda: section_se3(args.pause),
        "pointcloud": lambda: section_pointcloud(args.pause),
        "primitives": lambda: section_primitives(args.pause),
        "pybullet": lambda: section_pybullet(args.pybullet_mode, args.pybullet_seconds),
    }

    print("Running sections:", list(sections))
    for name in sections:
        handlers[name]()

    print("\nDone. Re-run with --pause to inspect internals step by step.")


if __name__ == "__main__":
    main()
