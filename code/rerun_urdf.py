"""
Utility functions for working with URDF files in Rerun visualization.

Uses rerun.urdf.UrdfTree (rerun >= 0.29) which exposes joint frames for
animation via rr.log(...). The transforms returned by
UrdfJoint.compute_transform() carry parent_frame / child_frame fields, so
rerun's TF system wires the kinematic chain regardless of entity path.
"""

from pathlib import Path
from typing import Dict, Tuple

import rerun as rr
from rerun.urdf import UrdfJoint, UrdfTree


def init_rerun_with_urdf(
    app_id: str, urdf_path: Path, spawn: bool = True
) -> Tuple[str, Dict[str, UrdfJoint]]:
    """
    Initialize Rerun and load a URDF file with animatable joint frames.

    Returns
    -------
    (robot_name, joints_by_name) where joints_by_name maps joint name -> UrdfJoint.
    Pass joints_by_name straight through to log_joint_angles each frame.
    """
    rr.init(app_id, spawn=spawn)
    tree = UrdfTree.from_file_path(str(urdf_path))
    tree.log_urdf_to_recording()
    joints_by_name = {j.name: j for j in tree.joints()}
    return tree.name, joints_by_name


def log_joint_angles(
    robot_name: str,
    joints_by_name: Dict[str, UrdfJoint],
    joint_names: list,
    joint_angles: list,
    clamp: bool = True,
) -> None:
    """
    Log joint angles to drive the URDF in the rerun viewer.

    The Transform3D carries the joint's parent_frame/child_frame, so the
    entity path here is just for organization in the viewer. Pass clamp=False
    to draw angles past the URDF limits: the real arm rests 17.8 degrees past
    Pitch's -180 (so101/urdf_map.json).
    """
    for name, angle in zip(joint_names, joint_angles):
        joint = joints_by_name.get(name)
        if joint is None or joint.joint_type == "fixed":
            continue
        transform = joint.compute_transform(float(angle), clamp=clamp)
        rr.log(f"{robot_name}/joint_transforms/{name}", transform)
