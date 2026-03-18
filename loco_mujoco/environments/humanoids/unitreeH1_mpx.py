from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import mujoco
from mujoco import MjSpec

from loco_mujoco.core import ObservationType, Observation
from .unitreeH1 import UnitreeH1


class MpxUnitreeH1(UnitreeH1):
    """
    Unitree H1 environment using the MJX API with the MjWarp backend (aka "MPX" here).

    This variant uses a local XML shipped in this repo:
    `loco_mujoco/xml_models/Unitree_H1/mjx_h1_walk_real_feet.xml`.
    That XML already contains simplified foot collision geometry, so we do not apply the
    additional MJX-specific spec modifications from `MjxUnitreeH1`.
    """

    mjx_enabled = True

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 5,
        use_mjwarp: bool = True,
        nconmax: Optional[int] = None,
        njmax: Optional[int] = None,
        **kwargs,
    ) -> None:
        if "model_option_conf" not in kwargs:
            model_option_conf = dict(
                iterations=2,
                ls_iterations=4,
                disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
            )
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            model_option_conf=model_option_conf,
            use_mjwarp=use_mjwarp,
            nconmax=nconmax,
            njmax=njmax,
            **kwargs,
        )

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        loco_root = Path(__file__).resolve().parents[2]  # .../loco_mujoco
        return str(loco_root / "xml_models" / "Unitree_H1" / "mjx_h1_walk_real_feet.xml")

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        # The referenced XML already includes simplified foot collision geometry and contact setup.
        return spec

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Observation spec compatible with `mjx_h1_walk_real_feet.xml`.

        The base `UnitreeH1` env expects a different joint naming convention (e.g. `back_bkz`, `l_arm_shy`, ...).
        This custom XML uses the joint names:
        - torso (instead of back_bkz)
        - left/right_shoulder_{pitch,roll,yaw} (instead of l/r_arm_sh{y,x,z})
        - left/right_{hip_yaw,hip_roll,hip_pitch,knee,ankle} (instead of hip_flexion/adduction/rotation, ...)
        """

        return [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),

            ObservationType.JointPos("q_back_bkz", xml_name="torso"),

            ObservationType.JointPos("q_l_arm_shy", xml_name="left_shoulder_pitch"),
            ObservationType.JointPos("q_l_arm_shx", xml_name="left_shoulder_roll"),
            ObservationType.JointPos("q_l_arm_shz", xml_name="left_shoulder_yaw"),
            ObservationType.JointPos("q_left_elbow", xml_name="left_elbow"),

            ObservationType.JointPos("q_r_arm_shy", xml_name="right_shoulder_pitch"),
            ObservationType.JointPos("q_r_arm_shx", xml_name="right_shoulder_roll"),
            ObservationType.JointPos("q_r_arm_shz", xml_name="right_shoulder_yaw"),
            ObservationType.JointPos("q_right_elbow", xml_name="right_elbow"),

            # Right leg (keep original obs names for compatibility)
            ObservationType.JointPos("q_hip_flexion_r", xml_name="right_hip_pitch"),
            ObservationType.JointPos("q_hip_adduction_r", xml_name="right_hip_roll"),
            ObservationType.JointPos("q_hip_rotation_r", xml_name="right_hip_yaw"),
            ObservationType.JointPos("q_knee_angle_r", xml_name="right_knee"),
            ObservationType.JointPos("q_ankle_angle_r", xml_name="right_ankle"),

            # Left leg
            ObservationType.JointPos("q_hip_flexion_l", xml_name="left_hip_pitch"),
            ObservationType.JointPos("q_hip_adduction_l", xml_name="left_hip_roll"),
            ObservationType.JointPos("q_hip_rotation_l", xml_name="left_hip_yaw"),
            ObservationType.JointPos("q_knee_angle_l", xml_name="left_knee"),
            ObservationType.JointPos("q_ankle_angle_l", xml_name="left_ankle"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),

            ObservationType.JointVel("dq_back_bkz", xml_name="torso"),

            ObservationType.JointVel("dq_l_arm_shy", xml_name="left_shoulder_pitch"),
            ObservationType.JointVel("dq_l_arm_shx", xml_name="left_shoulder_roll"),
            ObservationType.JointVel("dq_l_arm_shz", xml_name="left_shoulder_yaw"),
            ObservationType.JointVel("dq_left_elbow", xml_name="left_elbow"),

            ObservationType.JointVel("dq_r_arm_shy", xml_name="right_shoulder_pitch"),
            ObservationType.JointVel("dq_r_arm_shx", xml_name="right_shoulder_roll"),
            ObservationType.JointVel("dq_r_arm_shz", xml_name="right_shoulder_yaw"),
            ObservationType.JointVel("dq_right_elbow", xml_name="right_elbow"),

            ObservationType.JointVel("dq_hip_flexion_r", xml_name="right_hip_pitch"),
            ObservationType.JointVel("dq_hip_adduction_r", xml_name="right_hip_roll"),
            ObservationType.JointVel("dq_hip_rotation_r", xml_name="right_hip_yaw"),
            ObservationType.JointVel("dq_knee_angle_r", xml_name="right_knee"),
            ObservationType.JointVel("dq_ankle_angle_r", xml_name="right_ankle"),

            ObservationType.JointVel("dq_hip_flexion_l", xml_name="left_hip_pitch"),
            ObservationType.JointVel("dq_hip_adduction_l", xml_name="left_hip_roll"),
            ObservationType.JointVel("dq_hip_rotation_l", xml_name="left_hip_yaw"),
            ObservationType.JointVel("dq_knee_angle_l", xml_name="left_knee"),
            ObservationType.JointVel("dq_ankle_angle_l", xml_name="left_ankle"),
        ]

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Action spec compatible with `mjx_h1_walk_real_feet.xml`.

        In this XML, actuators are motors named after their joint names (e.g. `torso`, `left_hip_yaw`, ...).
        """
        return [
            "torso",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
            "right_hip_pitch",
            "right_hip_roll",
            "right_hip_yaw",
            "right_knee",
            "right_ankle",
            "left_hip_pitch",
            "left_hip_roll",
            "left_hip_yaw",
            "left_knee",
            "left_ankle",
        ]
