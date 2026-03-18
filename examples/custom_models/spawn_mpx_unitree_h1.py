from __future__ import annotations

import argparse

import numpy as np
import mujoco
import jax
import jax.numpy as jnp

from loco_mujoco import LocoEnv
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn and render the custom Unitree H1 (MPX) model by creating a "
            "standing-still trajectory from the initial state and replaying it."
        )
    )
    parser.add_argument("--steps", type=int, default=5000, help="Number of simulation steps to run.")
    parser.add_argument(
        "--mjwarp",
        action="store_true",
        help="Enable MJX warp backend (experimental). If omitted, uses standard MJX backend.",
    )
    parser.add_argument("--nconmax", type=int, default=None, help="Warp-only: max contacts buffer.")
    parser.add_argument("--njmax", type=int, default=None, help="Warp-only: max constraints buffer.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the generated standing trajectory (.npz).",
    )
    args = parser.parse_args()

    # Create environment via registry (same pattern as standing_humanoid.py).
    env = LocoEnv.make(
        "MpxUnitreeH1",
        init_state_type="DefaultInitialStateHandler",
        terrain_type="FlatFloorTerrain",
        use_mjwarp=bool(args.mjwarp),
        nconmax=args.nconmax,
        njmax=args.njmax,
    )

    # Reset and grab initial state.
    key = jax.random.PRNGKey(0)
    env.reset(key)
    model = env.get_model()
    data = env.get_data()

    qpos0 = np.array(data.qpos)
    qvel0 = np.array(data.qvel)

    # torques ["back_bkz_actuator", "l_arm_shy_actuator", "l_arm_shx_actuator", "l_arm_shz_actuator", "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator", "r_arm_shz_actuator", "right_elbow_actuator", "hip_flexion_r_actuator", "hip_adduction_r_actuator", "hip_rotation_r_actuator", "knee_angle_r_actuator", "ankle_angle_r_actuator", "hip_flexion_l_actuator", "hip_adduction_l_actuator", "hip_rotation_l_actuator", "knee_angle_l_actuator", "ankle_angle_l_actuator"]
    torque = np.array(data.actuator_force) # size [18]
   

    # Tile into a standing trajectory.
    n_steps = int(args.steps)
    qpos = np.tile(qpos0, (n_steps, 1))
    qvel = np.tile(qvel0, (n_steps, 1))

    njnt = model.njnt
    jnt_type = model.jnt_type
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
    traj_info = TrajectoryInfo(
        jnt_names,
        model=TrajectoryModel(njnt, jnp.array(jnt_type)),
        frequency=1 / env.dt,
    )
    traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, n_steps]))
    traj = Trajectory(traj_info, traj_data)

    if args.out is not None:
        traj.save(args.out)

    # Load and replay.
    env.load_trajectory(traj)
    env.play_trajectory(n_steps_per_episode=n_steps)


if __name__ == "__main__":
    main()

