"""Trajectory-tracking mission runner for controller/estimator comparisons."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

from massaware.classify import classify_mass
from massaware.config import load_config
from massaware.controllers.attachment import TrackingAttachment
from massaware.controllers.ik import make_tracking_ik
from massaware.controllers.profiles import (
    TrackingProfile,
    controller_gains,
    tracking_profile,
)
from massaware.controllers import (
    InverseDynamicsTrackingController,
    JointReference,
    TrackingPIDController,
)
from massaware.controllers.trajectory import (
    JointTrajectory,
    make_pick_weigh_trajectory,
    make_release_trajectory,
)
from massaware.estimators import build as build_estimator
from massaware.estimators import inverse_dynamics as _inverse_dynamics  # noqa: F401
from massaware.estimators import lyapunov as _lyapunov  # noqa: F401
from massaware.estimators import momentum_observer as _momentum_observer  # noqa: F401
from massaware.estimators import pid_error as _pid_error  # noqa: F401
from massaware.estimators.base import Estimator, EstimatorObs
from massaware.mujoco_env import MujocoEnv
from massaware.robot import Robot
from massaware.tick_loop import Gripper

CUBE_BODY = "cube"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controller", choices=["pid_tracking", "inverse_dynamics"], default="pid_tracking")
    parser.add_argument("--estimator", choices=["pid_error", "lyapunov", "momentum_observer", "inverse_dynamics"], default="lyapunov")
    parser.add_argument("--profile", choices=["tracking", "main"], default="tracking")
    parser.add_argument("--mass", type=float, default=0.5)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--move-to-grasp-time", type=float, default=3.0)
    parser.add_argument("--lift-to-weigh-time", type=float, default=2.0)
    parser.add_argument("--move-to-release-time", type=float, default=None)
    parser.add_argument("--weigh-time", type=float, default=None)
    parser.add_argument("--calibration-time", type=float, default=3.0)
    parser.add_argument("--start-delay", type=float, default=0.0)
    parser.add_argument(
        "--disable-controller-overrides",
        action="store_true",
        help="ignore estimator-requested gain overrides during weighing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    profile = tracking_profile(args.profile, cfg)
    env = MujocoEnv()
    env.set_body_mass(CUBE_BODY, args.mass)
    env.reset(arm_qpos=cfg["poses"]["home_qpos"])
    robot = Robot(env)
    gripper = Gripper(env)
    controller = make_controller(
        args.controller,
        env,
        robot,
        cfg,
        profile,
        allow_overrides=not args.disable_controller_overrides,
    )
    estimator = make_estimator(args.estimator, cfg, profile, args.controller)

    cube_xyz = env.data.xpos[env.model.body(CUBE_BODY).id].copy()
    trajectory = make_pick_weigh_plan(
        env,
        robot,
        cfg,
        profile,
        cube_xyz,
        move_to_grasp_time=args.move_to_grasp_time,
        lift_to_weigh_time=args.lift_to_weigh_time,
        weigh_time=args.weigh_time,
    )

    if estimator.requires_calibration:
        run_calibration(
            env,
            robot,
            controller,
            estimator,
            trajectory.final_q,
            hold_time=args.calibration_time,
        )
        env.reset(arm_qpos=cfg["poses"]["home_qpos"])
        cube_xyz = env.data.xpos[env.model.body(CUBE_BODY).id].copy()
        trajectory = make_pick_weigh_plan(
            env,
            robot,
            cfg,
            profile,
            cube_xyz,
            move_to_grasp_time=args.move_to_grasp_time,
            lift_to_weigh_time=args.lift_to_weigh_time,
            weigh_time=args.weigh_time,
        )

    print(
        f"tracking mission: controller={args.controller}, estimator={args.estimator}, "
        f"profile={profile.name}, mass={args.mass:.3f}kg"
    )
    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            if args.start_delay > 0.0:
                t0 = time.time()
                while viewer.is_running() and time.time() - t0 < args.start_delay:
                    viewer.sync()
                    time.sleep(0.02)
            run_tracking_mission(
                env,
                robot,
                gripper,
                controller,
                estimator,
                cfg,
                trajectory,
                payload_mass=args.mass,
                move_to_release_time=release_time(args.move_to_release_time, profile),
                profile=profile,
                viewer=viewer,
            )
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    else:
        run_tracking_mission(
            env,
            robot,
            gripper,
            controller,
            estimator,
            cfg,
            trajectory,
            payload_mass=args.mass,
            move_to_release_time=release_time(args.move_to_release_time, profile),
            profile=profile,
        )
    return 0


def make_controller(
    name: str,
    env: MujocoEnv,
    robot: Robot,
    cfg: dict,
    profile: TrackingProfile | None = None,
    *,
    allow_overrides: bool = True,
):
    del cfg
    active_profile = profile
    if active_profile is None:
        raise ValueError("make_controller requires a tracking profile")
    kp, ki, kd = controller_gains(active_profile, name)
    if name == "pid_tracking":
        return TrackingPIDController(
            env,
            robot,
            kp=kp,
            ki=ki,
            kd=kd,
            allow_overrides=allow_overrides,
        )
    if name == "inverse_dynamics":
        return InverseDynamicsTrackingController(
            env,
            robot,
            kp=kp,
            kd=kd,
            allow_overrides=allow_overrides,
        )
    raise ValueError(f"Unknown controller '{name}'")


def make_estimator(
    name: str,
    cfg: dict,
    profile: TrackingProfile | None = None,
    controller_name: str = "pid_tracking",
) -> Estimator:
    est_cfg = cfg.get("estimator", {}) or {}
    runtime_cfg = dict(est_cfg.get(name, {}) or {})
    if profile is None:
        runtime_cfg["controller_kp"] = np.asarray(cfg["controller"]["kp"], dtype=float)
        runtime_cfg["controller_kd"] = np.asarray(cfg["controller"]["kd"], dtype=float)
    else:
        kp, _ki, kd = controller_gains(profile, controller_name)
        runtime_cfg["controller_kp"] = kp.copy()
        runtime_cfg["controller_kd"] = kd.copy()
    return build_estimator(name, runtime_cfg)


def make_pick_weigh_plan(
    env: MujocoEnv,
    robot: Robot,
    cfg: dict,
    profile: TrackingProfile,
    cube_xyz: np.ndarray,
    *,
    move_to_grasp_time: float,
    lift_to_weigh_time: float,
    weigh_time: float | None,
) -> JointTrajectory:
    q_home = np.asarray(cfg["poses"]["home_qpos"], dtype=float)
    q_weigh = np.asarray(cfg["poses"]["weigh_qpos"], dtype=float)
    target_orientation = env.ee_pose()[1]
    ik_solver = make_tracking_ik(env) if profile.use_analytical_ik else None
    grasp_xyz = profile.grasp_xyz if profile.grasp_xyz is not None else cube_xyz
    if profile.weigh_xyz is not None:
        q_weigh = solve_ik(
            robot,
            profile.weigh_xyz,
            q_home,
            "weigh",
            ik_solver=ik_solver,
            target_orientation=target_orientation,
        )
    q_grasp = solve_ik(
        robot,
        grasp_xyz,
        q_home,
        "grasp",
        ik_solver=ik_solver,
        target_orientation=target_orientation,
    )
    return make_pick_weigh_trajectory(
        q_initial=q_home,
        q_grasp=q_grasp,
        q_weigh=q_weigh,
        move_to_grasp_time=move_to_grasp_time,
        grasp_hold_time=profile.grasp_hold_time,
        lift_to_weigh_time=lift_to_weigh_time,
        weigh_hold_time=(
            weigh_time
            if weigh_time is not None
            else max(float(cfg["weigh"]["hold_seconds"]), profile.weigh_hold_time_min)
        ),
        blend_time_fraction=profile.blend_time_fraction,
        collect_lift_samples=profile.collect_lift_samples,
    )


def run_tracking_mission(
    env: MujocoEnv,
    robot: Robot,
    gripper: Gripper,
    controller,
    estimator: Estimator,
    cfg: dict,
    trajectory: JointTrajectory,
    *,
    payload_mass: float,
    move_to_release_time: float = 3.0,
    profile: TrackingProfile,
    viewer=None,
) -> dict:
    estimator.reset()
    estimate_result = None
    release_bin = None
    active = trajectory
    start_time = env.data.time
    phase = "pick_weigh"
    last_stage = None
    overrides_active = False
    attachment = TrackingAttachment(env, robot, CUBE_BODY)

    try:
        while True:
            if viewer is not None and not viewer.is_running():
                return {
                    "m_hat": np.nan,
                    "sigma": np.nan,
                    "bin_label": None,
                    "completed": False,
                }
            sample = active.sample(env.data.time - start_time)
            if sample.done:
                if phase == "pick_weigh":
                    if overrides_active:
                        controller.clear_overrides()
                        overrides_active = False
                    estimate_result = estimator.estimate()
                    release_bin = classify_mass(
                        estimate_result.m_hat,
                        float(cfg["classifier"]["mass_threshold"]),
                    )
                    print(
                        f"  [WEIGH] m_hat={estimate_result.m_hat:.4f}kg -> {release_bin}"
                    )
                    q_start = env.get_arm_qpos()
                    release_xyz = release_target(profile, cfg, release_bin)
                    release_approach_xyz = release_approach_target(profile, release_xyz)
                    target_orientation = env.ee_pose()[1]
                    ik_solver = (
                        make_tracking_ik(env)
                        if profile.use_analytical_ik
                        else None
                    )
                    q_release_approach = solve_ik(
                        robot,
                        release_approach_xyz,
                        q_start,
                        f"{release_bin}_release_approach",
                        ik_solver=ik_solver,
                        target_orientation=target_orientation,
                    )
                    q_release = solve_ik(
                        robot,
                        release_xyz,
                        q_release_approach,
                        f"{release_bin}_release",
                        ik_solver=ik_solver,
                        target_orientation=target_orientation,
                    )
                    active = make_release_trajectory(
                        q_initial=q_start,
                        q_release_approach=q_release_approach,
                        q_release=q_release,
                        move_to_release_time=move_to_release_time,
                        release_hold_time=profile.release_hold_time,
                        blend_time_fraction=profile.blend_time_fraction,
                    )
                    start_time = env.data.time
                    phase = "release"
                    last_stage = None
                    continue
                break

            if sample.stage != last_stage:
                print(f"  [TRACK] stage={sample.stage}")
                if sample.collect_mass_samples:
                    overrides_active = controller.apply_overrides(
                        estimator.controller_overrides()
                    )
                elif overrides_active:
                    controller.clear_overrides()
                    overrides_active = False
                last_stage = sample.stage

            attachment.set_attached(sample.gripper_closed)
            if sample.stage in {"release", "done"}:
                attachment.set_attached(False)
            gravity_mask = (
                estimator.gravity_comp_mask(6)
                if sample.collect_mass_samples
                else np.ones(6)
            )
            output = step_reference(
                env,
                gripper,
                controller,
                sample.q,
                sample.q_dot,
                sample.q_ddot,
                sample.gripper_closed,
                gravity_mask=gravity_mask,
                attachment=attachment,
                payload_mass=payload_mass,
                controller_payload_mass=(
                    estimate_result.m_hat
                    if estimate_result is not None
                    and sample.compensate_payload
                    and not sample.collect_mass_samples
                    else 0.0
                ),
            )
            if sample.collect_mass_samples:
                estimator.update(build_obs(env, robot, sample.q, output.tau_cmd))
            if viewer is not None:
                viewer.sync()
                time.sleep(env.dt)
    finally:
        attachment.restore()

    if estimate_result is not None:
        print(
            f"done: estimated={estimate_result.m_hat:.4f}kg, "
            f"classified={release_bin}"
        )
        return {
            "m_hat": float(estimate_result.m_hat),
            "sigma": float(estimate_result.sigma)
            if estimate_result.sigma is not None
            else np.nan,
            "bin_label": release_bin,
            "completed": True,
        }
    return {
        "m_hat": np.nan,
        "sigma": np.nan,
        "bin_label": None,
        "completed": False,
    }


def run_calibration(
    env: MujocoEnv,
    robot: Robot,
    controller,
    estimator: Estimator,
    q_weigh: np.ndarray,
    *,
    hold_time: float,
) -> None:
    print(f"calibrating estimator '{estimator.name}' for {hold_time:.1f}s")
    q_weigh = np.asarray(q_weigh, dtype=float)
    zero = np.zeros_like(q_weigh)
    try:
        for _ in range(int(2.0 / env.dt)):
            step_reference(env, None, controller, q_weigh, zero, zero, False)
        ctx = SimpleNamespace(weigh_qpos=q_weigh)
        estimator.start_calibration(ctx)
        controller.apply_overrides(estimator.controller_overrides())
        gravity_mask = estimator.gravity_comp_mask(6)
        for _ in range(max(1, int(1.0 / env.dt))):
            step_reference(
                env,
                None,
                controller,
                q_weigh,
                zero,
                zero,
                False,
                gravity_mask=gravity_mask,
            )
        for _ in range(max(1, int(hold_time / env.dt))):
            output = step_reference(
                env,
                None,
                controller,
                q_weigh,
                zero,
                zero,
                False,
                gravity_mask=gravity_mask,
            )
            estimator.update_calibration(build_obs(env, robot, q_weigh, output.tau_cmd))
        estimator.load_calibration(estimator.finish_calibration())
    finally:
        controller.clear_overrides()


def step_reference(
    env: MujocoEnv,
    gripper: Gripper | None,
    controller,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    gripper_closed: bool,
    *,
    gravity_mask: np.ndarray | None = None,
    attachment: TrackingAttachment | None = None,
    payload_mass: float = 0.0,
    controller_payload_mass: float = 0.0,
):
    reference = JointReference(q=q, q_dot=q_dot, q_ddot=q_ddot)
    env.data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    if attachment is not None:
        attachment.update(payload_mass)
    output = controller.command(
        reference,
        gravity_mask=gravity_mask,
        payload_mass=controller_payload_mass,
    )
    if gripper is not None:
        gripper.apply(gripper_cmd(gripper_closed))
    mujoco.mj_step(env.model, env.data)
    return output


def build_obs(env: MujocoEnv, robot: Robot, q_ref: np.ndarray, tau_cmd: np.ndarray) -> EstimatorObs:
    ee_xyz, _ = env.ee_pose()
    q = env.get_arm_qpos()
    return EstimatorObs(
        t=float(env.data.time),
        q=q,
        q_dot=env.get_arm_qvel(),
        tau_cmd=tau_cmd.copy(),
        tau_meas=env.actuator_force,
        qfrc_bias=env.qfrc_bias,
        jacobian_ee=robot.jacobian_ee(q),
        q_ref=q_ref.copy(),
        ee_xyz=ee_xyz,
        M=env.mass_matrix(),
    )


def gripper_cmd(closed: bool):
    from massaware.tick_loop import GripperCmd

    return GripperCmd.CLOSE if closed else GripperCmd.OPEN


def solve_ik(
    robot: Robot,
    xyz: np.ndarray,
    seed: np.ndarray,
    label: str,
    *,
    ik_solver=None,
    target_orientation: np.ndarray | None = None,
) -> np.ndarray:
    if ik_solver is not None:
        q, err = ik_solver.solve_task_waypoint(
            np.asarray(xyz, dtype=float),
            np.asarray(seed, dtype=float),
            target_orientation,
            acceptable_error=0.03,
        )
        if err > 0.03:
            print(f"  [IK] warning: {label} error={err:.4f}m/rad")
        return q
    q, ok = robot.ik(np.asarray(xyz, dtype=float), q_seed=np.asarray(seed, dtype=float))
    if not ok:
        raise RuntimeError(f"IK failed for {label} target {np.round(xyz, 3)}")
    return q


def release_time(cli_value: float | None, profile: TrackingProfile) -> float:
    return profile.move_to_release_time if cli_value is None else float(cli_value)


def release_target(profile: TrackingProfile, cfg: dict, release_bin: str) -> np.ndarray:
    if release_bin == "heavy":
        target = profile.heavy_release_xyz
        fallback = cfg["poses"]["heavy_bin_drop"]
    else:
        target = profile.light_release_xyz
        fallback = cfg["poses"]["light_bin_drop"]
    return np.asarray(target if target is not None else fallback, dtype=float).copy()


def release_approach_target(profile: TrackingProfile, release_xyz: np.ndarray) -> np.ndarray:
    approach = np.asarray(release_xyz, dtype=float).copy()
    if profile.weigh_xyz is not None:
        approach[2] = float(profile.weigh_xyz[2])
    else:
        approach[2] = approach[2] + 0.18
    return approach


if __name__ == "__main__":
    raise SystemExit(main())
