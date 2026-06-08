"""Plot tracking-controller joint, EE, and torque errors in the main scene."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

from massaware.classify import classify_mass
from massaware.config import load_config
from massaware.controllers.attachment import TrackingAttachment
from massaware.controllers.ik import make_external_ik
from massaware.controllers.profiles import TrackingProfile, tracking_profile
from massaware.controllers.references import JointReference
from massaware.controllers.trajectory import JointTrajectory, make_release_trajectory
from massaware.estimators import inverse_dynamics as _inverse_dynamics  # noqa: F401
from massaware.estimators import build as build_estimator
from massaware.estimators.base import Estimator, EstimatorObs
from massaware.mujoco_env import MujocoEnv, UR5E_JOINTS
from massaware.robot import Robot
from massaware.tick_loop import Gripper, GripperCmd

from mission_tracking import (
    CUBE_BODY,
    build_obs,
    make_controller,
    make_pick_weigh_plan,
    release_approach_target,
    release_target,
    release_time,
    solve_ik,
)

AXIS_LABELS = ["x", "y", "z"]
AXIS_COLORS = ["tab:red", "tab:green", "tab:blue"]
JOINT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
]


@dataclass(frozen=True)
class TrackingTrace:
    controller: str
    profile: str
    mass: float
    payload_comp_mode: str
    joint_names: list[str]
    time: np.ndarray
    q: np.ndarray
    q_dot: np.ndarray
    q_ref: np.ndarray
    q_dot_ref: np.ndarray
    tau: np.ndarray
    torque_min: np.ndarray
    torque_max: np.ndarray
    stage_transitions: list[tuple[float, str]]
    actual_position: np.ndarray
    desired_position: np.ndarray
    orientation_error: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["pid_tracking", "inverse_dynamics"],
        default=["pid_tracking", "inverse_dynamics"],
    )
    parser.add_argument("--profile", choices=["tracking", "external", "main"], default="tracking")
    parser.add_argument("--mass", type=float, default=0.5)
    parser.add_argument("--move-to-grasp-time", type=float, default=3.0)
    parser.add_argument("--lift-to-weigh-time", type=float, default=2.0)
    parser.add_argument("--move-to-release-time", type=float, default=None)
    parser.add_argument("--weigh-time", type=float, default=None)
    parser.add_argument(
        "--payload-comp-mode",
        choices=["oracle", "estimated", "none"],
        default="oracle",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SOFTWARE_ROOT / "output" / "tracking_controller_error_plots"),
    )
    parser.add_argument("--joint-threshold-deg", type=float, default=2.0)
    parser.add_argument("--ee-threshold-mm", type=float, default=5.0)
    parser.add_argument("--skip-ee-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    profile = tracking_profile(args.profile, cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces: dict[str, TrackingTrace] = {}
    joint_errors: dict[str, np.ndarray] = {}

    for controller_name in args.controllers:
        trace = run_controller_trace(
            cfg,
            profile,
            controller_name,
            mass=args.mass,
            payload_comp_mode=args.payload_comp_mode,
            move_to_grasp_time=args.move_to_grasp_time,
            lift_to_weigh_time=args.lift_to_weigh_time,
            move_to_release_time=release_time(args.move_to_release_time, profile),
            weigh_time=args.weigh_time,
        )
        traces[controller_name] = trace
        prefix = output_dir / controller_name
        plot_joint_angles_velocities(trace, prefix.with_name(f"{controller_name}_joint_angles_velocities.png"))
        joint_errors[controller_name] = plot_joint_errors(
            trace,
            prefix.with_name(f"{controller_name}_joint_position_error.png"),
            threshold_deg=args.joint_threshold_deg,
        )
        plot_joint_torques(trace, prefix.with_name(f"{controller_name}_joint_torques.png"))

        if not args.skip_ee_plots:
            plot_ee_position(trace, prefix.with_name(f"{controller_name}_ee_position.png"))
            plot_ee_error(
                trace,
                prefix.with_name(f"{controller_name}_ee_position_error.png"),
                threshold_m=args.ee_threshold_mm / 1000.0,
            )
            plot_orientation_error(
                trace,
                prefix.with_name(f"{controller_name}_ee_orientation_error.png"),
            )

    if "pid_tracking" in traces and "inverse_dynamics" in traces:
        plot_joint_error_comparison(
            traces["pid_tracking"],
            joint_errors["pid_tracking"],
            traces["inverse_dynamics"],
            joint_errors["inverse_dynamics"],
            output_dir / "pid_tracking_vs_inverse_dynamics_joint_error_comparison.png",
            threshold_deg=args.joint_threshold_deg,
        )

    print(f"wrote plots to {output_dir}")
    return 0


def run_controller_trace(
    cfg: dict,
    profile: TrackingProfile,
    controller_name: str,
    *,
    mass: float,
    payload_comp_mode: str,
    move_to_grasp_time: float,
    lift_to_weigh_time: float,
    move_to_release_time: float,
    weigh_time: float | None,
) -> TrackingTrace:
    env = MujocoEnv()
    env.set_body_mass(CUBE_BODY, mass)
    env.reset(arm_qpos=cfg["poses"]["home_qpos"])
    robot = Robot(env)
    gripper = Gripper(env)
    controller = make_controller(controller_name, env, robot, cfg, profile)
    estimator = build_estimator("inverse_dynamics", {"min_samples": 30, "g": 9.81})

    cube_xyz = env.data.xpos[env.model.body(CUBE_BODY).id].copy()
    active = make_pick_weigh_plan(
        env,
        robot,
        cfg,
        profile,
        cube_xyz,
        move_to_grasp_time=move_to_grasp_time,
        lift_to_weigh_time=lift_to_weigh_time,
        weigh_time=weigh_time,
    )

    times: list[float] = []
    q_values: list[np.ndarray] = []
    q_dot_values: list[np.ndarray] = []
    q_ref_values: list[np.ndarray] = []
    q_dot_ref_values: list[np.ndarray] = []
    tau_values: list[np.ndarray] = []
    actual_positions: list[np.ndarray] = []
    desired_positions: list[np.ndarray] = []
    orientation_errors: list[np.ndarray] = []
    stage_transitions: list[tuple[float, str]] = []

    phase = "pick_weigh"
    start_time = env.data.time
    last_stage = None
    estimated_mass = 0.0
    release_bin = classify_mass(mass, float(cfg["classifier"]["mass_threshold"]))
    attachment = TrackingAttachment(env, robot, CUBE_BODY)

    try:
        while True:
            elapsed = env.data.time - start_time
            sample = active.sample(elapsed)

            if sample.done:
                if phase == "pick_weigh":
                    estimated_mass = estimate_payload_mass(estimator, default=mass)
                    if payload_comp_mode == "estimated":
                        release_bin = classify_mass(
                            estimated_mass,
                            float(cfg["classifier"]["mass_threshold"]),
                        )
                    q_start = env.get_arm_qpos()
                    release_xyz = release_target(profile, cfg, release_bin)
                    release_approach_xyz = release_approach_target(profile, release_xyz)
                    target_orientation = env.ee_pose()[1]
                    ik_solver = (
                        make_external_ik(env, robot, np.asarray(cfg["poses"]["home_qpos"], dtype=float))
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
                stage_transitions.append((env.data.time, sample.stage))
                last_stage = sample.stage

            attachment.set_attached(sample.gripper_closed)
            if sample.stage in {"release", "done"}:
                attachment.set_attached(False)

            env.data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            attachment.update(mass)

            desired_position, desired_rotation = robot.fk(sample.q)
            actual_position, actual_rotation = env.ee_pose()
            q = env.get_arm_qpos()
            q_dot = env.get_arm_qvel()
            reference = JointReference(sample.q, sample.q_dot, sample.q_ddot)
            controller_payload_mass = controller_payload(
                payload_comp_mode,
                sample,
                true_mass=mass,
                estimated_mass=estimated_mass,
            )
            output = controller.command(reference, payload_mass=controller_payload_mass)
            gripper.apply(GripperCmd.CLOSE if sample.gripper_closed else GripperCmd.OPEN)

            times.append(env.data.time)
            q_values.append(q)
            q_dot_values.append(q_dot)
            q_ref_values.append(sample.q.copy())
            q_dot_ref_values.append(sample.q_dot.copy())
            tau_values.append(output.tau_cmd.copy())
            actual_positions.append(actual_position)
            desired_positions.append(desired_position)
            orientation_errors.append(
                orientation_error_xyz_euler(desired_rotation, actual_rotation)
            )

            mujoco.mj_step(env.model, env.data)

            if sample.collect_mass_samples:
                estimator.update(build_obs(env, robot, sample.q, output.tau_cmd))
    finally:
        attachment.restore()

    return TrackingTrace(
        controller=controller_name,
        profile=profile.name,
        mass=mass,
        payload_comp_mode=payload_comp_mode,
        joint_names=list(UR5E_JOINTS),
        time=relative_time(np.array(times)),
        q=np.vstack(q_values),
        q_dot=np.vstack(q_dot_values),
        q_ref=np.vstack(q_ref_values),
        q_dot_ref=np.vstack(q_dot_ref_values),
        tau=np.vstack(tau_values),
        torque_min=actuator_ctrl_limits(env)[0],
        torque_max=actuator_ctrl_limits(env)[1],
        stage_transitions=relative_stage_times(stage_transitions, times[0] if times else 0.0),
        actual_position=np.vstack(actual_positions),
        desired_position=np.vstack(desired_positions),
        orientation_error=np.vstack(orientation_errors),
    )


def estimate_payload_mass(estimator: Estimator, default: float) -> float:
    try:
        return float(estimator.estimate().m_hat)
    except RuntimeError:
        return float(default)


def controller_payload(
    payload_comp_mode: str,
    sample,
    *,
    true_mass: float,
    estimated_mass: float,
) -> float:
    if not sample.compensate_payload or sample.collect_mass_samples:
        return 0.0
    if payload_comp_mode == "oracle":
        return true_mass
    if payload_comp_mode == "estimated":
        return estimated_mass
    return 0.0


def actuator_ctrl_limits(env: MujocoEnv) -> tuple[np.ndarray, np.ndarray]:
    ctrlrange = env.model.actuator_ctrlrange[env._ur5e_ctrl_adr]
    return ctrlrange[:, 0].copy(), ctrlrange[:, 1].copy()


def relative_time(times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return times
    return times - times[0]


def relative_stage_times(
    stage_transitions: list[tuple[float, str]],
    start_time: float,
) -> list[tuple[float, str]]:
    return [(time - start_time, stage) for time, stage in stage_transitions]


def xyz_euler_from_rotation(rotation: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    singular = sy < 1e-9
    if singular:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        y = np.arctan2(-rotation[2, 0], sy)
        z = 0.0
    else:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        y = np.arctan2(-rotation[2, 0], sy)
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([x, y, z])


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def orientation_error_xyz_euler(
    desired_rotation: np.ndarray,
    actual_rotation: np.ndarray,
) -> np.ndarray:
    return wrap_angle(
        xyz_euler_from_rotation(desired_rotation)
        - xyz_euler_from_rotation(actual_rotation)
    )


def mark_stage_transitions(ax: plt.Axes, trace: TrackingTrace) -> None:
    for index, (time, stage) in enumerate(trace.stage_transitions):
        if time <= trace.time[0] + 1e-9:
            continue
        label = "stage change" if index == 1 else None
        ax.axvline(time, linestyle="--", color="black", linewidth=0.6, alpha=0.35, label=label)


def plot_joint_angles_velocities(trace: TrackingTrace, output_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for joint_index, _joint_name in enumerate(trace.joint_names):
        color = JOINT_COLORS[joint_index % len(JOINT_COLORS)]
        label = f"joint {joint_index + 1}"
        ax1.plot(trace.time, np.rad2deg(trace.q[:, joint_index]), color=color, linewidth=1.2, label=label)
        ax1.plot(trace.time, np.rad2deg(trace.q_ref[:, joint_index]), color=color, linestyle="--", linewidth=0.8)
        ax2.plot(trace.time, np.rad2deg(trace.q_dot[:, joint_index]), color=color, linewidth=1.2, label=label)
        ax2.plot(trace.time, np.rad2deg(trace.q_dot_ref[:, joint_index]), color=color, linestyle="--", linewidth=0.8)
    mark_stage_transitions(ax1, trace)
    mark_stage_transitions(ax2, trace)
    ax1.plot([], [], color="black", linestyle="-", linewidth=1.2, label="actual")
    ax1.plot([], [], color="black", linestyle="--", linewidth=0.8, label="desired")
    ax1.set_title(title(trace, "Joint Angles vs Time"))
    ax1.set_ylabel("Joint Angle [deg]")
    ax1.legend(ncol=4, fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax2.set_title(title(trace, "Joint Velocities vs Time"))
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Joint Velocity [deg/s]")
    ax2.legend(ncol=3, fontsize=7)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_joint_errors(
    trace: TrackingTrace,
    output_path: Path,
    threshold_deg: float = 2.0,
) -> np.ndarray:
    err_deg = np.rad2deg(trace.q_ref - trace.q)
    fig, ax = plt.subplots(figsize=(10, 5))
    for joint_index in range(len(trace.joint_names)):
        color = JOINT_COLORS[joint_index % len(JOINT_COLORS)]
        ax.plot(trace.time, err_deg[:, joint_index], color=color, linewidth=1.2, label=f"joint {joint_index + 1}")
    ax.axhline(threshold_deg, linestyle="--", color="black", linewidth=1.0, label=f"+/-{threshold_deg:.0f} deg")
    ax.axhline(-threshold_deg, linestyle="--", color="black", linewidth=1.0)
    mark_stage_transitions(ax, trace)
    ax.set_title(title(trace, "Joint Position Error vs Time"))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [deg]")
    ax.legend(ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"{trace.controller}: peak |joint error| [deg] {np.max(np.abs(err_deg), axis=0).round(4)}")
    return err_deg


def plot_joint_torques(trace: TrackingTrace, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for joint_index in range(len(trace.joint_names)):
        color = JOINT_COLORS[joint_index % len(JOINT_COLORS)]
        ax.plot(trace.time, trace.tau[:, joint_index], color=color, linewidth=1.2, label=fr"$\tau_{joint_index + 1}$")
        ax.axhline(trace.torque_max[joint_index], linestyle=":", color="gray", linewidth=0.6)
        ax.axhline(trace.torque_min[joint_index], linestyle=":", color="gray", linewidth=0.6)
    mark_stage_transitions(ax, trace)
    ax.set_title(title(trace, "Joint Torque Commands vs Time"))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque [N-m]")
    ax.legend(ncol=4, fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ee_position(trace: TrackingTrace, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(trace.time, trace.actual_position[:, index], color="red", linewidth=1.5, label="actual")
        ax.plot(trace.time, trace.desired_position[:, index], color="black", linestyle="--", linewidth=1.0, label="desired")
        ax.set_ylabel(f"{AXIS_LABELS[index]} [m]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(title(trace, "End-Effector Position vs Time"))
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ee_error(
    trace: TrackingTrace,
    output_path: Path,
    threshold_m: float = 0.005,
) -> np.ndarray:
    err_m = trace.desired_position - trace.actual_position
    fig, ax = plt.subplots(figsize=(10, 5))
    for index in range(3):
        ax.plot(trace.time, err_m[:, index], color=AXIS_COLORS[index], linewidth=1.4, label=f"{AXIS_LABELS[index]} error")
    ax.axhline(threshold_m, linestyle="--", color="black", linewidth=1.0, label=f"+/-{threshold_m * 1000.0:.0f} mm")
    ax.axhline(-threshold_m, linestyle="--", color="black", linewidth=1.0)
    ax.set_title(title(trace, "End-Effector Position Error vs Time"))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"{trace.controller}: final |EE position error| [mm] {(np.abs(err_m[-1]) * 1000.0).round(3)}")
    return err_m


def plot_orientation_error(
    trace: TrackingTrace,
    output_path: Path,
    threshold_deg: float = 1.0,
) -> np.ndarray:
    err_deg = np.rad2deg(trace.orientation_error)
    err_norm_deg = np.rad2deg(np.linalg.norm(trace.orientation_error, axis=1))
    fig, ax = plt.subplots(figsize=(10, 5))
    for index in range(3):
        ax.plot(trace.time, err_deg[:, index], color=AXIS_COLORS[index], linewidth=1.4, label=f"{AXIS_LABELS[index]} error")
    ax.plot(trace.time, err_norm_deg, color="black", linestyle="--", linewidth=1.0, label="norm")
    ax.axhline(threshold_deg, linestyle=":", color="black", linewidth=0.8, label=f"+/-{threshold_deg:.1f} deg")
    ax.axhline(-threshold_deg, linestyle=":", color="black", linewidth=0.8)
    ax.set_title(title(trace, "End-Effector Orientation Error vs Time"))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("XYZ Euler Error [deg]")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return err_deg


def plot_joint_error_comparison(
    trace_a: TrackingTrace,
    err_a: np.ndarray,
    trace_b: TrackingTrace,
    err_b: np.ndarray,
    output_path: Path,
    threshold_deg: float = 2.0,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for joint_index in range(len(trace_a.joint_names)):
        color = JOINT_COLORS[joint_index % len(JOINT_COLORS)]
        ax.plot(trace_a.time, err_a[:, joint_index], color=color, linestyle="--", alpha=0.5, linewidth=0.9)
        ax.plot(trace_b.time, err_b[:, joint_index], color=color, linestyle="-", linewidth=1.2)
    ax.plot([], [], "k--", linewidth=0.9, label=trace_a.controller)
    ax.plot([], [], "k-", linewidth=1.2, label=trace_b.controller)
    ax.axhline(threshold_deg, linestyle="--", color="black", linewidth=1.0)
    ax.axhline(-threshold_deg, linestyle="--", color="black", linewidth=1.0, label=f"+/-{threshold_deg:.0f} deg")
    mark_stage_transitions(ax, trace_a)
    ax.set_title(f"Joint Tracking Error Comparison - {trace_a.controller} vs {trace_b.controller}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [deg]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def title(trace: TrackingTrace, text: str) -> str:
    return (
        f"{trace.controller} - {text} "
        f"(profile={trace.profile}, mass={trace.mass:.2f}kg, payload={trace.payload_comp_mode})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
