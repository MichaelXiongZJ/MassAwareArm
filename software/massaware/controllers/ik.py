"""UR5e task-space IK helpers for tracking experiments."""

from __future__ import annotations

from contextlib import contextmanager

import mujoco
import numpy as np

from massaware.mujoco_env import EE_SITE, MujocoEnv


class RobotIKAdapter:
    """Expose the small robot API needed by the analytical UR5e solver."""

    def __init__(self, env: MujocoEnv) -> None:
        self.env = env
        self.ee_site_id = env.model.site(EE_SITE).id
        self.qpos_ids = env._ur5e_qpos_adr
        self.qvel_ids = env._ur5e_dof_adr

    @property
    def dof(self) -> int:
        return len(self.qpos_ids)

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.env.set_arm_qpos(qpos)
        mujoco.mj_forward(self.env.model, self.env.data)

    def ee_position(self) -> np.ndarray:
        return self.env.data.site_xpos[self.ee_site_id].copy()

    def ee_orientation(self) -> np.ndarray:
        return self.env.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        q_min = np.empty(self.dof)
        q_max = np.empty(self.dof)
        for i in range(self.dof):
            joint_id = int(self.env.model.dof_jntid[self.qvel_ids[i]])
            if self.env.model.jnt_limited[joint_id]:
                q_min[i], q_max[i] = self.env.model.jnt_range[joint_id]
            else:
                q_min[i], q_max[i] = -np.inf, np.inf
        return q_min, q_max

    @contextmanager
    def preserve_state(self):
        qpos = self.env.data.qpos.copy()
        qvel = self.env.data.qvel.copy()
        ctrl = self.env.data.ctrl.copy()
        try:
            yield
        finally:
            self.env.data.qpos[:] = qpos
            self.env.data.qvel[:] = qvel
            self.env.data.ctrl[:] = ctrl
            mujoco.mj_forward(self.env.model, self.env.data)


class UR5eAnalyticalIK:
    """Full-pose UR5e analytical IK for tracking scripts."""

    D1 = 0.163
    A2 = -0.425
    A3 = -0.392
    D4 = 0.134
    D5 = 0.1
    D6 = 0.1
    BASE_HEIGHT = 0.25
    TOOL_OFFSET = 0.1558
    Q_OFFSETS = np.array([np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    TOOL_ROTATION = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    def __init__(
        self,
        robot: RobotIKAdapter,
        orientation_gain: float = 0.5,
        acceptable_error: float = 3e-2,
    ) -> None:
        self.robot = robot
        self.orientation_gain = float(orientation_gain)
        self.acceptable_error = float(acceptable_error)

    def solve_task_waypoint(
        self,
        target_position: np.ndarray,
        seed_q: np.ndarray,
        target_orientation: np.ndarray | None = None,
        acceptable_error: float | None = None,
    ) -> tuple[np.ndarray, float]:
        candidates = self.analytical_solutions(
            np.asarray(target_position, dtype=float),
            np.asarray(seed_q, dtype=float),
            target_orientation,
        )
        if not candidates:
            raise RuntimeError("Analytical IK found no valid UR5e solutions")

        acceptable = self.acceptable_error if acceptable_error is None else acceptable_error
        feasible = [candidate for candidate in candidates if candidate[1] <= acceptable]
        if feasible:
            return min(feasible, key=lambda item: np.linalg.norm(wrap_to_pi(item[0] - seed_q)))
        return min(candidates, key=lambda item: item[1])

    def analytical_solutions(
        self,
        target_position: np.ndarray,
        seed_q: np.ndarray,
        target_orientation: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        with self.robot.preserve_state():
            if target_orientation is None:
                self.robot.set_qpos(seed_q)
                target_rotation = self.robot.ee_orientation()
            else:
                target_rotation = np.asarray(target_orientation, dtype=float)

            target_transform = make_transform(target_rotation, target_position)
            ur_transform = (
                self._world_to_ur_base_transform()
                @ target_transform
                @ np.linalg.inv(self._flange_to_site_transform())
            )

            q_min, q_max = self.robot.joint_limits()
            candidates = []
            for q_ur in self._inverse_ur(ur_transform, q6_des=seed_q[5]):
                q = wrap_to_pi(q_ur - self.Q_OFFSETS)
                if np.any(q < q_min - 1e-8) or np.any(q > q_max + 1e-8):
                    continue
                error = self._mujoco_fk_error(q, target_position, target_rotation)
                candidates.append((q, error))
            return candidates

    def _mujoco_fk_error(
        self,
        q: np.ndarray,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
    ) -> float:
        self.robot.set_qpos(q)
        position_error = target_position - self.robot.ee_position()
        orientation_error = orientation_error_vector(
            target_orientation,
            self.robot.ee_orientation(),
        )
        return float(
            np.linalg.norm(
                np.concatenate((position_error, self.orientation_gain * orientation_error))
            )
        )

    def _world_to_ur_base_transform(self) -> np.ndarray:
        transform = np.eye(4)
        transform[2, 3] = -self.BASE_HEIGHT
        return transform

    def _flange_to_site_transform(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.TOOL_ROTATION
        transform[:3, 3] = np.array([self.TOOL_OFFSET, 0.0, 0.0])
        return transform

    def _inverse_ur(self, transform: np.ndarray, q6_des: float = 0.0) -> list[np.ndarray]:
        zero_thresh = 1e-8
        pi = np.pi
        values = transform.reshape(-1)
        t02 = -values[0]
        t00 = values[1]
        t01 = values[2]
        t03 = -values[3]
        t12 = -values[4]
        t10 = values[5]
        t11 = values[6]
        t13 = -values[7]
        t22 = values[8]
        t20 = -values[9]
        t21 = -values[10]
        t23 = values[11]

        q1 = []
        a = self.D6 * t12 - t13
        b = self.D6 * t02 - t03
        radius_sq = a * a + b * b
        if abs(a) < zero_thresh:
            div = -sign(self.D4) * sign(b) if abs(abs(self.D4) - abs(b)) < zero_thresh else -self.D4 / b
            arcsin = np.arcsin(np.clip(div, -1.0, 1.0))
            arcsin = 0.0 if abs(arcsin) < zero_thresh else arcsin
            q1 = [arcsin + 2.0 * pi if arcsin < 0.0 else arcsin, pi - arcsin]
        elif abs(b) < zero_thresh:
            div = sign(self.D4) * sign(a) if abs(abs(self.D4) - abs(a)) < zero_thresh else self.D4 / a
            arccos = np.arccos(np.clip(div, -1.0, 1.0))
            q1 = [arccos, 2.0 * pi - arccos]
        elif self.D4 * self.D4 > radius_sq:
            return []
        else:
            arccos = np.arccos(np.clip(self.D4 / np.sqrt(radius_sq), -1.0, 1.0))
            arctan = np.arctan2(-b, a)
            pos = arccos + arctan
            neg = -arccos + arctan
            pos = 0.0 if abs(pos) < zero_thresh else pos
            neg = 0.0 if abs(neg) < zero_thresh else neg
            q1 = [pos if pos >= 0.0 else 2.0 * pi + pos, neg if neg >= 0.0 else 2.0 * pi + neg]

        q5 = []
        for q1_i in q1:
            numer = t03 * np.sin(q1_i) - t13 * np.cos(q1_i) - self.D4
            div = sign(numer) * sign(self.D6) if abs(abs(numer) - abs(self.D6)) < zero_thresh else numer / self.D6
            arccos = np.arccos(np.clip(div, -1.0, 1.0))
            q5.append([arccos, 2.0 * pi - arccos])

        solutions = []
        for i, q1_i in enumerate(q1):
            for q5_ij in q5[i]:
                c1, s1 = np.cos(q1_i), np.sin(q1_i)
                c5, s5 = np.cos(q5_ij), np.sin(q5_ij)
                if abs(s5) < zero_thresh:
                    q6 = q6_des % (2.0 * pi)
                else:
                    q6 = np.arctan2(
                        sign(s5) * -(t01 * s1 - t11 * c1),
                        sign(s5) * (t00 * s1 - t10 * c1),
                    )
                    q6 = 0.0 if abs(q6) < zero_thresh else q6
                    q6 = q6 + 2.0 * pi if q6 < 0.0 else q6

                c6, s6 = np.cos(q6), np.sin(q6)
                x04x = -s5 * (t02 * c1 + t12 * s1) - c5 * (
                    s6 * (t01 * c1 + t11 * s1) - c6 * (t00 * c1 + t10 * s1)
                )
                x04y = c5 * (t20 * c6 - t21 * s6) - t22 * s5
                p13x = (
                    self.D5 * (s6 * (t00 * c1 + t10 * s1) + c6 * (t01 * c1 + t11 * s1))
                    - self.D6 * (t02 * c1 + t12 * s1)
                    + t03 * c1
                    + t13 * s1
                )
                p13y = t23 - self.D1 - self.D6 * t22 + self.D5 * (t21 * c6 + t20 * s6)
                c3 = (
                    p13x * p13x
                    + p13y * p13y
                    - self.A2 * self.A2
                    - self.A3 * self.A3
                ) / (2.0 * self.A2 * self.A3)
                if abs(abs(c3) - 1.0) < zero_thresh:
                    c3 = sign(c3)
                elif abs(c3) > 1.0:
                    continue

                arccos = np.arccos(np.clip(c3, -1.0, 1.0))
                q3 = [arccos, 2.0 * pi - arccos]
                denom = self.A2 * self.A2 + self.A3 * self.A3 + 2.0 * self.A2 * self.A3 * c3
                s3 = np.sin(arccos)
                a_term = self.A2 + self.A3 * c3
                b_term = self.A3 * s3
                q2 = [
                    np.arctan2(
                        (a_term * p13y - b_term * p13x) / denom,
                        (a_term * p13x + b_term * p13y) / denom,
                    ),
                    np.arctan2(
                        (a_term * p13y + b_term * p13x) / denom,
                        (a_term * p13x - b_term * p13y) / denom,
                    ),
                ]

                q4 = []
                for q2_k, q3_k in zip(q2, q3):
                    c23 = np.cos(q2_k + q3_k)
                    s23 = np.sin(q2_k + q3_k)
                    q4.append(np.arctan2(c23 * x04y - s23 * x04x, x04x * c23 + x04y * s23))

                for k in range(2):
                    q2_k = 0.0 if abs(q2[k]) < zero_thresh else q2[k]
                    q4_k = 0.0 if abs(q4[k]) < zero_thresh else q4[k]
                    q2_k = q2_k + 2.0 * pi if q2_k < 0.0 else q2_k
                    q4_k = q4_k + 2.0 * pi if q4_k < 0.0 else q4_k
                    solutions.append(np.array([q1_i, q2_k, q3[k], q4_k, q5_ij, q6]))

        return solutions


def make_tracking_ik(env: MujocoEnv) -> UR5eAnalyticalIK:
    return UR5eAnalyticalIK(RobotIKAdapter(env))


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def sign(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


def make_transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def orientation_error_vector(
    target_orientation: np.ndarray,
    current_orientation: np.ndarray,
) -> np.ndarray:
    error_matrix = target_orientation @ current_orientation.T
    return 0.5 * np.array(
        [
            error_matrix[2, 1] - error_matrix[1, 2],
            error_matrix[0, 2] - error_matrix[2, 0],
            error_matrix[1, 0] - error_matrix[0, 1],
        ]
    )
