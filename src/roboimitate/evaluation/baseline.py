"""Scripted expert controller and manipulation baseline for RoboImitate.

Implements a multi-phase state machine controller using privileged environment state:
1. APPROACH: Move end-effector above the object at safe clearance height
2. DESCEND: Lower gripper down so fingers straddle the object
3. GRASP: Close gripper fingers securely around object
4. LIFT: Lift object off table surface
5. TRANSIT: Transport object horizontally to target zone
6. DESCEND_PLACE: Lower object toward target placement surface
7. RELEASE: Open gripper to release object inside target zone
8. RETRACT: Move end-effector upward to complete task
"""

from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ControllerPhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSIT = auto()
    DESCEND_PLACE = auto()
    RELEASE = auto()
    RETRACT = auto()
    DONE = auto()


class ScriptedPickAndPlaceExpert:
    """Multi-phase deterministic expert controller for tabletop pick-and-place."""

    def __init__(
        self,
        max_delta_pos: float = 0.03,
        hover_height: float = 0.12,
        grasp_ee_z: float = 0.460,
        lift_ee_z: float = 0.540,
        place_ee_z: float = 0.465,
        close_gripper_steps: int = 4,
        open_gripper_steps: int = 4,
    ):
        self.max_delta_pos = max_delta_pos
        self.hover_height = hover_height
        self.grasp_ee_z = grasp_ee_z
        self.lift_ee_z = lift_ee_z
        self.place_ee_z = place_ee_z
        self.close_gripper_steps = close_gripper_steps
        self.open_gripper_steps = open_gripper_steps

        self.phase = ControllerPhase.APPROACH
        self.wait_counter = 0

    def reset(self) -> None:
        """Reset the internal controller state machine."""
        self.phase = ControllerPhase.APPROACH
        self.wait_counter = 0

    def get_action(self, privileged_state: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the next 4D action [dx, dy, dz, gripper_cmd] from privileged state.

        Args:
            privileged_state: Dictionary containing:
                - 'ee_pos': [x, y, z] of end-effector
                - 'ee_target': [x, y, z] of current reference target
                - 'object_pos': [x, y, z] of object
                - 'target_pos': [x, y, z] of target drop zone

        Returns:
            np.ndarray of shape (4,) within action bounds.
        """
        ee_pos = privileged_state["ee_pos"]
        obj_pos = privileged_state["object_pos"]
        target_pos = privileged_state["target_pos"]

        target_xyz = ee_pos.copy()
        gripper_cmd = -1.0 # default open

        # State machine transitions and targets
        if self.phase == ControllerPhase.APPROACH:
            target_xyz = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.hover_height], dtype=np.float32)
            gripper_cmd = -1.0
            dist_xy = np.linalg.norm(ee_pos[:2] - target_xyz[:2])
            dist_z = abs(ee_pos[2] - target_xyz[2])
            if dist_xy < 0.02 and dist_z < 0.03:
                self.phase = ControllerPhase.DESCEND

        elif self.phase == ControllerPhase.DESCEND:
            target_xyz = np.array([obj_pos[0], obj_pos[1], self.grasp_ee_z], dtype=np.float32)
            gripper_cmd = -1.0
            dist = np.linalg.norm(ee_pos - target_xyz)
            if dist < 0.018 or ee_pos[2] <= (self.grasp_ee_z + 0.015):
                self.phase = ControllerPhase.GRASP
                self.wait_counter = 0

        elif self.phase == ControllerPhase.GRASP:
            target_xyz = np.array([obj_pos[0], obj_pos[1], self.grasp_ee_z], dtype=np.float32)
            gripper_cmd = 1.0
            self.wait_counter += 1
            if self.wait_counter >= self.close_gripper_steps:
                self.phase = ControllerPhase.LIFT

        elif self.phase == ControllerPhase.LIFT:
            target_xyz = np.array([ee_pos[0], ee_pos[1], self.lift_ee_z], dtype=np.float32)
            gripper_cmd = 1.0
            if ee_pos[2] >= (self.lift_ee_z - 0.03):
                self.phase = ControllerPhase.TRANSIT

        elif self.phase == ControllerPhase.TRANSIT:
            target_xyz = np.array([target_pos[0], target_pos[1], self.lift_ee_z], dtype=np.float32)
            gripper_cmd = 1.0
            dist_xy = np.linalg.norm(ee_pos[:2] - target_xyz[:2])
            if dist_xy < 0.025:
                self.phase = ControllerPhase.DESCEND_PLACE

        elif self.phase == ControllerPhase.DESCEND_PLACE:
            target_xyz = np.array([target_pos[0], target_pos[1], self.place_ee_z], dtype=np.float32)
            gripper_cmd = 1.0
            if ee_pos[2] <= (self.place_ee_z + 0.02):
                self.phase = ControllerPhase.RELEASE
                self.wait_counter = 0

        elif self.phase == ControllerPhase.RELEASE:
            target_xyz = np.array([target_pos[0], target_pos[1], self.place_ee_z], dtype=np.float32)
            gripper_cmd = -1.0
            self.wait_counter += 1
            if self.wait_counter >= self.open_gripper_steps:
                self.phase = ControllerPhase.RETRACT

        elif self.phase == ControllerPhase.RETRACT:
            target_xyz = np.array([target_pos[0], target_pos[1], self.lift_ee_z], dtype=np.float32)
            gripper_cmd = -1.0
            if ee_pos[2] >= (self.lift_ee_z - 0.03):
                self.phase = ControllerPhase.DONE

        elif self.phase == ControllerPhase.DONE:
            target_xyz = ee_pos.copy()
            gripper_cmd = -1.0

        # Compute delta position relative to reference target for monotonic integration
        ee_ref = privileged_state.get("ee_target", ee_pos)
        delta = target_xyz - ee_ref
        delta_clipped = np.clip(delta, -self.max_delta_pos, self.max_delta_pos)

        action = np.array([
            delta_clipped[0],
            delta_clipped[1],
            delta_clipped[2],
            gripper_cmd,
        ], dtype=np.float32)

        return action
