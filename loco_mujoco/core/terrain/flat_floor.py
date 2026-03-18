from __future__ import annotations

from typing import Any, Union, Tuple
from types import ModuleType

import mujoco
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.terrain import Terrain
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class FlatFloorTerrain(Terrain):
    """
    Adds a simple flat floor plane to the worldbody if none exists.

    This is useful when working with custom robot XMLs that only define the robot
    and assets, but no environment geometry (e.g., no `floor` geom).
    """

    def __init__(
        self,
        env: Any,
        floor_size: float = 10.0,
        floor_friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001),
        floor_rgba: Tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0),
        add_overhead_light: bool = True,
        **terrain_config,
    ):
        super().__init__(env, **terrain_config)
        self.floor_size = float(floor_size)
        self.floor_friction = tuple(float(x) for x in floor_friction)
        self.floor_rgba = tuple(float(x) for x in floor_rgba)
        self.add_overhead_light = bool(add_overhead_light)

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        wb = spec.worldbody

        has_floor = any(g.name == "floor" for g in spec.geoms)
        if not has_floor:
            wb.add_geom(
                name="floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=(0.0, 0.0, 0.0),
                size=(self.floor_size, self.floor_size, 0.1),
                friction=self.floor_friction,
                rgba=self.floor_rgba,
                contype=1,
                conaffinity=1,
                group=2,
            )

        if self.add_overhead_light and not any(l.name == "overhead_light" for l in spec.lights):
            wb.add_light(
                name="overhead_light",
                pos=(0.0, 0.0, 4.0),
                dir=(0.0, 0.0, -1.0),
                diffuse=(1.0, 1.0, 1.0),
                ambient=(0.6, 0.6, 0.6),
                specular=(0.2, 0.2, 0.2),
            )

        return spec

    def reset(
        self,
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[Union[MjData, Data], Any]:
        assert_backend_is_supported(backend)
        return data, carry

    def update(
        self,
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        assert_backend_is_supported(backend)
        return model, data, carry

    @property
    def is_dynamic(self) -> bool:
        return False

