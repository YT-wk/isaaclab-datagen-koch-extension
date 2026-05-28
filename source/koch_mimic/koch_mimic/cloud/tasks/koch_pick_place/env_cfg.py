# Copyright (c) 2026, Custom project example
# SPDX-License-Identifier: Apache-2.0

"""自定义 Koch Mimic 环境的配置定义。

这个模块把场景搭建、任务观测、重置事件以及资产生成逻辑集中在一起，
方便在接入自定义 USD 资产时统一排查问题。设计目标主要有三点：

1. 让大场景房间 USD 与 Isaac Lab 默认的桌面堆叠场景解耦。
2. 在重置和随机化逻辑附近保留足够多的说明，便于定位自定义资产不稳定时的启动问题。
3. 对 Koch 机器人采用固定底座的操作语义，使微分 IK 动作项与当前任务的物理结构保持一致。
"""

from __future__ import annotations

from dataclasses import MISSING
import math

import torch

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceCfg, DevicesCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    JointVelocityActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, FrameTransformer, RayCasterCamera, TiledCamera
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import bind_physics_material, clone, get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR,ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from koch_mimic.shared.configuration import as_tuple, get_active_runtime_config, get_config_section, resolve_config_path
from koch_mimic.shared.constants import CLOUD_PROFILE

from .latched_differential_ik_action import (
    LatchedDifferentialInverseKinematicsActionCfg,
    PositionPriorityDifferentialInverseKinematicsActionCfg,
)
from .mecanum_position_only_keyboard_device import MecanumPositionOnlyIKKeyboardCfg
from .position_only_keyboard_device import PositionOnlyIKKeyboardCfg


@configclass
class KochLargeSceneCfg(InteractiveSceneCfg):
    """大房间风格 USD 场景的布局配置。

    原始 Stack 任务默认依赖桌子与地面的模板场景。这里改为把机器人和物体直接放进
    预先制作好的房间 USD 中，从而让场景资源与操作任务逻辑独立替换。
    """

    stage = AssetBaseCfg(
        prim_path="/World/Stage",
        spawn=UsdFileCfg(usd_path="/media/robot/ef64217c-7820-452d-931f-2253a903882d/robot/Files/usd_files_new/gauss_with_ground0409v1.usd"),
    )

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/ExtraLight",
        spawn=sim_utils.DomeLightCfg(color=(0.78, 0.78, 0.78), intensity=2200.0),
    )


def spawn_usd_with_physics_fallback(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """生成 USD 资产，并确保 PhysX 能把它当作刚体处理。

    有些自定义 mesh/USD 资产只包含可视化内容，没有挂载 ``RigidBodyAPI`` 或
    ``CollisionAPI``。而 ``RigidObjectCfg`` 要求目标必须具备刚体属性，因此这里
    会在缺失时自动补齐物理 API，使任务物体仍然可以被代码重置和随机化。

    这里有一个关键点：
    对抓取/放置类物体，我们刻意使用“动态刚体”作为兜底，而不是运动学刚体。
    如果兜底成运动学刚体，物体会变成无法推动的障碍物，在 IK 驱动接触时更容易
    产生过大的接触力，进而影响机械臂稳定性。
    """

    prim = sim_utils.spawn_from_usd(
        prim_path=prim_path,
        cfg=cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    matched_paths = sim_utils.find_matching_prim_paths(prim_path)
    for matched_path in matched_paths:
        fallback_rigid_props = cfg.rigid_props or sim_utils.RigidBodyPropertiesCfg()
        kinematic_enabled = (
            False if fallback_rigid_props.kinematic_enabled is None else fallback_rigid_props.kinematic_enabled
        )
        # 保留资产原有的关节结构。
        # 如果在这里粗暴禁用所有关节，带子结构的模型可能会出现“部件脱离”或“直接消失”的现象。
        # 若某个资产确实存在 root_joint 警告，优先修正该资产本身，或将其作为 articulation 处理，
        # 不要在这里做全局性的关节禁用。

        rigid_body_prims = sim_utils.get_all_matching_child_prims(
            matched_path,
            predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI),
            traverse_instance_prims=False,
        )
        if len(rigid_body_prims) == 0:
            # 尽量复用调用方传入的刚体参数，让兜底后的行为与普通 RigidObjectCfg 的生成方式保持一致。
            fallback_rigid_props = cfg.rigid_props or sim_utils.RigidBodyPropertiesCfg()
            sim_utils.define_rigid_body_properties(
                matched_path,
                sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=kinematic_enabled,
                    disable_gravity=(
                        False if fallback_rigid_props.disable_gravity is None else fallback_rigid_props.disable_gravity
                    ),
                    max_angular_velocity=fallback_rigid_props.max_angular_velocity,
                    max_linear_velocity=fallback_rigid_props.max_linear_velocity,
                    max_depenetration_velocity=fallback_rigid_props.max_depenetration_velocity,
                    solver_position_iteration_count=fallback_rigid_props.solver_position_iteration_count,
                    solver_velocity_iteration_count=fallback_rigid_props.solver_velocity_iteration_count,
                ),
            )

        collision_prims = sim_utils.get_all_matching_child_prims(
            matched_path,
            predicate=lambda p: p.HasAPI(UsdPhysics.CollisionAPI),
            traverse_instance_prims=False,
        )
        collider_mesh_prims = sim_utils.get_all_matching_child_prims(
            matched_path,
            predicate=lambda p: p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh),
            traverse_instance_prims=False,
        )
        if len(collision_prims) == 0:
            mesh_prims = sim_utils.get_all_matching_child_prims(
                matched_path,
                predicate=lambda p: p.IsA(UsdGeom.Gprim),
                traverse_instance_prims=False,
            )
            for mesh_prim in mesh_prims:
                sim_utils.define_collision_properties(
                    mesh_prim.GetPath().pathString,
                    sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                )
            collider_mesh_prims = sim_utils.get_all_matching_child_prims(
                matched_path,
                predicate=lambda p: p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh),
                traverse_instance_prims=False,
            )

        # Dynamic rigid bodies cannot use triangle-mesh style collision approximations.
        # Office/scanned assets at tiny scales can also fail SDF/convex cooking, so use
        # a conservative proxy that is much more likely to yield a valid PhysX shape.
        if not kinematic_enabled:
            for mesh_prim in collider_mesh_prims:
                sim_utils.define_mesh_collision_properties(
                    mesh_prim.GetPath().pathString,
                    # sim_utils.SDFMeshPropertiesCfg(),
                    sim_utils.BoundingCubePropertiesCfg(),
                )

    return prim


def _mesh_collision_cfg_from_name(name: str):
    collision_cfg_map = {
        "convexdecomposition": sim_utils.ConvexDecompositionPropertiesCfg,
        "convexhull": sim_utils.ConvexHullPropertiesCfg,
        "meshsimplification": sim_utils.TriangleMeshSimplificationPropertiesCfg,
        "sdf": sim_utils.SDFMeshPropertiesCfg,
        "boundingcube": sim_utils.BoundingCubePropertiesCfg,
        "boundingsphere": sim_utils.BoundingSpherePropertiesCfg,
    }
    normalized = name.strip().lower()
    if normalized == "none":
        return None
    cfg_class = collision_cfg_map.get(normalized)
    if cfg_class is None:
        valid = ", ".join(sorted(["none", *collision_cfg_map.keys()]))
        raise ValueError(f"Unsupported mesh collision approximation: {name!r}. Expected one of: {valid}")
    return cfg_class()


@clone
def spawn_usd_with_custom_physics(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn USD and optionally bind a rigid-body material plus compliant contact overrides.

    This extends the project-local fallback spawn helper with two practical behaviors used by custom grasp objects:

    1. Allow a caller-chosen mesh collision approximation instead of always forcing a bounding cube.
    2. Bind rigid-body material / compliant-contact material directly on the spawned object root so cloud-side
       custom assets can be tuned from YAML without hand-editing the USD file.
    """

    prim = spawn_usd_with_physics_fallback(
        prim_path=prim_path,
        cfg=cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    stage = get_current_stage()
    material_cfg = getattr(cfg, "physics_material", None)
    compliant_stiffness = getattr(cfg, "compliant_contact_stiffness", None)
    compliant_damping = getattr(cfg, "compliant_contact_damping", None)

    if material_cfg is not None:
        if not hasattr(material_cfg, "func") or material_cfg.func is None:
            raise ValueError("physics_material must be a valid RigidBodyMaterialCfg with a spawn function.")
        if compliant_stiffness is not None:
            material_cfg.compliant_contact_stiffness = compliant_stiffness
        if compliant_damping is not None:
            material_cfg.compliant_contact_damping = compliant_damping
        material_path = f"{prim_path}/physicsMaterial"
        material_cfg.func(material_path, material_cfg)
        bind_physics_material(prim_path, material_path, stage=stage)

    collision_cfg = getattr(cfg, "mesh_collision_props", None)
    if collision_cfg is not None:
        matched_paths = sim_utils.find_matching_prim_paths(prim_path)
        for matched_path in matched_paths:
            collider_mesh_prims = sim_utils.get_all_matching_child_prims(
                matched_path,
                predicate=lambda p: p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh),
                traverse_instance_prims=False,
            )
            for mesh_prim in collider_mesh_prims:
                sim_utils.define_mesh_collision_properties(
                    mesh_prim.GetPath().pathString,
                    collision_cfg,
                )

    return prim


@configclass
class UsdFileWithCustomPhysicsCfg(UsdFileCfg):
    """Project-local USD spawn config with extra rigid-body material controls.

    Isaac Lab's stock ``UsdFileCfg`` supports rigid/mass/collision properties but not an explicit per-asset
    rigid-body material field. This extension keeps the runtime behavior local to the project and avoids requiring
    manual USD edits for custom scanned objects.
    """

    physics_material: object | None = None
    mesh_collision_props: object | None = None
    compliant_contact_stiffness: float | None = None
    compliant_contact_damping: float | None = None


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    """从相机传感器中读取图像张量。"""
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]

    if data_type == "distance_to_camera" and convert_perspective_to_orthogonal:
        # 对透视深度做正交化，便于后续在统一深度定义下使用。
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
        elif "distance_to" in data_type or "depth" in data_type:
            # 深度图里的无穷远值会影响后续网络或可视化处理，这里统一压成 0。
            images = images.clone()
            images[images == float("inf")] = 0.0
        elif data_type == "normals":
            images = (images + 1.0) * 0.5

    return images.clone()


def rigid_pose_obs(env: ManagerBasedEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回物体位姿观测，并减去环境原点偏移。"""
    rigid_object = env.scene[object_cfg.name]
    return torch.cat((rigid_object.data.root_pos_w - env.scene.env_origins, rigid_object.data.root_quat_w), dim=-1)


def koch_gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """读取 Koch 风格单自由度夹爪的位置。"""
    robot = env.scene[robot_cfg.name]
    gripper_joint_ids, _ = robot.find_joints(env.cfg.koch_gripper_joint_names)
    return robot.data.joint_pos[:, gripper_joint_ids[0] : gripper_joint_ids[0] + 1]


def _gripper_close_progress(env: ManagerBasedRLEnv, joint_pos: torch.Tensor) -> torch.Tensor:
    """Return normalized gripper progress where 0 means open and 1 means closed."""
    open_val = torch.tensor(env.cfg.koch_gripper_open_command, dtype=torch.float32, device=env.device)
    close_val = torch.tensor(env.cfg.koch_gripper_close_command, dtype=torch.float32, device=env.device)
    travel = close_val - open_val
    if torch.isclose(travel, torch.zeros_like(travel)):
        raise ValueError("gripper open/close command values must be different.")
    return torch.clamp((joint_pos - open_val) / travel, 0.0, 1.0)


def _is_gripper_closed(env: ManagerBasedRLEnv, joint_pos: torch.Tensor) -> torch.Tensor:
    """根据归一化开合进度判断夹爪是否已闭合。"""
    progress = _gripper_close_progress(env, joint_pos)
    threshold = float(getattr(env.cfg, "koch_gripper_closed_progress_threshold", 0.65))
    return progress >= threshold


def _is_gripper_open(env: ManagerBasedRLEnv, joint_pos: torch.Tensor) -> torch.Tensor:
    """根据归一化开合进度判断夹爪是否已张开。"""
    progress = _gripper_close_progress(env, joint_pos)
    threshold = float(getattr(env.cfg, "koch_gripper_open_progress_threshold", 0.35))
    return progress <= threshold


def koch_object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.12,
    lift_height_threshold: float = 0.015,
    require_gripper_closed: bool = True,
    lifted_requires_gripper_closed: bool = False,
) -> torch.Tensor:
    """为单自由度夹爪机器人生成启发式抓取成功信号。"""
    robot = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    lifted = object_pos[:, 2] > (env.scene.env_origins[:, 2] + env.cfg.object_spawn_z + lift_height_threshold)
    grasp_candidate = torch.logical_or(pose_diff < diff_threshold, lifted)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.koch_gripper_joint_names)
    gripper_joint_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]
    # Near-object contact should still require a closed gripper, while lift is
    # already strong evidence that the object was captured earlier in the episode.
    if require_gripper_closed:
        gripper_closed = _is_gripper_closed(env, gripper_joint_pos)
        near_and_closed = torch.logical_and(pose_diff < diff_threshold, gripper_closed)
        if lifted_requires_gripper_closed:
            lifted = torch.logical_and(lifted, gripper_closed)
        return torch.logical_or(near_and_closed, lifted)
    return grasp_candidate


def koch_object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.008,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """为单自由度夹爪的抓放任务生成启发式堆叠成功信号。"""
    robot = env.scene[robot_cfg.name]
    upper_object = env.scene[upper_object_cfg.name]
    lower_object = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.koch_gripper_joint_names)
    gripper_joint_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]
    # 放置成功的判据包括三部分：XY 对齐、高度差接近期望值，以及夹爪已经松开。
    return torch.logical_and(stacked, _is_gripper_open(env, gripper_joint_pos))


def koch_object_inside_container(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    xy_half_size: tuple[float, float] = (0.12, 0.12),
    z_range: tuple[float, float] = (0.0, 0.20),
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    require_gripper_open: bool = True,
) -> torch.Tensor:
    """Return true when object center is inside a configurable box region of the container."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    container = env.scene[container_cfg.name]

    object_pos_w = obj.data.root_pos_w
    container_pos_w = container.data.root_pos_w
    container_quat_w = container.data.root_quat_w
    center_offset_t = torch.tensor(center_offset, dtype=torch.float32, device=env.device).unsqueeze(0)

    rel_pos_w = object_pos_w - container_pos_w
    rel_pos_container = math_utils.quat_apply_inverse(container_quat_w, rel_pos_w) - center_offset_t

    xy_half_size_t = torch.tensor(xy_half_size, dtype=torch.float32, device=env.device).unsqueeze(0)
    z_min, z_max = z_range
    inside_xy = torch.all(torch.abs(rel_pos_container[:, :2]) <= xy_half_size_t, dim=1)
    inside_z = torch.logical_and(rel_pos_container[:, 2] >= z_min, rel_pos_container[:, 2] <= z_max)
    inside = torch.logical_and(inside_xy, inside_z)

    if require_gripper_open:
        gripper_joint_ids, _ = robot.find_joints(env.cfg.koch_gripper_joint_names)
        gripper_joint_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]
        inside = torch.logical_and(inside, _is_gripper_open(env, gripper_joint_pos))

    return inside


def _yaw_from_quaternion(quat_wxyz: tuple[float, float, float, float]) -> float:
    """返回四元数的 yaw 分量，供平面摆位使用。"""
    qw, qx, qy, qz = quat_wxyz
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _position_from_base_polar(
    base_xy: tuple[float, float],
    base_quat: tuple[float, float, float, float],
    forward_yaw_offset: float,
    radius: float,
    relative_angle: float,
    object_z: float,
) -> tuple[float, float, float]:
    """根据机器人底座位姿，生成位于前方极坐标系中的世界坐标。"""
    base_yaw = _yaw_from_quaternion(base_quat) + forward_yaw_offset
    local_x = radius * math.cos(relative_angle)
    local_y = radius * math.sin(relative_angle)

    world_x = base_xy[0] + local_x * math.cos(base_yaw) - local_y * math.sin(base_yaw)
    world_y = base_xy[1] + local_x * math.sin(base_yaw) + local_y * math.cos(base_yaw)
    return (world_x, world_y, object_z)


def _default_front_pair_positions(
    base_xy: tuple[float, float],
    base_quat: tuple[float, float, float, float],
    forward_yaw_offset: float,
    radius: float,
    pair_distance: float,
    object_z: float,
    forward_angle_range: tuple[float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """为 A/B 两个任务物体生成满足前方范围与相互间距约束的默认摆位。"""
    forward_angle_low, forward_angle_high = forward_angle_range
    if radius <= 0.0:
        raise ValueError(f"Invalid object radius: {radius}")
    if forward_angle_high <= forward_angle_low:
        raise ValueError(f"Invalid forward angle range: {forward_angle_range}")
    if pair_distance < 0.0 or pair_distance > 2.0 * radius:
        raise ValueError(
            "Pair distance is infeasible for the requested default placement: "
            f"{pair_distance=} with {radius=}."
        )

    center_angle = 0.5 * (forward_angle_low + forward_angle_high)
    half_span = 0.5 * (forward_angle_high - forward_angle_low)
    half_angle = math.asin(min(1.0, pair_distance / (2.0 * radius)))
    if half_angle > half_span:
        raise ValueError(
            "Forward angle range is too narrow for the requested default placement: "
            f"{forward_angle_range=} can not satisfy {pair_distance=} with {radius=}."
        )

    object_a_pos = _position_from_base_polar(
        base_xy, base_quat, forward_yaw_offset, radius, center_angle - half_angle, object_z
    )
    object_b_pos = _position_from_base_polar(
        base_xy, base_quat, forward_yaw_offset, radius, center_angle + half_angle, object_z
    )
    return object_a_pos, object_b_pos


def randomize_object_pose_around_base(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfgs: list[SceneEntityCfg],
    base_xy: tuple[float, float] = (0.0, 0.0),
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    forward_yaw_offset: float = math.pi / 2.0,
    radius_range: tuple[float, float] = (0.28, 0.33),
    object_z: float = 0.0203,
    forward_angle_range: tuple[float, float] = (-0.6, 0.6),
    yaw_range: tuple[float, float] = (-1.0, 1.0),
    pair_distance_range: tuple[float, float] = (0.15, 0.25),
    max_sample_tries: int = 200,
) -> None:
    """在机器人前方随机摆放刚体物体，并满足距离底座与 A/B 间距约束。"""
    if env_ids is None or env_ids == slice(None):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    base_x, base_y = base_xy
    radius_low, radius_high = radius_range
    forward_angle_low, forward_angle_high = forward_angle_range
    yaw_low, yaw_high = yaw_range
    pair_distance_low, pair_distance_high = pair_distance_range
    num_objects = len(asset_cfgs)

    if radius_low < 0.0 or radius_high < radius_low:
        raise ValueError(f"Invalid radius range: {radius_range}")
    if forward_angle_high <= forward_angle_low:
        raise ValueError(f"Invalid forward angle range: {forward_angle_range}")
    if pair_distance_low < 0.0 or pair_distance_high < pair_distance_low:
        raise ValueError(f"Invalid pair distance range: {pair_distance_range}")

    for cur_env in env_ids.tolist():
        sampled_positions: list[tuple[float, float, float]] | None = None
        for _ in range(max_sample_tries):
            candidates: list[tuple[float, float, float]] = []
            for _ in asset_cfgs:
                # 在机器人前方扇区内采样，再转换成世界坐标。
                radius = torch.empty(1, device=env.device).uniform_(radius_low, radius_high).item()
                angle = torch.empty(1, device=env.device).uniform_(forward_angle_low, forward_angle_high).item()
                candidates.append(
                    _position_from_base_polar(base_xy, base_quat, forward_yaw_offset, radius, angle, object_z)
                )

            valid = True
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    pair_distance = math.dist(candidates[i][:2], candidates[j][:2])
                    if pair_distance < pair_distance_low or pair_distance > pair_distance_high:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                sampled_positions = candidates
                break

        if sampled_positions is None:
            # 当前任务只需要 A/B 两个物体；兜底时使用前方对称摆位，
            # 让它们同时满足距离底座与相互间距范围。
            if num_objects == 1:
                fallback_radius = 0.5 * (radius_low + radius_high)
                fallback_angle = 0.5 * (forward_angle_low + forward_angle_high)
                sampled_positions = [
                    _position_from_base_polar(
                        base_xy, base_quat, forward_yaw_offset, fallback_radius, fallback_angle, object_z
                    )
                ]
            elif num_objects == 2:
                fallback_radius = 0.5 * (radius_low + radius_high)
                fallback_pair_distance = 0.5 * (pair_distance_low + pair_distance_high)
                object_a_pos, object_b_pos = _default_front_pair_positions(
                    base_xy=base_xy,
                    base_quat=base_quat,
                    forward_yaw_offset=forward_yaw_offset,
                    radius=fallback_radius,
                    pair_distance=fallback_pair_distance,
                    object_z=object_z,
                    forward_angle_range=forward_angle_range,
                )
                sampled_positions = [object_a_pos, object_b_pos]
            else:
                raise RuntimeError(
                    "Fallback placement currently supports up to two task objects, "
                    f"but received {num_objects}."
                )

        for i, asset_cfg in enumerate(asset_cfgs):
            asset = env.scene[asset_cfg.name]
            pos = torch.tensor([sampled_positions[i]], device=env.device, dtype=torch.float32)
            pos_world = pos + env.scene.env_origins[cur_env : cur_env + 1, 0:3]

            yaw = torch.empty(1, device=env.device).uniform_(yaw_low, yaw_high)
            quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
            )

            root_pose = torch.cat([pos_world, quat], dim=-1)
            env_id_tensor = torch.tensor([cur_env], device=env.device, dtype=torch.long)
            asset.write_root_pose_to_sim(root_pose, env_ids=env_id_tensor)
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_id_tensor)


@configclass
class EventCfg:
    """自定义任务的重置事件配置。

    这里的顺序很重要：
    1. ``reset_all`` 先把机器人和物体状态恢复到配置中的默认值。
    2. ``randomize_object_ab_positions`` 再只对任务物体做扰动。

    如果缺少 ``reset_all``，单独调用 ``scene.reset()`` 只会清空缓冲区，之前回合中
    已经失稳的机器人状态可能会泄漏到下一回合。
    """

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    randomize_object_ab_positions = EventTerm(
        func=randomize_object_pose_around_base,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2")],
            "base_xy": (0.0, 0.0),
            "base_quat": (1.0, 0.0, 0.0, 0.0),
            "forward_yaw_offset": math.pi / 2.0,
            "radius_range": (0.28, 0.33),
            "object_z": 0.0203,
            "forward_angle_range": (-0.6, 0.6),
            "yaw_range": (-1.0, 1.0),
            "pair_distance_range": (0.15, 0.25),
        },
    )


@configclass
class ActionsCfg:
    """自定义操作环境的动作项配置。"""

    base_action: JointVelocityActionCfg | None = None
    arm_action: LatchedDifferentialInverseKinematicsActionCfg = MISSING
    wrist_action: RelativeJointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg | JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """策略学习与 Mimic 标注共用的观测组配置。"""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=koch_gripper_pos)
        object_a_pose = ObsTerm(func=rigid_pose_obs, params={"object_cfg": SceneEntityCfg("cube_1")})
        object_b_pose = ObsTerm(func=rigid_pose_obs, params={"object_cfg": SceneEntityCfg("cube_2")})

        def __post_init__(self):
            # 这里关闭观测噪声并保持字典结构，便于 Mimic 直接读取各字段。
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        cam_up_rgb = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("cam_up"), "data_type": "rgb", "normalize": False},
        )
        cam_arm_rgb = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("cam_arm"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            # 图像观测同样保留原始键，避免后续视觉策略取值时再拆分拼接。
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp_obj_a = ObsTerm(
            func=koch_object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        place_obj_a_on_b = ObsTerm(
            func=koch_object_inside_container,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube_1"),
                "container_cfg": SceneEntityCfg("cube_2"),
            },
        )

        def __post_init__(self):
            # 子任务信号需要以显式名称暴露给 Mimic/SkillGen 进行分段。
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """终止条件配置。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_a_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")},
    )
    object_b_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )
    success = DoneTerm(
        func=koch_object_inside_container,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("cube_1"),
            "container_cfg": SceneEntityCfg("cube_2"),
        },
    )


@configclass
class KochPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """适用于大场景资产的 Isaac Lab Koch 抓放环境配置。

    该版本不依赖 ``StackEnvCfg``，更适合直接接入独立的 USD/USDZ 世界资源。
    """

    # Mimic 采集数据时通常需要较多并行环境；如果只是可视化调试，可以先改成 1 提升迭代速度。
    scene: KochLargeSceneCfg = KochLargeSceneCfg(num_envs=128, env_spacing=8.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    # 替换为你自己的大场景世界资产路径。
    world_usdz_path: str = "/media/robot/ef64217c-7820-452d-931f-2253a903882d/robot/Files/usd_files_new/gauss_with_ground0409v1.usd"
    # 替换为你自己的 Koch 机器人 USD 路径。
    koch_robot_prim_path: str = "{ENV_REGEX_NS}/roal_wheel_robot_0325v1"
    # koch_robot_usd_path: str = "/media/robot/ef64217c-7820-452d-931f-2253a903882d/robot/Files/usd_files_new/roal_wheel_robot_0325v1/roal_wheel_robot_0325v1.usd"
    koch_robot_usd_path: str = "/media/robot/ef64217c-7820-452d-931f-2253a903882d/robot/Files/usd_files_new/orin_roal_wheel_robot_0325v1/roal_wheel_robot_0325v1.usd"
    # koch_robot_usd_path: str = "/media/robot/ef64217c-7820-452d-931f-2253a903882d/robot/Files/usd_files_new/arm_robot_v2_0701/arm_robot_v2_0701.usd"
    # 替换为任务物体路径。
    # object_a_usd_path 表示被抓取物体，object_b_usd_path 表示目标容器/承载物。
    object_a_usd_path: str = "/media/robot/74E24312E242D7CE/isaacsim5_0_assets/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/Props/SM_BottleB.usd"
    # object_a_usd_path: str = f"/media/robot/74E24312E242D7CE/isaacsim5_0_assets/isaacsim_assets/Assets/Isaac/5.1/Isaac/IsaacLab/Objects/ToyTruck/toy_truck.usd"
    object_b_usd_path: str = f"/media/robot/74E24312E242D7CE/isaacsim5_0_assets/isaacsim_assets/Assets/Isaac/5.1/Isaac/IsaacLab/Objects/Box/box.usd"
    # 资产缩放倍率通常是排查模型尺寸不匹配时最先要调的参数。
    world_scale: tuple[float, float, float] | None = None
    koch_robot_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # object_a_scale: tuple[float, float, float] = (0.0005, 0.0005, 0.0005)
    object_a_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)
    object_b_scale: tuple[float, float, float] = (0.5, 0.5, 0.3)
    # object_a can be tuned independently for custom scanned/converted grasp objects such as a textured banana USD.
    object_a_mass_kg: float | None = None
    object_a_density_kg_m3: float | None = None
    object_a_static_friction: float = 1.2
    object_a_dynamic_friction: float = 1.0
    object_a_restitution: float = 0.02
    object_a_friction_combine_mode: str = "multiply"
    object_a_restitution_combine_mode: str = "min"
    object_a_linear_damping: float = 1.5
    object_a_angular_damping: float = 2.0
    object_a_max_linear_velocity: float = 10.0
    object_a_max_angular_velocity: float = 720.0
    object_a_max_depenetration_velocity: float = 1.0
    object_a_solver_position_iteration_count: int = 16
    object_a_solver_velocity_iteration_count: int = 4
    object_a_contact_offset: float = 0.003
    object_a_rest_offset: float = 0.0
    object_a_torsional_patch_radius: float = 0.01
    object_a_min_torsional_patch_radius: float = 0.005
    object_a_mesh_collision_approximation: str = "convexDecomposition"
    object_a_compliant_contact_enabled: bool = False
    object_a_compliant_contact_stiffness: float = 8.0e4
    object_a_compliant_contact_damping: float = 2.5e3
    object_b_kinematic_enabled: bool = True
    object_b_mesh_collision_approximation: str = "meshSimplification"

    # # 替换为你自己的大场景世界资产路径。
    # world_usdz_path: str = "/root/gpufree-data/UsdFiles/mygauss.usd"
    # # 替换为你自己的 Koch 机器人 USD 路径。
    # koch_robot_usd_path: str = "/root/gpufree-data/UsdFiles/0417cylinder_mesh_road_wheel_robot_v1.usd"
    # # 替换为任务物体路径。
    # # 默认语义：object_a 是抓取物体，object_b 是盒子容器。
    # object_a_usd_path: str = "/root/gpufree-data/UsdFiles/banana_0305/banana/banana.usd"
    # # object_a_usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_BottleB.usd"
    # object_b_usd_path: str = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Box/box.usd"
    # # 资产缩放倍率通常是排查模型尺寸不匹配时最先要调的参数。
    # world_scale: tuple[float, float, float] | None = None
    # koch_robot_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # object_a_scale: tuple[float, float, float] = (0.7, 0.7, 0.7)
    # object_b_scale: tuple[float, float, float] = (0.5, 0.5, 0.3)

    # IK 稳定性相关参数。
    # 当前默认使用“前 4 关节做 position-only IK + 第 5 关节单独键控”的混合控制，
    # 这样抓取时可以一边移动末端 XYZ，一边手动微调 wrist_roll。
    arm_action_scale: float = 0.2
    arm_ik_k_val: float = 1.0
    wrist_action_scale: float = 1.0

    # Koch 机械臂在参考文件里的关节语义依次为：
    # shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper
    koch_arm_joint_names: tuple[str, ...] = (
        "arm_j1_v2_joint",
        "arm_j2_v2_joint",
        "arm_j3_v2_joint",
        "arm_j4_v2_joint",
        "arm_j5_v2_joint",
    )
    koch_ik_joint_names: tuple[str, ...] = (
        "arm_j1_v2_joint",
        "arm_j2_v2_joint",
        "arm_j3_v2_joint",
        "arm_j4_v2_joint",
    )
    koch_wrist_joint_names: tuple[str, ...] = ("arm_j5_v2_joint",)
    koch_gripper_joint_names: tuple[str, ...] = ("arm_j6_v2_joint",)

    # koch_arm_joint_names: tuple[str, ...] = (
    #     "Joint_1_joint",
    #     "Joint_2_joint",
    #     "Joint_3_joint",
    #     "Joint_4_joint",
    #     "Joint_5_joint",
    # )
    # koch_ik_joint_names: tuple[str, ...] = (
    #     "Joint_1_joint",
    #     "Joint_2_joint",
    #     "Joint_3_joint",
    #     "Joint_4_joint",
    # )
    # koch_wrist_joint_names: tuple[str, ...] = ("Joint_5_joint",)
    # koch_gripper_joint_names: tuple[str, ...] = ("Joint_Gripper_joint",)

    # 仅在混合底盘+机械臂 teleop 模式下使用。
    # 这里给出的是待核对的默认占位名；如果你的 USD 名称不同，可在脚本里通过
    # --base-wheel-joint-names 覆盖，避免影响当前固定底盘接口。
    koch_base_wheel_joint_names: tuple[str, ...] = (
        "j1_joint",
        "j2_joint",
        "j3_joint",
        "j4_joint",
    )
    koch_base_wheel_radius_m: float = 0.05
    koch_base_wheel_half_length_m: float = 0.18
    koch_base_wheel_half_width_m: float = 0.16
    koch_base_wheel_velocity_signs: tuple[float, float, float, float] = (1.0, -1.0, 1.0, 1.0)

    # 这里的刚体名和 frame 路径必须与实际 Koch USD articulation 保持一致。
    koch_ee_body_name: str = "motor6_fixer_v3_link"
    koch_base_frame_prim_path: str = "{ENV_REGEX_NS}/roal_wheel_robot_0325v1/base_link"
    koch_ee_frame_prim_path: str = (
        "{ENV_REGEX_NS}/roal_wheel_robot_0325v1/motor6_fixer_v3_link"
    )
    # koch_ee_body_name: str = "Gripper_link"
    # koch_base_frame_prim_path: str = "{ENV_REGEX_NS}/arm_robot_v2_0701/base_link"
    # koch_ee_frame_prim_path: str = (
    #     "{ENV_REGEX_NS}/arm_robot_v2_0701/Gripper_link"
    # )


    # 如果 TCP 不在末端刚体原点上，可以在这里补位置和姿态偏置。
    koch_ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    koch_ee_offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # 这个操作任务要求机器人固定在世界坐标系中。
    # 微分 IK 动作实现会根据是否为 fixed-base 选择不同的 Jacobian 索引逻辑。
    robot_fix_root_link: bool = True

    # 执行 reset 事件时，顺带清理上一回合动作管理器残留的关节目标值。
    reset_joint_targets_on_reset: bool = True

    # 机器人底座在世界坐标系中的初始位姿。
    # 当前默认生成位置为世界坐标 [0.0, 0.0, 0.05] 米。
    robot_base_pos: tuple[float, float, float] = (-0.94787, 0.24217, 0.06)
    robot_base_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # 任务物体相对于机器人底座的初始生成与随机化参数。
    # 当前约束为：A/B 都位于机器人前方 0.28~0.33m 的工作带内，
    # 且两者的平面直线距离保持在 0.15~0.25m。
    # 对当前 Koch 资产，前向默认按底座局部 +Y 轴解释，因此这里给一个 +pi/2 的偏置。
    object_spawn_z: float = 0.0203
    object_forward_yaw_offset_rad: float = math.pi / 2.0
    object_init_distance_to_base_m: float = 0.305
    object_init_ab_distance_m: float = 0.20
    object_randomize_radius_range_m: tuple[float, float] = (0.28, 0.33)
    object_forward_angle_range_rad: tuple[float, float] = (-0.6, 0.6)
    object_randomize_yaw_range: tuple[float, float] = (-1.0, 1.0)
    object_ab_distance_range_m: tuple[float, float] = (0.15, 0.25)
    object_randomize_max_sample_tries: int = 200
    object_drop_minimum_height: float = -0.05
    grasp_success_distance_threshold: float = 0.12
    grasp_success_lift_height_threshold: float = 0.015
    grasp_success_require_gripper_closed: bool = True
    grasp_success_lifted_requires_gripper_closed: bool = True
    container_success_xy_half_size: tuple[float, float] = (0.12, 0.12)
    container_success_z_range: tuple[float, float] = (0.0, 0.20)
    container_success_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    container_success_require_gripper_open: bool = True

    koch_gripper_open_command: float = math.radians(80.0)
    koch_gripper_close_command: float = math.radians(-10.0)
    koch_gripper_threshold: float = 0.005
    koch_gripper_open_progress_threshold: float = 0.35
    koch_gripper_closed_progress_threshold: float = 0.65
    external_master_arm_gripper_close_delta: float | None = None
    external_master_arm_gripper_close_direction: str = "negative"
    external_master_arm_stream_host: str = "127.0.0.1"
    external_master_arm_stream_port: int = 55000
    external_master_arm_joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    external_master_arm_joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    external_master_arm_zero_on_first_frame: bool = True
    external_master_arm_stale_timeout: float = 1.0
    mimic_wrist_target_bias_rad: float = 0.0
    mimic_generation_control_mode: str = "position_only"
    mimic_pose_ik_joint_names: tuple[str, ...] = ()
    mimic_orientation_weight: float = 0.25
    mimic_orientation_max_step_rad: float = 0.15
    mimic_ik_damping: float = 0.01

    # 当前键盘映射默认值。直接改这些字段即可调整按键，不需要再去改设备源码。
    teleop_pos_sensitivity: float = 0.05
    teleop_wrist_sensitivity: float = 0.05
    teleop_x_positive_key: str = "D"
    teleop_x_negative_key: str = "A"
    teleop_y_positive_key: str = "W"
    teleop_y_negative_key: str = "S"
    teleop_z_positive_key: str = "Q"
    teleop_z_negative_key: str = "E"
    teleop_wrist_positive_key: str = "Z"
    teleop_wrist_negative_key: str = "X"
    teleop_gripper_toggle_key: str = "K"
    teleop_clear_buffer_key: str = "L"
    teleop_base_vx_sensitivity: float = 0.4
    teleop_base_vy_sensitivity: float = 0.4
    teleop_base_omega_sensitivity: float = 0.8
    teleop_fixed_base: bool = True
    teleop_arm_source: str = "keyboard"

    # 大场景中的相机摆位配置。
    cam_up_pos: tuple[float, float, float] = (0, 0.12595, 0.48127)
    cam_up_rot: tuple[float, float, float, float] = (0.97346, 0.22888, 0, 0)
    cam_arm_pos: tuple[float, float, float] = (0, 0.04052, 0.03643)
    cam_arm_rot: tuple[float, float, float, float] = (0.9996128, -0.0278257, 0, 0)
    cam_up_parent_prim_path: str = "{ENV_REGEX_NS}/roal_wheel_robot_0325v1/base_link"
    cam_arm_parent_prim_path: str = "{ENV_REGEX_NS}/roal_wheel_robot_0325v1/motor5_fixer_v3_link"

    def _apply_runtime_config(self):
        runtime_config = get_active_runtime_config(CLOUD_PROFILE, require_user_local=False)

        self.scene.stage.prim_path = str(
            get_config_section(runtime_config, "scene", "stage_prim_path", default=self.scene.stage.prim_path)
        )
        self.scene.light.prim_path = str(
            get_config_section(runtime_config, "scene", "light_prim_path", default=self.scene.light.prim_path)
        )
        light_color = as_tuple(
            get_config_section(runtime_config, "scene", "light_color", default=list(self.scene.light.spawn.color))
        )
        if light_color is not None:
            self.scene.light.spawn.color = light_color
        self.scene.light.spawn.intensity = float(
            get_config_section(runtime_config, "scene", "light_intensity", default=self.scene.light.spawn.intensity)
        )
        self.scene.num_envs = int(get_config_section(runtime_config, "scene", "num_envs", default=self.scene.num_envs))
        self.scene.env_spacing = float(
            get_config_section(runtime_config, "scene", "env_spacing", default=self.scene.env_spacing)
        )
        self.scene.replicate_physics = bool(
            get_config_section(runtime_config, "scene", "replicate_physics", default=self.scene.replicate_physics)
        )

        self.world_usdz_path = str(
            resolve_config_path(
                str(get_config_section(runtime_config, "assets", "world_usd_path", default=self.world_usdz_path)),
                runtime_config,
            )
        )
        self.world_scale = as_tuple(get_config_section(runtime_config, "assets", "world_scale", default=self.world_scale))
        self.koch_robot_prim_path = str(
            get_config_section(runtime_config, "robot", "prim_path", default=self.koch_robot_prim_path)
        )
        self.koch_robot_usd_path = str(
            resolve_config_path(
                str(get_config_section(runtime_config, "assets", "robot_usd_path", default=self.koch_robot_usd_path)),
                runtime_config,
            )
        )
        self.koch_robot_scale = as_tuple(
            get_config_section(runtime_config, "assets", "robot_scale", default=list(self.koch_robot_scale))
        ) or self.koch_robot_scale
        self.object_a_usd_path = str(
            resolve_config_path(
                str(get_config_section(runtime_config, "assets", "object_a_usd_path", default=self.object_a_usd_path)),
                runtime_config,
            )
        )
        self.object_b_usd_path = str(
            resolve_config_path(
                str(get_config_section(runtime_config, "assets", "object_b_usd_path", default=self.object_b_usd_path)),
                runtime_config,
            )
        )
        self.object_a_scale = as_tuple(
            get_config_section(runtime_config, "assets", "object_a_scale", default=list(self.object_a_scale))
        ) or self.object_a_scale
        self.object_b_scale = as_tuple(
            get_config_section(runtime_config, "assets", "object_b_scale", default=list(self.object_b_scale))
        ) or self.object_b_scale
        object_a_mass = get_config_section(runtime_config, "assets", "object_a_mass_kg", default=self.object_a_mass_kg)
        self.object_a_mass_kg = None if object_a_mass is None else float(object_a_mass)
        object_a_density = get_config_section(
            runtime_config,
            "assets",
            "object_a_density_kg_m3",
            default=self.object_a_density_kg_m3,
        )
        self.object_a_density_kg_m3 = None if object_a_density is None else float(object_a_density)
        self.object_a_static_friction = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_static_friction",
                default=self.object_a_static_friction,
            )
        )
        self.object_a_dynamic_friction = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_dynamic_friction",
                default=self.object_a_dynamic_friction,
            )
        )
        self.object_a_restitution = float(
            get_config_section(runtime_config, "assets", "object_a_restitution", default=self.object_a_restitution)
        )
        self.object_a_friction_combine_mode = str(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_friction_combine_mode",
                default=self.object_a_friction_combine_mode,
            )
        )
        self.object_a_restitution_combine_mode = str(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_restitution_combine_mode",
                default=self.object_a_restitution_combine_mode,
            )
        )
        self.object_a_linear_damping = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_linear_damping",
                default=self.object_a_linear_damping,
            )
        )
        self.object_a_angular_damping = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_angular_damping",
                default=self.object_a_angular_damping,
            )
        )
        self.object_a_max_linear_velocity = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_max_linear_velocity",
                default=self.object_a_max_linear_velocity,
            )
        )
        self.object_a_max_angular_velocity = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_max_angular_velocity",
                default=self.object_a_max_angular_velocity,
            )
        )
        self.object_a_max_depenetration_velocity = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_max_depenetration_velocity",
                default=self.object_a_max_depenetration_velocity,
            )
        )
        self.object_a_solver_position_iteration_count = int(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_solver_position_iteration_count",
                default=self.object_a_solver_position_iteration_count,
            )
        )
        self.object_a_solver_velocity_iteration_count = int(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_solver_velocity_iteration_count",
                default=self.object_a_solver_velocity_iteration_count,
            )
        )
        self.object_a_contact_offset = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_contact_offset",
                default=self.object_a_contact_offset,
            )
        )
        self.object_a_rest_offset = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_rest_offset",
                default=self.object_a_rest_offset,
            )
        )
        self.object_a_torsional_patch_radius = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_torsional_patch_radius",
                default=self.object_a_torsional_patch_radius,
            )
        )
        self.object_a_min_torsional_patch_radius = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_min_torsional_patch_radius",
                default=self.object_a_min_torsional_patch_radius,
            )
        )
        self.object_a_mesh_collision_approximation = str(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_mesh_collision_approximation",
                default=self.object_a_mesh_collision_approximation,
            )
        )
        self.object_a_compliant_contact_enabled = bool(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_compliant_contact_enabled",
                default=self.object_a_compliant_contact_enabled,
            )
        )
        self.object_a_compliant_contact_stiffness = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_compliant_contact_stiffness",
                default=self.object_a_compliant_contact_stiffness,
            )
        )
        self.object_a_compliant_contact_damping = float(
            get_config_section(
                runtime_config,
                "assets",
                "object_a_compliant_contact_damping",
                default=self.object_a_compliant_contact_damping,
            )
        )
        self.object_b_kinematic_enabled = bool(
            get_config_section(
                runtime_config,
                "assets",
                "object_b_kinematic_enabled",
                default=self.object_b_kinematic_enabled,
            )
        )
        self.object_b_mesh_collision_approximation = str(
            get_config_section(
                runtime_config,
                "assets",
                "object_b_mesh_collision_approximation",
                default=self.object_b_mesh_collision_approximation,
            )
        )

        self.arm_action_scale = float(
            get_config_section(runtime_config, "robot", "arm_action_scale", default=self.arm_action_scale)
        )
        self.arm_ik_k_val = float(get_config_section(runtime_config, "robot", "arm_ik_k_val", default=self.arm_ik_k_val))
        self.wrist_action_scale = float(
            get_config_section(runtime_config, "robot", "wrist_action_scale", default=self.wrist_action_scale)
        )
        self.koch_arm_joint_names = as_tuple(
            get_config_section(runtime_config, "robot", "arm_joint_names", default=list(self.koch_arm_joint_names))
        ) or self.koch_arm_joint_names
        self.koch_ik_joint_names = as_tuple(
            get_config_section(runtime_config, "robot", "ik_joint_names", default=list(self.koch_ik_joint_names))
        ) or self.koch_ik_joint_names
        self.koch_wrist_joint_names = as_tuple(
            get_config_section(runtime_config, "robot", "wrist_joint_names", default=list(self.koch_wrist_joint_names))
        ) or self.koch_wrist_joint_names
        self.koch_gripper_joint_names = as_tuple(
            get_config_section(runtime_config, "robot", "gripper_joint_names", default=list(self.koch_gripper_joint_names))
        ) or self.koch_gripper_joint_names
        self.koch_ee_body_name = str(
            get_config_section(runtime_config, "robot", "ee_body_name", default=self.koch_ee_body_name)
        )
        self.koch_base_frame_prim_path = str(
            get_config_section(runtime_config, "robot", "base_frame_prim_path", default=self.koch_base_frame_prim_path)
        )
        self.koch_ee_frame_prim_path = str(
            get_config_section(runtime_config, "robot", "ee_frame_prim_path", default=self.koch_ee_frame_prim_path)
        )
        self.koch_ee_offset = as_tuple(
            get_config_section(runtime_config, "robot", "ee_offset", default=list(self.koch_ee_offset))
        ) or self.koch_ee_offset
        self.koch_ee_offset_rot = as_tuple(
            get_config_section(runtime_config, "robot", "ee_offset_rot", default=list(self.koch_ee_offset_rot))
        ) or self.koch_ee_offset_rot
        self.robot_fix_root_link = bool(
            get_config_section(runtime_config, "robot", "fix_root_link", default=self.robot_fix_root_link)
        )
        self.robot_base_pos = as_tuple(
            get_config_section(runtime_config, "robot", "base_pos", default=list(self.robot_base_pos))
        ) or self.robot_base_pos
        self.robot_base_rot = as_tuple(
            get_config_section(runtime_config, "robot", "base_rot", default=list(self.robot_base_rot))
        ) or self.robot_base_rot
        self.koch_gripper_open_command = float(
            get_config_section(runtime_config, "robot", "gripper_open_command_rad", default=self.koch_gripper_open_command)
        )
        self.koch_gripper_close_command = float(
            get_config_section(runtime_config, "robot", "gripper_close_command_rad", default=self.koch_gripper_close_command)
        )
        self.koch_gripper_threshold = float(
            get_config_section(runtime_config, "robot", "gripper_threshold", default=self.koch_gripper_threshold)
        )
        self.koch_gripper_open_progress_threshold = float(
            get_config_section(
                runtime_config,
                "robot",
                "gripper_open_progress_threshold",
                default=self.koch_gripper_open_progress_threshold,
            )
        )
        self.koch_gripper_closed_progress_threshold = float(
            get_config_section(
                runtime_config,
                "robot",
                "gripper_closed_progress_threshold",
                default=self.koch_gripper_closed_progress_threshold,
            )
        )
        self.reset_joint_targets_on_reset = bool(
            get_config_section(
                runtime_config,
                "robot",
                "reset_joint_targets_on_reset",
                default=self.reset_joint_targets_on_reset,
            )
        )

        self.koch_base_wheel_joint_names = as_tuple(
            get_config_section(runtime_config, "base", "wheel_joint_names", default=list(self.koch_base_wheel_joint_names))
        ) or self.koch_base_wheel_joint_names
        self.koch_base_wheel_radius_m = float(
            get_config_section(runtime_config, "base", "wheel_radius_m", default=self.koch_base_wheel_radius_m)
        )
        self.koch_base_wheel_half_length_m = float(
            get_config_section(runtime_config, "base", "wheel_half_length_m", default=self.koch_base_wheel_half_length_m)
        )
        self.koch_base_wheel_half_width_m = float(
            get_config_section(runtime_config, "base", "wheel_half_width_m", default=self.koch_base_wheel_half_width_m)
        )
        self.koch_base_wheel_velocity_signs = as_tuple(
            get_config_section(
                runtime_config,
                "base",
                "wheel_velocity_signs",
                default=list(self.koch_base_wheel_velocity_signs),
            )
        ) or self.koch_base_wheel_velocity_signs

        self.object_spawn_z = float(get_config_section(runtime_config, "objects", "spawn_z", default=self.object_spawn_z))
        self.object_forward_yaw_offset_rad = float(
            get_config_section(runtime_config, "objects", "forward_yaw_offset_rad", default=self.object_forward_yaw_offset_rad)
        )
        self.object_init_distance_to_base_m = float(
            get_config_section(
                runtime_config,
                "objects",
                "init_distance_to_base_m",
                default=self.object_init_distance_to_base_m,
            )
        )
        self.object_init_ab_distance_m = float(
            get_config_section(runtime_config, "objects", "init_ab_distance_m", default=self.object_init_ab_distance_m)
        )
        self.object_randomize_radius_range_m = as_tuple(
            get_config_section(
                runtime_config,
                "objects",
                "randomize_radius_range_m",
                default=list(self.object_randomize_radius_range_m),
            )
        ) or self.object_randomize_radius_range_m
        self.object_forward_angle_range_rad = as_tuple(
            get_config_section(
                runtime_config,
                "objects",
                "forward_angle_range_rad",
                default=list(self.object_forward_angle_range_rad),
            )
        ) or self.object_forward_angle_range_rad
        self.object_randomize_yaw_range = as_tuple(
            get_config_section(
                runtime_config,
                "objects",
                "randomize_yaw_range_rad",
                default=list(self.object_randomize_yaw_range),
            )
        ) or self.object_randomize_yaw_range
        self.object_ab_distance_range_m = as_tuple(
            get_config_section(
                runtime_config,
                "objects",
                "ab_distance_range_m",
                default=list(self.object_ab_distance_range_m),
            )
        ) or self.object_ab_distance_range_m
        self.object_randomize_max_sample_tries = int(
            get_config_section(
                runtime_config,
                "objects",
                "randomize_max_sample_tries",
                default=self.object_randomize_max_sample_tries,
            )
        )
        self.object_drop_minimum_height = float(
            get_config_section(
                runtime_config,
                "objects",
                "drop_minimum_height",
                default=self.object_drop_minimum_height,
            )
        )
        self.grasp_success_distance_threshold = float(
            get_config_section(
                runtime_config,
                "success",
                "grasp_distance_threshold_m",
                default=self.grasp_success_distance_threshold,
            )
        )
        self.grasp_success_lift_height_threshold = float(
            get_config_section(
                runtime_config,
                "success",
                "grasp_lift_height_threshold_m",
                default=self.grasp_success_lift_height_threshold,
            )
        )
        self.grasp_success_require_gripper_closed = bool(
            get_config_section(
                runtime_config,
                "success",
                "grasp_require_gripper_closed",
                default=self.grasp_success_require_gripper_closed,
            )
        )
        self.grasp_success_lifted_requires_gripper_closed = bool(
            get_config_section(
                runtime_config,
                "success",
                "grasp_lifted_requires_gripper_closed",
                default=self.grasp_success_lifted_requires_gripper_closed,
            )
        )
        self.container_success_xy_half_size = as_tuple(
            get_config_section(
                runtime_config,
                "success",
                "container_xy_half_size_m",
                default=list(self.container_success_xy_half_size),
            )
        ) or self.container_success_xy_half_size
        self.container_success_z_range = as_tuple(
            get_config_section(
                runtime_config,
                "success",
                "container_z_range_m",
                default=list(self.container_success_z_range),
            )
        ) or self.container_success_z_range
        self.container_success_center_offset = as_tuple(
            get_config_section(
                runtime_config,
                "success",
                "container_center_offset_m",
                default=list(self.container_success_center_offset),
            )
        ) or self.container_success_center_offset
        self.container_success_require_gripper_open = bool(
            get_config_section(
                runtime_config,
                "success",
                "require_gripper_open",
                default=self.container_success_require_gripper_open,
            )
        )
        self.mimic_wrist_target_bias_rad = float(
            get_config_section(
                runtime_config,
                "mimic",
                "wrist_target_bias_rad",
                default=self.mimic_wrist_target_bias_rad,
            )
        )
        self.mimic_generation_control_mode = str(
            get_config_section(
                runtime_config,
                "mimic",
                "generation_control_mode",
                default=self.mimic_generation_control_mode,
            )
        )
        pose_ik_joint_names = get_config_section(runtime_config, "mimic", "pose_ik_joint_names", default=None)
        self.mimic_pose_ik_joint_names = as_tuple(pose_ik_joint_names) if pose_ik_joint_names is not None else ()
        self.mimic_orientation_weight = float(
            get_config_section(
                runtime_config,
                "mimic",
                "orientation_weight",
                default=self.mimic_orientation_weight,
            )
        )
        self.mimic_orientation_max_step_rad = float(
            get_config_section(
                runtime_config,
                "mimic",
                "orientation_max_step_rad",
                default=self.mimic_orientation_max_step_rad,
            )
        )
        self.mimic_ik_damping = float(
            get_config_section(
                runtime_config,
                "mimic",
                "ik_damping",
                default=self.mimic_ik_damping,
            )
        )

        self.teleop_pos_sensitivity = float(
            get_config_section(runtime_config, "teleop", "pos_sensitivity", default=self.teleop_pos_sensitivity)
        )
        self.teleop_wrist_sensitivity = float(
            get_config_section(runtime_config, "teleop", "wrist_sensitivity", default=self.teleop_wrist_sensitivity)
        )
        self.teleop_x_positive_key = str(
            get_config_section(runtime_config, "teleop", "keys", "x_positive", default=self.teleop_x_positive_key)
        )
        self.teleop_x_negative_key = str(
            get_config_section(runtime_config, "teleop", "keys", "x_negative", default=self.teleop_x_negative_key)
        )
        self.teleop_y_positive_key = str(
            get_config_section(runtime_config, "teleop", "keys", "y_positive", default=self.teleop_y_positive_key)
        )
        self.teleop_y_negative_key = str(
            get_config_section(runtime_config, "teleop", "keys", "y_negative", default=self.teleop_y_negative_key)
        )
        self.teleop_z_positive_key = str(
            get_config_section(runtime_config, "teleop", "keys", "z_positive", default=self.teleop_z_positive_key)
        )
        self.teleop_z_negative_key = str(
            get_config_section(runtime_config, "teleop", "keys", "z_negative", default=self.teleop_z_negative_key)
        )
        self.teleop_wrist_positive_key = str(
            get_config_section(runtime_config, "teleop", "keys", "wrist_positive", default=self.teleop_wrist_positive_key)
        )
        self.teleop_wrist_negative_key = str(
            get_config_section(runtime_config, "teleop", "keys", "wrist_negative", default=self.teleop_wrist_negative_key)
        )
        self.teleop_gripper_toggle_key = str(
            get_config_section(runtime_config, "teleop", "keys", "gripper_toggle", default=self.teleop_gripper_toggle_key)
        )
        self.teleop_clear_buffer_key = str(
            get_config_section(runtime_config, "teleop", "keys", "clear_buffer", default=self.teleop_clear_buffer_key)
        )
        self.teleop_base_vx_sensitivity = float(
            get_config_section(runtime_config, "teleop", "base_vx_sensitivity", default=self.teleop_base_vx_sensitivity)
        )
        self.teleop_base_vy_sensitivity = float(
            get_config_section(runtime_config, "teleop", "base_vy_sensitivity", default=self.teleop_base_vy_sensitivity)
        )
        self.teleop_base_omega_sensitivity = float(
            get_config_section(
                runtime_config,
                "teleop",
                "base_omega_sensitivity",
                default=self.teleop_base_omega_sensitivity,
            )
        )
        self.teleop_fixed_base = bool(
            get_config_section(runtime_config, "teleop", "fixed_base", default=self.teleop_fixed_base)
        )
        self.teleop_arm_source = str(
            get_config_section(runtime_config, "teleop", "arm_source", default=self.teleop_arm_source)
        )
        self.external_master_arm_stream_host = str(
            get_config_section(
                runtime_config,
                "teleop",
                "stream",
                "host",
                default=self.external_master_arm_stream_host,
            )
        )
        self.external_master_arm_stream_port = int(
            get_config_section(
                runtime_config,
                "teleop",
                "stream",
                "port",
                default=self.external_master_arm_stream_port,
            )
        )
        self.external_master_arm_joint_signs = as_tuple(
            get_config_section(
                runtime_config,
                "teleop",
                "stream",
                "joint_signs",
                default=list(self.external_master_arm_joint_signs),
            )
        ) or self.external_master_arm_joint_signs
        self.external_master_arm_joint_offsets = as_tuple(
            get_config_section(
                runtime_config,
                "teleop",
                "stream",
                "joint_offsets",
                default=list(self.external_master_arm_joint_offsets),
            )
        ) or self.external_master_arm_joint_offsets
        close_delta = get_config_section(
            runtime_config,
            "teleop",
            "stream",
            "gripper_close_delta",
            default=self.external_master_arm_gripper_close_delta,
        )
        self.external_master_arm_gripper_close_delta = None if close_delta is None else float(close_delta)
        self.external_master_arm_gripper_close_direction = str(
            get_config_section(
                runtime_config,
                "teleop",
                "stream",
                "gripper_close_direction",
                default=self.external_master_arm_gripper_close_direction,
            )
        )
        self.external_master_arm_zero_on_first_frame = bool(
            get_config_section(
                runtime_config,
                "teleop",
                "stream",
                "zero_on_first_frame",
                default=self.external_master_arm_zero_on_first_frame,
            )
        )
        self.external_master_arm_stale_timeout = float(
            get_config_section(
                runtime_config,
                "teleop",
                "stream",
                "stale_timeout",
                default=self.external_master_arm_stale_timeout,
            )
        )

        self.cam_up_pos = as_tuple(
            get_config_section(runtime_config, "cameras", "cam_up", "pos", default=list(self.cam_up_pos))
        ) or self.cam_up_pos
        self.cam_up_rot = as_tuple(
            get_config_section(runtime_config, "cameras", "cam_up", "rot", default=list(self.cam_up_rot))
        ) or self.cam_up_rot
        self.cam_up_parent_prim_path = str(
            get_config_section(runtime_config, "cameras", "cam_up", "parent_prim_path", default=self.cam_up_parent_prim_path)
        )
        self.cam_arm_pos = as_tuple(
            get_config_section(runtime_config, "cameras", "cam_arm", "pos", default=list(self.cam_arm_pos))
        ) or self.cam_arm_pos
        self.cam_arm_rot = as_tuple(
            get_config_section(runtime_config, "cameras", "cam_arm", "rot", default=list(self.cam_arm_rot))
        ) or self.cam_arm_rot
        self.cam_arm_parent_prim_path = str(
            get_config_section(runtime_config, "cameras", "cam_arm", "parent_prim_path", default=self.cam_arm_parent_prim_path)
        )

        return runtime_config


    def __post_init__(self):
        """在所有可覆写字段就位后，完成自定义场景的最终配置。"""
        # 这些仿真参数会同时影响任务逻辑和 RTX 相机渲染。
        runtime_config = self._apply_runtime_config()
        self.decimation = int(get_config_section(runtime_config, "simulation", "decimation", default=5))
        if self.external_master_arm_gripper_close_delta is None:
            self.external_master_arm_gripper_close_delta = abs(
                self.koch_gripper_close_command - self.koch_gripper_open_command
            )
            if self.external_master_arm_gripper_close_delta <= 0.0:
                self.external_master_arm_gripper_close_delta = 1.0
        if self.external_master_arm_gripper_close_direction not in ("positive", "negative"):
            raise ValueError(
                "external_master_arm_gripper_close_direction must be 'positive' or 'negative', "
                f"got {self.external_master_arm_gripper_close_direction!r}"
            )
        if self.teleop_arm_source not in ("keyboard", "remote_master_arm"):
            raise ValueError(
                "teleop_arm_source must be 'keyboard' or 'remote_master_arm', "
                f"got {self.teleop_arm_source!r}"
            )

        self.episode_length_s = float(
            get_config_section(runtime_config, "simulation", "episode_length_s", default=30.0)
        )
        self.sim.dt = float(get_config_section(runtime_config, "simulation", "dt", default=0.01))
        self.sim.render_interval = int(get_config_section(runtime_config, "simulation", "render_interval", default=2))
        self.sim.physx.bounce_threshold_velocity = float(
            get_config_section(runtime_config, "simulation", "bounce_threshold_velocity", default=0.01)
        )
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = int(
            get_config_section(
                runtime_config,
                "simulation",
                "gpu_found_lost_aggregate_pairs_capacity",
                default=1024 * 1024 * 4,
            )
        )
        self.sim.physx.gpu_total_aggregate_pairs_capacity = int(
            get_config_section(runtime_config, "simulation", "gpu_total_aggregate_pairs_capacity", default=16 * 1024)
        )
        self.sim.physx.friction_correlation_distance = float(
            get_config_section(runtime_config, "simulation", "friction_correlation_distance", default=0.00625)
        )

        self.scene.stage.spawn.usd_path = self.world_usdz_path
        self.scene.stage.spawn.scale = self.world_scale

        # 默认物体摆位是确定性的，真正进入 episode 后会再围绕同一个“以机器人为中心”的前方工作区做随机化。
        base_x, base_y, _ = self.robot_base_pos
        obj_a_pos, obj_b_pos = _default_front_pair_positions(
            base_xy=(base_x, base_y),
            base_quat=self.robot_base_rot,
            forward_yaw_offset=self.object_forward_yaw_offset_rad,
            radius=self.object_init_distance_to_base_m,
            pair_distance=self.object_init_ab_distance_m,
            object_z=self.object_spawn_z,
            forward_angle_range=self.object_forward_angle_range_rad,
        )

        # 让机械臂从零位、夹爪张开开始，便于 Mimic 示教从中性的预抓取姿态起步。
        joint_init = {joint_name: 0.0 for joint_name in self.koch_arm_joint_names}
        for wheel_joint in self.koch_base_wheel_joint_names:
            joint_init[wheel_joint] = 0.0
        for gripper_joint in self.koch_gripper_joint_names:
            joint_init[gripper_joint] = self.koch_gripper_open_command

        # 对当前导入的 USD 机器人而言，隐式执行器是最简单且相对稳定的配置。
        robot_actuators: dict[str, ImplicitActuatorCfg] = {
            "koch_base_wheels": ImplicitActuatorCfg(
                joint_names_expr=list(self.koch_base_wheel_joint_names),
                effort_limit_sim=400.0,
                stiffness=0.0,
                # Hybrid mode drives these joints through velocity targets.
                damping=50.0,
            ),
            "koch_arm": ImplicitActuatorCfg(
                joint_names_expr=list(self.koch_arm_joint_names),
                effort_limit_sim=20.0,
                stiffness=400.0,
                damping=40.0,
            ),
            "koch_gripper": ImplicitActuatorCfg(
                joint_names_expr=list(self.koch_gripper_joint_names),
                effort_limit_sim=20.0,
                stiffness=600.0,
                damping=50.0,
            ),
        }

        self.scene.robot = ArticulationCfg(
            prim_path=self.koch_robot_prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.koch_robot_usd_path,
                scale=self.koch_robot_scale,
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    fix_root_link=self.robot_fix_root_link,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.robot_base_pos,
                rot=self.robot_base_rot,
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
                joint_pos=joint_init,
            ),
            actuators=robot_actuators,
            soft_joint_pos_limit_factor=1.0,
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        self.actions.arm_action = LatchedDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=list(self.koch_ik_joint_names),
            body_name=self.koch_ee_body_name,
            controller=DifferentialIKControllerCfg(
                # 调试版改为只控制末端位置，不再强约束完整姿态。
                # 同时把最后一个 wrist_roll 关节让出来做手动控制，避免抓取时朝向不够灵活。
                command_type="position",
                use_relative_mode=True,
                # 注意：当前 Isaac Lab 里的 SVD 解法默认假设机械臂自由度不少于 6。
                # 对 4 自由度这类欠驱动机械臂，用 pinv 更稳妥，也更容易调试。
                ik_method="pinv",
                ik_params={"k_val": self.arm_ik_k_val},
            ),
            scale=self.arm_action_scale,
            # 零输入时保持上一帧末端目标位姿，避免微小下沉被误当成新的“当前目标”。
            zero_action_tolerance=1e-6,
            body_offset=LatchedDifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=list(self.koch_ee_offset),
                rot=self.koch_ee_offset_rot,
            ),
        )
        self.actions.wrist_action = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(self.koch_wrist_joint_names),
            scale=self.wrist_action_scale,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(self.koch_gripper_joint_names),
            open_command_expr={name: self.koch_gripper_open_command for name in self.koch_gripper_joint_names},
            close_command_expr={name: self.koch_gripper_close_command for name in self.koch_gripper_joint_names},
        )

        self.gripper_joint_names = list(self.koch_gripper_joint_names)
        self.gripper_open_val = self.koch_gripper_open_command
        self.gripper_threshold = self.koch_gripper_threshold

        container_success_params = {
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("cube_1"),
            "container_cfg": SceneEntityCfg("cube_2"),
            "xy_half_size": self.container_success_xy_half_size,
            "z_range": self.container_success_z_range,
            "center_offset": self.container_success_center_offset,
            "require_gripper_open": self.container_success_require_gripper_open,
        }
        self.observations.subtask_terms.grasp_obj_a.params.update(
            {
                "diff_threshold": self.grasp_success_distance_threshold,
                "lift_height_threshold": self.grasp_success_lift_height_threshold,
                "require_gripper_closed": self.grasp_success_require_gripper_closed,
                "lifted_requires_gripper_closed": self.grasp_success_lifted_requires_gripper_closed,
            }
        )
        self.observations.subtask_terms.place_obj_a_on_b.params = container_success_params
        self.terminations.success.params = container_success_params
        self.terminations.object_a_dropping.params["minimum_height"] = self.object_drop_minimum_height
        self.terminations.object_b_dropping.params["minimum_height"] = self.object_drop_minimum_height

        # 默认打开 position-only 键盘遥操作路径。
        # 这样输出维度会与当前的 position-only IK 动作项保持一致：XYZ + gripper。
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": PositionOnlyIKKeyboardCfg(
                    pos_sensitivity=self.teleop_pos_sensitivity,
                    wrist_sensitivity=self.teleop_wrist_sensitivity,
                    x_positive_key=self.teleop_x_positive_key,
                    x_negative_key=self.teleop_x_negative_key,
                    y_positive_key=self.teleop_y_positive_key,
                    y_negative_key=self.teleop_y_negative_key,
                    z_positive_key=self.teleop_z_positive_key,
                    z_negative_key=self.teleop_z_negative_key,
                    wrist_positive_key=self.teleop_wrist_positive_key,
                    wrist_negative_key=self.teleop_wrist_negative_key,
                    gripper_toggle_key=self.teleop_gripper_toggle_key,
                    clear_buffer_key=self.teleop_clear_buffer_key,
                    sim_device=self.sim.device,
                ),
                "keyboard_mecanum": MecanumPositionOnlyIKKeyboardCfg(
                    pos_sensitivity=self.teleop_pos_sensitivity,
                    wrist_sensitivity=self.teleop_wrist_sensitivity,
                    x_positive_key=self.teleop_x_positive_key,
                    x_negative_key=self.teleop_x_negative_key,
                    y_positive_key=self.teleop_y_positive_key,
                    y_negative_key=self.teleop_y_negative_key,
                    z_positive_key=self.teleop_z_positive_key,
                    z_negative_key=self.teleop_z_negative_key,
                    wrist_positive_key=self.teleop_wrist_positive_key,
                    wrist_negative_key=self.teleop_wrist_negative_key,
                    gripper_toggle_key=self.teleop_gripper_toggle_key,
                    clear_buffer_key=self.teleop_clear_buffer_key,
                    base_vx_sensitivity=self.teleop_base_vx_sensitivity,
                    base_vy_sensitivity=self.teleop_base_vy_sensitivity,
                    base_omega_sensitivity=self.teleop_base_omega_sensitivity,
                    wheel_radius=self.koch_base_wheel_radius_m,
                    wheel_base_half_length=self.koch_base_wheel_half_length_m,
                    wheel_base_half_width=self.koch_base_wheel_half_width_m,
                    wheel_velocity_signs=self.koch_base_wheel_velocity_signs,
                    sim_device=self.sim.device,
                ),
            }
        )

        # 这个 frame transformer 定义了末端执行器参考系。
        # 观测项和 Mimic 的动作/位姿转换都依赖这里的定义。
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=self.koch_base_frame_prim_path,
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=self.koch_ee_frame_prim_path,
                    name="end_effector",
                    offset=OffsetCfg(pos=self.koch_ee_offset, rot=self.koch_ee_offset_rot),
                ),
            ],
        )

        rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        object_a_rigid_props = RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=self.object_a_linear_damping,
            angular_damping=self.object_a_angular_damping,
            max_linear_velocity=self.object_a_max_linear_velocity,
            max_angular_velocity=self.object_a_max_angular_velocity,
            max_depenetration_velocity=self.object_a_max_depenetration_velocity,
            solver_position_iteration_count=self.object_a_solver_position_iteration_count,
            solver_velocity_iteration_count=self.object_a_solver_velocity_iteration_count,
        )
        object_a_collision_props = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=self.object_a_contact_offset,
            rest_offset=self.object_a_rest_offset,
            torsional_patch_radius=self.object_a_torsional_patch_radius,
            min_torsional_patch_radius=self.object_a_min_torsional_patch_radius,
        )
        object_a_mass_props = sim_utils.MassPropertiesCfg(
            mass=self.object_a_mass_kg,
            density=self.object_a_density_kg_m3,
        )
        object_a_physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.object_a_static_friction,
            dynamic_friction=self.object_a_dynamic_friction,
            restitution=self.object_a_restitution,
            friction_combine_mode=self.object_a_friction_combine_mode,
            restitution_combine_mode=self.object_a_restitution_combine_mode,
            compliant_contact_stiffness=(
                self.object_a_compliant_contact_stiffness if self.object_a_compliant_contact_enabled else 0.0
            ),
            compliant_contact_damping=(
                self.object_a_compliant_contact_damping if self.object_a_compliant_contact_enabled else 0.0
            ),
        )
        object_a_mesh_collision_props = _mesh_collision_cfg_from_name(self.object_a_mesh_collision_approximation)
        # 两个任务物体都复用同一个生成辅助函数，
        # 这样即便原始 USD 不带物理属性，也能在运行时补成可用的刚体。
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_A",
            init_state=RigidObjectCfg.InitialStateCfg(pos=obj_a_pos, rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileWithCustomPhysicsCfg(
                func=spawn_usd_with_custom_physics,
                usd_path=self.object_a_usd_path,
                scale=self.object_a_scale,
                rigid_props=object_a_rigid_props,
                collision_props=object_a_collision_props,
                mass_props=object_a_mass_props,
                physics_material=object_a_physics_material,
                mesh_collision_props=object_a_mesh_collision_props,
                compliant_contact_stiffness=(
                    self.object_a_compliant_contact_stiffness if self.object_a_compliant_contact_enabled else None
                ),
                compliant_contact_damping=(
                    self.object_a_compliant_contact_damping if self.object_a_compliant_contact_enabled else None
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                semantic_tags=[("class", "object_a")],
            ),
        )
        object_b_rigid_props = RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=self.object_b_kinematic_enabled,
            disable_gravity=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
        )
        object_b_collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        object_b_mesh_collision_props = _mesh_collision_cfg_from_name(self.object_b_mesh_collision_approximation)
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_B",
            init_state=RigidObjectCfg.InitialStateCfg(pos=obj_b_pos, rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileWithCustomPhysicsCfg(
                func=spawn_usd_with_custom_physics,
                usd_path=self.object_b_usd_path,
                scale=self.object_b_scale,
                rigid_props=object_b_rigid_props,
                collision_props=object_b_collision_props,
                mesh_collision_props=object_b_mesh_collision_props,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                semantic_tags=[("class", "object_b")],
            ),
        )

        # 保证 reset 随机化工作区始终围绕机器人当前配置的底座位置展开。
        self.events.reset_all.params["reset_joint_targets"] = self.reset_joint_targets_on_reset
        self.events.randomize_object_ab_positions.params["base_xy"] = (base_x, base_y)
        self.events.randomize_object_ab_positions.params["base_quat"] = self.robot_base_rot
        self.events.randomize_object_ab_positions.params["forward_yaw_offset"] = self.object_forward_yaw_offset_rad
        self.events.randomize_object_ab_positions.params["radius_range"] = self.object_randomize_radius_range_m
        self.events.randomize_object_ab_positions.params["object_z"] = self.object_spawn_z
        self.events.randomize_object_ab_positions.params["forward_angle_range"] = self.object_forward_angle_range_rad
        self.events.randomize_object_ab_positions.params["yaw_range"] = self.object_randomize_yaw_range
        self.events.randomize_object_ab_positions.params["pair_distance_range"] = self.object_ab_distance_range_m
        self.events.randomize_object_ab_positions.params["max_sample_tries"] = self.object_randomize_max_sample_tries

        # 配置两个固定外部视角相机，供 Mimic 或视觉策略读取。
        self.scene.cam_up = CameraCfg(
            prim_path=f"{self.cam_up_parent_prim_path}/cam_up",
            update_period=0.0,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.14756,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10000.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=self.cam_up_pos,
                rot=self.cam_up_rot,
                convention="opengl",
            ),
        )
        self.scene.cam_arm = CameraCfg(
            prim_path=f"{self.cam_arm_parent_prim_path}/cam_arm",
            update_period=0.0,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.14756,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10000.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=self.cam_arm_pos,
                rot=self.cam_arm_rot,
                convention="opengl",
            ),
        )

        # reset 之后额外重渲染 1 次就足够刷新 RTX 相机输出，
        # 同时也能避免调试时过于明显的“多帧重置”视觉跳变。
        self.num_rerenders_on_reset = int(
            get_config_section(runtime_config, "simulation", "num_rerenders_on_reset", default=1)
        )
        self.sim.render.antialiasing_mode = str(
            get_config_section(runtime_config, "simulation", "antialiasing_mode", default="DLAA")
        )
        self.image_obs_list = list(
            get_config_section(runtime_config, "cameras", "image_obs_list", default=["cam_up", "cam_arm"])
        )
