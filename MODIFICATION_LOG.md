# Modification Log

## 2026-05-27 - Mimic 生成成功率 0 诊断与 wrist 修复

### 1. 修改了哪些文件

- `scripts/local/analyze_hdf5_dataset.py`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/mimic_env.py`
- `README.md`
- `MODIFICATION_LOG.md`

### 2. 每个文件为什么改

- `scripts/local/analyze_hdf5_dataset.py`  
  增加 `--mimic-diagnose`，按 `cloud.defaults.yaml -> cloud.user.local.yaml -> --cloud-config` 的有效配置读取成功阈值，输出动作维度、wrist 分布、gripper 状态、物体最大抬升、最终容器相对位置和按当前配置重算的成功结果。

- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/mimic_env.py`  
  修复 Mimic 生成阶段 wrist 被固定为 0 的问题。`actions_to_gripper_actions()` 现在会把 Koch wrist 信息作为 gripper_action 的附加元数据传给 Mimic waypoint；`target_eef_pose_to_action()` 再把该 wrist 元数据恢复成环境的 5D 动作 `[dx, dy, dz, wrist, gripper]`。直接关节示教的 wrist 会按绝对关节目标处理，IK 示教的 wrist 会按相对增量处理。

- `README.md`  
  重写项目运行说明，覆盖安装、配置优先级、本地 HDF5 分析、云端录制、回放、Mimic 标注、Mimic 生成、teleop、leader arm SSH 推流和本地调试。云端 IsaacLab 相关命令统一使用 `~/isaaclab/_isaac_sim/python.sh`。

- `MODIFICATION_LOG.md`  
  新增本文件，用于记录本次以及后续修改的文件、原因、验证结果、未验证项和需要人工确认的事项。

### 3. 运行了哪些验证

- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe -m py_compile scripts/local/analyze_hdf5_dataset.py`：通过。
- `python -m py_compile source\koch_mimic\koch_mimic\cloud\tasks\koch_pick_place\mimic_env.py`：通过。
- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe scripts/local/analyze_hdf5_dataset.py --dataset-dir D:\Codes\RoboticsProject\datasets\0527 --mimic-diagnose -n 0 --no-values --precision 5`：通过，并确认有效配置加载了 `cloud.defaults.yaml` 与 `cloud.user.local.yaml`。
- 同一次完整诊断确认：`koch_mimic_generated_100.hdf5` 成功 demo 数为 0；`koch_mimic_generated_100_failed.hdf5` 的 8 条失败轨迹均为 5D action，wrist 全 0，`object_a max_lift=0`，按有效配置成功数为 0。
- 同一次完整诊断确认：`koch_raw_remote_0526.hdf5` 的 10 条 raw 轨迹均为 10D action，wrist 非零，`object_a max_lift` 为 0.07383 到 0.17578 m，按有效配置成功数为 10。
- 追加短诊断命令 `--mimic-diagnose -n 1 --no-values --precision 5`：通过，确认输出较小样本时仍能显示配置覆盖、wrist、lift、container 和 success 结论。

### 4. 还有哪些没验证

- 尚未在 Isaac Sim / IsaacLab 中运行小批量 Mimic 生成。
- 尚未 replay/validate 新生成的成功 HDF5。
- 尚未确认修复后真实生成数据的成功率是否大于 0。

### 5. 哪些地方需要我人工确认

- 请确认云端实际 IsaacLab 路径确实为 `~/isaaclab/_isaac_sim/python.sh`。
- 请确认 `configs/cloud.user.local.yaml` 中的成功阈值仍符合当前任务目标，尤其是 `container_xy_half_size_m`、`container_z_range_m` 和 `require_gripper_open`。
- 请在云端跑一次小批量无噪声 Mimic 生成，确认 wrist 修复后是否能产生非空成功 HDF5。

## 2026-05-27 - 0527 raw 数据 grasp_obj_a 标注修复

### 1. 修改了哪些文件

- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py`
- `scripts/local/analyze_hdf5_dataset.py`
- `configs/cloud.defaults.yaml`
- `configs/cloud.user.example.yaml`
- `configs/cloud.user.local.yaml`
- `README.md`
- `MODIFICATION_LOG.md`

### 2. 每个文件为什么改

- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py`  
  调整 `koch_object_grasped()` 的 `grasp_obj_a` 判定：近距离抓取仍要求 gripper closed，但物体已被抬升时可以单独作为抓取完成证据。这样可以避免 raw 轨迹中“确实夹起过物体，但 closed 判定和 lifted 判定没有落在同一帧”导致 Mimic 自动标注失败。

- `scripts/local/analyze_hdf5_dataset.py`  
  增加 `grasp_obj_a` 诊断输出，报告 near、closed、lifted、near_and_closed、lifted_and_closed、grasp frame 数量和首次 grasp 帧，方便定位标注失败到底卡在距离、抬升还是夹爪闭合。

- `configs/cloud.defaults.yaml` / `configs/cloud.user.example.yaml` / `configs/cloud.user.local.yaml`  
  增加 `success.grasp_lifted_requires_gripper_closed: false`。默认情况下，lifted 作为强抓取证据，不再强制要求同一帧 gripper closed；如后续想恢复严格判定，可改为 `true`。同时把本机 `configs/cloud.user.local.yaml` 的 `teleop.stream.gripper_close_direction` 改为 `positive`，与 0527 采集时真实硬件可打开仿真夹爪的方向保持一致。

- `README.md`  
  在 HDF5 Mimic 诊断说明中补充 `grasp_obj_a` 诊断用途。

- `MODIFICATION_LOG.md`  
  追加本次修改记录。

### 3. 运行了哪些验证

- `python -m py_compile source\koch_mimic\koch_mimic\cloud\tasks\koch_pick_place\env_cfg.py source\koch_mimic\koch_mimic\cloud\tasks\koch_pick_place\mimic_env.py`：通过。
- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe -m py_compile scripts/local/analyze_hdf5_dataset.py`：通过。
- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe scripts/local/analyze_hdf5_dataset.py --dataset-dir D:\Codes\RoboticsProject\datasets\0527 --file koch_raw_remote_0527.hdf5 --mimic-diagnose -n 0 --no-values --precision 5`：通过。
- 对 `koch_raw_remote_0527.hdf5` 的 3 条轨迹，本地诊断确认：按有效配置最终成功数为 3；`grasp_obj_a` 现在 3 条都有信号，frame_count_range 为 `[30, 54]`。
- 同一诊断也确认：3 条轨迹的 `near_frames=0`，最小 EEF-object 距离约 0.13052、0.13413、0.13893 m，略大于当前 `grasp_distance_threshold_m=0.12`；修复后由 lifted 分支触发 grasp。

### 4. 还有哪些没验证

- 尚未在云端重新运行 `annotate_koch_mimic_demos.py --auto` 验证 HDF5 标注文件是否成功导出。
- 尚未用新 annotated HDF5 运行 Mimic generation。
- 尚未 replay/validate 新生成数据。

### 5. 哪些地方需要我人工确认

- 请在云端重新运行 0527 raw 数据的自动标注命令，确认 `koch_mimic_demos_annotated_0527.hdf5` 能导出非 0 条 demo。
- 请确认 `gripper_close_direction` 最终应为你实际可控的 `positive`，并同步到云端 `configs/cloud.user.local.yaml`；该文件通常不进 Git，需要手动复制或在云端改同一项。
- 如果你希望抓取标注必须严格要求 lifted 与 gripper closed 同帧成立，可把 `success.grasp_lifted_requires_gripper_closed` 改回 `true`，但当前 0527 raw 数据会再次无法标注。

## 2026-05-27 - 夹爪开合语义与最终释放判定修正

### 1. 修改了哪些文件
- `configs/cloud.defaults.yaml`
- `configs/cloud.user.example.yaml`
- `configs/cloud.user.local.yaml`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py`
- `scripts/local/analyze_hdf5_dataset.py`
- `README.md`
- `MODIFICATION_LOG.md`

### 2. 每个文件为什么改

- `configs/cloud.defaults.yaml` / `configs/cloud.user.example.yaml` / `configs/cloud.user.local.yaml`  
  将当前 Koch 夹爪资产的语义改为 `gripper_open_command_rad=1.3962634016`、`gripper_close_command_rad=-0.1745329252`，并将 `grasp_lifted_requires_gripper_closed` 设为 `true`，避免仅靠抬升而夹爪已经打开时仍被当作抓取完成。`cloud.user.local.yaml` 同时把 `teleop.stream.gripper_close_direction` 改回 `negative`，让真实 leader 夹爪收紧时映射到新的 close command。
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py`  
  同步环境类默认值，确保没有 YAML 覆盖时也按“较大角度打开、较小角度闭合”的语义创建 BinaryJointPositionAction、初始化夹爪和计算成功/子任务条件。
- `scripts/local/analyze_hdf5_dataset.py`  
  同步本地诊断脚本默认 open/close 命令，并打印 open/closed progress 阈值，方便检查旧数据是否仍被错误判为释放成功。
- `README.md`  
  补充夹爪语义说明，明确最终成功必须在放入容器后真正打开夹爪释放。
- `MODIFICATION_LOG.md`  
  追加本次修改记录。

### 3. 运行了哪些验证
- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe -m py_compile scripts/local/analyze_hdf5_dataset.py source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/mimic_env.py`：通过。
- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe scripts/local/analyze_hdf5_dataset.py --dataset-dir D:\Codes\RoboticsProject\datasets\0527 --file koch_raw_remote_0527.hdf5 --mimic-diagnose -n 0 --no-values --precision 5`：通过。有效配置显示 `gripper_open_command_rad=1.3962634016`、`gripper_close_command_rad=-0.1745329252`，旧 0527 raw 的 3 条轨迹均因 `final gripper is not open enough` 不再按新语义判为成功。
- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe scripts/local/analyze_hdf5_dataset.py --dataset-dir D:\Codes\RoboticsProject\datasets\0527 --file koch_mimic_generated_0527_failed.hdf5 --mimic-diagnose -n 0 --no-values --precision 5`：通过。15 条 failed 轨迹仍为 0 成功，主要失败原因包含物体未抬升、XY 未进入容器以及最终夹爪未打开。

### 4. 还有哪些没验证
- 尚未在云端 IsaacLab 中重新录制并确认真实 leader 夹爪开合方向。
- 尚未重新运行 Mimic annotation / generation。
- 尚未 replay/validate 新生成数据。

### 5. 哪些地方需要我人工确认
- 请在云端用 `cloud.user.local.yaml` 重新启动 teleop，确认真实打开夹爪时仿真夹爪打开，真实收紧夹爪时仿真夹爪闭合。
- 新采集数据时，请确认放入容器后要打开夹爪释放，保持闭合不应再记录为 success。

## 2026-05-28 - Mimic 生成 wrist 45 度姿态偏置

### 1. 修改了哪些文件
- `configs/cloud.defaults.yaml`
- `configs/cloud.user.example.yaml`
- `configs/cloud.user.local.yaml`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/mimic_env.py`
- `README.md`
- `MODIFICATION_LOG.md`

### 2. 每个文件为什么改

- `configs/cloud.defaults.yaml` / `configs/cloud.user.example.yaml`  
  新增 `mimic.wrist_target_bias_rad`，默认值为 `0.0`，让 wrist 姿态偏置成为可配置项，不影响默认行为。
- `configs/cloud.user.local.yaml`  
  将 `mimic.wrist_target_bias_rad` 设置为 `0.7853981634`，即 45 度，用于让 Mimic 生成阶段夹爪开口从朝下调整为斜向下。
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py`  
  从有效 cloud 配置读取 `mimic.wrist_target_bias_rad`，保存到环境配置 `mimic_wrist_target_bias_rad`。
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/mimic_env.py`  
  在 Mimic 生成把源轨迹的绝对 wrist 目标恢复成 5D 动作时加入该偏置；不切换到 6D pose IK，也不改变 raw 采集或 annotation replay 的动作语义。
- `README.md`  
  增加 `mimic.wrist_target_bias_rad` 的使用说明，以及方向相反时改为负值的提示。
- `MODIFICATION_LOG.md`  
  追加本次修改记录。

### 3. 运行了哪些验证
- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe -m py_compile source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/mimic_env.py scripts/local/analyze_hdf5_dataset.py`：通过。
- `rg --line-number "wrist_target_bias_rad|mimic_wrist_target_bias_rad|wrist_bias" configs source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place`：通过，确认配置项、环境读取和 Mimic wrist 偏置应用点均存在。

### 4. 还有哪些没验证
- 尚未在云端 Isaac Sim 可视化确认 `+0.7853981634` 的方向是否正好是你想要的斜向下 45 度。
- 尚未重新运行 Mimic generation 检查成功率是否从 0 提升。
- 尚未 replay/validate 新生成数据。

### 5. 哪些地方需要我人工确认
- 请在云端生成时观察夹爪开口方向；如果朝反方向倾斜，把 `mimic.wrist_target_bias_rad` 改成 `-0.7853981634`。
- 请确认 45 度偏置不会让 wrist 关节接近极限或造成自碰撞。
## 2026-05-28 - Position-priority 5DOF pose-assisted Mimic generation

### 1. Modified files

- `configs/cloud.defaults.yaml`
- `configs/cloud.user.example.yaml`
- `configs/cloud.user.local.yaml`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/latched_differential_ik_action.py`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/mimic_env_cfg.py`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/mimic_env.py`
- `source/koch_mimic/koch_mimic/cloud/scripts/mimic_action_compat.py`
- `scripts/local/analyze_hdf5_dataset.py`
- `README.md`
- `MODIFICATION_LOG.md`

### 2. Why each file changed

- Config files: add `mimic.generation_control_mode: position_priority_pose_ik`, `pose_ik_joint_names`, `orientation_weight`, `orientation_max_step_rad`, and `ik_damping`; reset legacy `wrist_target_bias_rad` to `0.0`.
- `latched_differential_ik_action.py`: add a local position-priority IK action term. It solves XYZ first, then uses the position nullspace for orientation assistance.
- `env_cfg.py`: read the new Mimic generation parameters from the effective cloud config.
- `mimic_env_cfg.py`: switch Mimic generation to the new 7D position-priority pose IK action space when configured.
- `mimic_env.py`: keep using annotated `target_rot` for pose action conversion, and strip old wrist metadata when no manual wrist action exists.
- `mimic_action_compat.py`: preserve annotation replay compatibility by restoring 5D position-only IK plus manual wrist for old keyboard/IK datasets.
- `analyze_hdf5_dataset.py`: diagnose 7D generated actions as position and rotation deltas instead of mislabeling the second-last dimension as wrist.
- `README.md`: document the 7D position-priority generation interface and clarify that `wrist_target_bias_rad` is legacy wrist-only behavior.
- `MODIFICATION_LOG.md`: append this change record.

### 3. Validation run

- `python -m py_compile source\koch_mimic\koch_mimic\cloud\tasks\koch_pick_place\latched_differential_ik_action.py source\koch_mimic\koch_mimic\cloud\tasks\koch_pick_place\env_cfg.py source\koch_mimic\koch_mimic\cloud\tasks\koch_pick_place\mimic_env_cfg.py source\koch_mimic\koch_mimic\cloud\tasks\koch_pick_place\mimic_env.py source\koch_mimic\koch_mimic\cloud\scripts\mimic_action_compat.py scripts\local\analyze_hdf5_dataset.py`: passed.
- `rg -n "generation_control_mode|position_priority_pose_ik|orientation_weight|orientation_max_step_rad|ik_damping|wrist_target_bias_rad" ...`: passed; no remaining `wrist_target_bias_rad: 0.785...` config was found.
- `git diff --check -- ...`: passed; only existing Git line-ending warnings were printed.
- Attempted local config instantiation with plain `python -c ...`; blocked because this local Python environment does not have `gymnasium` / IsaacLab runtime dependencies.

### 4. Not yet validated

- Cloud Isaac Sim generation has not been run yet.
- Small-batch generation with `koch_mimic_demos_annotated_0527v2.hdf5` has not been run yet.
- Replay/validate of newly generated 7D actions has not been run yet.

### 5. Needs manual confirmation

- In cloud visualization, confirm the gripper pose is closer to the collected demonstration while the end effector still reaches the object accurately.
- If XYZ accuracy visibly degrades, reduce `mimic.orientation_weight` or `mimic.orientation_max_step_rad`.
- Confirm downstream replay/training uses the same 7D generation configuration.

## 2026-05-28 - Mimic 生成数据加入视觉训练字段

### 1. 修改了哪些文件

- `source/koch_mimic/koch_mimic/cloud/scripts/_isaaclab_mimic_runner.py`
- `source/koch_mimic/koch_mimic/cloud/scripts/mimic_generation_compat.py`
- `source/koch_mimic/koch_mimic/cloud/recorders/recorders.py`
- `source/koch_mimic/koch_mimic/cloud/recorders/recorders_cfg.py`
- `source/koch_mimic/koch_mimic/cloud/tasks/koch_pick_place/env_cfg.py`
- `README.md`
- `MODIFICATION_LOG.md`

### 2. 每个文件为什么改

- `_isaaclab_mimic_runner.py`
  在执行上游 IsaacLab Mimic 生成脚本前读取有效 cloud 配置，并在 `app.enable_cameras=true` 且 CLI 未显式传入 `--enable_cameras` 时自动补上该参数；同时把生成脚本的 `setup_env_config` 包装为项目本地版本。

- `mimic_generation_compat.py`
  新增 `configure_generation_recorder()`，让 Mimic generation 使用 `KochActionStateRecorderManagerCfg`，从而导出 `obs/rgb_camera/*` 视觉观测；保留原有失败上限 wrapper。

- `recorders.py` / `recorders_cfg.py`
  更新注释和说明，把 recorder 的用途从“只保存 teleop RGB”扩展为“保存 teleop 与 Mimic generation 的相机观测”。

- `env_cfg.py`
  在 `rgb_camera` 观测组中加入 `cam_up_depth` 和 `cam_arm_depth`，与相机 sensor 已启用的 `distance_to_image_plane` 数据类型对齐。生成数据现在应包含 RGB 和 depth 两类视觉训练字段。

- `README.md`
  说明生成 HDF5 会包含低维字段与视觉字段，并补充用本地分析脚本检查 `obs/rgb_camera/*` 的命令。

- `MODIFICATION_LOG.md`
  追加本次修改记录。

### 3. 运行了哪些验证

- `python -m py_compile source\koch_mimic\koch_mimic\cloud\scripts\_isaaclab_mimic_runner.py source\koch_mimic\koch_mimic\cloud\scripts\mimic_generation_compat.py source\koch_mimic\koch_mimic\cloud\recorders\recorders.py source\koch_mimic\koch_mimic\cloud\recorders\recorders_cfg.py source\koch_mimic\koch_mimic\cloud\tasks\koch_pick_place\env_cfg.py`：通过。
- `D:\Env_IDE\anaconda3_202303\envs\pytorch\python.exe scripts\local\analyze_hdf5_dataset.py --dataset-dir D:\Codes\RoboticsProject\datasets\0527 --file koch_mimic_generated_0527v3.hdf5 -n 1 --tree --include-images --no-values`：通过，确认旧 v3 generated 文件没有 `obs/rgb_camera/*` 字段，只有低维 obs、states、actions 和 processed_actions。
- `rg -n "configure_generation_recorder|KochActionStateRecorderManagerCfg|cam_up_depth|cam_arm_depth|enable_cameras|rgb_camera" ...`：通过，确认生成 wrapper、recorder 和 RGB/depth 观测配置均存在。

### 4. 还有哪些没验证

- 尚未在云端 IsaacLab / Isaac Sim 中重新运行 Mimic generation。
- 尚未实际打开新生成的 HDF5 确认 `obs/rgb_camera/cam_up_rgb`、`obs/rgb_camera/cam_arm_rgb`、`obs/rgb_camera/cam_up_depth`、`obs/rgb_camera/cam_arm_depth` 已写入。
- 尚未用包含视觉字段的新 generated HDF5 跑训练脚本或 replay/validate。

### 5. 哪些地方需要我人工确认

- 请在云端重新生成一个小批量文件，并用 README 中的 `--tree --include-images --no-values` 命令确认 `obs/rgb_camera/*` 字段存在。
- 请确认训练代码期望的图像 key 名称是否接受 `obs/rgb_camera/cam_up_rgb` 这类嵌套路径；如果训练框架要求平铺 key，需要再加一个导出/转换脚本。
- 请留意 HDF5 文件大小；RGB+depth 两个视角会明显增加磁盘占用，正式批量生成前建议先用 1-3 条轨迹估算容量。
