# Koch Mimic

Koch Mimic 是一个可安装的 IsaacLab 外部项目，用于 Koch 抓放任务、遥操作录制、Isaac Lab Mimic 标注与数据生成，以及本地 leader arm 串口/SSH 推流。

项目按三层组织：

- `source/koch_mimic/koch_mimic/cloud`：云端 IsaacLab / Isaac Sim 任务、Mimic 环境、录制、回放、标注和生成入口。
- `source/koch_mimic/koch_mimic/local`：本地 leader arm 串口读取、Dynamixel 调试和 SSH 推流。
- `source/koch_mimic/koch_mimic/shared`：两端共用的 YAML 配置加载、joint JSONL 协议和常量。

## 安装

云端 IsaacLab 环境：

```bash
cd /abs/path/to/isaaclab-datagen-koch-extension
~/isaaclab/_isaac_sim/python.sh -m pip install -e source/koch_mimic[cloud]
```

本地 leader arm 环境：

```bash
cd D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension
python -m pip install -e source/koch_mimic[local]
```

## 配置

提交到仓库的模板配置：

- `configs/cloud.defaults.yaml`
- `configs/cloud.user.example.yaml`
- `configs/local.defaults.yaml`
- `configs/local.user.example.yaml`

本地私有配置：

- `configs/cloud.user.local.yaml`
- `configs/local.user.local.yaml`

初始化方式：

```bash
cp configs/cloud.user.example.yaml configs/cloud.user.local.yaml
cp configs/local.user.example.yaml configs/local.user.local.yaml
```

运行时配置优先级：

1. `*.defaults.yaml`
2. `*.user.local.yaml`
3. `--config <yaml>` 指定的额外覆盖文件
4. 显式 CLI 参数

也就是说，同一字段优先使用 user local；user local 没配置时才回退到 defaults。

当前 Koch 夹爪资产的语义是 `gripper_open_command_rad=1.3962634016`、`gripper_close_command_rad=-0.1745329252`。`success.require_gripper_open=true` 会要求物体进入容器后夹爪已经真正打开释放；如果放入容器后仍保持闭合，轨迹不会被判为成功。

## 云端运行

云端如使用 IsaacLab 的 Python，统一使用：

```bash
~/isaaclab/_isaac_sim/python.sh <script> [args...]
```

默认任务 ID：

- `Isaac-Koch-Mimic-PickPlace-v0`
- 兼容别名：`Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0`

### 录制示教

```bash
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/record_koch_mimic_demos.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0
```

四种遥操作模式：

四种设备遥操效果视频：

- [Fixed base + keyboard arm](https://github.com/user-attachments/assets/79490d78-5274-4205-91a6-4af35ee0e04f)
- [Mobile base + keyboard arm](https://github.com/user-attachments/assets/735bbd1c-0710-4d0f-8b93-ab4ad5a14406)
- [Fixed base + remote master arm](https://github.com/user-attachments/assets/b69c1bc0-4e4c-4093-bdd0-1148e4e4abaa)
- [Mobile base + remote master arm](https://github.com/user-attachments/assets/a2497755-ddfa-422c-82af-fe3a9e748360)

```bash
# Fixed base + keyboard arm
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/record_koch_mimic_demos.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0 \
  --teleop_fixed_base \
  --arm_teleop_source keyboard
```

```bash
# Mobile base + keyboard arm
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/record_koch_mimic_demos.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0 \
  --no-teleop_fixed_base \
  --arm_teleop_source keyboard
```

```bash
# Fixed base + remote master arm
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/record_koch_mimic_demos.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0 \
  --teleop_fixed_base \
  --arm_teleop_source remote_master_arm
```

```bash
# Mobile base + remote master arm
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/record_koch_mimic_demos.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0 \
  --no-teleop_fixed_base \
  --arm_teleop_source remote_master_arm
```

### 只运行 Teleop

```bash
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/run_koch_mimic_teleop.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0
```

### 回放示教或生成数据

```bash
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/replay_koch_mimic_demos.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0 \
  --dataset_file /root/gpufree-data/datasets/koch_mimic_demos_0526.hdf5
```

### Mimic 自动标注

```bash
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/annotate_koch_mimic_demos.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0 \
  --input_file /root/gpufree-data/datasets/koch_mimic_demos_0526.hdf5 \
  --output_file /root/gpufree-data/datasets/koch_mimic_demos_0526_annotated.hdf5
```

### Mimic 数据生成

```bash
~/isaaclab/_isaac_sim/python.sh /abs/path/to/repo/scripts/cloud/generate_koch_mimic_dataset.py \
  --config /abs/path/to/repo/configs/cloud.user.local.yaml \
  --task Isaac-Koch-Mimic-PickPlace-v0 \
  --input_file /root/gpufree-data/datasets/koch_mimic_demos_0526_annotated.hdf5 \
  --output_file /root/gpufree-data/datasets/koch_mimic_generated.hdf5
```

生成失败样本会按 Mimic 配置写到类似 `<output>_failed.hdf5` 的文件中，便于用本地分析脚本排查。
当前默认 `mimic.generation_control_mode=position_priority_pose_ik`，生成动作接口为 7D：`[dx, dy, dz, dRx, dRy, dRz, gripper]`。其中 `dRx/dRy/dRz` 是 axis-angle 姿态误差接口，不代表 Koch 具备完整 6D 末端自由度；本地 IK 会优先满足 XYZ 位置，再用剩余可动空间尽量贴近示教姿态。`mimic.wrist_target_bias_rad` 仅是旧的单 wrist 关节偏置，应保持 `0.0`，不要再用它处理夹爪整体前倾。

生成入口会按有效配置自动读取 `app.enable_cameras`；该值为 `true` 时，即使命令里没有显式写 `--enable_cameras`，也会打开 IsaacLab 相机渲染。生成 HDF5 会保留低维训练字段和相机字段：

- 低维字段：`actions`、`processed_actions`、`obs/joint_pos`、`obs/joint_vel`、`obs/eef_pos`、`obs/eef_quat`、`obs/gripper_pos`、`obs/object_a_pose`、`obs/object_b_pose`、`states/*`。
- 视觉字段：`obs/rgb_camera/cam_up_rgb`、`obs/rgb_camera/cam_arm_rgb`、`obs/rgb_camera/cam_up_depth`、`obs/rgb_camera/cam_arm_depth`。

视觉字段尺寸会显著增大数据集；如果只想调试成功率，可以临时把 `configs/cloud.user.local.yaml` 里的 `app.enable_cameras` 改为 `false` 后再运行生成。

生成数据集的成功轨迹效果视频：

- [Mimic generated successful trajectory](https://github.com/user-attachments/assets/d625aa62-d13c-4c68-8252-c770a3b37760)

## 本地运行

本地脚本不依赖 IsaacLab，使用本地 Python 或 Conda 环境即可。

### HDF5 数据集分析

```powershell
python D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\scripts\local\analyze_hdf5_dataset.py `
  --dataset-dir D:\Codes\RoboticsProject\datasets\0527 `
  -n 3 `
  --stats
```

查看数据集是否包含视觉训练字段：

```powershell
python D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\scripts\local\analyze_hdf5_dataset.py `
  --dataset-dir D:\Codes\RoboticsProject\datasets\0527 `
  --file koch_mimic_generated_0527v4.hdf5 `
  -n 1 `
  --tree `
  --include-images `
  --no-values
```

Mimic 失败诊断会按 `cloud.defaults.yaml -> cloud.user.local.yaml -> --cloud-config` 的有效配置重算成功条件：

```powershell
python D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\scripts\local\analyze_hdf5_dataset.py `
  --dataset-dir D:\Codes\RoboticsProject\datasets\0527 `
  --mimic-diagnose `
  -n 0 `
  --no-values
```

诊断输出会包含 `grasp_obj_a` 的 near、closed、lifted、grasp frame 计数。若 Mimic 标注提示 `Did not detect completion for the subtask "grasp_obj_a"`，优先用该命令检查是末端距离、物体抬升、还是 gripper closed 判定没有满足。
若旧数据在“关闭夹爪”后被判成功，优先检查诊断输出里的 `gripper_open_command_rad`、`gripper_close_command_rad` 和 `final_progress`；当前语义下 final gripper 需要接近 open command 才能通过最终释放判定。

### leader arm SSH 推流

```powershell
python D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\scripts\local\stream_koch_leader_over_ssh.py `
  --config D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\configs\local.user.local.yaml
```

### Dynamixel 总线调试

```powershell
python D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\scripts\local\debug_koch_leader_bus.py `
  --config D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\configs\local.user.local.yaml `
  status
```

### 串口监视

```powershell
python D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\scripts\local\test_koch_leader_serial.py `
  --config D:\Codes\RoboticsProject\isaaclab-datagen-koch-extension\isaaclab-datagen-koch-extension\configs\local.user.local.yaml
```

## 说明

- `remote_master_arm` 模式需要先在本地启动 `scripts/local/stream_koch_leader_over_ssh.py`。
- `keyboard` 模式不需要本地 leader arm 推流。
- `external/IsaacLab` 只作为参考和兼容资源；项目运行入口在 `scripts/` 和 `source/koch_mimic`。
- 项目支持从任意工作目录运行，前提是已安装 `source/koch_mimic[...]` 并传入正确配置路径。
