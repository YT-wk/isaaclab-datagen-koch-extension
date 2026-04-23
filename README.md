# isaaclab-datagen-koch-extension
0418 done,four devices for teleop

local runs:

```
cd D:\Codes\RoboticsProject\LeadArmData2Cloud\my_scripts

python koch_leader_ssh_streamer.py --password qzarjzq0 --stream-port 55000 --port COM5
```

cloud platform runs:

**1. fixed base + keyboard control arm**

```
./isaaclab.sh -p scripts/tools/record_demos_remote_master_arm.py   --task Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0   --dataset_file /root/gpufree-data/datasets/koch_raw_keyboard_fixed.hdf5  --teleop_fixed_base   --arm_teleop_source keyboard   --num_demos 20   --num_success_steps 10   --enable_cameras 
```

**2. movable base + keyboard control arm**

```
./isaaclab.sh -p scripts/tools/record_demos_remote_master_arm.py   --task Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0   --dataset_file /root/gpufree-data/datasets/koch_raw_keyboard_mobile.hdf5   --no-teleop_fixed_base   --arm_teleop_source keyboard   --num_demos 20   --num_success_steps 10   --enable_cameras 
```

**3. fixed base + JSONL control arm**

```
./isaaclab.sh -p scripts/tools/record_demos_remote_master_arm.py   --task Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0   --dataset_file /root/gpufree-data/datasets/koch_raw_remote_fixed.hdf5   --teleop_fixed_base   --arm_teleop_source remote_master_arm   --num_demos 20   --num_success_steps 10   --enable_cameras   --stream-port 55000   --joint-signs 1,1,1,1,-1   --joint-offsets 0.139626,-0.785398,-1.570796,1.396263,-1.570796 
```

**4. movable base + JSONL control arm**

```
./isaaclab.sh -p scripts/tools/record_demos_remote_master_arm.py   --task Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0   --dataset_file /root/gpufree-data/datasets/koch_raw_remote_mobile.hdf5   --no-teleop_fixed_base   --arm_teleop_source remote_master_arm   --num_demos 20   --num_success_steps 10   --enable_cameras   --stream-port 55000   --joint-signs 1,1,1,1,-1   --joint-offsets 0.139626,-0.785398,-1.570796,1.396263,-1.570796 
```


