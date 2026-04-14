# isaaclab-datagen-koch-extension
0414 done,only koch arm teleop

local runs:

```
cd D:\Codes\RoboticsProject\LeadArmData2Cloud\my_scripts

python koch_leader_ssh_streamer.py --password qzarjzq0 --stream-port 55000 --port COM5
```

cloud platform runs:

```
./isaaclab.sh -p scripts/tools/record_demos_remote_master_arm.py   --task Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0   --teleop_device external_master_arm   --dataset_file /root/gpufree-data/datasets/koch_raw.hdf5   --num_demos 20   --num_success_steps 10   --enable_cameras   --stream-port 55000   --joint-signs 1,1,1,1,-1  --joint-offsets 0.139626,-0.785398,-1.570796,1.396263,-1.57079
```

