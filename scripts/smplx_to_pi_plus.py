import argparse
import pathlib
import os
import time
import csv

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openlong", "pi_football"],
        default="pi_football",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--offset_to_ground",
        default=False,
        action="store_true",
        help="Automatically adjust human motion so the lowest foot is 0.1m above ground.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]
        
        # Print current frame number for tracking
        print(f"Processing frame {i}/{len(smplx_data_frames)-1}")

        # retarget
        qpos = retarget.retarget(smplx_data, offset_to_ground=args.offset_to_ground)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, -0.0]),  
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        
        # 重排序关节角度：从 左腿→左臂→右腿→右臂 到 左腿→右腿→左臂→右臂
        # 当前顺序: 0-5(左腿), 6-10(左臂), 11-16(右腿), 17-21(右臂)
        # 期望顺序: 0-5(左腿), 6-11(右腿), 12-16(左臂), 17-21(右臂)
        reorder_indices = [
            # 左腿 (保持原位)
            0, 1, 2, 3, 4, 5,           # l_hip_pitch → l_ankle_roll
            # 右腿 (从位置11-16移到6-11) 
            11, 12, 13, 14, 15, 16,     # r_hip_pitch → r_ankle_roll
            # 左臂 (从位置6-10移到12-16)
            6, 7, 8, 9, 10,             # l_shoulder_pitch → l_wrist  
            # 右臂 (从位置17-21移到17-21)
            17, 18, 19, 20, 21          # r_shoulder_pitch → r_wrist
        ]
        dof_pos = dof_pos[:, reorder_indices]

        # Check if save as CSV format
        if args.save_path.endswith('.csv'):
            # 保存为CSV格式
            with open(args.save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头 
                header = ['frame']
                header.extend([f'root pos {i}' for i in ['x', 'y','z']])
                header.extend([f'root rot {i}' for i in ['x','y','z','w']])
                
                # pi_plus机器人关节名称 (22个自由度，无腰部和头部关节)
                joint_names = [
                    # 左腿
                    'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
                    # 右腿  
                    'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
                    # 左臂
                    'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
                    # 右臂
                    'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist'
                ]
                header.extend(joint_names)
                writer.writerow(header)

                # 按帧写入数据
                for frame_idx in range(len(qpos_list)):
                    row = [frame_idx]
                    row.extend(root_pos[frame_idx].tolist())
                    row.extend(root_rot[frame_idx].tolist())
                    row.extend(dof_pos[frame_idx].tolist())
                    writer.writerow(row)
            print(f"Saved to {args.save_path}")
        else:
            # Save as pickle format (original behavior)
            import pickle
            local_body_pos = None
            body_names = None
            
            motion_data = {
                "fps": aligned_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                # "local_body_pos": local_body_pos,
                # "link_body_list": body_names,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()
