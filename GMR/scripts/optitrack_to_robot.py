from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import threading
import argparse

def main(args):
    # Check if firewall is disabled on this machine
    print("Make sure to disable firewall on both machines:")
    print("On OptiTrack computer: Disable Windows Firewall")
    print("On this computer: sudo ufw disable")

    client = setup_optitrack(
        server_address=args.server_ip,
        client_address=args.client_ip,
        use_multicast=args.use_multicast,
    )

    # start a thread to client.run()
    thread = threading.Thread(target=client.run)
    thread.start()

    if not client:
        print("Failed to setup OptiTrack client")
        exit(1)

    print(f"OptiTrack client connected: {client.connected()}")
    print("Starting motion retargeting...")

    retarget = GMR(
            src_human="fbx",
            tgt_robot=args.robot,
            actual_human_height=1.6,
        )
    viewer = RobotMotionViewer(robot_type="unitree_g1")

    while True:
        frame = client.get_frame()
        frame_number = client.get_frame_number()
        qpos = retarget.retarget(frame)
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            rate_limit=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.200.160")
    parser.add_argument("--client_ip", type=str, default="192.168.200.117")
    parser.add_argument("--use_multicast", type=bool, default=False)
    parser.add_argument("--robot", choices=["unitree_g1", "booster_t1", "stanford_toddy","hightorque_hi","pi_football"], type=str, default="unitree_g1")
    args = parser.parse_args()
    main(args)
    