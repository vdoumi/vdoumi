### python timeline_controller.py --timeline '[{"gestures": "small-nod", "duration_sec": 10}, {"gestures": "small-nod", "duration_sec": 2}, {"gestures": "think", "duration_sec": 2}, {"gestures": "think", "duration_sec": 2}, {"gestures": "small-nod", "duration_sec": 2}, {"gestures": "think", "duration_sec": 10}]'
import json
import socket
import time
from threading import Thread
import math
import random
import sys

from stabilizer import Stabilizer

# --- Stabilizer Setup ---
stabilizers = {
    'pitch': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'yaw': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'roll': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'mar': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'ear_left': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'ear_right': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'x_ratio_left': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'y_ratio_left': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'x_ratio_right': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'y_ratio_right': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'mouth_distance': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'ahoge': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'front': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'side': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'back': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
    'sideup': Stabilizer(state_num=2, measure_num=1, cov_process=0.01, cov_measure=0.1),
}

def init_TCP():
    port = 5066

    # '127.0.0.1' = 'localhost' = your computer internal data transmission IP
    address = ('127.0.0.1', port)
    # address = ('192.168.0.107', port)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        # print(socket.gethostbyname(socket.gethostname()) + "::" + str(port))
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)
        
        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

sock = init_TCP()

def send_pose_to_unity(pitch, yaw, roll, ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right, mar, mouth_distance, ahoge, front, side, back, sideup):
    """Sends pose data to Unity via TCP."""
    msg = f"{pitch} {yaw} {roll} {ear_left} {ear_right} {x_ratio_left} {y_ratio_left} {x_ratio_right} {y_ratio_right} {mar} {mouth_distance} {ahoge} {front} {side} {back} {sideup}"
    
    try:
        sock.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("error while sending :: " + str(e))

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

def interpolate(from_val, to_val, duration):
    """Smoothly interpolates between two values over a given duration."""
    steps = int(duration * 60) # Assuming 60 FPS
    if steps == 0:
        return [to_val]
    
    delta = (to_val - from_val) / steps
    return [from_val + i * delta for i in range(steps)]

def animation_thread(duration, pitch_vals, yaw_vals, roll_vals, gesture_duration=None):
    """Simulates lip-sync and sends pose data to Unity."""
    end_time = time.time() + duration
    
    pitch_len = len(pitch_vals)
    yaw_len = len(yaw_vals)
    roll_len = len(roll_vals)
    
    blink_counter = 0

    if gesture_duration == None:
        gesture_duration = duration
    else:
        pitch_vals += interpolate(pitch_vals[-1], 0, duration - gesture_duration)
        yaw_vals += interpolate(yaw_vals[-1], 0, duration - gesture_duration)
        roll_vals += interpolate(roll_vals[-1], 0, duration - gesture_duration) 

    i = 0
    while time.time() < end_time:
        # Simulate mar value changing with speech using a sine wave
        mar = 0.5 + 0.5 * math.sin(time.time() * 20)
        stabilizers['mar'].update([[mar]])
        
        # Simulate mouth distance based on mar
        mouth_distance = mar * 50 + random.uniform(-1, 1)
        stabilizers['mouth_distance'].update([[mouth_distance]])

        # Simulate eye blinking
        if blink_counter > 0:
            ear = 0.0
            blink_counter -= 1
        elif random.random() < 0.01:
            ear = 0.0
            blink_counter = 5 # Blink for 5 frames
        else:
            ear = 1.0
            
        stabilizers['ear_left'].update([[ear]])
        stabilizers['ear_right'].update([[ear]])

        # Simulate eye rolling
        x_ratio = math.sin(time.time() * 0.5) * 0.1 + 0.5
        y_ratio = math.cos(time.time() * 0.5) * 0.1 + 0.5
        stabilizers['x_ratio_left'].update([[x_ratio]])
        stabilizers['y_ratio_left'].update([[y_ratio]])
        stabilizers['x_ratio_right'].update([[x_ratio]])
        stabilizers['y_ratio_right'].update([[y_ratio]])

        # Simulate hair movement
        ahoge = math.sin(time.time() * 2) * 0.5
        front = math.sin(time.time() * 2.5) * 0.2
        side = math.sin(time.time() * 3) * 0.4
        back = math.sin(time.time() * 1.5) * 0.1
        sideup = math.sin(time.time() * 3.5) * 0.2
        
        stabilizers['ahoge'].update([[ahoge]])
        stabilizers['front'].update([[front]])
        stabilizers['side'].update([[side]])
        stabilizers['back'].update([[back]])
        stabilizers['sideup'].update([[sideup]])

        # Get the pose values for the current frame
        pitch = pitch_vals[i] if i < pitch_len else pitch_vals[-1]
        yaw = yaw_vals[i] if i < yaw_len else yaw_vals[-1]
        roll = roll_vals[i] if i < roll_len else roll_vals[-1]

        stabilizers['pitch'].update([[pitch]])
        stabilizers['yaw'].update([[yaw]])
        stabilizers['roll'].update([[roll]])
        
        send_pose_to_unity(
            stabilizers['pitch'].get()[0][0],
            stabilizers['yaw'].get()[0][0],
            stabilizers['roll'].get()[0][0],
            stabilizers['ear_left'].get()[0][0],
            stabilizers['ear_right'].get()[0][0],
            stabilizers['x_ratio_left'].get()[0][0],
            stabilizers['y_ratio_left'].get()[0][0],
            stabilizers['x_ratio_right'].get()[0][0],
            stabilizers['y_ratio_right'].get()[0][0],
            stabilizers['mar'].get()[0][0],
            stabilizers['mouth_distance'].get()[0][0],
            stabilizers['ahoge'].get()[0][0],
            stabilizers['front'].get()[0][0],
            stabilizers['side'].get()[0][0],
            stabilizers['back'].get()[0][0],
            stabilizers['sideup'].get()[0][0]
        )
        
        i += 1
        time.sleep(1/60)

def main(timeline, poses):
    """Main function to process the timeline."""
    for event in timeline:
        print(f"Processing event: {event}")

        # Get gesture and duration
        gesture = event.get("gestures")
        duration = event.get("duration_sec")

        if gesture:
            pose_data = poses.get(gesture)
            gesture_duration = pose_data.get("duration")
            print(gesture_duration)
            if gesture_duration != None and gesture_duration > duration:
                gesture_duration = None # this is maximum guideline, so it's okay to follow entire duration
            if pose_data:
                # Get the current pose from the stabilizers
                from_pitch = stabilizers['pitch'].get()[0][0]
                from_yaw = stabilizers['yaw'].get()[0][0]
                from_roll = stabilizers['roll'].get()[0][0]

                if "from" in pose_data and "to" in pose_data:
                    # Interpolate from "from" to "to"
                    from_pose = pose_data["from"]
                    to_pose = pose_data["to"]
                    
                    interpolate_duration = duration if gesture_duration == None else gesture_duration
                    pitch_vals = interpolate(from_pose[0], to_pose[0], interpolate_duration)
                    yaw_vals = interpolate(from_pose[1], to_pose[1], interpolate_duration)
                    roll_vals = interpolate(from_pose[2], to_pose[2], interpolate_duration)

                elif "to" in pose_data:
                    # Move directly to "to"
                    to_pose = pose_data["to"]
                    pitch_vals = interpolate(from_pitch, to_pose[0], 1) # 1 second transition
                    yaw_vals = interpolate(from_yaw, to_pose[1], 1)
                    roll_vals = interpolate(from_roll, to_pose[2], 1)
                
                else:
                    # No pose data, do nothing
                    pitch_vals = [from_pitch]
                    yaw_vals = [from_yaw]
                    roll_vals = [from_roll]

                animation_thread(duration, pitch_vals, yaw_vals, roll_vals, gesture_duration)

        else:
            # If no gesture, just wait for the duration
            time.sleep(duration)

import argparse
def _parse_args(argv=None):
  p = argparse.ArgumentParser(description="Turn solver output into a lecture-style script.")
  p.add_argument("--timeline", required=True, help="json timeline")
  return p.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args()
    timeline = json.loads(args.timeline)
    
    # Load the timeline of events from timeline.json
    # with open('timeline.json') as f:
    #     timeline = json.load(f)

    # Load poses from pose.json
    with open('pose.json') as f:
        poses = json.load(f)

    main(timeline, poses)
