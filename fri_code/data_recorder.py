import os
import threading
import time

import cv2
import intera_interface
import numpy as np
import pyrealsense2 as rs
import rospy
from matplotlib import pyplot as plt
from pynput import keyboard
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LEROBOT_HOME

IMG_X = 640  # Camera observation width
IMG_Y = 480  # Camera observation height


class RealSenseCamera:
    """Class to interface with Intel RealSense camera and observe image observations."""

    def __init__(self):
        """Initialize RealSense camera pipeline to store RGB images with dimensions (IMG_X, IMG_Y)."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, IMG_X, IMG_Y, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

    def get_frame(self):
        """Capture a frame and return it as a NumPy array with dimensions (IMG_X, IMG_Y, 3)"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return color_image_rgb

    def close(self):
        """Shuts down the RealSense camera pipeline."""
        self.pipeline.stop()


class Recorder:
    """Class to record states, actions, and observations of the Sawyer Robot's end effector in LeRobot format"""

    def __init__(self, dataset_name: str, camera: RealSenseCamera, sample_interval: float,
                 store_observations: bool = False, side="right"):
        """Initialize the recorder with LeRobot dataset, sampling interval, and demonstration recording parameters"""
        self.dataset_name = dataset_name
        self.sample_interval = sample_interval
        self.store_observations = store_observations
        
        # Create the LeRobot dataset
        self.dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            robot_type="sawyer",
            fps=1/sample_interval,
            features={
                "observation.image": {
                    "dtype": "image",
                    "shape": (IMG_Y, IMG_X, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state.position": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                },
                "observation.state.orientation": {
                    "dtype": "float32",
                    "shape": (4,),
                    "names": ["x", "y", "z", "w"],
                },
                "observation.state.gripper_status": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["gripper_status"],
                }
                "action.delta_position": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                },
                "action.delta_orientation": {
                    "dtype": "float32",
                    "shape": (4,),
                    "names": ["x", "y", "z", "w"],
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["seconds"],
                },
            },
            image_writer_threads=5,
        )

        self.limb = intera_interface.Limb(side)
        self.gripper = intera_interface.Gripper(side)
        self.camera = camera

        # Demonstration recording initialization
        self.demo_num = 0
        self.recording = False
        self.sample_count = 0
        self.prev_state = None
        self.record_thread = None
        self.start_time = None

    def start_recording(self):
        """Begin recording data by initializing a thread to collect samples every sample_interval seconds."""
        self.recording = True
        description = f"Sample demonstration {self.demo_num}"
        # Uncomment the line below to add descriptions to demonstrations
        # description = input("Enter the description for this demonstration...\n")
        self.start_time = time.time()
        self.sample_count = 0
        self.prev_state = None
        self.record_thread = threading.Thread(target=self.record_sample_thread, args=())
        print(f"Recording demonstration {self.demo_num}. Press <q> to end the recording.")
        self.record_thread.start()

    def stop_recording(self):
        """Stop the recording process and finalize the data in the current demonstration."""
        if self.recording:
            self.recording = False
            self.record_thread.join()
            
            # Save the episode with a task description
            description = f"Sample demonstration {self.demo_num}"
            self.dataset.save_episode(task=description)
            
            self.demo_num += 1
            print(f"\nDemonstration {self.demo_num} recorded.")
            print("Press <ENTER> to start recording another demonstration or press <q> to exit the program.")

    def record_sample(self):
        """Store endpoint state, action, and observation as a frame in LeRobot dataset."""
        if self.recording and not rospy.is_shutdown():
            timestamp_time = time.time() - self.start_time
            endpoint_pose = self.limb.endpoint_pose()
            position = endpoint_pose["position"]
            orientation = endpoint_pose["orientation"]
            gripper_status = self.gripper.state()
            observation = self.camera.get_frame()

            # Calculate delta actions
            if self.prev_state is None:  # The first action should just be the first state
                delta_position = position
                delta_orientation = orientation
            else:
                curr_x, curr_y, curr_z = endpoint_pose["position"]
                prev_x, prev_y, prev_z = self.prev_state["position"]
                delta_position = intera_interface.Limb.Point(curr_x - prev_x, curr_y - prev_y, curr_z - prev_z)

                # Quaternion subtraction is different from regular, element-wise subtraction
                curr_w, curr_x, curr_y, curr_z = orientation
                prev_w, prev_x, prev_y, prev_z = self.prev_state["orientation"]

                # Compute the conjugate of the previous orientation
                prev_conj_w = prev_w
                prev_conj_x = -prev_x
                prev_conj_y = -prev_y
                prev_conj_z = -prev_z

                # Quaternion multiplication (prev_conjugate * current_orientation)
                delta_w = prev_conj_w * curr_w - prev_conj_x * curr_x - prev_conj_y * curr_y - prev_conj_z * curr_z
                delta_x = prev_conj_w * curr_x + prev_conj_x * curr_w + prev_conj_y * curr_z - prev_conj_z * curr_y
                delta_y = prev_conj_w * curr_y - prev_conj_x * curr_z + prev_conj_y * curr_w + prev_conj_z * curr_x
                delta_z = prev_conj_w * curr_z + prev_conj_x * curr_y - prev_conj_y * curr_x + prev_conj_z * curr_w

                delta_orientation = intera_interface.Limb.Quaternion(delta_x, delta_y, delta_z, delta_w)

            self.prev_state = endpoint_pose
            
            # Add frame to LeRobot dataset
            self.dataset.add_frame({
                "observation.image": observation,
                "observation.state.position": np.array(position, dtype=np.float32),
                "observation.state.orientation": np.array(orientation, dtype=np.float32),
                "observation.state.gripper_status": np.array([gripper_status], dtype=np.float32),
                "action.delta_position": np.array(delta_position, dtype=np.float32),
                "action.delta_orientation": np.array(delta_orientation, dtype=np.float32),
                "timestamp": np.array([timestamp_time], dtype=np.float32),
            })
            
            # Optionally save images separately
            if self.store_observations:
                # Create output directory if it doesn't exist
                os.makedirs(f"{self.dataset_name}/demo_{self.demo_num}", exist_ok=True)
                try:
                    plt.imsave(f"{self.dataset_name}/demo_{self.demo_num}/sample_{self.sample_count}.png", observation)
                except FileNotFoundError:
                    pass

            print(f"Sample {self.sample_count}:")
            print(f"\tTimestamp: {timestamp_time:.2f}")
            print(f"\tPosition: {position}")
            print(f"\tOrientation: {orientation}")
            print(f"\tGripper Status: {gripper_status}")
            print(f"\tDelta Position: {delta_position}")
            print(f"\tDelta Orientation: {delta_orientation}")
            print(f"\tObservation shape: {observation.shape}\n")
            self.sample_count += 1

    def record(self):
        """Monitor keyboard events to control demonstration recording events."""
        with keyboard.Events() as events:
            for event in events:
                if isinstance(event, keyboard.Events.Press):
                    if event.key == keyboard.Key.enter and not self.recording:
                        self.start_recording()
                    elif event.key == keyboard.KeyCode.from_char('q'):
                        if self.recording:
                            self.stop_recording()
                        else:
                            print("\nQuitting program...")
                            break  # Exit the loop and quit if <q> is pressed when not recording
                    elif event.key == keyboard.KeyCode.from_char('\x03'):
                        if self.recording:
                            self.stop_recording()
                        print("<Ctrl+C> pressed. Quitting program...")
                        exit(0)

    def record_sample_thread(self):
        """Thread to record samples to also allow for monitoring keyboard input"""
        while self.recording:
            self.record_sample()
            time.sleep(self.sample_interval)
            
    def finalize(self):
        """Consolidate the dataset when recording is complete"""
        self.dataset.consolidate(run_compute_stats=True)
        print(f"Dataset consolidated and saved to {LEROBOT_HOME}/{self.dataset_name}")


def main():
    dataset_name = input("Enter the name for the demonstration dataset:\n")
    sample_interval = float(input("Enter the sampling interval (in seconds) for data collection:\n"))
    camera = RealSenseCamera()
    rospy.init_node("endpoint_recorder")
    
    recorder = Recorder(dataset_name, camera, sample_interval, store_observations=True)
    print("Press <ENTER> to start recording a demonstration or press <q> to exit the program.")
    recorder.record()
    
    # Finalize and consolidate the dataset
    recorder.finalize()
    print(f"Total demonstrations recorded: {recorder.demo_num}")
    camera.close()


if __name__ == "__main__":
    main()
