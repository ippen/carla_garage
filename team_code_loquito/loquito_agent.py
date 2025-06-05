import os
import time
import json
import carla
import torch
import numpy as np
from collections import deque
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from leaderboard.autoagents.autonomous_agent_local import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track
from utils import convert_numpy_image_to_tensor
from navigator import Navigator
# from debug import print_grid

from loquito.lib.utils import get_images_tensor, load_torchscript_model
from loquito.lib.transforms import process_waypoints_in_cs0, get_waypoint_diff


LOQUITO_RECORDING_DIR = os.getenv("LOQUITO_RECORDING_DIR")
LOQUITO_LIVE_RESULTS = os.getenv("LOQUITO_LIVE_RESULTS")

def get_entry_point():
    return 'LoquitoAgent'

class LoquitoAgent(AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        """
        Initialize agent, load model, and setup configurations.
        """
        self.track = Track.SENSORS
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        model_path = "/work/data01/ippen/data/models/loquito_v1m_hd/2025-04-04_23-22-30/model_epoch_3_torchscript.pt"

        self.model, extra_files = load_torchscript_model(model_path, self.device)
        self.image_size = eval(extra_files["image_size"])

        self.supersample_factor = 10 # Corresponds to delta_seconds/supersample_factor = 0.05 seconds per timestep (default of carla leaderboard)
        self.delta_seconds = 0.5 / self.supersample_factor
        self.image_buffer = deque(maxlen=self.supersample_factor)
        
        self.navigator = Navigator(self.org_dense_route_world_coord)

        # x, y, yaw = self.navigator.dense_route_world_coord[0][0].location.x, self.navigator.dense_route_world_coord[0][0].location.y, self.navigator.dense_route_world_coord[0][0].rotation.yaw
        # print_grid(x, y, yaw)

        # Parameters for Loquito Driving Behavior
        self.min_velocity = 0.5
        self.max_velocity = 6.5
        self.min_actuation = 0.25
        self.deadband = 0.05
        self.stopping_threshold = 0.9
        self.velocity_error_gain = 0.5

        if LOQUITO_RECORDING_DIR:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.recording_dir = os.path.join(LOQUITO_RECORDING_DIR, "loquito", time_str, route_index)
            os.makedirs(self.recording_dir, exist_ok=True)
            for subdir in ["rgb_front", "rgb_left", "rgb_right", "rgb_rear", "rgb_follow", "rgb_topdown", "data"]:
                os.makedirs(os.path.join(self.recording_dir, subdir), exist_ok=True)
            
            # Save the route of the navigator
            with open(os.path.join(self.recording_dir, "route.json"), "w") as f:
                json.dump(self.navigator.to_dict(), f, indent=4)

            # Save metadata
            metadata = {
                "route_index": route_index,
                "model_path": model_path,
                "image_size": self.image_size,
                "supersample_factor": self.supersample_factor,
                "delta_seconds": self.delta_seconds,
                "parameters": {
                    "min_velocity": self.min_velocity,
                    "max_velocity": self.max_velocity,
                    "min_actuation": self.min_actuation,
                    "deadband": self.deadband,
                    "stopping_threshold": self.stopping_threshold,
                    "velocity_error_gain": self.velocity_error_gain
                }
            }
            with open(os.path.join(self.recording_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
                num_workers = max(1, min(32, os.cpu_count()))
                print(f"Using {num_workers} workers for recording")
                self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        self.counter_live_result = 0


    def sensors(self):
        """Define and return the sensor configuration for the agent."""
        sensors = [
            {
                "type": "sensor.camera.rgb", "id": "rgb_front",
                "x": 1.3, "y": 0.0, "z": 2.3, 'roll': 0.0, 'pitch': 0.0, "yaw": 0, "fov": 100, "width": 800, "height": 600,
                "lens_circle_multiplier": 3.0, "lens_circle_falloff": 3.0, "chromatic_aberration_intensity": 0.5, "chromatic_aberration_offset": 0.0
            },
            {
                "type": "sensor.camera.rgb", "id": "rgb_left",
                "x": 1.3, "y": 0.0, "z": 2.3, 'roll': 0.0, 'pitch': 0.0, "yaw": -60, "fov": 100, "width": 800, "height": 600,
                "lens_circle_multiplier": 3.0, "lens_circle_falloff": 3.0, "chromatic_aberration_intensity": 0.5, "chromatic_aberration_offset": 0.0
            },
            {
                "type": "sensor.camera.rgb", "id": "rgb_right",
                "x": 1.3, "y": 0.0, "z": 2.3, 'roll': 0.0, 'pitch': 0.0, "yaw": 60, "fov": 100, "width": 800, "height": 600,
                "lens_circle_multiplier": 3.0, "lens_circle_falloff": 3.0, "chromatic_aberration_intensity": 0.5, "chromatic_aberration_offset": 0.0
            },
            {
                "type": "sensor.camera.rgb", "id": "rgb_rear",
                "x": -1.3, "y": 0.0, "z": 2.3, 'roll': 0.0, 'pitch': 0.0, "yaw": 180, "fov": 100, "width": 800, "height": 600,
                "lens_circle_multiplier": 3.0, "lens_circle_falloff": 3.0, "chromatic_aberration_intensity": 0.5, "chromatic_aberration_offset": 0.0
            },
            {
                'type': 'sensor.speedometer', 'id': 'Speed'
            },
            {
                'type': 'sensor.egolocation', 'id': 'EgoLocation'
            }
        ]
        if LOQUITO_RECORDING_DIR:
            sensors.append({
                "type": "sensor.camera.rgb", "id": "rgb_follow",
                "x": -6.0, "y": 0.0, "z": 3.0, 'roll': 0.0, 'pitch': -15, "yaw": 0, "fov": 100, "width": 1280, "height": 720
            })
            sensors.append({
                "type": "sensor.camera.rgb", "id": "rgb_topdown",
                "x": 0.0, "y": 0.0, "z": 15.0, 'roll': 0.0, 'pitch': -90.0, "yaw": 0, "fov": 120, "width": 1024, "height": 1024
            })
        if LOQUITO_LIVE_RESULTS:
            sensors.append({
                "type": "sensor.camera.rgb", "id": "rgb_follow_live",
                "x": -6.0, "y": 0.0, "z": 3.0, 'roll': 0.0, 'pitch': -15, "yaw": 0, "fov": 100, "width": 640, "height": 360
            })

        return sensors
    
    def save(self, input_data, idx, timestamp, target_waypoint, target_waypoint_in_cs0, throttle, steering, brake,
             pred_waypoint_deltas=None, pred_waypoints=None, pred_stop=None, pred_steer=None, embedding_history=None, attention_weights=None):
        
        # Save the numpy image to disk
        for sensor_id in ["rgb_front", "rgb_left", "rgb_right", "rgb_rear", "rgb_follow", "rgb_topdown"]:
            if sensor_id in input_data:
                image = input_data[sensor_id][1]
                # BGR to RGB
                image = image[:, :, [2, 1, 0]]
                image = Image.fromarray(image)
                image.save(os.path.join(self.recording_dir, sensor_id, f"{timestamp:.3f}.jpg"))
        
        # Save the ego location
        _, ego_location = input_data["EgoLocation"]
        ego_x, ego_y, ego_yaw = ego_location["x"], ego_location["y"], ego_location["yaw"]

        _, current_velocity = input_data["Speed"]
        current_velocity = current_velocity["speed"]

        target_x, target_y = target_waypoint[0], target_waypoint[1]
        target_x_in_cs0, target_y_in_cs0 = target_waypoint_in_cs0[0], target_waypoint_in_cs0[1]

        data = {
            "idx": int(idx),
            "timestamp": float(timestamp),
            "ego_location": {
                "x": float(ego_x),
                "y": float(ego_y),
                "yaw": float(ego_yaw)
            },
            "control": {
                "throttle": float(throttle),
                "steering": float(steering),
                "brake": float(brake)
            },
            "current_velocity": float(current_velocity),
            "target_waypoint": {
                "x": float(target_x),
                "y": float(target_y)
            },
            "target_waypoint_in_cs0": {
                "x": float(target_x_in_cs0),
                "y": float(target_y_in_cs0)
            },
            "navigator": {
                    "idx": int(self.navigator.current_index),
                    "location": {
                        "x": float(self.navigator.locations[self.navigator.current_index][0]),
                        "y": float(self.navigator.locations[self.navigator.current_index][1])
                    },
                    "command": int(self.navigator.commands[self.navigator.current_index]),
                },
            "model_outputs": {
                "waypoint_deltas": pred_waypoint_deltas.tolist() if pred_waypoint_deltas is not None else None,
                "waypoints": pred_waypoints.tolist() if pred_waypoints is not None else None,
                "stop": float(pred_stop) if pred_stop is not None else None,
                "steer": float(pred_steer) if pred_steer is not None else None,
                "embedding_history": embedding_history.tolist() if embedding_history is not None else None,
                "attention_weights": attention_weights.tolist() if attention_weights is not None else None
            }
        }

        with open(os.path.join(self.recording_dir, "data", f"{timestamp:.3f}.json"), "w") as f:
            json.dump(data, f, indent=4)
            

    def run_step(self, input_data, timestamp, sensors=None):
        """
        Execute one step of the autonomous driving loop.
        """
        images = {
            sensor_id: input_data[sensor_id][1]
            for sensor_id in ["rgb_front", "rgb_left", "rgb_right", "rgb_rear"]
        }

        idx, ego_location = input_data["EgoLocation"]
        _, current_velocity = input_data["Speed"]
        current_velocity = current_velocity["speed"]

        target_waypoint = self.navigator.get_next_target_waypoint(ego_location["x"], ego_location["y"], distance=20.0)

        target_x_in_cs0, target_y_in_cs0, _ = get_waypoint_diff(ego_location["x"], ego_location["y"], ego_location["yaw"] / 180 * np.pi,
                                                                target_waypoint[0], target_waypoint[1]) 
        
        tensor_target_waypoint_in_cs0 = torch.tensor([target_x_in_cs0, target_y_in_cs0], device=self.device, dtype=torch.float32)

        tensor_images_t = torch.cat([convert_numpy_image_to_tensor(images[sensor_id]).unsqueeze(0) for sensor_id in ["rgb_front", "rgb_left", "rgb_right", "rgb_rear"]], dim=0)
    
        # Wait until we have enough history
        if len(self.image_buffer) < self.image_buffer.maxlen:
            self.image_buffer.append(tensor_images_t)
            throttle, steer, brake= 0.0, 0.0, 0.0
            pred_waypoint_deltas, pred_waypoints, pred_stop, pred_steer, embedding_history, attention_weights = None, None, None, None, None, None
        else:
            tensor_images_tm1 = self.image_buffer[0]
            self.image_buffer.append(tensor_images_t)

            pred_waypoint_deltas, pred_waypoints, pred_stop, pred_steer, embedding_history, attention_weights = self._inference(tensor_images_tm1, tensor_images_t, tensor_target_waypoint_in_cs0)

            throttle, steer, brake = self._compute_control(pred_waypoint_deltas, pred_stop, pred_steer, current_velocity)

        if LOQUITO_RECORDING_DIR:
            self.executor.submit(self.save, input_data, idx, timestamp, target_waypoint,
                     [target_x_in_cs0, target_y_in_cs0], throttle, steer, brake,
                     pred_waypoint_deltas, pred_waypoints, pred_stop, pred_steer,
                     embedding_history, attention_weights)
        
        if LOQUITO_LIVE_RESULTS:
            if self.counter_live_result % 4 == 0:
                # Save the numpy image to disk
                image = input_data["rgb_follow_live"][1]
                # BGR to RGB
                image = image[:, :, [2, 1, 0]]
                image = Image.fromarray(image)
                image.save("live_results.jpg")
            self.counter_live_result += 1

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

        return control


    def _inference(self, tensor_images_tm1, tensor_images_t, tensor_target_waypoint):
        """Perform model inference to predict waypoints."""
        tensor_images_tm1 = get_images_tensor(tensor_images_tm1, img_size=self.image_size).to(self.device).unsqueeze(0)
        tensor_images_t = get_images_tensor(tensor_images_t, img_size=self.image_size).to(self.device).unsqueeze(0)
        tensor_target_waypoint = tensor_target_waypoint.to(self.device).unsqueeze(0)

        with torch.no_grad():
            pred_waypoint_deltas, pred_tasks, embedding_history, attention_weights = self.model(
                tensor_images_tm1, tensor_images_t, tensor_target_waypoint
            )
        pred_waypoints = np.array(process_waypoints_in_cs0(pred_waypoint_deltas).squeeze(0).cpu())
        pred_waypoint_deltas = np.array(pred_waypoint_deltas.squeeze(0).cpu())
        pred_tasks = np.array(pred_tasks.squeeze(0).cpu())
        pred_stop, pred_steer = pred_tasks[0], pred_tasks[1]
        embedding_history = np.array(embedding_history.squeeze(0).cpu())
        attention_weights = np.array(attention_weights.squeeze(0).cpu())
        
        return pred_waypoint_deltas, pred_waypoints, pred_stop, pred_steer, embedding_history, attention_weights

 
    def _compute_control(self, waypoints, stopping, steering, current_velocity):
        """Compute throttle, steering, and braking values for vehicle control."""

        # Compute weighted desired velocity from waypoints
        velocities = []
        for dx, dy, _ in waypoints[:4]:
            distance = np.sqrt(dx**2 + dy**2)
            velocity = 2.0 * distance   # waypoints are in 0.5 second intervals
            velocities.append(velocity)

        desired_velocity = sum(v * w for v, w in zip(velocities, [0.4, 0.3, 0.2, 0.1]))

        desired_velocity = min(desired_velocity, self.max_velocity)

        # Stopping logic
        if stopping >= self.stopping_threshold:
            throttle = 0.0
            brake = 1.0
            desired_velocity = 0.0
        else:
            velocity_error = desired_velocity - current_velocity
            accel_cmd = self.velocity_error_gain * velocity_error   # Simple proportional scaling

            if accel_cmd > self.deadband:
                throttle = np.clip(accel_cmd, self.min_actuation, 1.0)
                brake = 0.0
            elif accel_cmd < -self.deadband:
                throttle = 0.0
                brake = np.clip(-accel_cmd, self.min_actuation, 1.0)
            else:
                throttle = 0.0
                brake = 0.0

        steering = float(steering)

        # Logging (if enabled)
        if LOQUITO_LIVE_RESULTS:
            log_lines = [
                "  Loquito Control",
                "-" * 40,
                f"  Predicted Stop     : {stopping:.3f}",
                *[
                    f"  Desired Velocity {i} : {v:.3f} m/s"
                    for i, v in enumerate(velocities)
                ],
                f"  Desired Velocity   : {desired_velocity:.3f} m/s",
                f"  Current Velocity   : {current_velocity:.3f} m/s",
                f"  Delta Velocity     : {desired_velocity - current_velocity:.3f} m/s",
                f"  Throttle           : {throttle:.3f}",
                f"  Steering           : {steering:.3f}",
                f"  Brake              : {brake:.3f}",
                f"  Acceleration Cmd   : {accel_cmd:.3f}" if stopping < self.stopping_threshold else "  PID Out            : N/A"
            ]

            with open("live_results.txt", "w") as log_file:
                log_file.write("\n".join(log_lines))

        return throttle, steering, brake

    
    def destroy(self, results=None):
        """
        Gets called after a route finished.
        The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
        Also writes logging files to disk.
        """
        if LOQUITO_RECORDING_DIR:
            print("Waiting for all recording threads to finish")
            self.executor.shutdown(wait=True)
            print("All recording threads finished")

        print("Destroying agent")
        super().destroy()

