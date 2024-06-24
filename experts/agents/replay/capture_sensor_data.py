#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to the capture sensor data script, a script that provides users with a baseline for data collection,
which they can later modify to their specific needs, easying the process of creating a database.

This script will start with a CARLA recorder log, spawning the desired sensor configuration at the ego vehicle,
and saving their data into a folder. The exact parameters can be found at the very top of the script and include:

- SENSORS: List of all the sensors tha will be spawned in the simulation
- WEATHER: Weather of the simulation
- RECORDER_INFO: List of all the CARLA recorder logs that will be run. Each recorder has three elements:
    Â· folder: path to the folder with the recorder files
    Â· name: name of the endpoint folder
    Â· start_time: start time of the recorder
    Â· duration: duration of the recorder. 0 to replay it until the end
- DESTINATION_FOLDER: folder where all sensor data will be stored
"""

import time
import os
import carla
import argparse
import random
import json
import threading
import glob
import argparse
import cv2 as cv 
import numpy as np

from queue import Queue, Empty

from leaderboard.envs.sensor_interface import SpeedometerReader, OpenDriveMapReader
from generate_recorder_info import generate_recorder_info

################### User simulation configuration ####################
# 1) Choose the sensors
IMG_WIDTH = 1080
IMG_HEIGHT = 720
def get_sensors():
    
    sensors = [
        [
            'rgb_front',
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            },
        ],
        [
            'seg_front',
            {
                'bp': 'sensor.camera.semantic_segmentation',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            },
        ],
        [
            'depth_front',
            {
                'bp': 'sensor.camera.depth',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            },
        ],
        [
            'rgb_rear',
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': -1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0
            },
        ],
        [
            'rgb_left',
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.30, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0
            },  
        ],
        [
            'seg_left',
            {
                'bp': 'sensor.camera.semantic_segmentation',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.30, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0
            },
        ],
        [
            'depth_left',
            {
                'bp': 'sensor.camera.depth',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.30, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0
            },
        ],
        [
            'rgb_right',
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.30, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0
            },
        ],
        [
            'seg_right',
            {
                'bp': 'sensor.camera.semantic_segmentation',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.30, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0
            },
            
        ],
        [
            'depth_right',
            {
                'bp': 'sensor.camera.depth',
                'image_size_x': IMG_WIDTH, 'image_size_y': IMG_HEIGHT, 'fov': 100,
                'x': 1.3, 'y': 0.0, 'z': 2.30, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0
            },
        ],
        [
            'lidar',
            {
                'bp': 'sensor.lidar.ray_cast',
                'x': 1.3, 'y': 0.0, 'z': 2.50, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'range': 85, 'rotation_frequency': 10, 'channels': 64, 'upper_fov': 10,
                'lower_fov': -30, 'points_per_second': 600000, 'atmosphere_attenuation_rate': 0.004,
                'dropoff_general_rate': 0.45, 'dropoff_intensity_limit': 0.8, 'dropoff_zero_intensity': 0.4
            }
        ],
        [
            'radar',
            {
                'bp': 'sensor.other.radar',
                'x': 1.3, 'y': 0.0, 'z': 2.30, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'points_per_second': 1500, 'range': 100

            }
        ],
        [
            'gnss',
            {
                'bp': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'noise_alt_stddev': 0.000005, 'noise_lat_stddev': 0.000005, 'noise_lon_stddev': 0.000005,
                'noise_alt_bias': 0.0, 'noise_lat_bias': 0.0, 'noise_lon_bias': 0.0
            }
        ],
        [
            'imu',
            {
                'bp': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'noise_accel_stddev_x': 0.001, 'noise_accel_stddev_y': 0.001, 'noise_accel_stddev_z': 0.015,
                'noise_gyro_stddev_x': 0.001,'noise_gyro_stddev_y': 0.001, 'noise_gyro_stddev_z': 0.001
            }
        ],
        #[
        #    'speed',
        #    {
        #        "bp": "sensor.speedometer", 
        #        "reading_frequency": 20
        #    }
        #]
    ]

    return sensors

SENSORS = get_sensors()

# 2) Weatger variants
WEATHERS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "ClearNight": carla.WeatherParameters(5.0,0.0,0.0,10.0,-1.0,-90.0,60.0,75.0,1.0,0.0),
    "CloudyNight": carla.WeatherParameters(60.0,0.0,0.0,10.0,-1.0,-90.0,60.0,0.75,0.1,0.0),
    "WetNight": carla.WeatherParameters(5.0,0.0,50.0,10.0,-1.0,-90.0,60.0,75.0,1.0,60.0),
    "WetCloudyNight": carla.WeatherParameters(60.0,0.0,50.0,10.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    "SoftRainNight": carla.WeatherParameters(60.0,30.0,50.0,30.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    "MidRainyNight": carla.WeatherParameters(80.0,60.0,60.0,60.0,-1.0,-90.0,60.0,0.75,0.1,80.0),
    "HardRainNight": carla.WeatherParameters(100.0,100.0,90.0,100.0,-1.0,-90.0,100.0,0.75,0.1,100.0),
}
WEATHERS_IDS = list(WEATHERS)

# 3) Choose the recorder files
#RECORDER_INFO = generate_recorder_info()

RECORDER_INFO = [
    {
        'folder': "ScenarioLogs/EnterActorFlow_fast",
        'name': 'EnterActorFlow_fast',
        'start_time': 0,
        'duration': 2
    }
]

# 4) Choose the destination folder
DESTINATION_FOLDER = "data_collection"
################# End user simulation configuration ##################

FPS = 20
THREADS = 20
CURRENT_THREADS = 0
AGENT_TICK_DELAY = 10

def create_folders(endpoint, sensors):
    for sensor_id, sensor_bp in sensors:
        sensor_endpoint = f"{endpoint}/{sensor_id}"
        if not os.path.exists(sensor_endpoint):
            os.makedirs(sensor_endpoint)

        if 'gnss' in sensor_bp:
            sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
            with open(sensor_endpoint, 'w') as data_file:
                data_txt = f"Frame,Altitude,Latitude,Longitude\n"
                data_file.write(data_txt)

        if 'imu' in sensor_bp:
            sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
            with open(sensor_endpoint, 'w') as data_file:
                data_txt = f"Frame,Accelerometer X,Accelerometer y,Accelerometer Z,Compass,Gyroscope X,Gyroscope Y,Gyroscope Z\n"
                data_file.write(data_txt)


def add_listener(sensor, sensor_queue, sensor_id):
    sensor.listen(lambda data: sensor_listen(data, sensor_queue, sensor_id))


def sensor_listen(data, sensor_queue, sensor_id):
    sensor_queue.put((sensor_id, data.frame, data))
    return


def get_ego_id(recorder_file):
    found_lincoln = False
    found_id = None

    for line in recorder_file.split("\n"):

        # Check the role_name for hero
        if found_lincoln:
            if not line.startswith("  "):
                found_lincoln = False
                found_id = None
            else:
                data = line.split(" = ")
                if 'role_name' in data[0] and 'hero' in data[1]:
                    return found_id

        # Search for all lincoln vehicles
        if not found_lincoln and line.startswith(" Create ") and 'vehicle.lincoln' in line:
            found_lincoln = True
            found_id =  int(line.split(" ")[2][:-1])

    return found_id


def save_data_to_disk(sensor_id, frame, data, imu_data, endpoint):
    """
    Saves the sensor data into file:
    - Images                        ->              '.png', one per frame, named as the frame id
    - Lidar:                        ->              '.ply', one per frame, named as the frame id
    - SemanticLidar:                ->              '.ply', one per frame, named as the frame id
    - RADAR:                        ->              '.csv', one per frame, named as the frame id
    - GNSS:                         ->              '.csv', one line per frame, named 'gnss_data.csv'
    - IMU:                          ->              '.csv', one line per frame, named 'imu_data.csv'
    """
    global CURRENT_THREADS
    CURRENT_THREADS += 1
    if isinstance(data, carla.Image):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.png"
        if 'rgb' in sensor_id:
            data.save_to_disk(sensor_endpoint, carla.ColorConverter.Raw)
        elif 'seg' in sensor_id:
            data.save_to_disk(sensor_endpoint, carla.ColorConverter.CityScapesPalette)
        elif 'depth' in sensor_id:
            data.save_to_disk(sensor_endpoint, carla.ColorConverter.Depth)

    elif isinstance(data, carla.LidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.SemanticLidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.RadarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.csv"
        data_txt = f"Altitude,Azimuth,Depth,Velocity\n"
        for point_data in data:
            data_txt += f"{point_data.altitude},{point_data.azimuth},{point_data.depth},{point_data.velocity}\n"
        with open(sensor_endpoint, 'w') as data_file:
            data_file.write(data_txt)

    elif isinstance(data, carla.GnssMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{data.altitude},{data.latitude},{data.longitude}\n"
            data_file.write(data_txt)

    elif isinstance(data, carla.IMUMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{imu_data[0][0]},{imu_data[0][1]},{imu_data[0][2]},{data.compass},{imu_data[1][0]},{imu_data[1][1]},{imu_data[1][2]}\n"
            data_file.write(data_txt)
    elif sensor_id == 'speed':
        sensor_endpoint = f"{endpoint}/{sensor_id}/speed_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{data}\n"
            data_file.write(data_txt)
    else:
        print(f"WARNING: Ignoring sensor '{sensor_id}', as no callback method is known for data of type '{type(data)}'.")

    CURRENT_THREADS -= 1


def add_agent_delay(recorder_log):
    """
    The agent logs are delayed from the simulation recorder, which depends on the leaderboard setup.
    As the vehicle is stopped at the beginning, fake them with all 0 values, and the initial transform
    """

    init_tran = recorder_log['records'][0]['state']['transform']
    for _ in range(AGENT_TICK_DELAY):

        elem = {}
        elem['control'] = {
            'brake': 0.0, 'gear': 0, 'hand_brake': False, 'manual_gear_shift': False,
            'reverse': False, 'steer': 0.0, 'throttle': 0.0
        }
        elem['state'] = {
            'acceleration': {'value': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular_velocity': { 'value': 0.0, 'x': -0.0, 'y': 0.0, 'z': 0.0},
            'velocity': {'value': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'transform': {
                'pitch': init_tran['pitch'], 'yaw': init_tran['yaw'], 'roll': init_tran['roll'],
                'x': init_tran['x'], 'y': init_tran['y'], 'z': init_tran['z']
            }
        }
        recorder_log['records'].insert(0, elem)

    return recorder_log


def extract_imu_data(recorder_logs):

    records = recorder_logs['records']
    log_data = []

    for record in records:
        acceleration_data = record['state']['acceleration']
        acceleration_vector = [acceleration_data['x'], acceleration_data['y'], acceleration_data['z']]

        # TODO: Remove this (don't use logs without angular velocity)
        if 'angular_velocity' in record['state']:
            angular_data = record['state']['angular_velocity']
            angular_vector = [angular_data['x'], angular_data['y'], angular_data['z']]
        else:
            angular_vector = [random.random(), random.random(), random.random()]

        log_data.append([acceleration_vector, angular_vector])

    return log_data


def save_recorded_data(endpoint, info, logs, start, duration, weather):
    captured_logs = logs['records'][int(FPS*start):int(FPS*(start + duration))]
    saved_logs = {"records": captured_logs}
    print(f"\033[1m> Saving the logs in '{endpoint}'\033[0m")
    
    with open(f'{endpoint}/ego_logs.json', 'w') as fd:
        json.dump(saved_logs, fd, indent=4)
    with open(f'{endpoint}/sensors.json', 'w') as fd:
        json.dump(SENSORS, fd, indent=4)
    with open(f'{endpoint}/simulation.json', 'w') as fd:
        simulation_info = info
        simulation_info.pop('name')
        simulation_info['input_data'] = simulation_info.pop('folder')
        simulation_info['weather'] = {
            'sun_azimuth_angle': weather.sun_azimuth_angle, 'sun_altitude_angle': weather.sun_altitude_angle,
            'cloudiness': weather.cloudiness, 'wind_intensity': weather.sun_azimuth_angle,
            'precipitation': weather.precipitation, 'precipitation_deposits': weather.precipitation_deposits, 'wetness': weather.wetness,
            'fog_density':weather.fog_density, 'fog_distance': weather.fog_distance, 'fog_falloff': weather.fog_falloff,
        }
        json.dump(simulation_info, fd, indent=4)

    future_waypoints = logs['future_waypoints'][int(FPS*start):int(FPS*(start + duration))]
    with open(f'{endpoint}/future_waypoints.json', 'w') as fd:
        json.dump(future_waypoints, fd, indent=4)

def set_endpoint(recorder_info):
    def get_new_endpoint(endpoint):
        i = 2
        new_endpoint = endpoint + "_" + str(i)
        while os.path.isdir(new_endpoint):
            i += 1
            new_endpoint = endpoint + "_" + str(i)
        return new_endpoint

    endpoint = f"{DESTINATION_FOLDER}/{recorder_info['name']}"
    if os.path.isdir(endpoint):
        old_endpoint = endpoint
        endpoint = get_new_endpoint(old_endpoint)
        print(f"\033[93mWARNING: Given endpoint already exists, changing {old_endpoint} to {endpoint}\033[0m")

    os.makedirs(endpoint)
    return endpoint

def preprocess_sensor_specs(sensor):
    # Extract the data from the sesor configuration
    # return: sensor_id, transform, attributes
    sensor_id, attributes = sensor
    if sensor_id == "speed":
        sensor_transform = carla.Transform()
    else:   
        sensor_transform = carla.Transform(
            carla.Location(x=attributes.get('x'), y=attributes.get('y'), z=attributes.get('z')),
            carla.Rotation(pitch=attributes.get('pitch'), roll=attributes.get('roll'), yaw=attributes.get('yaw'))
        )
    return sensor_id, sensor_transform, attributes

def _generate_future_waypoints(map, logs, max_distance = 50, distance = 5):
    """waypoints for future 50 meters ahead at an interval of 5 meters
    Reverse the vehicle state and get the future waypoints for the vehicle
        
    Args:
        world (_type_): _description_
        vehicle (_type_): _description_
        distance (_type_): _description_
    """
    print("Getting future waypoints ----------------")
    captured_logs = logs['records']
    num_futures = max_distance // distance
    future_waypoints = []
    for i in range(len(captured_logs)):
        record = captured_logs[i]
        
        # vehicle transform
        transform = record['state']['transform']
        location = carla.Location(x=transform['x'], y=transform['y'], z=transform['z'])
        rotation = carla.Rotation(pitch=transform['pitch'], roll=transform['roll'], yaw=transform['yaw'])
        transform = carla.Transform(location, rotation)
        waypoint = map.get_waypoint(location, project_to_road=True)
        
        j = 1
        h = 1
        prev_transform = transform
        future_waypoints_i = [waypoint]
        dist = 0
        while i+j < len(captured_logs):
            # future transform
            next_transform = captured_logs[i+j]['state']['transform']
            next_location = carla.Location(x=next_transform['x'], y=next_transform['y'], z=next_transform['z'])
            next_rotation = carla.Rotation(pitch=next_transform['pitch'], roll=next_transform['roll'], yaw=next_transform['yaw'])
            next_transform = carla.Transform(next_location, next_rotation)
            
            # estimate distance along road
            dist += prev_transform.location.distance(next_transform.location)
            
            # save waypoints at an interval
            if dist >= h*distance:
                waypoint = map.get_waypoint(next_transform.location, project_to_road=True)
                future_waypoints_i.append(waypoint)
                h += 1
                
            # if we have enough future waypoints
            if len(future_waypoints_i) == num_futures + 1:
                break
            
            # update 
            prev_transform = next_transform
            j += 1
        
        # for the tailing frames, where there are not enough future waypoints in record, we will use the waypoints from map
        needs = num_futures + 1 - len(future_waypoints_i)
        while needs > 0:
            waypoint = future_waypoints_i[-1].next(distance)
            if waypoint is not None:
                waypoint = waypoint[0]
                future_waypoints_i.append(waypoint)
            needs -= 1
            
        # save for each frame
        future_waypoints.append([_from_carla_transform(waypoint.transform) for waypoint in future_waypoints_i[1:]])
    
    # save to logs
    logs['future_waypoints'] = future_waypoints
    return logs

def _get_camera_calibration(camera_attributes):
    """Get camera instrinsic matrix K

    Args:
        camera_attributes (_type_): _description_
        is_behind_camera (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    w, h, fov = camera_attributes['image_size_x'], camera_attributes['image_size_y'], camera_attributes['fov']
    K = np.identity(3)
    
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    
    # if the camera is behind the vehicle, the focal length should be negative
    K[0, 0] = focal
    K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    
    return K
    
def _from_carla_transform(transform):
    """convert carla transform to dict"""
    return {
        'x': transform.location.x,
        'y': transform.location.y,
        'z': transform.location.z,
        'pitch': transform.rotation.pitch,
        'roll': transform.rotation.roll,
        'yaw': transform.rotation.yaw
    }

def _to_carla_transform(transform):
    """Convert dict to carla transform
    """
    return carla.Transform(
        carla.Location(x=transform['x'], y=transform['y'], z=transform['z']),
        carla.Rotation(pitch=transform['pitch'], roll=transform['roll'], yaw=transform['yaw'])
    )
def _get_3d_bboxes(world, ego):
    bboxes = {
        "traffic_light": [],
        "stop_sign": [],
        "vehicle": [],
        "pedestrian": [],
    }

    bounding_boxes = world.get_actors().filter('vehicle.*')

def _find_obstacle_bbox(obstacle_type, world, ego, sensor_id, camera_instrincs, max_distance=50):
    """Returns a list of 3D bounding boxes of the queried obstacle type within the max_distance

    Args:
        world (_type_): _description_
        obstacle_type (_type_): _description_
        max_distance (int, optional): _description_. Defaults to 50.
    """
    obst = []
    actors = world.get_actors()
    obstacles = actors.filter(obstacle_type)
    
    for obstacle in obstacles:
        if obstacle.id != ego.id:
            # find 3d bounding box
            bbox_3d_world = _bbox_to_world(obstacle)
            bbox_3d_sensor = _bbox_world_to_sensor(bbox_3d_world, ego.get_sensor(sensor_id))
            bbox_3d_img = _bbox_sensor_to_image(bbox_3d_sensor, camera_instrincs)
            obst.append(bbox_3d_img)
    return obst

def _bbox_to_world(actor):
    """Get the 8 vertices of the bounding box in world coordinate
    
    return: 
        cords_world (8, 3): 8 vertices of the bounding box in world coordinate
    """
    vertices_carla = actor.bounding_box.get_world_vertices(actor.get_transform())
    vertices = np.array([[v.x, v.y, v.z] for v in vertices_carla])
    
    return vertices

def _bbox_world_to_sensor(cords_world, sensor):
    """Convert the bounding box from world coordinate to sensor coordinate
    
    return: 
        cords_sensor (8, 3): 8 vertices of the bounding box in sensor coordinate
    """
    world_to_sensor_matrix = np.array(sensor.get_transform().get_inverse_matrix())
    # homogenous coordinates
    cords_world = np.hstack((cords_world, np.ones((8, 1))))
    # transform to sensor coordinate
    cords_sensor = np.dot(world_to_sensor_matrix, cords_world.T).T
    
    return cords_sensor[:, :3]

def _bbox_sensor_to_image(cords_sensor, camera_instrinsic):
    """Convert the bounding box from sensor coordinate to image coordinate
    
    return: 
        cords_image (8, 3): 8 vertices of the bounding box in homogenous image coordinate
    """
    # the cords_sensor is from UE4's coordinate system
    # convert to a right-hand system
    # (x, y, z) -> (y, -z, x)
    cords_sensor = cords_sensor[:, [1, 2, 0]]
    cords_sensor[:, 1] *= -1
    
    # project to image plane
    cords_image = np.dot(camera_instrinsic, cords_sensor.T).T
    cords_image = cords_image[:, :2] / cords_image[:, 2]
    cords_image.astype(int)
    
    # note if the point is behind the camera, the z value will be negative
    return cords_image


def main():
    # running carla from docker container
    CARLA_IN_DOCKER = True 

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--replay-dir', default="experts/agents/replay", help='Path to the replay directory in terms of Carla server')
    argparser.add_argument('--weather-id', 
                    default=10, 
                    type=int, 
                    help='Weather id to use in the simulation: [0, 21]')
    args = argparser.parse_args()
    print(__doc__)

    active_sensors = {}
    sensor_calibrations = {}
    
    try:

        # Initialize the simulation
        client = carla.Client(args.host, args.port)
        client.set_timeout(120.0)
        world = client.get_world()

        for recorder_info in RECORDER_INFO:

            print(f"\n\033[1m> Getting the recorder information\033[0m")
            recorder_folder = recorder_info['folder']
            recorder_start = recorder_info['start_time']
            recorder_duration = recorder_info['duration']
            recorder_path_list = glob.glob(f"{root_path}/{args.replay_dir}/{recorder_folder}/*.log")

            if recorder_path_list:
                recorder_path = recorder_path_list[0]
                # need read from server
                if CARLA_IN_DOCKER:
                    recorder_path = recorder_path.replace(root_path, '/mnt/shared')
                print(f"\033[1m> Running recorder '{recorder_path}'\033[0m")
            else:
                print(f"\033[91mCouldn't find the recorder file for the folder '{recorder_folder}'\033[0m")
                continue

            endpoint = set_endpoint(recorder_info)

            print(f"\033[1m> Preparing the world. This may take a while...\033[0m")
            print(recorder_path)
            recorder_str = client.show_recorder_file_info(recorder_path, False)
            recorder_map = recorder_str.split("\n")[1][5:]
            world = client.load_world(recorder_map)
            world.tick()

            weather = WEATHERS[WEATHERS_IDS[args.weather_id]]
            world.set_weather(weather)
            settings = world.get_settings()
            settings.fixed_delta_seconds = 1 / FPS
            settings.synchronous_mode = True
            world.apply_settings(settings)
            world_map = world.get_map()
            world.tick()

            max_duration = float(recorder_str.split("\n")[-2].split(" ")[1])
            if recorder_duration == 0:
                recorder_duration = max_duration
            elif recorder_start + recorder_duration > max_duration:
                print("\033[93mWARNING: Found a duration that exceeds the recorder length. Reducing it...\033[0m")
                recorder_duration = max_duration - recorder_start
            if recorder_start >= max_duration:
                print("\033[93mWARNING: Found a start point that exceeds the recoder duration. Ignoring it...\033[0m")
                continue
            
            recorder_log_list = glob.glob(f"{root_path}/{args.replay_dir}/{recorder_folder}/log.json")
            recorder_log_path = recorder_log_list[0] if recorder_log_list else None
            print("\033[50m recorder_log_path: ", recorder_log_path, "\033[0m]")
            if recorder_log_path:
                with open(recorder_log_path) as fd:
                    recorder_log = json.load(fd)
                recorder_log = add_agent_delay(recorder_log)
                imu_logs = extract_imu_data(recorder_log)
                # add future waypoints
                recorder_log = _generate_future_waypoints(world_map, recorder_log, max_distance=50, distance=5)
                save_recorded_data(endpoint, recorder_info, recorder_log, recorder_start, recorder_duration, weather)
            else:
                imu_logs = None

            client.replay_file(recorder_path, recorder_start, recorder_duration, get_ego_id(recorder_str), False)
            
            # need write to local
            if CARLA_IN_DOCKER:
                recorder_path = recorder_path.replace('/mnt/shared', root_path)
            with open(f"{recorder_path[:-4]}.txt", 'w') as fd:
                fd.write(recorder_str)
            world.tick()

            hero = None
            while hero is None:
                possible_vehicles = world.get_actors().filter('vehicle.*')
                for vehicle in possible_vehicles:
                    if vehicle.attributes['role_name'] == 'hero':
                        hero = vehicle
                        break
                time.sleep(1)

            print(f"\033[1m> Creating the sensors\033[0m")
            create_folders(endpoint, [[s[0], s[1].get('bp')] for s in SENSORS])
            blueprint_library = world.get_blueprint_library()
            sensor_queue = Queue()
            for sensor in SENSORS:
                
                # Extract the data from the sesor configuration
                sensor_id, sensor_transform, attributes = preprocess_sensor_specs(sensor)
                
                # TODO: speed meter seems not working here. Therefore disabled in the sensor list
                if sensor_id == 'speed':
                    sensor = SpeedometerReader(hero, attributes['reading_frequency'])
                else:
                    blueprint_name = attributes.get('bp')
                    # Get the blueprint and add the attributes
                    blueprint = blueprint_library.find(blueprint_name)
                    for key, value in attributes.items():
                        if key in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                            continue
                        blueprint.set_attribute(str(key), str(value))

                    # Create the sensors and its callback
                    sensor = world.spawn_actor(blueprint, sensor_transform, hero)
                    # add calibration if it is a camera
                    if 'rgb' in sensor_id:
                        sensor_calibrations[sensor_id] = _get_camera_calibration(attributes)
                        
                add_listener(sensor, sensor_queue, sensor_id)
                #active_sensors.append(sensor)
                # save sensor map
                active_sensors[sensor_id] = sensor

            for _ in range(10):
                world.tick()

            print(f"\033[1m> Running the replayer\033[0m")
            start_time = world.get_snapshot().timestamp.elapsed_seconds
            start_frame = world.get_snapshot().frame
            sensor_amount = len(SENSORS)

            max_threads = THREADS
            results = []

            while True:
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                if current_duration >= recorder_duration:
                    print(f">>>>>  Running recorded simulation: 100.00%  completed  <<<<<")
                    break

                completion = format(round(current_duration / recorder_duration * 100, 2), '3.2f')
                print(f">>>>>  Running recorded simulation: {completion}%  completed  <<<<<", end="\r")

                # Get and save the sensor data from the queue.
                missing_sensors = sensor_amount
                while True:

                    frame = world.get_snapshot().frame
                    try:
                        sensor_data = sensor_queue.get(True, 2.0)
                        if sensor_data[1] != frame: continue  # Ignore previous frame data
                        missing_sensors -= 1
                    except Empty:
                        raise ValueError("A sensor took too long to send their data")

                    ## get sensor data
                    sensor_id = sensor_data[0]
                    frame_diff = sensor_data[1] - start_frame
                    data = sensor_data[2]
                    imu_data = [[0,0,0], [0,0,0]] if not imu_logs else imu_logs[int(FPS*recorder_start + frame_diff)]

                    # get bbox of obstacles for rgb cameras
                    # 
                    
                    
                    # save data
                    res = threading.Thread(target=save_data_to_disk, args=(sensor_id, frame_diff, data, imu_data, endpoint))
                    results.append(res)
                    res.start()

                    if CURRENT_THREADS > max_threads:
                        for res in results:
                            res.join()
                        results = []

                    if missing_sensors <= 0:
                        break

                world.tick()

            for res in results:
                res.join()

            for _, sensor in active_sensors.items():
                sensor.stop()
                sensor.destroy()
            active_sensors = {}

            for _ in range(50):
                world.tick()

    # End the simulation
    finally:
        # stop and remove cameras
        for _, sensor in active_sensors.items():
            sensor.stop()
            sensor.destroy()

        # set fixed time step length
        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')