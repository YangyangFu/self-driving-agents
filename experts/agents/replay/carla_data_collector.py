import carla
import numpy as np
import os
import json
import time
from queue import Queue, Empty
import threading


from leaderboard.envs.sensor_interface import SpeedometerReader, OpenDriveMapReader
from generate_recorder_info import generate_recorder_info

class CarlaDataCollector:
    def __init__(self, client, world, hero_vehicle, destination_folder, fps=20, max_threads=20):
        self.client = client
        self.world = world
        self.hero_vehicle = hero_vehicle
        self.sensors = self.get_sensors()
        self.destination_folder = destination_folder
        self.fps = fps
        self.max_threads = max_threads
        self.sensor_queue = Queue()
        self.active_sensors = {}
        self.current_threads = 0
        self.results = []

    def setup(self):
        self.create_folders()
        self.spawn_sensors()

    def get_sensors(self):
        IMG_WIDTH, IMG_HEIGHT = 1080, 720
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

    def create_folders(self):
        for sensor_id, _ in self.sensors:
            sensor_endpoint = f"{self.destination_folder}/{sensor_id}"
            if not os.path.exists(sensor_endpoint):
                os.makedirs(sensor_endpoint)

            if 'gnss' in sensor_id:
                sensor_endpoint = f"{self.destination_folder}/{sensor_id}/gnss_data.csv"
                with open(sensor_endpoint, 'w') as data_file:
                    data_file.write("Frame,Altitude,Latitude,Longitude\n")

            if 'imu' in sensor_id:
                sensor_endpoint = f"{self.destination_folder}/{sensor_id}/imu_data.csv"
                with open(sensor_endpoint, 'w') as data_file:
                    data_file.write("Frame,Accelerometer X,Accelerometer y,Accelerometer Z,Compass,Gyroscope X,Gyroscope Y,Gyroscope Z\n")

    def spawn_sensors(self):
        blueprint_library = self.world.get_blueprint_library()
        for sensor in self.sensors:
            sensor_id, sensor_transform, attributes = self.preprocess_sensor_specs(sensor)
            
            if sensor_id == 'speed':
                sensor = SpeedometerReader(self.hero_vehicle, attributes['reading_frequency'])
            else:
                blueprint = blueprint_library.find(attributes.get('bp'))
                for key, value in attributes.items():
                    if key not in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                        blueprint.set_attribute(str(key), str(value))
                sensor = self.world.spawn_actor(blueprint, sensor_transform, self.hero_vehicle)

            sensor.listen(lambda data, sensor_id=sensor_id: self.sensor_callback(data, sensor_id))
            self.active_sensors[sensor_id] = sensor

    def preprocess_sensor_specs(self, sensor):
        sensor_id, attributes = sensor
        if sensor_id == "speed":
            sensor_transform = carla.Transform()
        else:   
            sensor_transform = carla.Transform(
                carla.Location(x=attributes.get('x'), y=attributes.get('y'), z=attributes.get('z')),
                carla.Rotation(pitch=attributes.get('pitch'), roll=attributes.get('roll'), yaw=attributes.get('yaw'))
            )
        return sensor_id, sensor_transform, attributes

    def sensor_callback(self, data, sensor_id):
        self.sensor_queue.put((sensor_id, data.frame, data))

    def tick(self, start_frame):
        missing_sensors = len(self.sensors)
        while True:
            frame = self.world.get_snapshot().frame
            try:
                sensor_data = self.sensor_queue.get(True, 2.0)
                if sensor_data[1] != frame:
                    continue
                missing_sensors -= 1
            except Empty:
                raise ValueError("A sensor took too long to send their data")

            sensor_id, frame_diff, data = sensor_data[0], sensor_data[1] - start_frame, sensor_data[2]
            imu_data = [[0,0,0], [0,0,0]]  # Replace with actual IMU data if available

            res = threading.Thread(target=self.save_data_to_disk, args=(sensor_id, frame_diff, data, imu_data))
            self.results.append(res)
            res.start()

            if self.current_threads > self.max_threads:
                for res in self.results:
                    res.join()
                self.results = []

            if missing_sensors <= 0:
                break

    def save_data_to_disk(self, sensor_id, frame, data, imu_data):
        self.current_threads += 1
        # Your existing save_data_to_disk logic here
        endpoint=self.destination_folder
        
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

        self.current_threads -= 1

    def cleanup(self):
        for sensor in self.active_sensors.values():
            sensor.stop()
            sensor.destroy()

# Usage example:
# client = carla.Client('localhost', 2000)
# world = client.get_world()
# hero_vehicle = world.get_actors().filter('vehicle.*')[0]  # Assume the first vehicle is the hero
# sensors = get_sensors()  # Your existing get_sensors function
# collector = CarlaDataCollector(client, world, hero_vehicle, sensors, "data_collection")
# collector.setup()
# collector.collect_data(duration=60)  # Collect data for 60 seconds
# collector.cleanup()