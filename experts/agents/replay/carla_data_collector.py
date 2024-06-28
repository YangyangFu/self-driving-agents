import carla
import numpy as np
import os
import json
import time
from queue import Queue, Empty
import threading
from PIL import Image 
import cv2 

from image_converter import labels_to_array, labels_to_cityscapes_palette
from leaderboard.envs.sensor_interface import SpeedometerReader, OpenDriveMapReader
from generate_recorder_info import generate_recorder_info

class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 2 # default: 2

        #TODO: consider adding map data later
        self._opendrive_tag = None


    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            raise ValueError("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor

        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, timestamp):
        # print("Updating {} - {}".format(tag, timestamp))
        if tag not in self._sensors_objects:
            raise ValueError("The sensor with tag [{}] has not been created!".format(tag))

        self._new_data_buffers.put((tag, timestamp, data))

    def get_data(self, frame):
        try: 
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):                
                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    # print("Ignoring opendrive sensor")
                    break
                
                sensor_data = self._new_data_buffers.get(True, self._queue_timeout)
                if sensor_data[1] != frame:
                    continue
                
                data_dict[sensor_data[0]] = (sensor_data[1], sensor_data[2])

        except Empty:
            raise TimeoutError("A sensor took too long to send their data")

        return data_dict


class CarlaDataCollector:
    def __init__(self, client, world, hero_vehicle, destination_folder, max_threads=20):
        self.client = client
        self.world = world
        self.hero_vehicle = hero_vehicle
        self.camera_instrinsics = self.get_sensor_instrinsics()
        self.destination_folder = destination_folder
        self.max_threads = max_threads
        self.active_sensors = {}
        self.current_threads = 0
        self.results = []
        
        # skip a few frames
        self._start_frame = 0
        
        # debug mode: plotting bounding boxes on image
        self.debug = True
        
    def setup(self):
        self.create_output_folders()
        self.create_sensor_interface()
        self.setup_sensors()
    
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
            #[
            #    'imu',
            #    {
            #        'bp': 'sensor.other.imu',
            #        'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #        'noise_accel_stddev_x': 0.001, 'noise_accel_stddev_y': 0.001, 'noise_accel_stddev_z': 0.015,
            #        'noise_gyro_stddev_x': 0.001,'noise_gyro_stddev_y': 0.001, 'noise_gyro_stddev_z': 0.001
            #    }
            #],
            #[
            #    'speed',
            #    {
            #        "bp": "sensor.speedometer", 
            #        "reading_frequency": 20
            #    }
            #]
        ]
        return sensors       

    def get_sensor_instrinsics(self):
        camera_instrinsics = {}
        for sensor_id, attr in self.get_sensors():
            if 'rgb' in sensor_id or 'seg' in sensor_id or 'depth' in sensor_id:
                width = attr['image_size_x']
                height = attr['image_size_y']
                fov = attr['fov']
                
                focal = width / (2 * np.tan(fov * np.pi / 360))
                K = np.identity(3)
                K[0, 0] = K[1, 1] = focal
                K[0, 2] = width / 2
                K[1, 2] = height / 2
                
                camera_instrinsics[sensor_id] = K
        return camera_instrinsics

    def create_sensor_interface(self):
        self.sensor_interface = SensorInterface()

    def create_output_folders(self):
        for sensor_id, _ in self.get_sensors():
            sensor_endpoint = f"{self.destination_folder}/{sensor_id}"
            if not os.path.exists(sensor_endpoint):
                os.makedirs(sensor_endpoint)

            # create bounding box folder for segmentation camera
            if 'seg' in sensor_id:
                sensor_endpoint = f"{self.destination_folder}/2d_bbs_{sensor_id.split('_')[1]}"
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

    def setup_sensors(self):
        blueprint_library = self.world.get_blueprint_library()
        for sensor in self.get_sensors():
            sensor_id, sensor_transform, attributes = self.preprocess_sensor_specs(sensor)
            
            if sensor_id == 'speed':
                sensor = SpeedometerReader(self.hero_vehicle, attributes['reading_frequency'])
            else:
                blueprint = blueprint_library.find(attributes.get('bp'))
                for key, value in attributes.items():
                    if key not in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                        blueprint.set_attribute(str(key), str(value))
                sensor = self.world.spawn_actor(blueprint, sensor_transform, self.hero_vehicle)
            
            self.sensor_interface.register_sensor(sensor_id, attributes['bp'], sensor)
            sensor.listen(lambda data, sensor_id=sensor_id: self.sensor_interface.update_sensor(sensor_id, data, data.frame))
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

    def tick(self):
        frame = self.world.get_snapshot().frame
        data_dict = self.sensor_interface.get_data(frame)
        
        return data_dict
    
    def save_data(self, data_dict):
        
        # save data to disk multi-threaded
        for sensor_id in data_dict.keys():
            frame, data = data_dict[sensor_id]
            frame_diff = frame - self._start_frame
            res = threading.Thread(target=self._save_data_to_disk, args=(sensor_id, frame_diff, data))
            self.results.append(res)
            res.start()

            if self.current_threads > self.max_threads:
                for res in self.results:
                    res.join()
                self.results = []
        
        # save 2d bounding boxes
        for sensor_id in data_dict.keys():
            if 'seg' in sensor_id:
                _, data = data_dict[sensor_id]
                seg_label_img = labels_to_array(data)
                bbs_3d = self._get_3d_bbs_world()
                bbs_2d = self._get_2d_bbs_img(sensor_id, bbs_3d, seg_label_img)
                self._save_2d_bbs(bbs_2d, frame_diff, sensor_id)
            
                if self.debug:
                    # need plot the image to visualize the bounding box
                    seg_img = labels_to_cityscapes_palette(data)
                    # RGB to BGR
                    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
                    # 3d
                    seg_img = self._draw_3d_bbs(bbs_3d, sensor_id, seg_img)
                    # 2d 
                    seg_img = self._draw_2d_bbs(bbs_2d, sensor_id, seg_img)
        
                    # save or display 
                    pos = sensor_id.split('_')[1]
                    cv2.imwrite(f"{self.destination_folder}/2d_bbs_{pos}/{frame_diff}.png", seg_img)
        
    def cleanup(self):
        for sensor in self.active_sensors.values():
            sensor.stop()
            sensor.destroy()

    def _save_data_to_disk(self, sensor_id, frame, data):
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

    def _save_2d_bbs(self, bbs_2d, frame, seg_camera_id):
        
        sensor_type = '2d_bbs'
        position = seg_camera_id.split('_')[1]
        save_path = f"{self.destination_folder}/{sensor_type}_{position}"
        with open(f"{save_path}/{frame}.json", "w") as f:
            json.dump(bbs_2d, f)

    def _get_3d_bbs_world(self, max_distance=50):
        """Get 3d bbox in world coordinates

        Args:
            max_distance (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
        """
        bounding_boxes = {
            "traffic_lights": [],
            "stop_signs": [],
            "vehicles": [],
            "pedestrians": [],
        }

        bounding_boxes["traffic_lights"] = self._get_3d_bbs_cords_world(
            "*traffic_light*", max_distance
        )
        bounding_boxes["stop_signs"] = self._get_3d_bbs_cords_world("*stop*", max_distance)
        bounding_boxes["vehicles"] = self._get_3d_bbs_cords_world("*vehicle*", max_distance)
        bounding_boxes["pedestrians"] = self._get_3d_bbs_cords_world(
            "*walker*", max_distance
        )

        return bounding_boxes

    def _get_3d_bbs_cords_world(self, obstacle_type, max_distance=50):
        """Returns a list of 3d bounding boxes coordinates of type obstacle_type in world coordinates

        Args:
            obstacle_type (String): Regular expression
            max_distance (int, optional): max search distance. Returns all bbs in this radius. Defaults to 50.

        Returns:
            List: List of Boundingboxes
        """
        obst = list()

        _actors = self.world.get_actors()
        _obstacles = _actors.filter(obstacle_type)

        for _obstacle in _obstacles:
            distance_to_car = _obstacle.get_transform().location.distance(
                self.hero_vehicle.get_location()
            )

            if distance_to_car <= max_distance:

                if hasattr(_obstacle, "bounding_box"):
                    # center of bbox in vehicle coordinates
                    vertice_world = [v for v in _obstacle.bounding_box.get_world_vertices(_obstacle.get_transform())]
                    vertice_world = [[v.x, v.y, v.z] for v in vertice_world]

                #else:
                #    loc = _obstacle.get_transform().location
                #    bb = np.array([[loc.x, loc.y, loc.z], [0.5, 0.5, 2]])

                obst.append(vertice_world)

        return obst
    
    def _get_2d_bbs_img(self, seg_camera_id, bb_3d, seg_img):
        """Returns a dict of all 2d boundingboxes given a camera position, affordances and 3d bbs

        Args:
            seg_cam ([type]): [description]
            affordances ([type]): [description]
            bb_3d ([type]): [description]

        Returns:
            [type]: [description]
        """

        # sensor has to be a segmentation camera
        assert 'seg' in seg_camera_id, "Sensor has to be a segmentation camera to find bboxes that are not occluded."
        
        # initialize
        bounding_boxes = {
            "sensor_id": seg_camera_id,
            "boxes": {
                "traffic_light": list(),
                "stop_sign": list(),
                "vehicles": list(),
                "pedestrians": list(),
            }
        }

        #if affordances["stop_sign"]:
        #    baseline = self._get_2d_bb_baseline(self._target_stop_sign)
        #    bb = self._baseline_to_box(baseline, seg_cam)

        #    if bb is not None:
        #        bounding_boxes["stop_sign"].append(bb)

        #if affordances["traffic_light"] is not None:
        #    baseline = self._get_2d_bb_baseline(
        #        self.hero_vehicle.get_traffic_light(), distance=8
        #    )

        #    tl_bb = self._baseline_to_box(baseline, seg_cam, height=0.5)

        #    if tl_bb is not None:
        #        bounding_boxes["traffic_light"].append(
        #            {
        #                "bb": tl_bb,
        #                "state": self._translate_tl_state(
        #                    self._vehicle.get_traffic_light_state()
        #                ),
        #            }
        #        )

        for vehicle in bb_3d["vehicles"]:
            # (3, 8) -> (4, 8)
            bbox3d_vertice_world = np.vstack([np.array(vehicle).T, np.ones((1, 8))])
            bbox3d_vertice_sensor = self._world_to_sensor(
                bbox3d_vertice_world, seg_camera_id, False
            )

            veh_bb = self._coords_sensor_to_2d_bbs(bbox3d_vertice_sensor[:3, :], seg_camera_id)

            # use segmentation camera to check if actor is occlued
            if veh_bb is not None:
                if np.any(
                    seg_img[veh_bb[0][1] : veh_bb[1][1], veh_bb[0][0] : veh_bb[1][0]]
                    == 10
                ):
                    bounding_boxes['boxes']["vehicles"].append(veh_bb)

        for pedestrian in bb_3d["pedestrians"]:

            trig_loc_world = np.vstack([np.array(pedestrian).T, np.ones((1,8))])
            cords_x_y_z = self._world_to_sensor(
                trig_loc_world, seg_camera_id, False
            )

            cords_x_y_z = np.array(cords_x_y_z)[:3, :]

            ped_bb = self._coords_sensor_to_2d_bbs(cords_x_y_z)

            if ped_bb is not None:
                if np.any(
                    seg_img[ped_bb[0][1] : ped_bb[1][1], ped_bb[0][0] : ped_bb[1][0]]
                    == 4
                ):
                    bounding_boxes['boxes']["pedestrians"].append(ped_bb)

        return bounding_boxes

    def _create_3d_bbs_coords_world(self, bb: dict):
        """
        Returns 3D bounding box world coordinates.
        
        Args:
            bb: dict
                Bounding box information.
        
        Returns:
            np.ndarray (4, 8): 3D bounding box world coordinates.
            
        """

        cords = np.zeros((8, 4))
        extent = bb['extent']
        loc = bb['loc']
        cords[0, :] = np.array(
            [loc[0] + extent[0], loc[1] + extent[1], loc[2] - extent[2], 1]
        )
        cords[1, :] = np.array(
            [loc[0] - extent[0], loc[1] + extent[1], loc[2] - extent[2], 1]
        )
        cords[2, :] = np.array(
            [loc[0] - extent[0], loc[1] - extent[1], loc[2] - extent[2], 1]
        )
        cords[3, :] = np.array(
            [loc[0] + extent[0], loc[1] - extent[1], loc[2] - extent[2], 1]
        )
        cords[4, :] = np.array(
            [loc[0] + extent[0], loc[1] + extent[1], loc[2] + extent[2], 1]
        )
        cords[5, :] = np.array(
            [loc[0] - extent[0], loc[1] + extent[1], loc[2] + extent[2], 1]
        )
        cords[6, :] = np.array(
            [loc[0] - extent[0], loc[1] - extent[1], loc[2] + extent[2], 1]
        )
        cords[7, :] = np.array(
            [loc[0] + extent[0], loc[1] - extent[1], loc[2] + extent[2], 1]
        )
        return cords.T

    def _coords_sensor_to_2d_bbs(self, cords, sensor_id):
        """Returns coords of a 2d box given points in sensor coords

        Args:
            cords ([type]): [description]
            sensor_id (str): has to be a segmentation camera

        Returns:
            [type]: [description]
        """
        # assertion
        assert 'seg' in sensor_id, "Sensor has to be a segmentation camera to find bboxes that are not occluded."
        
        # from UE4 coordinate (left-handed sytem) to standard coordinate (right-handed system)
        # x, y, z -> y, -z, x
        cords_y_minus_z_x = np.vstack((cords[1, :], -cords[2, :], cords[0, :]))

        bbox = (self.camera_instrinsics[sensor_id] @ cords_y_minus_z_x).T

        camera_bbox = np.vstack(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]]
        ).T

        # get image bounds
        for id, attr in self.get_sensors():
            if sensor_id == id:
                img_width = attr["image_size_x"]
                img_height = attr["image_size_y"]
        
        # only keep boxes that are in front of camera
        if np.any(camera_bbox[:, 2] > 0):

            camera_bbox = np.array(camera_bbox)
            _positive_bb = camera_bbox[camera_bbox[:, 2] > 0]

            min_x = int(
                np.clip(np.min(_positive_bb[:, 0]), 0, img_width)
            )
            min_y = int(
                np.clip(np.min(_positive_bb[:, 1]), 0, img_height)
            )
            max_x = int(
                np.clip(np.max(_positive_bb[:, 0]), 0, img_width)
            )
            max_y = int(
                np.clip(np.max(_positive_bb[:, 1]), 0, img_height)
            )

            return [(min_x, min_y), (max_x, max_y)]
        else:
            return None

    def _draw_3d_bbs(self, bbs_3d, sensor_id, img):
        edges = [[0,1], 
                 [1,3], 
                 [3,2], 
                 [2,0], 
                 [0,4], 
                 [4,5], 
                 [5,1], 
                 [5,7], 
                 [7,6], 
                 [6,4], 
                 [6,2], 
                 [7,3]]
        
        for bb in bbs_3d['vehicles']:
            # (4, 8)
            cords_world = np.vstack([np.array(bb).T, np.ones((1, 8))])
            # (4, 8)
            cords_sensor = self._world_to_sensor(cords_world, sensor_id, False)
            # to right-handed system: x, y, z -> y, -z, x
            cords_sensor = np.vstack((cords_sensor[1, :], -cords_sensor[2, :], cords_sensor[0, :]))
            cords_img = self.camera_instrinsics[sensor_id] @ cords_sensor
            cords_img[0,:] = cords_img[0,:] / cords_img[2,:]
            cords_img[1,:] = cords_img[1,:] / cords_img[2,:]
            
            lines = []
            for edge in edges:
                p1 = (int(cords_img[0, edge[0]]), int(cords_img[1, edge[0]]))
                p2 = (int(cords_img[0, edge[1]]), int(cords_img[1, edge[1]]))
                lines.append((p1, p2))
            
            # we shouldn't draw the bounding box if any point are not in front of the camera
            if np.all(cords_img[2, :] > 0):
                for p1, p2 in lines:
                    img = cv2.line(img, p1, p2, (255, 0, 0), 2)
        
        return img
    
    def _draw_2d_bbs(self, bbs_2d, sensor_id, img):
        colors = {
            "traffic_light": (255, 0, 0),
            "stop_sign": (0, 255, 0),
            "vehicles": (0, 0, 255),
            "pedestrians": (255, 255, 0),
        }
        
        # check if sensor type is correct
        if bbs_2d['sensor_id'] == sensor_id:
            for obstacle_type in bbs_2d['boxes'].keys():
                for bb in bbs_2d['boxes'][obstacle_type]:
                            x1, y1 = bb[0]
                            x2, y2 = bb[1]
                            img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[obstacle_type], 2)
        return img 
    
    def _world_to_sensor(self, cords_in_world, sensor_id, move_cords=False):
        """
        Transforms world coordinates to sensor.
        
        Args:    
            cords_in_world: np.array (4, n)
            sensor_id: str
            move_cords: bool. If True, the cords are moved to the sensor plane if they are behind the sensor.
        
        Returns:
            np.array (4, n)
        """
        sensor = self.active_sensors[sensor_id]
        world_sensor_matrix = np.array(sensor.get_transform().get_inverse_matrix())
        cords_in_sensor = np.dot(world_sensor_matrix, cords_in_world)

        if move_cords:
            _num_cords = range(cords_in_sensor.shape[1])
            modified_cords = np.array([])
            for i in _num_cords:
                if cords_in_sensor[0, i] < 0:
                    for j in _num_cords:
                        if cords_in_sensor[0, j] > 0:
                            _direction = cords_in_sensor[:, i] - cords_in_sensor[:, j]
                            _distance = -cords_in_sensor[0, j] / _direction[0]
                            new_cord = (
                                cords_in_sensor[:, j]
                                + _distance[0, 0] * _direction * 0.9999
                            )
                            modified_cords = (
                                np.hstack([modified_cords, new_cord])
                                if modified_cords.size
                                else new_cord
                            )
                else:
                    modified_cords = (
                        np.hstack([modified_cords, cords_in_sensor[:, i]])
                        if modified_cords.size
                        else cords_in_sensor[:, i]
                    )

            return modified_cords
        else:
            return cords_in_sensor


# Usage example:
# client = carla.Client('localhost', 2000)
# world = client.get_world()
# hero_vehicle = world.get_actors().filter('vehicle.*')[0]  # Assume the first vehicle is the hero
# sensors = get_sensors()  # Your existing get_sensors function
# collector = CarlaDataCollector(client, world, hero_vehicle, sensors, "data_collection")
# collector.setup()
# collector.collect_data(duration=60)  # Collect data for 60 seconds
# collector.cleanup()