import carla
import numpy as np
import os
import json
import time
from queue import Queue, Empty
import threading
from PIL import Image, ImageDraw

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from leaderboard.envs.sensor_interface import OpenDriveMapReader


class DisplayManager:
    def __init__(self, grid_size, window_size, sensor_manager):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        self.grid_size = grid_size
        self.window_size = window_size # (width, height)
        self.sensor_manager = sensor_manager
        
        # display position of sensors
        self.render_sensors_config = {'rgb_front': [0, 0],
                                      'rgb_left': [0, 1], 
                                      'rgb_right': [0, 2],
                                      'seg_front': [1, 0], 
                                      'seg_left': [1, 1], 
                                      'seg_right': [1, 2]
                                    }
        
    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def get_sensors(self) -> dict:
        return self.sensor_manager.get_sensors()

    def render(self, data_dict):
        if not self.render_enabled():
            return
        width, height = self.get_display_size()
        for id in self.render_sensors_config.keys():
            sensor = self.get_sensors()[id]
            # make surface for displaying data
            arr = data_dict[id][1]
            sensor.make_surface(arr.swapaxes(0,1), (width, height))            
            # render
            sensor.render()

            pygame.display.flip()

    def destroy(self):
        for s in self.sensor_manager.get_sensors().values():
            s.destroy()

    def render_enabled(self):
        return self.display != None

class Sensor:
    def __init__(self, world, vehicle, sensor_id, sensor_config, display_man = None):
        # world
        self.world = world
        self.hero_vehicle = vehicle

        # sensor config and sensor creation
        self.sensor_id = sensor_id
        self.sensor_config = sensor_config
        self.sensor = self._create_sensor(sensor_id, sensor_config)

        # for display
        self.surface = None
        self.display_man = display_man
        if self.display_man:
            self.display_pos = display_man.render_sensors_config.get(sensor_id)
        
        # timer: None

    def _create_sensor(self, sensor_id, sensor_config):
        sensor_id, sensor_transform, attributes = self._preprocess_sensor_specs(sensor_id, sensor_config)
        

        blueprint = self.world.get_blueprint_library().find(attributes.get('bp'))
        for key, value in attributes.items():
            if key not in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                blueprint.set_attribute(str(key), str(value))
        sensor = self.world.spawn_actor(blueprint, sensor_transform, self.hero_vehicle)
                
        return sensor

    def _preprocess_sensor_specs(self, sensor_id, attributes):
        
        sensor_transform = carla.Transform(
            carla.Location(x=attributes.get('x'), y=attributes.get('y'), z=attributes.get('z')),
            carla.Rotation(pitch=attributes.get('pitch'), roll=attributes.get('roll'), yaw=attributes.get('yaw'))
        )
        return sensor_id, sensor_transform, attributes
    
    def get_sensor(self):
        return self.sensor

    def make_surface(self, data, size):
        # size = (width, height)
        self.surface = pygame.surfarray.make_surface(data)
        # scale surface to target size
        self.surface = pygame.transform.scale(self.surface, size)
        
    def listen(self, callback):
        return self.sensor.listen(callback)
    
    def render(self):
        if self.surface is not None and self.display_man is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()   

class SensorsManager(object):
    def __init__(self, world, hero_vehicle):
        self.world = world
        self.hero_vehicle = hero_vehicle
        
        # 
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 10 # default: 2

        #TODO: consider adding map data later
        self._opendrive_tag = None
        
    def get_sensors_configs(self):
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
                    'noise_accel_stddev_x': 0.001, 'noise_accel_stddev_y': 0.001, 'noise_accel_stddev_z': 0.001,
                    'noise_gyro_stddev_x': 0.001,'noise_gyro_stddev_y': 0.001, 'noise_gyro_stddev_z': 0.001
                }
            ],
        ]
        return sensors       

    def setup_sensors(self, display):
        for sensor_id, sensor_config in self.get_sensors_configs():
            sensor = Sensor(self.world, self.hero_vehicle, sensor_id, sensor_config, display)
            sensor.listen(lambda data, sensor_id=sensor_id: self.update_sensor(sensor_id, data, data.frame))
            self.register_sensor(sensor_id, sensor_config['bp'], sensor)            
                    
    def get_sensors(self):
        return self._sensors_objects
    
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
                
                # process data
                processed_data = self._process_sensor_data(sensor_data[0], sensor_data[2])
                data_dict[sensor_data[0]] = (sensor_data[1], processed_data)

        except Empty:
            raise TimeoutError("A sensor took too long to send their data")

        return data_dict

    def destroy(self):
        for sensor in self._sensors_objects.values():
            sensor.destroy()
    
    def _process_sensor_data(self, sensor_id, data):
        if 'rgb' in sensor_id:
            data.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1] # BGR to RGB
            return array
        elif 'seg' in sensor_id:
            data.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            return array
        elif 'depth' in sensor_id:
            data.convert(carla.ColorConverter.Depth)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            return array
        elif 'gnss' in sensor_id:
            array = {"latitude": data.latitude, "longitude": data.longitude, "altitude": data.altitude}
            return array
        
        elif 'imu' in sensor_id:
            array = {"accelerometer": [data.accelerometer.x, data.accelerometer.y, data.accelerometer.z],
                     "gyroscope": [data.gyroscope.x, data.gyroscope.y, data.gyroscope.z],
                     "compass": data.compass}
            return array
        else: 
            return data

class CarlaDataCollector:
    def __init__(self, client, world, hero_vehicle, destination_folder, max_threads=20, debug=False):
        self.client = client
        self.world = world
        self.hero_vehicle = hero_vehicle
        self.destination_folder = destination_folder
        self.max_threads = max_threads
        self.active_sensors = {}
        self.current_threads = 0
        self.results = []
        
        # skip a few frames
        self._start_frame = 0
        
        # debug mode: plotting bounding boxes on image
        self.debug = debug
        self.display_manager = None
    def setup(self):
        
        self.sensor_manager = SensorsManager(self.world, self.hero_vehicle)
        if self.debug:
            self.display_manager = DisplayManager([2, 3], [1920, 1080], self.sensor_manager)
        self.sensor_manager.setup_sensors(self.display_manager)
        self.camera_instrinsics = self.get_sensor_instrinsics()
        self.create_output_folders()
        
    def get_sensor_instrinsics(self):
        camera_instrinsics = {}
        for sensor_id, attr in self.sensor_manager.get_sensors_configs():
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


    def create_output_folders(self):
        for sensor_id, _ in self.sensor_manager.get_sensors_configs():
            sensor_endpoint = f"{self.destination_folder}/{sensor_id}"
            if not os.path.exists(sensor_endpoint):
                os.makedirs(sensor_endpoint)

            # create bounding box folder for segmentation camera
            if 'seg' in sensor_id:
                sensor_endpoint = f"{self.destination_folder}/2d_bbs_{sensor_id.split('_')[1]}"
                if not os.path.exists(sensor_endpoint):
                    os.makedirs(sensor_endpoint)
                           
    def tick(self):
        frame = self.world.get_snapshot().frame
        data_dict = self.sensor_manager.get_data(frame)
                
        return data_dict
    
    def render(self, data_dict):
        if self.display_manager and self.debug:
            self.display_manager.render(data_dict)
    
    def save_data(self, data_dict):
        
        # save data to disk multi-threaded
        for sensor_id in data_dict.keys():
            frame, data = data_dict[sensor_id]
            frame_diff = frame - self._start_frame
            thread = threading.Thread(target=self._save_data_to_disk, args=(sensor_id, frame_diff, data))
            thread.start()
            self.results.append(thread)
            
            # draw and save bounding boxes
            if 'seg' in sensor_id:
                sensor = self.sensor_manager.get_sensors()[sensor_id].sensor
                world_sensor_matrix = np.array(sensor.get_transform().get_inverse_matrix())
                thread = threading.Thread(target=self._save_and_draw_bbs, args=(sensor_id, frame_diff, data, world_sensor_matrix))
                thread.start()
                self.results.append(thread)

            #print(f"Current threads: {self.current_threads}")
            # wait for all threads to finish
            if self.current_threads >= self.max_threads:
                for res in self.results:
                    res.join()
                self.results = []
        
    def _save_and_draw_bbs(self, sensor_id, frame, seg_img, world_sensor_matrix):
        
            self.current_threads += 1
            
            seg_label_img = self._get_segmentation_labels(seg_img)
            bbs_3d = self._get_3d_bbs_world()
            bbs_2d = self._get_2d_bbs_img(sensor_id, world_sensor_matrix, bbs_3d, seg_label_img)
                
            # extrack bounding boxes using segmentation camera
            self._save_2d_bbs(bbs_2d, frame, sensor_id)
            # plot boundding boxes on image
            # convert array to image
            #seg_img = Image.fromarray(seg_img)
            # 3d
            #seg_img = self._draw_3d_bbs(bbs_3d, world_sensor_matrix, sensor_id, seg_img)
            # 2d 
            seg_img = self._draw_2d_bbs(bbs_2d, sensor_id, seg_img)

            # save or display
            pos = sensor_id.split('_')[1] 
            endpoint = f"{self.destination_folder}/2d_bbs_{pos}/{frame}.png"
            self._save_image(seg_img, endpoint)
            
            self.current_threads -= 1
            
    def cleanup(self):
        self.sensor_manager.destroy()

    def _save_image(self, image_array, endpoint):
        image = Image.fromarray(image_array)
        image.save(endpoint)
        
    def _save_data_to_disk(self, sensor_id, frame, data):
        self.current_threads += 1
        # Your existing save_data_to_disk logic here
        endpoint=self.destination_folder
        
        
        if 'rgb' in sensor_id:
            sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.png"
            self._save_image(data, sensor_endpoint)
            
        elif 'seg' in sensor_id:
            sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.png"
            self._save_image(data, sensor_endpoint)
            
        elif 'depth' in sensor_id:
            sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.png"
            self._save_image(data, sensor_endpoint)

        elif isinstance(data, carla.LidarMeasurement):
            sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
            data.save_to_disk(sensor_endpoint)

        elif isinstance(data, carla.SemanticLidarMeasurement):
            sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
            data.save_to_disk(sensor_endpoint)

        elif 'radar' in sensor_id:
            sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.csv"
            data_txt = f"Altitude,Azimuth,Depth,Velocity\n"
            for point_data in data:
                data_txt += f"{point_data.altitude},{point_data.azimuth},{point_data.depth},{point_data.velocity}\n"
            with open(sensor_endpoint, 'w') as data_file:
                data_file.write(data_txt)

        elif 'gnss' in sensor_id:
            sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.json"
            with open(sensor_endpoint, 'w') as data_file:
                json.dump(data, data_file)
                
        elif 'imu' in sensor_id:
            sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.json"
            with open(sensor_endpoint, 'w') as data_file:
                json.dump(data, data_file)

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
    
    def _get_2d_bbs_img(self, seg_camera_id, world_sensor_matrix, bb_3d, seg_img):
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
        # vehicles will include vehicles, bycicles, motorcycles, etc.
        for vehicle in bb_3d["vehicles"]:
            # (3, 8) -> (4, 8)
            bbox3d_vertice_world = np.vstack([np.array(vehicle).T, np.ones((1, 8))])
            bbox3d_vertice_sensor = self._world_to_sensor(
                bbox3d_vertice_world, world_sensor_matrix, False
            )

            veh_bb = self._coords_sensor_to_2d_bbs(bbox3d_vertice_sensor[:3, :], seg_camera_id)

            # use segmentation camera to check if actor is occlued
            if veh_bb is not None:
                if np.any(
                    seg_img[veh_bb[0][1] : veh_bb[1][1], veh_bb[0][0] : veh_bb[1][0]]
                    == 14
                ):
                    bounding_boxes['boxes']["vehicles"].append(veh_bb)

        for pedestrian in bb_3d["pedestrians"]:

            trig_loc_world = np.vstack([np.array(pedestrian).T, np.ones((1,8))])
            cords_x_y_z = self._world_to_sensor(
                trig_loc_world, world_sensor_matrix, False
            )

            cords_x_y_z = np.array(cords_x_y_z)[:3, :]

            ped_bb = self._coords_sensor_to_2d_bbs(cords_x_y_z)

            if ped_bb is not None:
                if np.any(
                    seg_img[ped_bb[0][1] : ped_bb[1][1], ped_bb[0][0] : ped_bb[1][0]]
                    == 12
                ):
                    bounding_boxes['boxes']["pedestrians"].append(ped_bb)

        return bounding_boxes

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
        for id, attr in self.sensor_manager.get_sensors_configs():
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

    def _draw_3d_bbs(self, bbs_3d, world_sensor_matrix, sensor_id, img):
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
        # array to image
        img = Image.fromarray(img)
        
        for bb in bbs_3d['vehicles']:
            # (4, 8)
            cords_world = np.vstack([np.array(bb).T, np.ones((1, 8))])
            # (4, 8)
            cords_sensor = self._world_to_sensor(cords_world, world_sensor_matrix, False)
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
                    ImageDraw.Draw(img).line([p1, p2], fill=(255, 0, 0), width=2)
            
        return np.array(img)
    
    def _draw_2d_bbs(self, bbs_2d, sensor_id, img):
        
        img = Image.fromarray(img)
        
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
                            ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], outline=colors[obstacle_type])
                            #img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[obstacle_type], 2)
        
        return np.array(img) 
    
    def _world_to_sensor(self, cords_in_world, world_sensor_matrix, move_cords=False):
        """
        Transforms world coordinates to sensor.
        
        Args:    
            cords_in_world: np.array (4, n)
            sensor_id: str
            move_cords: bool. If True, the cords are moved to the sensor plane if they are behind the sensor.
        
        Returns:
            np.array (4, n)
        """
        # this relies on call of current sensor location in the server, which might be wrong transform in an async environment
        #sensor = self.sensor_manager.get_sensors()[sensor_id].sensor
        #world_sensor_matrix = np.array(sensor.get_transform().get_inverse_matrix())
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

    def _get_segmentation_labels(self, seg_img):
        """ Returns the segmentation labels of the image using cityscapes palette
        
        Args:
            seg_img (np.array, (height, width, 3)): segmentation image in RGB format
            
        Returns:
            np.array, (height, width): segmentation labels based on cityscapes palette
        """
        seg_img_cp = np.zeros_like(seg_img)
        for rgb, label in self._cityscapes_palette_to_labels().items():
            seg_img_cp[(seg_img == rgb).all(axis=2)] = label
        
        return seg_img_cp[:, :, 0]
     
    def _cityscapes_palette_to_labels(self):
        """Converts segmentation image in cityscapes palette to labels based on 
        https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera.
        
        """
        hmap = {(0, 0, 0): 0, # unlabeled
                (128, 64, 128): 1, # road
                (244, 35, 232): 2, # sidewalk
                (70, 70, 70): 3, # building
                (102, 102, 156): 4, # wall
                (190, 153, 153): 5, # fence
                (153, 153, 153): 6, # pole
                (250, 170, 30): 7, # traffic light
                (220, 220, 0): 8, # traffic sign
                (107, 142, 35): 9, # vegetation
                (152, 251, 152): 10, # terrain
                (70, 130, 180): 11, # sky 
                (220, 20, 60): 12, # person
                (255, 0, 0): 13, # rider
                (0, 0, 142): 14, # car
                (0, 0, 70): 15, # truck
                (0, 60, 100): 16, # bus
                (0, 80, 100): 17, # train
                (0, 0, 230): 18, # motorcycle
                (119, 11, 32): 19, # bicycle
                (110, 190, 160): 20, # static
                (170, 120, 50): 21, # dynamic
                (55, 90, 80): 22, # other
                (45, 60, 150): 23, # water
                (157, 234, 50): 24, # roadline
                (81, 0, 81): 25, # groud
                (150, 100, 100): 26, # bridge
                (230, 150, 140): 27, # railtrack
                (180, 165, 180): 28 # guardrail
            } 
        return hmap
   
# Usage example:
# client = carla.Client('localhost', 2000)
# world = client.get_world()
# hero_vehicle = world.get_actors().filter('vehicle.*')[0]  # Assume the first vehicle is the hero
# sensors = get_sensors()  # Your existing get_sensors function
# collector = CarlaDataCollector(client, world, hero_vehicle, sensors, "data_collection")
# collector.setup()
# collector.collect_data(duration=60)  # Collect data for 60 seconds
# collector.cleanup()