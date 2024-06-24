#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random
import queue 
import cv2


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# add seed to carla 

client = carla.Client('localhost', 2000)
world  = client.reload_world()
bp_lib = world.get_blueprint_library()
# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# set up traffic manager to get deterministic results
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_random_device_seed(0)
random.seed(0)

# spawn vehicle
vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# spawn camera
# camera attributes
width, height = 800, 600
camera_bp = bp_lib.find('sensor.camera.rgb')
depth_bp = bp_lib.find('sensor.camera.depth') # depth image to filter occuluded objects
camera_bp.set_attribute('image_size_x', str(width))
camera_bp.set_attribute('image_size_y', str(height))
camera_bp.set_attribute('fov', '110')
depth_bp.set_attribute('image_size_x', str(width))
depth_bp.set_attribute('image_size_y', str(height))
depth_bp.set_attribute('fov', '110')

camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
depth = world.spawn_actor(depth_bp, camera_init_trans, attach_to=vehicle)
vehicle.set_autopilot(True)

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.1
world.apply_settings(settings)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

depth_queue = queue.Queue()
depth.listen(depth_queue.put)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    if not isinstance(image, carla.Image):
        raise ValueError("Argument must be a carla.sensor.Image")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array

def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]
    
# Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)


# Retrieve all bounding boxes for traffic lights within the level
bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)

edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]


# add some vehicles
for i in range(30):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if npc:
        npc.set_autopilot(True)

# Retrieve the first image
world.tick()
image = image_queue.get(True, 2)
depth_image = depth_queue.get(True, 2)

# Reshape the raw data into an RGB array
img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 
depth_img = depth_to_array(depth_image)
depth_img = (depth_img * 255.999).astype(np.uint8)
#depth_img = depth_image.convert(carla.ColorConverter.Depth)
print(depth_image, depth_img.shape)
d = np.dstack((depth_img, depth_img, depth_img))
print(d.shape)

combined_img = np.concatenate((img[:, :, :3], d), axis=1)

# Display the image in an OpenCV display window
cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
cv2.imshow('ImageWindowName',combined_img)
cv2.waitKey(1)

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

while True:
    # Retrieve and reshape the image
    world.tick()
    image = image_queue.get()
    depth_image = depth_queue.get()
    
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    depth_img = depth_to_array(depth_image)
    depth_img = (depth_img * 255.999).astype(np.uint8)
    d = np.dstack((depth_img, depth_img, depth_img))
    # Get the camera matrix 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())


    for npc in world.get_actors().filter('*vehicle*'):

        # Filter out the ego vehicle
        if npc.id != vehicle.id:

            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

            # check if occuluded
            # if npc location in the depth image is less than the distance, then it is occuluded
            npc_world_coord = np.array([npc.get_transform().location.x, 
                                        npc.get_transform().location.y, 
                                        npc.get_transform().location.z,
                                        1]).reshape(1, 4)
            npc_depth_coord = np.dot(
                np.array(depth.get_transform().get_inverse_matrix()),
                npc_world_coord.T
            ).T
            npc_depth_coord = npc_depth_coord[0, [1,2,0]]
            npc_depth_coord[1] *= -1
            
            npc_depth_img = np.dot(K, npc_depth_coord[:3].T).T
            npc_depth_img[0] /= npc_depth_img[2]
            npc_depth_img[1] /= npc_depth_img[2]
            
            if 0 <= npc_depth_img[0] < image_w and 0 <= npc_depth_img[1] < image_h:
                dist_in_depth = depth_img[int(npc_depth_img[1]), int(npc_depth_img[0])]/266 * 1000
                print("dist: ", dist, "dist_in_depth: ", dist_in_depth)
            
            # Filter for the vehicles within 50m
            if dist < 100 and dist_in_depth >= dist - 1:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the other vehicle. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                if forward_vec.dot(ray) > 0:
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    
                    # to camera coordinates
                    verts_world = np.array([[v.x, v.y, v.z, 1] for v in verts])
                    verts_camera = np.dot(world_2_camera, verts_world.T).T
                    verts_camera = verts_camera[:, [1,2,0]]
                    verts_camera[:, 1] *= -1
                    
                    # to image coordinates
                    verts_img = np.dot(K, verts_camera[:, :3].T).T
                    verts_img[:, 0] /= verts_img[:, 2]
                    verts_img[:, 1] /= verts_img[:, 2]
                    verts_img = verts_img[:, :2]
                    
                    # bound to image size: no need
                    #verts_img = np.clip(verts_img, 0, [image_w, image_h])
                    #verts_img = verts_img.astype(np.int32)
                    
                    for edge in edges:
                        p1 = verts_img[edge[0], :]
                        p2 = verts_img[edge[1], :]                            
                            
                        ray0 = verts[edge[0]] - camera.get_transform().location
                        ray1 = verts[edge[1]] - camera.get_transform().location
                        cam_forward_vec = camera.get_transform().get_forward_vector()
                        
                        """
                        # One of the qvertex is behind the camera
                        if not (cam_forward_vec.dot(ray0) > 0):
                            #print("vertex behind camera: ", verts[edge[0]])
                            p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                        if not (cam_forward_vec.dot(ray1) > 0):
                            #print("vertex behind camera: ", verts[edge[1]])
                            p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)
                        """
                        # if occuluded, then skip
                        #if point_in_canvas(p1, image_h, image_w) and point_in_canvas(p2, image_h, image_w) and d[int(p1[1]), int(p1[0])] < dist-5 and d[int(p2[1]), int(p2[0])] < dist-5:
                        #    continue
                        
                        # if behind camera, then skip
                        if not (cam_forward_vec.dot(ray0) > 0) or not (cam_forward_vec.dot(ray1) > 0):
                            continue
                        
                        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)
                        cv2.line(d, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)         
                        
                        combined_img = np.concatenate((img[:, :, :3], d), axis=1)
                        
    cv2.imshow('ImageWindowName',combined_img)
    if cv2.waitKey(1) == ord('q'):
        camera.destroy()
        depth.destroy()
        break
cv2.destroyAllWindows()