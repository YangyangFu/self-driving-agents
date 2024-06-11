#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function
from PIL import Image
import pathlib 

import carla
from lib.basic_agent import BasicAgent
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from lib.misc import get_speed
import os
import lmdb
import datetime
import numpy as np
from experts.utils.utils import bcolors as bc

def get_entry_point():
    return 'AutoAgent'

class AutoAgent(AutonomousAgent):

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        self.origin_global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in range(len(global_plan_world_coord))]
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def setup(self, config):
        #super().setup(config)
        self.track = Track.SENSORS
        self._route_assigned = False
        self._agent = None
        self.num_frames = 0
        self.stop_counter = 0
        self.config = config
        self.rgbs, self.sems, self.info, self.brak = [], [], [], []

        self._sensor_data = {"width": 400, "height": 300, "fov": 100}
        self._rgb_sensor_data = {"width": 800, "height": 600, "fov": 100}
        
        # data path 
        now = datetime.datetime.now()
        folder_name = ''
        time_now = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.output_dir = pathlib.Path(self.config.output_dir)/(folder_name+time_now)
        (self.output_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sem").mkdir(parents=True, exist_ok=True)
        
    def sensors(self):
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 1.5, 'y': 0.0, 'z': 2.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': self._rgb_sensor_data['width'], 'height': self._rgb_sensor_data['height'], 'fov': self._rgb_sensor_data['fov'], 'id': 'RGB'},
            {'type': 'sensor.camera.semantic_segmentation', 'x': 1.5, 'y': 0.0, 'z': 2.4,  'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': self._rgb_sensor_data['width'], 'height': self._rgb_sensor_data['height'], 'fov': self._rgb_sensor_data['fov'], 'id': 'SEM'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        control = carla.VehicleControl(steer=0, throttle=0, brake=1)

        if not self._agent:
            self.init_set_routes()
            return control  

        control, obs_actor, light_actor, walker = self._agent.run_step()

        # get sensor data 
        _, rgb = input_data.get('RGB')
        _, sem = input_data.get('SEM')
        _, ego = input_data.get('EGO')
        spd = ego.get('speed')

        if spd < 0.5:
            self.stop_counter += 1
        else:
            self.stop_counter = 0


        if self.config.save_output and self.num_frames % 5 == 0 and self.stop_counter < self.config.max_stop_num:
            vel = get_speed(self._vehicle)/3.6 #  m/s
            is_junction = self._map.get_waypoint(self._vehicle.get_transform().location).is_junction
            self.rgbs.append(rgb[...,:3])
            self.sems.append(sem[...,2,])
            self.info.append([vel, is_junction, self.config.weather_change])

            # save data
            self.save_data(input_data)
            
            # change weather
            if not self.config.debug_print and self.num_frames % 50 == 0:
                self.change_weather()
                
        self.num_frames += 1
        if self.stop_counter>500:
            self.force_destory_actor(obs_actor, light_actor, walker)

        return control

    def force_destory_actor(self, obs, light, walker):
        if obs:
            self._world.get_actor(obs.id).destroy()
            self.stop_counter = 0
            print(f"{self.num_frames}, {bc.WARNING}ATTENTION:{bc.ENDC} force to detroy actor {obs.id} stopping for a long time")
        elif walker:
            self._world.get_actor(walker.id).destroy()
            self.stop_counter = 0
            print(f"{self.num_frames}, {bc.WARNING}ATTENTION:{bc.ENDC} force to detroy actor {walker.id} stopping for a long time")
        elif light:
            light.set_green_time(10.0)
            light.set_state(carla.TrafficLightState.Green)
            self.stop_counter = 0
            print(f"{self.num_frames}, {bc.WARNING}ATTENTION:{bc.ENDC} force to setting green light {light.id}")
        else:
            print(f"{bc.WARNING}==========>{bc.ENDC}  error!!!! None factor trigger the stop!!!")
            return

    def init_set_routes(self):
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._agent = BasicAgent(self._vehicle)
        
        plan = []
        for transform, roadoption in self.origin_global_plan_world_coord:
            wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
            plan.append((wp, roadoption))

        self._agent.set_global_plan(plan)

        if self.config.debug_print:
            for i, point in enumerate(plan):
                self._world.debug.draw_string(point, str(i), life_time=999, color=carla.Color(0,0,255))

    def change_weather(self):
        # TODO
        return

    def save_data(self, input_data):
        _, rgb = input_data.get('RGB')
        _, sem = input_data.get('SEM')
        
        rgb = rgb[...,:3]
        sem = sem[...,2,]
        
        # save to folder
        Image.fromarray(rgb).save(
            self.output_dir/"rgb"/("%04d.jpg" % self.num_frames)
        )
        
        Image.fromarray(sem).save(
            self.output_dir/"sem"/("%04d.jpg" % self.num_frames)
        )
        
    def destroy(self):
        if len(self.rgbs) == 0:
            return
        
        self.rgbs.clear()
        self.sems.clear()
        self.info.clear()
        
        return