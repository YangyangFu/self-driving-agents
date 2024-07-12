import time
import os
import carla
import argparse
import random
import json
import glob
import argparse

from generate_recorder_info import generate_recorder_info
from carla_data_collector import CarlaDataCollector

################# simulation configuration
# 1) choose weather
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

# 2) Choose the recorder files
#RECORDER_INFO = generate_recorder_info()

# read from file
#with open('./experts/agents/replay/recorder_info.json', 'r') as f:
#    RECORDER_INFO = json.load(f) 
RECORDER_INFO = [
    {
        "folder": "ScenarioLogs/InterurbanActorFlow_fast",
        "name": "InterurbanActorFlow_fast",
        "start_time": 0,
        "duration": 0
    },
]
################# End user simulation configuration ##################

FPS = 20
THREADS = 20
CURRENT_THREADS = 0
AGENT_TICK_DELAY = 10


def set_endpoint(root_path, recorder_info):
    def get_new_endpoint(endpoint):
        i = 2
        new_endpoint = endpoint + "_" + str(i)
        while os.path.isdir(new_endpoint):
            i += 1
            new_endpoint = endpoint + "_" + str(i)
        return new_endpoint

    endpoint = f"{root_path}/{recorder_info['name']}"
    if os.path.isdir(endpoint):
        old_endpoint = endpoint
        endpoint = get_new_endpoint(old_endpoint)
        print(f"\033[93mWARNING: Given endpoint already exists, changing {old_endpoint} to {endpoint}\033[0m")

    os.makedirs(endpoint)
    return endpoint

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

    # create output path
    output_dir = os.path.join("./data_collection", "weather_" + str(args.weather_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Start the simulation
    try:

        # Initialize the simulation
        client = carla.Client(args.host, args.port)
        client.set_timeout(600.0)
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

            endpoint = set_endpoint(output_dir, recorder_info)

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
            
            print(f"\033[1m> Saving recorder info \033[0m")
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

            print(f"\033[1m> Setting up data collector \033[0m")
            DEBUG = False
            data_collector = CarlaDataCollector(client, world, hero, endpoint, max_threads=20, debug=DEBUG)
            data_collector.setup()
            
            # 
            for _ in range(AGENT_TICK_DELAY):
                world.tick()

            print(f"\033[1m> Running the replayer\033[0m")
            start_time = world.get_snapshot().timestamp.elapsed_seconds
            start_frame = world.get_snapshot().frame
            data_collector._start_frame = start_frame

            while True:
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                if current_duration >= recorder_duration:
                    print(f">>>>>  Running recorded simulation: 100.00%  completed  <<<<<")
                    break

                completion = format(round(current_duration / recorder_duration * 100, 2), '3.2f')
                print(f">>>>>  Running recorded simulation: {completion}%  completed  <<<<<", end="\r")
                data_dict, traffic_info = data_collector.tick()
                data_collector.render(data_dict)
                
                data_collector.save_data(data_dict, traffic_info)
                world.tick()

            # clean up
            data_collector.cleanup()

            for _ in range(100):
                world.tick()

    # End the simulation
    finally:

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