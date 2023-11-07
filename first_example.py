import carla 
import random

client = carla.Client('localhost', 2000)
world = client.get_world()

# load a map
client.load_world('Town05')

# adding vehicles and NPCs
# spawn vehicles
# get all the blueprints of the vehicles
vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
# each map provides predefined spawn points
spawn_points = world.get_map().get_spawn_points()
# spawn 50 vehicles randomly distributed around the map
# for each spawn point, we randomly choose a blueprint and spawn a vehicle
for i in range(50):
    spawn_point = random.choice(spawn_points)
    blueprint = random.choice(vehicle_blueprints)
    world.try_spawn_actor(blueprint, spawn_point)

# now we should add ego vehicle

    
