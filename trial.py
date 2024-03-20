import carla
import random

def set_spawn_point(world, spawn_point):
    """
    Set the spawn point of the vehicle.

    Parameters:
    - world: Carla world object.
    - spawn_point: Desired spawn point.
    """
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return vehicle

def generate_fixed_route(world, start_location, route_length=100, seed=42):
    """
    Generate fixed route waypoints.

    Parameters:
    - world: Carla world object.
    - start_location: Starting location for the route.
    - route_length: Number of waypoints in the route.
    - seed: Seed for randomization.

    Returns:
    - waypoints: List of waypoints.
    """
    random.seed(seed)

    waypoints = []
    current_location = start_location
    for _ in range(route_length):
        waypoint = world.get_map().get_waypoint(current_location)
        waypoints.append(waypoint)
        next_waypoints = waypoint.next(1.0)
        if next_waypoints:
            current_location = next_waypoints[0].transform.location
        else:
            break

    return waypoints

# Connect to Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Retrieve the Carla world
world = client.get_world()

# Set a fixed spawn point
fixed_spawn_point = carla.Transform(carla.Location(x=100, y=100, z=1), carla.Rotation())

# Spawn the vehicle at the fixed spawn point
vehicle = set_spawn_point(world, fixed_spawn_point)

# Generate fixed route waypoints
fixed_route_waypoints = generate_fixed_route(world, fixed_spawn_point.location)

# Print the waypoints (for demonstration purposes)
for idx, waypoint in enumerate(fixed_route_waypoints):
    print(f"Waypoint {idx + 1}: {waypoint.transform.location}")

# Do your simulation using the fixed_spawn_point and fixed_route_waypoints

# Destroy the vehicle (clean up)
vehicle.destroy()
