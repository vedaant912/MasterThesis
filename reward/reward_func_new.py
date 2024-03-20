# Reward Functions
from config import reward_params
import numpy as np
import time
import math

def get_weather_description(self):

    weather = self.world.get_weather()
    cloudiness = weather.cloudiness
    precipitation = weather.precipitation
    fog_density = weather.fog_density

    # Classify weather based on parameters
    if precipitation > 1.0:
        return "Rainy"
    elif fog_density > 5.0:
        return "Foggy"
    elif cloudiness > 70.0:
        return "Cloudy"
    elif cloudiness <= 20.0 and weather.sun_altitude_angle >= 45.0:
        return "Sunny"
    else:
        return "Clear"
    
def get_road_info(self):

    location = self.location
    waypoint = self.world.get_map().get_waypoint(location)

    road_inclination = waypoint.transform.rotation.pitch
    
    if self.world.get_weather().precipitation > 0.1:
        return "Wet"
    elif road_inclination > 5.0:
        return "Uphill"
    elif road_inclination < -5.0:
        return "Downhill"
    else:
        return "Dry"
    
def get_lane_marking(self):

    waypoint = self.world.get_map().get_waypoint(self.location)

    marking = waypoint.left_lane_marking.type

    if marking == 'Broken':
        return 'broken_line'
    elif marking == 'Solid':
        return 'solid_line'
    else:
        return 'double_line_broken_vehicle_side'
    
def get_angle_to_nearest_pedestrian(self):
    # Get the vehicle's location
    vehicle_location = self.vehicle.get_location()

    # Get all pedestrians in the world
    pedestrians = [actor for actor in self.world.get_actors() if 'walker' in actor.type_id]

    if not pedestrians:
        return None

    # Find the nearest pedestrian
    nearest_pedestrian = min(pedestrians, key=lambda p: vehicle_location.distance(p.get_location()))

    # Get the location of the nearest pedestrian
    pedestrian_location = nearest_pedestrian.get_location()

    # Calculate the angle between the vehicle and the pedestrian
    angle_to_pedestrian = math.degrees(math.atan2(pedestrian_location.y - vehicle_location.y, pedestrian_location.x - vehicle_location.x))

    return angle_to_pedestrian

def map_value(value, min1, max1, min2, max2):
    return (value - min1) * (max2 - min2) / (max1 - min1) + min2

def reward_with_risk_factor(self):

    reward = 0
    done = False
    
    ######## Risk Degree based on Environmental factors ########
    environment_hazard_degree_weather = {
        'Clear':0.3,
        'Sunny':0.45,
        'Cloudy':0.6,
        'Rainy':0.7,
        'Foggy':0.9,
    }

    weather_description = get_weather_description(self)
    weather_hazard_degree = environment_hazard_degree_weather[weather_description]

    environment_hazard_degree_road_surface = {
        'Wet':0.8,
        'Dry':0.45,
        'Uphill':0.45,
        'Downhill':0.75
    }

    road_info = get_road_info(self)
    road_hazard_degree = environment_hazard_degree_road_surface[road_info]

    environment_hazard_degree_road_centerline = {
        'broken_line':0.5,
        'solid_line':0.75,
        'double_line_broken_vehicle_side':0.55
    }

    lane_info = get_lane_marking(self)
    lane_hazard_degree = environment_hazard_degree_road_centerline[lane_info]

    environment_hazard_degree = np.mean([weather_hazard_degree, road_hazard_degree, lane_hazard_degree])

    ######## Risk Degree based on Vehicle ########
    vehicle_hazard = 1/(1 + np.exp(-self.speed)) 

    ######## Risk Degree based on Pedestrian ########
    pedestrian_angle = get_angle_to_nearest_pedestrian(self)
    pedestrian_hazard = map_value(pedestrian_angle, -20, 20, 0.4, 0.9)

    ######## Overall Risk Degree ########
    overall_risk = np.mean([environment_hazard_degree, vehicle_hazard, pedestrian_hazard])
    
    ######## Risk Degree ########
    K = 1.5 * overall_risk * 1/(1 + np.exp(-(-overall_risk))) 

    ######## Reward ########
    nearest_pedestrian_distance = self.nearest_pedestrian_distance()

    if K < 0.7 and nearest_pedestrian_distance < 2:
        reward = -100 / K
    elif K < 0.7 and (nearest_pedestrian_distance >= 2 or nearest_pedestrian_distance < 8):
        reward = K * ((nearest_pedestrian_distance - 2) + 10)
    elif K < 0.7 and nearest_pedestrian_distance >= 8:
        reward = 1 / K * ((12 - nearest_pedestrian_distance) + 10)
    elif K > 0.7 and nearest_pedestrian_distance < 2:
        reward = -10 * 1 / K
    elif K > 0.7 and (nearest_pedestrian_distance >= 2 or nearest_pedestrian_distance < 8):
        reward = 100 * K
    elif K > 0.7 and nearest_pedestrian_distance >= 8:
        reward = K * ((12 - nearest_pedestrian_distance) - 100)

    if len(self.collision_hist) != 0:
        done = True
        reward -= 10
        self.collision_flag = True    
    elif self.dis_to_left > self.max_distance_from_center or self.dis_to_right > self.max_distance_from_center:
        done = True
        self.distance_from_center_flag = True
    elif self.episode_start_time + 10 < time.time() and self.speed < 1.0:
        done = True
        reward -= 5
        self.low_speed_timer_flag = True

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
    # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
    angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

    if not done:
        if self.continuous_action_space:
            if self.speed < self.min_speed:
                reward += (self.speed / self.min_speed) * centering_factor * angle_factor    
            elif self.speed > self.target_speed:               
                reward += (1.0 - (self.speed-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
            else:                                         
                reward += 1.0 * centering_factor * angle_factor 
        else:
            reward += 1.0 * centering_factor * angle_factor

    # If steps_per_episodes are exceeded or waypoints are exceeded.
    if self.frame_step >= self.steps_per_episode:
        print('Completed all the frame_steps, done = True')
        reward += 20
        done = True


    return reward, done