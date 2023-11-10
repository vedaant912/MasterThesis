# Reward Functions
from config import reward_params
import numpy as np
import time

# min_speed = reward_params['reward_fn_5_default']["min_speed"]
# max_speed = reward_params['reward_fn_5_default']["max_speed"]
# target_speed = reward_params['reward_fn_5_default']["target_speed"]
# max_distance = reward_params['reward_fn_5_default']["max_distance"]
# max_std_center_lane = reward_params['reward_fn_5_default']["max_std_center_lane"]
# max_angle_center_lane = reward_params['reward_fn_5_default']["max_angle_center_lane"]
# penalty_reward = reward_params['reward_fn_5_default']["penalty_reward"]
# early_stop = reward_params['reward_fn_5_default']["early_stop"]
# reward_functions = {}


def reward_func1(self):

    reward = 0
    done = False

    # ##### Car Collision
    if len(self.collision_hist) != 0:
        #print('Collision Occurred , done =True')
        self.collision_flag = True
        done = True
        reward += -100
        self.collision_hist = []
        self.lane_invasion_hist = []
    else:
        reward += 5
    #####

    ##### Distance from center
    if self.distance_from_center > self.max_distance_from_center:
        # print('Distance from center exceeded, done = True')
        self.distance_from_center_flag = True
        done = True
        reward += -10
    else:
        reward += 5
    #####

    ##### Velocity       
    if self.speed > self.target_speed:
        self.speed_flag = True
        reward += 1.0 - (self.speed - self.target_speed) / self.max_speed - self.min_speed
    elif self.speed < self.min_speed:
        self.speed_flag = True
        reward += self.speed / self.min_speed
    else:
        reward += 1.0
    #####
    
    ##### Lane Invasion History
    if len(self.lane_invasion_hist) != 0:
        self.lane_invasion_flag = True
        reward += -10
        #print('Lane invaded')
    else:
        reward += 5
    #####

    if self.dist_from_start > 20 and self.speed > self.min_speed:
        reward += 0.2 * (self.dist_from_start + self.min_speed)

    ##### Calculating danger level based on nearest pedestrian distance
    nearest_pedestrian_distance = self.nearest_pedestrian_distance()
    # self.danger_level = self.calculate_danger_level(nearest_pedestrian_distance)
    if nearest_pedestrian_distance <= 5.0 :
        # print('Pedestrian too close . . .')
        self.pedestrian_flag = True
        reward -= 0.25*50
    elif nearest_pedestrian_distance > 5.0 and nearest_pedestrian_distance < 10.0:
        reward -= 0.25*10
    #####
                
    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
    # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
    angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

    if not done:
        if self.continuous_action_space:
            if self.speed < self.min_speed:
                reward += (self.speed / self.min_speed) * centering_factor * angle_factor    
            elif self.speed > self.target_speed:               
                reward += (1.0 - (self.speed - self.target_speed) / (self.max_speed - self.target_speed)) * centering_factor * angle_factor  
            else:                                         
                reward += 1.0 * centering_factor * angle_factor 
        else:
            reward += 1.0 * centering_factor * angle_factor
    
    if self.frame_step >= self.steps_per_episode:
        print('Completed all the frame_steps, done = True')
        
        if self.dist_from_start < 20 and self.speed < self.min_speed:
            reward -= 200
            print('Penalty for being stationary :', str(reward))
        else:
            reward += 5
        done = True
    elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
        print('Waypoint completed, done = True')
        done = True

    
    return reward, done


def reward_func2(self):

    reward, done = 0, False

    self.low_speed_timer += 1.0 / self.fps

    # Low Speed Timer
    if self.low_speed_timer > 10.0 and self.speed < 1.0 and self.dist_from_start < 10:
        reward -= 500
        done = True
        print('Low speed timer : ', str(self.low_speed_timer))
        self.low_speed_timer_flag = True
        
        return reward, done

    # Collision History
    if len(self.collision_hist) != 0:
        self.collision_flag = True
        done = True
        reward += -200
    else:
        reward += 1

    # Distance from center
    if self.distance_from_center > self.max_distance_from_center:
        self.distance_from_center_flag = True
        done = True
        reward -= 10
    else:
        reward += 1

    if len(self.lane_invasion_hist) != 0:
        self.lane_invasion_flag = True
        reward -= 1
    else:
        reward += 1

    if self.speed < self.min_speed:  # When speed is in [0, min_speed] range
        self.speed_flag = True
        speed_reward = self.speed / self.min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif self.speed > self.target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        self.speed_flag = True
        speed_reward = 1.0 - (self.speed - self.target_speed) / (self.max_speed - self.target_speed)
    else:  # Otherwise
        speed_reward = 1.0

    centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
    # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
    angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

    reward += speed_reward * centering_factor * angle_factor

    # Calculating danger level based on nearest pedestrian distance
    nearest_pedestrian_distance = self.nearest_pedestrian_distance()
    # self.danger_level = self.calculate_danger_level(nearest_pedestrian_distance)
    if nearest_pedestrian_distance <= 5.0 :
        # print('Pedestrian too close . . .')
        self.pedestrian_flag = True
        reward -= 0.25*50
    elif nearest_pedestrian_distance > 5.0 and nearest_pedestrian_distance < 10.0:
        reward -= 0.25*10

    # Interpolated from 1 when centered to 0 when 3 m from center
    reward += (self.current_waypoint_index - self.prev_waypoint_index) + speed_reward * centering_factor

    # If steps_per_episodes are exceeded or waypoints are exceeded.
    if self.frame_step >= self.steps_per_episode:
        print('Completed all the frame_steps, done = True')
        reward += 50
        done = True
    elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
        print('Waypoint completed, done = True')
        reward += 50
        done = True

    return reward, done


def reward_fn_following_lane(self):

    done = False
    reward = 0

    if len(self.collision_hist) != 0:
        done = True
        reward = -20
        self.collision_flag = True    
    elif self.dis_to_left > self.max_distance_from_center or self.dis_to_right > self.max_distance_from_center:
        done = True
        reward = -10
        self.distance_from_center_flag = True
    elif self.episode_start_time + 10 < time.time() and self.speed < 1.0:
        reward = -10
        done = True
        self.low_speed_timer_flag = True
    elif self.speed > self.max_speed:
        reward = -10
        done = True
        self.speed_flag = True
        

    # If steps_per_episodes are exceeded or waypoints are exceeded.
    if self.frame_step >= self.steps_per_episode:
        print('Completed all the frame_steps, done = True')
        done = True

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

    return reward, done


def reward_fn_waypoint(self):

    done = False
    reward = 0

    if len(self.collision_hist) != 0:
        done = True
        reward = -10
        self.collision_flag = True    
    elif self.dis_to_left > self.max_distance_from_center or self.dis_to_right > self.max_distance_from_center:
        done = True
        reward = -10
        self.distance_from_center_flag = True
    elif self.episode_start_time + 10 < time.time() and self.speed < 1.0:
        reward = -10
        done = True
        self.low_speed_timer_flag = True
    elif self.speed > self.max_speed:
        reward = -10
        done = True
        self.speed_flag = True
    elif len(self.lane_invasion_hist) != 0:
        self.lane_invasion_flag = True
        reward -= 0.01

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
    # elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
    #     print('Waypoint completed, done = True')
    #     reward += 20
    #     done = True

    return reward, done


def reward_fn_pedestrian(self):

    done = False
    reward = 0

    if len(self.collision_hist) != 0:
        done = True
        reward = -10
        self.collision_flag = True    
    elif self.dis_to_left > self.max_distance_from_center or self.dis_to_right > self.max_distance_from_center:
        done = True
        reward = -10
        self.distance_from_center_flag = True
    elif self.episode_start_time + 10 < time.time() and self.speed < 1.0:
        reward = -10
        done = True
        self.low_speed_timer_flag = True
    elif self.speed > self.max_speed:
        reward = -10
        done = True
        self.speed_flag = True
    elif len(self.lane_invasion_hist) != 0:
        self.lane_invasion_flag = True
        reward -= 0.01
   
    ##### Calculating danger level based on nearest pedestrian distance
    nearest_pedestrian_distance = self.nearest_pedestrian_distance()

    if nearest_pedestrian_distance <= 3.0 :
        # print('Pedestrian too close . . .')
        self.pedestrian_flag = True
        reward -= 0.25*50
    elif nearest_pedestrian_distance > 3.0 and nearest_pedestrian_distance < 5.0:
        reward -= 0.25*10

    #####

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
    # elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
    #     print('Waypoint completed, done = True')
    #     reward += 20
    #     done = True

    return reward, done