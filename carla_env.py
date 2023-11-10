import atexit
import cv2
import os
import signal
import sys
import carla
import gym
import time
import random
import numpy as np
import math
from queue import Queue
from misc import dist_to_roadline, exist_intersection
from gym import spaces
from setup import setup
from absl import logging
import graphics
import pygame
from simulation.sensors import CameraSensorEnv
from reward import reward_func1, reward_func2, reward_fn_following_lane, reward_fn_waypoint, reward_fn_pedestrian
import subprocess

logging.set_verbosity(logging.INFO)

# Carla environment
class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, action_type, enable_preview, steps_per_episode, create_pedestrian_flag, playing=False, timeout=60):

        super(CarlaEnv, self).__init__()

        self.client, self.world, self.frame, self.server = setup(town=town, fps=fps, client_timeout=timeout)
        self.client.set_timeout(20.0)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.lincoln = self.blueprint_library.filter('model3')[0]
        self.im_width = im_width
        self.im_height = im_height
        self.repeat_action = repeat_action
        self.action_type = action_type
        self.continuous_action_space = True
        self.start_transform_type = start_transform_type
        self.sensors = sensors
        self.actor_list = []
        self.preview_camera = None
        self.steps_per_episode = steps_per_episode
        self.playing = playing
        self.preview_camera_enabled = enable_preview
        self.episode_number = 0
        self.max_distance_covered = 0
        self.fresh_start=True
        self.render_flag = True
        self.fps = fps
        self.create_pedestrian_flag = create_pedestrian_flag

        # waypoint data
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.route_waypoints = None

        self.env_camera_obj = None

        self.walker_list = list()
        self.walker_list_id = list()
        self.walker_list_controller = list()
        
        if self.create_pedestrian_flag:
            self.create_pedestrians()


    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3), dtype=np.uint8)

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        if self.action_type == 'continuous':
            return gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        elif self.action_type == 'discrete':
            return gym.spaces.MultiDiscrete([4, 9])
        else:
            raise NotImplementedError()
        # TODO: Add discrete actions (here and anywhere else required)


    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
        return seed
    
    def get_gpu_usage(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting GPU usage: {e}")
            return None

    # Resets environment for new episode
    def reset(self):
        print('RESET')
        self._destroy_agents()
        
        logging.debug("Resetting environment")
        
        # Car, sensors, etc. We create them every episode then destroy
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.actor_list = []

        self.frame_step = 0
        self.out_of_loop = 0

        self.dist_from_start = 0
        self.speed = 0
        self.distance_from_center = float(0.0)
        self.max_distance_from_center = 3
        self.max_speed = 25.0
        self.min_speed = 10.0
        self.speed_accum = float(0.0)
        self.target_speed = 20.0
        self.center_lane_deviation = 0.0

        self.front_image_Queue = Queue()
        self.preview_image_Queue = Queue()

        self.episode_number += 1
        self.danger_level = 0
        self.episode_start_time = time.time()
        self.reward = 0
        self.done = False

        self.previous_steer = float(0.0)
        self.throttle = float(0.0)
        self.brake = float(0.0)
        self.previous_velocity = float(0.0)

        self.low_speed_timer = float(0.0)
        self.speed_time_flag = True

        self.total_reward = float(0.0)
        self.total_distance_traveled = float(0.0)


        #################### SPAWNING VEHICLE #########################
        spawn_start = time.time()
        while True:
            try:
                # Get random spot from a list from predefined spots and try to spawn a car there
                self.start_transform = self._get_start_transform()
                self.curr_loc = self.start_transform.location
                self.vehicle = self.world.spawn_actor(self.lincoln, self.start_transform)
                break
            except Exception as e:
                logging.error('Error carla 141 {}'.format(str(e)))
                time.sleep(0.01)

            # If that can't be self.done in 3 seconds - forgive (and allow main process to handle for this problem)
            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')
            
        # Append actor to a list of spawned actors, we need to remove them later
        if self.map.name == 'Town07':
                self.total_distance = 750
        elif self.map.name == 'Town02':
            self.total_distance = 780
        else:
            self.total_distance = 250

        self.actor_list.append(self.vehicle)

        # TODO: combine the sensors
        if 'rgb' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        elif 'semantic' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        else:
            raise NotImplementedError('unknown sensor type')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '90')
        bound_x = self.vehicle.bounding_box.extent.x
        bound_y = self.vehicle.bounding_box.extent.y

        transform_front = carla.Transform(carla.Location(x=bound_x, z=1.0))
        self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
        self.sensor_front.listen(self.front_image_Queue.put)
        self.actor_list.extend([self.sensor_front])

        while self.sensor_front is None:
            time.sleep(0.01)

        # Preview ("above the car") camera
        if self.preview_camera_enabled:
            # TODO: add the configs
            self.preview_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            self.preview_cam.set_attribute('image_size_x', '720')
            self.preview_cam.set_attribute('image_size_y', '720')
            self.preview_cam.set_attribute('fov', '100')
            transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
            self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
            self.preview_sensor.listen(self.preview_image_Queue.put)
            self.actor_list.append(self.preview_sensor)

        # Here's some workarounds.
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(4)

        # Collision history is a list callback is going to append to (we brake simulation on a collision)
        self.collision_hist = []
        self.lane_invasion_hist = []

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
        self.colsensor.listen(self._collision_data)
        self.lanesensor.listen(self._lane_invasion_data)
        self.actor_list.append(self.colsensor)
        self.actor_list.append(self.lanesensor)

        self.world.tick()

        # Wait for a camera to send first image (important at the beginning of first episode)
        while self.front_image_Queue.empty():
            logging.debug("waiting for camera to be ready")
            time.sleep(0.01)
            self.world.tick()

        # Disengage brakes
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.im_height, self.im_width, -1))
        image = image[:, :, :3]

        # WAYPOINT DATA
        # if self.fresh_start:
        #         self.current_waypoint_index = 0
        #         # Waypoint nearby angle and distance from it
        #         print('Creating Waypoints . . . . ')
        #         self.route_waypoints = list()
        #         self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
        #         current_waypoint = self.waypoint
        #         self.route_waypoints.append(current_waypoint)

        #         for x in range(self.total_distance):
        #             if self.map.name == "Town07":
        #                 if x < 650:
        #                     next_waypoint = current_waypoint.next(1.0)[0]
        #                 else:
        #                     next_waypoint = current_waypoint.next(1.0)[-1]
        #             elif self.map.name == "Town02":
        #                 if x < 650:
        #                     next_waypoint = current_waypoint.next(1.0)[-1]
        #                 else:
        #                     next_waypoint = current_waypoint.next(1.0)[0]
        #             else:
        #                 next_waypoint = current_waypoint.next(1.0)[0]
        #             self.route_waypoints.append(next_waypoint)
        #             current_waypoint = next_waypoint

        #         ###################### TRIAL CODE #############################

        #         # waypoint_radius = 0.2  # Adjust as needed
        #         # waypoint_color = carla.Color(r=255, g=0, b=0)

        #         # self.waypoint_markers = []  # List to store waypoint markers

        #         # for waypoint in self.route_waypoints:
        #         #     waypoint_location = waypoint.transform.location
        #         #     waypoint_marker = self.world.debug.draw_point(
        #         #         waypoint_location,
        #         #         size=waypoint_radius,
        #         #         color=waypoint_color,
        #         #         life_time=5  # Set to 0 for permanent markers, or adjust as needed
        #         #     )
        #         #     self.waypoint_markers.append(waypoint_marker) 

        #         ###############################################################
        # else:
        #     # Teleport vehicle to last checkpoint
        #     waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
        #     transform = waypoint.transform
        #     self.vehicle.set_transform(transform)
        #     self.current_waypoint_index = self.checkpoint_waypoint_index
        
        return image

    # Steps environment
    def step(self, action):
        
        if self.render_flag and self.preview_camera_enabled:
            self.render_flag = False
            mode='human'

            self._display, self._clock, self._font = graphics.setup(
                    width=720,
                    height=720,
                    render=(mode=="human"),
                )
            
        self.action = action

        self.world.tick()
        self.render()
            
        self.frame_step += 1
        self.fresh_start = False

        # Calculate speed in km/h from car's velocity (3D vector)
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        self.speed = kmh

        # Apply control to the vehicle based on an action
        if self.action_type == 'continuous':
            
            throttle = float(action[0]) if action[0] > 0 else 0.0
            brake = float(-action[0]) if action[0] < 0 else 0.0
            steer = action[1]
            steer = max(min(steer, 1.0), -1.0)

            
            self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, 
                                                            throttle=self.throttle*0.9 + throttle*0.1,
                                                            brake=self.brake*0.9 + 0.1*brake,
                                                    ))
            
            self.previous_steer = steer
            self.throttle = throttle
            self.brake = brake
        else:
            steer = self.action_space[action]
            if kmh < 20.0:
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
            else:
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
            self.previous_steer = steer
            self.throttle = 1.0
            
        loc = self.vehicle.get_location()
        new_dist_from_start = loc.distance(self.start_transform.location)
        square_dist_diff = new_dist_from_start ** 2 - self.dist_from_start ** 2
        self.dist_from_start = new_dist_from_start

        image = self.front_image_Queue.get()
        # image.save_to_disk('tutorial/output/%.6d.jpg' % image.frame)
        image = np.array(image.raw_data)
        image = image.reshape((self.im_height, self.im_width, -1))

        # TODO: Combine the sensors
        if 'rgb' in self.sensors:
            image = image[:, :, :3]
        if 'semantic' in self.sensors:
            image = image[:, :, 2]
            image = (np.arange(13) == image[..., None])
            image = np.concatenate((image[:, :, 2:3], image[:, :, 6:8]), axis=2)
            image = image * 255
            
            # logging.debug('{}'.format(image.shape))
            # assert image.shape[0] == self.im_height
            # assert image.shape[1] == self.im_width
            # assert image.shape[2] == 3

        self.dis_to_left, self.dis_to_right, self.waypoint = dist_to_roadline(self.map, self.vehicle)
        self.location = self.vehicle.get_location()

        self.distance_from_center = self.dis_to_left if self.dis_to_left > 0 else self.dis_to_right

        ############################ WAYPOINT CODE ################################

        # keep track of closest waypoint on the route
        # self.prev_waypoint_index = self.current_waypoint_index
        # waypoint_index = self.current_waypoint_index
        # for _ in range(len(self.route_waypoints)):

        #     # Check if we pased the next waypoint along the route
        #     next_waypoint_index = waypoint_index + 1
        #     wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
        #     dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
        #     if dot > 0.0:
        #         waypoint_index += 1
        #     else:
        #         break

        # self.current_waypoint_index = waypoint_index

        # # Calculate deviation from center of the lane
        # self.current_waypoint = self.route_waypoints[ self.current_waypoint_index % len(self.route_waypoints) ]
        # self.next_waypoint =  self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
        # self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
        # self.center_lane_deviation += self.distance_from_center

        self.center_lane_deviation += self.dis_to_left if self.dis_to_left > 0 else self.dis_to_right

        # Get angle difference between closest waypoint and vehicle forward vector
        fwd = self.vector(self.vehicle.get_velocity())
        wp_fwd = self.vector(self.waypoint.transform.rotation.get_forward_vector())
        self.angle = self.angle_diff(fwd, wp_fwd)

        ############################### REWARDS ##########################################
        
        self.done = False
        info = dict()

        self.collision_flag = False
        self.distance_from_center_flag = False
        self.speed_flag = False
        self.lane_invasion_flag = False
        self.pedestrian_flag = False
        self.low_speed_timer_flag = False

        if self.speed < 1:

            if self.speed_time_flag:

                self.episode_start_time = time.time()
                self.speed_time_flag = False
        else:
            
            self.speed_time_flag = True
        
        self.reward, self.done = reward_fn_pedestrian(self)

        self.speed_accum += self.speed
        self.total_reward += self.reward
        self.total_distance_traveled += self.dist_from_start
        gpu_usage = self.get_gpu_usage()

        if self.done:
            
            self.information_print(self.collision_flag, self.distance_from_center_flag, self.speed_flag, self.lane_invasion_flag, self.pedestrian_flag, self.low_speed_timer_flag)

            if self.dist_from_start > self.max_distance_covered:
                self.max_distance_covered = self.dist_from_start
                print('New maximum distance : ', str(self.max_distance_covered))
                self.reward += 20
            else:
                self.reward = self.reward - 10 + 0.2*self.dist_from_start
            
            self.fresh_start = True

            self.total_reward += self.reward

            self.world.tick()
            self.render()
            
            print('Total Rewards : ', str(self.total_reward))
            print('Last step reward : ', str(self.reward))
            print('Current distance : ' + str(self.dist_from_start) + ' Prev Max Distance :' + str(self.max_distance_covered))
            print("Env lasts {} steps, restarting ... ".format(self.frame_step))
            print(f'Throttle : {throttle} , Brake : {brake}')
            
        info = {
                'total_reward' : self.total_reward,
                'episode_number': self.episode_number,
                'avg_center_dev' : (self.center_lane_deviation / self.frame_step),
                'avg_speed' : (self.speed_accum / self.frame_step),
                'dist_from_start' : self.dist_from_start,
                'gpu_usage' : gpu_usage
            }
                
        return image, self.reward, self.done, info
    
    def information_print(self, col_flag, dist_flag, speed_flag, lane_invasion_flag, pedestrian_flag,low_speed_timer):
        
        print('################# REASONS #####################')
        print('Collision History : ',col_flag)
        print('Dist from Center  : ',dist_flag)
        print('Speed Exceeded    : ',speed_flag)
        print('Lane Invasion     : ',lane_invasion_flag)
        print('Pedestrian Near   : ',pedestrian_flag)
        print('Low Speed Timer   : ',low_speed_timer)

    def close(self):
        logging.info("Closes the CARLA server with process PID {}".format(self.server.pid))
        os.killpg(self.server.pid, signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))

    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom
    
    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])
    
    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle
    
    def render(self, mode='human'):
        # TODO: clean this
        # TODO: change the width and height to compat with the preview cam config

        if self.preview_camera_enabled:

            if not self.done:
            
                preview_img = self.preview_image_Queue.get()
                preview_img = np.array(preview_img.raw_data)
                preview_img = preview_img.reshape((720, 720, -1))
                preview_img = preview_img[:, :, :3]
                graphics.make_dashboard(
                    display=self._display,
                    font=self._font,
                    clock=self._clock,
                    observations={"preview_camera":preview_img},
                )

            #################### INFORMATION CODE ################################
            info_text = "Total Reward: " + str(self.total_reward) #+ "\nCurrent Reward : " + str(float(self.reward)) # Replace your_variable with the actual variable
            info_surface = self._font.render(info_text, True, (0, 0, 0))  # Render the text surface with white color

            info_text_2 = "Current Reward: " + str(self.reward) + " Distance : " + str(self.dist_from_start)
            info_surface_2 = self._font.render(info_text_2, True, (0, 0, 0))  # Render the text surface with white color


            # Blit the info surface onto the Pygame window
            info_position = (290, 10)  # Adjust the position where you want to display the text
            self._display.blit(info_surface, info_position)
            self._display.blit(info_surface_2, (290, 20))

            if self.done:
                time.sleep(2.0)

            #################### INFORMATION CODE ################################

            if mode == "human":
                # Update window display.
                pygame.display.flip()
            else:
                raise NotImplementedError()

    def _destroy_agents(self):
        
        print('Destroy Agent.')
        for actor in self.actor_list:

            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def _collision_data(self, event):
        # Add collision
        self.collision_hist.append(event)
    
    def _lane_invasion_data(self, event):
        # Change this function to filter lane invasions
        self.lane_invasion_hist.append(event)

    def _on_highway(self):
        goal_abs_lane_id = 4
        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        lane_id_sign = int(np.sign(vehicle_waypoint_closest_to_road.lane_id))
        assert lane_id_sign in [-1, 1]
        goal_lane_id = goal_abs_lane_id * lane_id_sign
        vehicle_s = vehicle_waypoint_closest_to_road.s
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        return not (goal_waypoint is None)

    def _get_start_transform(self):
        if self.start_transform_type == 'random':
            return random.choice(self.map.get_spawn_points())
        if self.start_transform_type == 'highway':
            if self.map.name == "Town02":
                for trial in range(10):
                    start_transform = random.choice(self.map.get_spawn_points())
                    start_waypoint = self.map.get_waypoint(start_transform.location)
                    if start_waypoint.road_id in list(range(35, 50)):
                        break

                return start_transform
            else:
                raise NotImplementedError

    # -------------------------------------------------
    # Removing Pedestrians from the world
    # -------------------------------------------------
    def remove_pedestrians(self):
        if len(self.walker_list) != 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list_id])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list_controller])


            self.walker_list = list()
            self.walker_list_id = list()
            self.walker_list_controller = list()

    # -------------------------------------------------
    # Creating and Spawning Pedestrians in our world |
    # -------------------------------------------------
    def create_pedestrians(self):
        # Our code for this method has been broken into 3 sections.

        # 1. Getting the available spawn points in  our world.
        # Random Spawn locations for the walker
        walker_spawn_points = []
        for i in range(200):
            spawn_point_ = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point_.location = loc
                walker_spawn_points.append(spawn_point_)

        # 2. We spawn the walker actor and ai controller
        # Also set their respective attributes
        for spawn_point_ in walker_spawn_points:
            walker_bp = random.choice(
                self.blueprint_library.filter('walker.pedestrian.*'))
            walker_controller_bp = self.blueprint_library.find(
                'controller.ai.walker')
            # Walkers are made visible in the simulation
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # They're all walking not running on their recommended speed
            if walker_bp.has_attribute('speed'):
                walker_bp.set_attribute(
                    'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
            else:
                walker_bp.set_attribute('speed', 0.0)
            walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
            if walker is not None:
                walker_controller = self.world.spawn_actor(
                    walker_controller_bp, carla.Transform(), walker)
                self.walker_list_id.append(walker_controller.id)
                self.walker_list_id.append(walker.id)
                self.walker_list.append(walker)
        all_actors = self.world.get_actors(self.walker_list_id)

        print('Spawned !!!')

        self.world.tick()

        # set how many pedestrians can cross the road
        #self.world.set_pedestrians_cross_factor(0.0)
        # 3. Starting the motion of our pedestrians
        for i in range(0, len(self.walker_list_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())

    # -------------------------------------------------
    # Calculating distance between vehicle and the nearest pedestrian spawned.
    # -------------------------------------------------
    def nearest_pedestrian_distance(self):

        vehicle_location = self.vehicle.get_location()

        list_of_pedestrians = []

        for i in self.walker_list:

            list_of_pedestrians.append(i.get_location())

        nearest_distance = float('inf')

        for i in list_of_pedestrians:
            dx = vehicle_location.x - i.x
            dy = vehicle_location.y - i.y
            dz = vehicle_location.z - i.z
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            if distance < nearest_distance:
                nearest_distance = distance

        return nearest_distance
    
    # -------------------------------------------------
    # Calculating the danger level based on the nearest_distance calculated
    # -------------------------------------------------
    def calculate_danger_level(self, nearest_distance):
        if nearest_distance <= 5.0 :
            return 50
        elif nearest_distance > 5.0 and nearest_distance < 10.0:
            return 10
        else:
            return 0
        
    def get_discrete_action_space(self):
        action_space = np.array([-0.50,-0.30,-0.10,0.0,0.10,0.30,0.50])
        return action_space