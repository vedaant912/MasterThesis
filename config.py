reward_params = {
    "reward_fn_5_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
     "reward_fn_5_no_early_stop": dict(
         early_stop=False,
         min_speed=20.0,  # km/h
         max_speed=35.0,  # km/h
         target_speed=25.0,  # kmh
         max_distance=3.0,  # Max distance from center before terminating
         max_std_center_lane=0.4,
         max_angle_center_lane=90,
         penalty_reward=-10,
     ),
    "reward_fn_5_best": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
}