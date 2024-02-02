import logging

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import compute_distance
from agents.tools.misc import get_speed
from agents.tools.misc import get_trafficlight_trigger_location
from agents.tools.misc import is_within_distance
from shapely.geometry import Polygon

from object_info import ObjectInfo


logger = logging.getLogger(__name__)


class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(
        self,
        vehicle,
        target_speed=20,
        opt_dict={},
        map_inst=None,
        grp_inst=None,
        detection=None,
    ):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._detection = detection
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                logger.warning("Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0

        # Change parameters according to the dictionary
        opt_dict["target_speed"] = target_speed
        if "ignore_traffic_lights" in opt_dict:
            self._ignore_traffic_lights = opt_dict["ignore_traffic_lights"]
        if "ignore_stop_signs" in opt_dict:
            self._ignore_stop_signs = opt_dict["ignore_stop_signs"]
        if "ignore_vehicles" in opt_dict:
            self._ignore_vehicles = opt_dict["ignore_vehicles"]
        if "use_bbs_detection" in opt_dict:
            self._use_bbs_detection = opt_dict["use_bbs_detection"]
        if "sampling_resolution" in opt_dict:
            self._sampling_resolution = opt_dict["sampling_resolution"]
        if "base_tlight_threshold" in opt_dict:
            self._base_tlight_threshold = opt_dict["base_tlight_threshold"]
        if "base_vehicle_threshold" in opt_dict:
            self._base_vehicle_threshold = opt_dict["base_vehicle_threshold"]
        if "detection_speed_ratio" in opt_dict:
            self._speed_ratio = opt_dict["detection_speed_ratio"]
        if "max_brake" in opt_dict:
            self._max_brake = opt_dict["max_brake"]
        if "offset" in opt_dict:
            self._offset = opt_dict["offset"]

        # Initialize the planners
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict=opt_dict, map_inst=self._map
        )
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                logger.warning("Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(
                    self._map, self._sampling_resolution
                )
        else:
            self._global_planner = GlobalRoutePlanner(
                self._map, self._sampling_resolution
            )

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = (
            {}
        )  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan, stop_waypoint_creation=stop_waypoint_creation, clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation."""
        hazard_detected = False

        # Retrieve all relevant actors
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = (
            self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        )
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(
            vehicle_list, max_vehicle_distance
        )
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = (
            self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        )
        affected_by_tlight, _ = self._affected_by_traffic_light(
            self._lights_list, max_tlight_distance
        )
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def lane_change(
        self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2
    ):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution,
        )
        if not path:
            logger.warning("Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if (
                trigger_wp.transform.location.distance(ego_vehicle_location)
                > max_distance
            ):
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(
                trigger_wp.transform,
                self._vehicle.get_transform(),
                max_distance,
                [0, 90],
            ):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected(
        self,
        vehicle_list=None,
        max_distance=None,
        up_angle_th=90,
        low_angle_th=0,
        lane_offset=0,
    ):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """

        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(
                    r_ext * r_vec.x, r_ext * r_vec.y
                )
                p2 = wp.transform.location + carla.Location(
                    l_ext * r_vec.x, l_ext * r_vec.y
                )
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        def get_est_bbs3D(vehicle):
            x_offset = []
            y_offset = []
            z_offset = []
            target_list = []
            location = vehicle.get_location()
            if vehicle.category == 0:
                x_offset.append(-0.5)
                x_offset.append(0.5)
                y_offset.append(-0.5)
                y_offset.append(0.5)
                z_offset.append(1.5)

            if vehicle.category == 1:
                x_offset.append(-1)
                x_offset.append(1)
                y_offset.append(-1)
                y_offset.append(1)
                z_offset.append(0)
                z_offset.append(1.5)

            for i in range(len(x_offset)):
                for j in range(len(y_offset)):
                    for k in range(len(z_offset)):
                        target_list.append(
                            [
                                location.x + x_offset[i],
                                location.y + y_offset[j],
                                location.z + z_offset[k],
                            ]
                        )

            return target_list

        if not vehicle_list:
            # vehicle_list = self._world.get_actors().filter("*vehicle*")
            return (False, None, -1)

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector()
        )

        opposite_invasion = (
            abs(self._offset) + self._vehicle.bounding_box.extent.y
            > ego_wpt.lane_width / 2
        )
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            # distance greater than safe distance
            if target_vehicle.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(
                target_vehicle.location, lane_type=carla.LaneType.Any
            )

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:
                target_bb = target_vehicle.bbox_3d
                # target_bb = get_est_bbs3D(target_vehicle)
                offset = [1, -1]
                target_list = []
                location = target_bb.location
                extent = target_bb.extent
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            target_list.append(
                                [
                                    location.x + offset[i] * extent.x,
                                    location.y + offset[j] * extent.y,
                                    location.z + offset[k] * extent.z,
                                ]
                            )

                # target_list = [[v.x, v.y, v.z] for v in target_vertices]

                # target_bb = target_vehicle.bounding_box
                # target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                # target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (
                        True,
                        target_vehicle,
                        compute_distance(target_vehicle.location, ego_location),
                    )

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:
                if (
                    target_wpt.road_id != ego_wpt.road_id
                    or target_wpt.lane_id != ego_wpt.lane_id + lane_offset
                ):
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(
                        steps=3
                    )[0]
                    if not next_wpt:
                        continue
                    if (
                        target_wpt.road_id != next_wpt.road_id
                        or target_wpt.lane_id != next_wpt.lane_id + lane_offset
                    ):
                        continue

                target_rear_transform = carla.Transform(
                    target_vehicle.location, carla.Rotation(0, 0, 0)
                )

                if is_within_distance(
                    target_rear_transform, ego_front_transform, max_distance
                ):
                    return (
                        True,
                        target_vehicle,
                        compute_distance(
                            target_vehicle.location, ego_transform.location
                        ),
                    )

        return (False, None, -1)

    def _generate_lane_change_path(
        self,
        waypoint,
        direction="left",
        distance_same_lane=10,
        distance_other_lane=25,
        lane_change_distance=25,
        check=True,
        lane_changes=1,
        step_distance=2,
    ):
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(
                plan[-1][0].transform.location
            )
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == "left":
            option = RoadOption.CHANGELANELEFT
        elif direction == "right":
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:
            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == "left":
                if check and str(next_wp.lane_change) not in ["Left", "Both"]:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ["Right", "Both"]:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(
                plan[-1][0].transform.location
            )
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan
