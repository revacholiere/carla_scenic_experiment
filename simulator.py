import logging
import os
import random

import carla
import numpy as np
from agents.tools.misc import get_speed

from config import cfg
from object_info import ObjectInfo


logger = logging.getLogger(__name__)


class CarlaSimulator:
    def __init__(self, seed):
        logger.debug(f"CARLA_HOST: {cfg.carla.host}")
        logger.debug(f"CARLA_PORT: {cfg.carla.port}")
        self.tm_port = cfg.carla.port + cfg.carla.tm_port_offset
        logger.debug(f"CARLA_TM_PORT: {self.tm_port}")

        logger.debug("Creating and connecting client...")
        self.client = carla.Client(cfg.carla.host, cfg.carla.port)
        logger.debug("Client connected!")
        self.client.set_timeout(cfg.carla.timeout_secs)

        logger.debug("Loading world...")
        world = self.client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = cfg.carla.fixed_delta_seconds
        settings.max_substep_delta_time = cfg.carla.max_substep_delta_time
        settings.max_substeps = cfg.carla.max_substeps

        world.apply_settings(settings)
        logger.debug(f"WORLD SETTINGS: {settings}")
        self.client.reload_world(False)

        self._vehicle_speeds = []
        self._vehicle_locations = []

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        logger.debug(f"TRAFFIC MANAGER PORT: {self.traffic_manager.get_port()}")

        # Set random seed
        self.traffic_manager.set_random_device_seed(seed)
        logger.debug(f"TRAFFIC MANAGER SEED: {seed}")

    def generate_world(self, epoch, trajectory, is_semi_random=False):
        # spawn vehicle(Our agent)
        world = self.client.get_world()
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        self.vehicle = None
        self.npcs = []
        self.controllers = []
        if is_semi_random:
            # index of raod with 4 lanes
            road_4_lanes = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                10,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]

            # random select a road
            selected_road_id = random.choice(road_4_lanes)

            # get all waypoints in the map
            waypoints = world.get_map().generate_waypoints(distance=2.0)

            # filter the waypoints within the given road
            filtered_waypoints = []
            for waypoint in waypoints:
                if waypoint.road_id == selected_road_id:
                    filtered_waypoints.append(waypoint)

            # spawn the ego vehicle a random selected waypoint
            length = int(len(filtered_waypoints) / 4)
            index = random.randint(0, length - 1)

            # the filtered point has the structure with same road, same location but 4 different lane together
            # index = 4*index is the point we want to spwan ego vehicle
            # index = 4*index+1 is the left lane
            # index = 4*idnex+2 is the oppsite lane 1
            # index = 4*idnex+2 is the oppsite lane 2

            random_waypoint = filtered_waypoints[4 * index]
            transform = random_waypoint.transform
            transform.location.z += 2
            self.vehicle = None
            while not self.vehicle:
                self.vehicle = world.try_spawn_actor(vehicle_bp, transform)

            # spawn a ped
            sidewalk = world.get_map().get_waypoint(
                random_waypoint.transform.location, lane_type=carla.LaneType.Sidewalk
            )
            ped_bp = random.choice(bp_lib.filter("walker.pedestrian.*"))

            ped_waypoint = sidewalk.next(15.0)
            if ped_waypoint:
                transform = ped_waypoint[0].transform
                transform.location.z += 2
                npc_ped = world.try_spawn_actor(ped_bp, transform)
                if npc_ped:
                    self.npcs.append(npc_ped)

            # spawn vehicle in front of selected waypoint
            next_random_waypoint = random_waypoint.next(8.0)[0]
            transform = next_random_waypoint.transform
            transform.location.z += 2
            vehicle_bp = random.choice(bp_lib.filter("vehicle"))
            vehicle1 = None
            while not vehicle1:
                vehicle1 = world.try_spawn_actor(vehicle_bp, transform)
            self.npcs.append(vehicle1)
            vehicle1.set_autopilot(True, self.tm_port)

            # spawn vehicle left to selected waypoint
            right_waypoint = filtered_waypoints[4 * index + 1]
            transform = right_waypoint.next(10.0)[0].transform
            transform.location.z += 2
            vehicle_bp = random.choice(bp_lib.filter("vehicle"))
            vehicle2 = None
            while not vehicle2:
                vehicle2 = world.try_spawn_actor(vehicle_bp, transform)
            self.npcs.append(vehicle2)
            vehicle2.set_autopilot(True, self.tm_port)

            # spawn vehicle in the oppsite direction lane 1 of selected waypoint
            left_waypoint = filtered_waypoints[4 * index + 2]
            transform = left_waypoint.previous(10.0)[0].transform
            transform.location.z += 2
            vehicle_bp = random.choice(bp_lib.filter("vehicle"))
            vehicle3 = None
            while not vehicle3:
                vehicle3 = world.try_spawn_actor(vehicle_bp, transform)
            self.npcs.append(vehicle3)
            vehicle3.set_autopilot(True, self.tm_port)

            # spawn vehicle in the oppsite direction lane 2 of selected waypoint
            left_waypoint = filtered_waypoints[4 * index + 3]
            transform = left_waypoint.previous(15.0)[0].transform
            transform.location.z += 2
            vehicle4 = None
            while not vehicle4:
                vehicle4 = world.try_spawn_actor(vehicle_bp, transform)
            self.npcs.append(vehicle4)
            vehicle4.set_autopilot(True, self.tm_port)

        else:
            spawn_points = world.get_map().get_spawn_points()
            while not self.vehicle:
                self.vehicle = world.try_spawn_actor(
                    vehicle_bp, random.choice(spawn_points)
                )

            # randomly generate vehicles on current environment
            for _ in range(cfg.carla.num_vehicles):
                vehicle_bp = random.choice(bp_lib.filter("vehicle"))
                npc_vehicle = world.try_spawn_actor(
                    vehicle_bp, random.choice(spawn_points)
                )
                if npc_vehicle:
                    self.npcs.append(npc_vehicle)
                    # npc_vehicle.set_autopilot(True)
                    npc_vehicle.set_autopilot(True, self.tm_port)

            # randomly generate pedestrians on current environment
            # Currently the pedestrian still is static, will add more code to enable the random behavior
            for _ in range(cfg.carla.num_pedestrians):
                ped_bp = random.choice(bp_lib.filter("walker.pedestrian.*"))
                npc_ped = world.try_spawn_actor(ped_bp, random.choice(spawn_points))
                if npc_ped:
                    self.npcs.append(npc_ped)

            # have to call world.tick() to do sync with all ped positions, otherwise the
            # ped AI controller would mess up the navigation
            world.tick()

            for npc in self.npcs:
                if isinstance(npc, carla.Walker):
                    walker_controller_bp = bp_lib.find("controller.ai.walker")
                    walker_controller = world.spawn_actor(
                        walker_controller_bp, carla.Transform(), npc
                    )
                    walker_controller.start()
                    walker_controller.go_to_location(
                        world.get_random_location_from_navigation()
                    )
                    walker_controller.set_max_speed(1 + random.random())
                    self.controllers.append(walker_controller)
        for _ in range(5):
            world.tick()
        return self.vehicle

    def step(self):
        self.client.get_world().tick()

        logger.debug(f"Vehicle id: {self.vehicle.id}")
        logger.debug(f"Vehicle velocity: {self.vehicle.get_velocity()}")
        vehicle_speed = get_speed(self.vehicle)
        logger.debug(f"Vehicle speed: {vehicle_speed}")
        self._vehicle_speeds.append(vehicle_speed)

        vehicle_location = self.vehicle.get_location()
        logger.debug(f"Vehicle location: {vehicle_location}")
        self._vehicle_locations.append(vehicle_location)

    def get_vehicle_states(self):
        return self._vehicle_speeds, self._vehicle_locations

    def destroy(self):
        logger.debug("Stopping controllers...")
        for controller in self.controllers:
            controller.stop()
        logger.debug("Destroying actors...")
        self.client.apply_batch(
            [
                carla.command.DestroyActor(x)
                for x in self.npcs + [self.vehicle] + self.controllers
            ]
        )
        logger.debug("Deleting client...")
        del self.client


def get_ground_truth_objects(vehicle, camera, distance_threshold=50):
    world = vehicle.get_world()
    logger.debug(f"vehicle_transform: {vehicle.get_transform()}")
    logger.debug(f"camera transform: {camera.get_transform()}")

    objects = []

    vehicles = world.get_actors().filter("*vehicle*")
    pedestrians = world.get_actors().filter("*walker.pedestrian*")
    npcs_and_categories = list(zip(vehicles, ["car"] * len(vehicles))) + list(
        zip(pedestrians, ["person"] * len(pedestrians))
    )

    for npc, category in npcs_and_categories:
        # Filter out the ego vehicle
        if npc.id == vehicle.id:
            logger.debug(f"Skipping ego vehicle: {npc}")
            continue

        # Filter for the vehicles within distance_threshold
        distance = npc.get_transform().location.distance(
            vehicle.get_transform().location
        )

        if distance < 1.0 or distance > distance_threshold:
            continue

        # Calculate the dot product between the forward vector
        # of the vehicle and the vector between the vehicle
        # and the other vehicle. We threshold this dot product
        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
        forward_vec = vehicle.get_transform().get_forward_vector()
        ray = npc.get_transform().location - vehicle.get_transform().location
        if forward_vec.dot(ray) < 0:
            continue

        npc_bbox_3d = npc.bounding_box

        world_verts = [
            [v.x, v.y, v.z] for v in npc_bbox_3d.get_world_vertices(npc.get_transform())
        ]

        # Project the 3D bounding box to the image
        image_points = project_points_world_to_image(world_verts, camera)
        x_min = min([p[0] for p in image_points])
        y_min = min([p[1] for p in image_points])
        x_max = max([p[0] for p in image_points])
        y_max = max([p[1] for p in image_points])

        speed = get_speed(npc)

        logger.debug(f"npc: {npc}")
        logger.debug(f"npc transform: {npc.get_transform()}")
        logger.debug(f"npc bounding box: {npc.bounding_box}")
        logger.debug(f"distance: {distance}")
        logger.debug(f"speed: {speed}")
        logger.debug(f"forward_vec: {forward_vec}")
        logger.debug(f"ray: {ray}")
        logger.debug(f"verts ({len(world_verts)}): {world_verts}")
        logger.debug(f"image_points ({len(image_points)}): {image_points}")
        logger.debug(f"bbox: {x_min}, {y_min}, {x_max}, {y_max}")

        obj = ObjectInfo(
            [x_min, y_min, x_max, y_max],
            category,
            1,
            npc.get_location(),
            distance,
            speed,
            npc_bbox_3d,
        )
        objects.append(obj)
    return objects


def build_projection_matrix_and_inverse(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    K_inv = np.linalg.inv(K)
    return K, K_inv


def clamp_point_to_image(point, image_width, image_height):
    x, y = point
    x = max(0, min(x, image_width))
    y = max(0, min(y, image_height))
    return x, y


def ue4_to_standard_coordinates(point):
    return np.array([point[1], -point[2], point[0]])


def standard_to_ue4_coordinates(point):
    return np.array([point[2], point[0], -point[1]])


def project_points_world_to_image(points, camera):
    image_width = int(camera.attributes["image_size_x"])
    image_height = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
    K, _ = build_projection_matrix_and_inverse(image_width, image_height, fov)
    w2c = np.array(camera.get_transform().get_inverse_matrix())

    # Add homogeneous coordinate
    points = [np.append(p, 1) for p in points]

    # Transform to camera coordinates
    points_camera = [np.dot(w2c, p) for p in points]

    # Transform to standard coordinates
    points_camera_standard = [ue4_to_standard_coordinates(p) for p in points_camera]
    points_image = [np.dot(K, p) for p in points_camera_standard]
    points_image_normalized = [p[0:2] / p[2] for p in points_image]
    return points_image_normalized


def reconstruct_points_image_depth_to_world(points, camera):
    image_width = int(camera.attributes["image_size_x"])
    image_height = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
    _, K_inv = build_projection_matrix_and_inverse(image_width, image_height, fov)
    c2w = np.array(camera.get_transform().get_matrix())

    points_world = []
    for x, y, depth in points:
        # Unproject to camera space
        point_camera_homogeneous = np.dot(K_inv, np.array([x, y, 1]))
        point_camera = point_camera_homogeneous[:3] * depth  # Scale by depth

        point_camera_ue4 = standard_to_ue4_coordinates(point_camera)
        point_world_homogeneous = np.dot(c2w, np.append(point_camera_ue4, 1))
        point_world = point_world_homogeneous[:3] / point_world_homogeneous[3]
        points_world.append(point_world)

    return points_world


def compute_locations_and_bboxes_3d(depth_image_meters, bboxes_xyxy, camera):
    logger.debug(f"depth_image_meters.shape: {depth_image_meters.shape}")
    logger.debug(
        f"depth_image_meters (min, max): ({np.min(depth_image_meters)}, {np.max(depth_image_meters)})"
    )
    locations = []
    bboxes_3d = []
    for bbox in bboxes_xyxy:
        x1, y1, x2, y2 = bbox
        xc = int((x1 + x2) / 2)
        yc = int((y1 + y2) / 2)

        depth = depth_image_meters[yc, xc]
        logger.debug(f"depth: {depth}")

        points_2d = [
            (xc, yc, depth),
            (x1, y1, depth),
            (x1, y2, depth),
            (x2, y1, depth),
            (x2, y2, depth),
        ]
        logger.debug(f"points_2d: {points_2d}")
        points_3d = reconstruct_points_image_depth_to_world(points_2d, camera)
        logger.debug(f"points_3d: {points_3d}")
        bbox_3d_center = points_3d[0]
        bbox_3d_corners = points_3d[1:]

        # Use camera transform to get world coordinates
        # location_vec = camera.get_transform().transform(carla.Location(*bbox_3d_center))
        # logger.debug(f"location_vec: {location_vec}")
        # location = carla.Location(x=location_vec.x, y=location_vec.y, z=location_vec.z)

        #######################################################################################################
        # The bbox_3d_center already is in world coordinate system, we don't need to do transformation again  #
        # Above three lines code can be deleted                                                               #
        #######################################################################################################
        location = carla.Location(
            x=bbox_3d_center[0], y=bbox_3d_center[1], z=bbox_3d_center[2]
        )
        logger.debug(f"location: {location}")
        locations.append(location)

        # Compute 3D bounding box extents
        extents = np.max(bbox_3d_corners, axis=0) - np.min(bbox_3d_corners, axis=0)
        logger.debug(f"extents: {extents}")
        bbox_3d = carla.BoundingBox()
        bbox_3d.location = location
        bbox_3d.extent.x, bbox_3d.extent.y, bbox_3d.extent.z = (
            extents / 2
        )  # Assuming center-aligned bbox
        logger.debug(f"bbox_3d: {bbox_3d}")
        bboxes_3d.append(bbox_3d)

    return locations, bboxes_3d


def polar2xyz(points):
    """
    This method convert the radar point from polar coordinate to xyz coordinate
    :param points: points from the radar data with format([[vel, azimuth, altitude, depth],...[,,,]])
    :return: The xyz locations with carla.Location format
    """
    points_location = []
    az_cos = np.cos(points[:, 1])
    az_sin = np.sin(points[:, 1])
    al_cos = np.cos(points[:, 2])
    al_sin = np.sin(points[:, 2])
    points_x = np.multiply(np.multiply(points[:, 3], al_cos), az_cos)
    points_y = np.multiply(np.multiply(points[:, 3], al_cos), az_sin)
    points_z = np.multiply(points[:, 3], al_sin)
    for i in range(points_x.shape[0]):
        x = points_x[i]
        y = points_y[i]
        z = points_z[i]
        point_location = carla.Location(float(x), float(y), float(z))
        # point_location.x = x
        # point_location.y = y
        # point_location.z = z
        # point_location = carla.Location(x=0, y=0,z = 0)
        points_location.append(point_location)
    return points_location


def nearest_point(radar_point, object_location):
    """
    This method find the nearest point of each object_location
    :param radar_point: The point captured from radar(carla.Location)
    :param object_location: The object location in the simulated world(carla.Location)
    :return: The searched location index
    """
    indices = []
    for i in range(len(object_location)):
        min_dist = 10000
        index = 0
        for j in range(len(radar_point)):
            ol = object_location[i]
            rp = radar_point[j]
            dist = ol.distance(rp)
            if dist < min_dist:
                index = j
                min_dist = dist
        indices.append(index)
    return indices
