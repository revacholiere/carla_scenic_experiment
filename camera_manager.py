import logging
from queue import Queue

import carla
import numpy as np

from config import cfg


logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, vehicle):
        world = vehicle.get_world()
        bp_lib = world.get_blueprint_library()

        # spawn ego (RGB) camera
        self._ego_camera_bp = bp_lib.find(cfg.carla.ego_camera.name)
        self._ego_camera_bp.set_attribute(
            "image_size_x", str(cfg.carla.ego_camera.image_width)
        )
        self._ego_camera_bp.set_attribute(
            "image_size_y", str(cfg.carla.ego_camera.image_height)
        )
        self._ego_camera_bp.set_attribute("sensor_tick", str(cfg.carla.ego_camera.tick))
        camera_init_trans = carla.Transform(
            carla.Location(x=cfg.carla.ego_camera.x, z=cfg.carla.ego_camera.z)
        )
        self._ego_camera = world.spawn_actor(
            self._ego_camera_bp, camera_init_trans, attach_to=vehicle
        )

        # spawn birdseye camera
        birdseye_transform = carla.Transform(
            carla.Location(
                x=cfg.carla.birdseye_camera.x, z=cfg.carla.birdseye_camera.z
            ),
            carla.Rotation(pitch=cfg.carla.birdseye_camera.pitch),
        )
        self._birdseye_camera_bp = bp_lib.find(cfg.carla.birdseye_camera.name)
        self._birdseye_camera_bp.set_attribute(
            "image_size_x", str(cfg.carla.birdseye_camera.image_width)
        )
        self._birdseye_camera_bp.set_attribute(
            "image_size_y", str(cfg.carla.birdseye_camera.image_height)
        )
        self._birdseye_camera_bp.set_attribute(
            "sensor_tick", str(cfg.carla.birdseye_camera.tick)
        )
        self._birdseye_camera = world.spawn_actor(
            self._birdseye_camera_bp, birdseye_transform, attach_to=vehicle
        )

        # spawn Depth Camera
        self._depth_camera_bp = bp_lib.find(cfg.carla.depth_camera.name)
        self._depth_camera_bp.set_attribute(
            "image_size_x", str(cfg.carla.depth_camera.image_width)
        )
        self._depth_camera_bp.set_attribute(
            "image_size_y", str(cfg.carla.depth_camera.image_height)
        )
        self._depth_camera_bp.set_attribute(
            "sensor_tick", str(cfg.carla.depth_camera.tick)
        )
        self._depth_camera = world.spawn_actor(
            self._depth_camera_bp, camera_init_trans, attach_to=vehicle
        )

        # spawn radar Camera
        self._radar_camera_bp = world.get_blueprint_library().find(
            cfg.carla.radar_camera.name
        )
        self._radar_camera_bp.set_attribute(
            "horizontal_fov", str(cfg.carla.radar_camera.horizontal_fov)
        )
        self._radar_camera_bp.set_attribute(
            "vertical_fov", str(cfg.carla.radar_camera.vertical_fov)
        )
        self._radar_camera_bp.set_attribute("range", str(cfg.carla.radar_camera.range))
        self._radar_camera_bp.set_attribute(
            "points_per_second", str(cfg.carla.radar_camera.points_per_second)
        )
        self._radar_camera_bp.set_attribute(
            "sensor_tick", str(cfg.carla.radar_camera.tick)
        )
        radar_camera_location = carla.Location(
            x=cfg.carla.radar_camera.x, z=cfg.carla.radar_camera.z
        )
        radar_camera_rotation = carla.Rotation(pitch=cfg.carla.radar_camera.pitch)
        rad_transform = carla.Transform(radar_camera_location, radar_camera_rotation)
        self._radar_camera = world.spawn_actor(
            self._radar_camera_bp, rad_transform, attach_to=vehicle
        )

        self._ego_image_queue = Queue()
        self._depth_image_queue = Queue()
        self._radar_image_queue = Queue()
        self._birds_eye_image_queue = Queue()

    def start_collecting_images(self):
        self._ego_camera.listen(self._ego_image_queue.put)
        self._depth_camera.listen(self._depth_image_queue.put)
        self._radar_camera.listen(self._radar_image_queue.put)
        self._birdseye_camera.listen(self._birds_eye_image_queue.put)

    def stop_collecting_images(self):
        self._ego_camera.stop()
        self._depth_camera.stop()
        self._radar_camera.stop()
        self._birdseye_camera.stop()

    def step(self):


        self._ego_image_raw = self._ego_image_queue.get()
        self._depth_image_raw = self._depth_image_queue.get()
        self._radar_image_raw = self._radar_image_queue.get()
        self._birds_eye_image_raw = self._birds_eye_image_queue.get()

    def destroy(self):
        self._ego_camera.destroy()
        self._depth_camera.destroy()
        self._radar_camera.destroy()
        self._birdseye_camera.destroy()

    @property
    def ego_camera(self):
        return self._ego_camera

    @property
    def ego_image_raw(self):
        return self._ego_image_raw

    @property
    def ego_image_array(self):
        ego_image = self.ego_image_raw
        ego_image = np.reshape(
            np.copy(ego_image.raw_data), (ego_image.height, ego_image.width, 4)
        )
        ego_image = ego_image[:, :, 0:3]
        return ego_image

    @property
    def ego_camera_fov(self):
        return self._ego_camera_bp.get_attribute("fov").as_float()

    @property
    def ego_camera_image_width(self):
        return self._ego_camera_bp.get_attribute("image_size_x").as_int()

    @property
    def ego_camera_image_height(self):
        return self._ego_camera_bp.get_attribute("image_size_y").as_int()

    @property
    def birdseye_camera(self):
        return self._birdseye_camera

    @property
    def birdseye_image_raw(self):
        return self._birds_eye_image_raw

    @property
    def birdseye_image_array(self):
        birdseye_image = self.birdseye_image_raw
        birdseye_image = np.reshape(
            np.copy(birdseye_image.raw_data),
            (birdseye_image.height, birdseye_image.width, 4),
        )
        birdseye_image = birdseye_image[:, :, 0:3]
        return birdseye_image

    @property
    def birdseye_camera_fov(self):
        return self._birdseye_camera_bp.get_attribute("fov").as_float()

    @property
    def birdseye_camera_image_width(self):
        return self._birdseye_camera_bp.get_attribute("image_size_x").as_int()

    @property
    def birdseye_camera_image_height(self):
        return self._birdseye_camera_bp.get_attribute("image_size_y").as_int()

    @property
    def depth_camera(self):
        return self._depth_camera

    @property
    def depth_image_raw(self):
        return self._depth_image_raw

    @property
    def depth_image_array(self):
        depth_image = self.depth_image_raw
        depth_image = np.reshape(
            np.copy(depth_image.raw_data), (depth_image.height, depth_image.width, 4)
        )
        depth_image = depth_image[:, :, 0:3]
        return depth_image

    @property
    def depth_image_normalized(self):
        depth_image_array = self.depth_image_array.astype(np.float32)
        # Extract the RGB channels
        B = depth_image_array[:, :, 0]
        G = depth_image_array[:, :, 1]
        R = depth_image_array[:, :, 2]

        # Apply the formula to normalize the depth values
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        return normalized

    @property
    def depth_image_meters(self):
        return self.depth_image_normalized * 1000

    @property
    def depth_camera_fov(self):
        return self._depth_camera_bp.get_attribute("fov").as_float()

    @property
    def depth_camera_image_width(self):
        return self._depth_camera_bp.get_attribute("image_size_x").as_int()

    @property
    def depth_camera_image_height(self):
        return self._depth_camera_bp.get_attribute("image_size_y").as_int()

    @property
    def radar_camera(self):
        return self._radar_camera

    @property
    def radar_image_raw(self):
        return self._radar_image_raw

    @property
    def radar_image_array(self):
        radar_image = self.radar_image_raw
        radar_image = np.reshape(
            np.copy(radar_image.raw_data), (radar_image.height, radar_image.width, 4)
        )
        return radar_image

    @property
    def radar_camera_fov(self):
        return self._radar_camera_bp.get_attribute("fov").as_float()

    @property
    def radar_camera_image_width(self):
        return self._radar_camera_bp.get_attribute("image_size_x").as_int()

    @property
    def radar_camera_image_height(self):
        return self._radar_camera_bp.get_attribute("image_size_y").as_int()
