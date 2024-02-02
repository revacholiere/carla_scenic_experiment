import logging
import os
from string import Template

import carla
import numpy as np
import torch
from carla import ColorConverter as cc
from PIL import Image
from PIL import ImageFile

from object_info import ObjectInfo


ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import cfg



path_templates = {
    "vehicle_state": Template(
        "/results/${experiment_name}/vehicle/state/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "vehicle_control": Template(
        "/results/${experiment_name}/vehicle/control/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "score": Template(
        "/results/${experiment_name}/rewards/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "ego_image": Template(
        "home/ekin/results/${experiment_name}/images/ego/${sample_or_baseline}_${epoch}_${trajectory}_${step}.png"
    ),
    "birdseye_image": Template(
        "/results/${experiment_name}/images/birdseye/${sample_or_baseline}_${epoch}_${trajectory}_${step}.png"
    ),
    "depth_image_raw": Template(
        "/results/${experiment_name}/images/depth_raw/${sample_or_baseline}_${epoch}_${trajectory}_${step}.png"
    ),
    "depth_image_log": Template(
        "/results/${experiment_name}/images/depth_log/${sample_or_baseline}_${epoch}_${trajectory}_${step}.png"
    ),
    "predicted_objects": Template(
        "/results/${experiment_name}/predictions/objects/${sample_or_baseline}_${epoch}_${trajectory}_${step}.txt"
    ),
    "predicted_captions": Template(
        "/results/${experiment_name}/predictions/captions/${sample_or_baseline}_${epoch}_${trajectory}_${step}.txt"
    ),
    "ground_truth_objects": Template(
        "/results/${experiment_name}/ground_truth/objects/${sample_or_baseline}_${epoch}_${trajectory}_${step}.txt"
    ),
    "ego_video": Template(
        "/results/${experiment_name}/videos/ego/${sample_or_baseline}_${epoch}_${trajectory}.mp4"
    ),
    "birdseye_video": Template(
        "/results/${experiment_name}/videos/birdseye/${sample_or_baseline}_${epoch}_${trajectory}.mp4"
    ),
}


def get_indices_from_filename(filename):
    filename = filename.split("/")[-1]
    filename = filename.split(".")[0]
    filename = filename.split("_")
    epoch = int(filename[1])
    trajectory = int(filename[2])
    if len(filename) > 3:
        step = int(filename[3])
        return epoch, trajectory, step
    else:
        return (
            epoch,
            trajectory,
        )


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def make_permissive_dir(dir):
    if not os.path.exists(dir):
        old_umask = os.umask(0)
        os.makedirs(dir, mode=0o777, exist_ok=True)
        os.umask(old_umask)


def fill_template(template, **kwargs):
    return template.substitute(**kwargs)


def sample_or_baseline(is_baseline):
    if is_baseline:
        return "baseline"
    else:
        return "sample"


def save_objects(objects, path):
    dir = os.path.dirname(path)
    make_dir(dir)

    with open(path, "w") as file:
        for object in objects:
            bbox_xyxy = object.bbox_xyxy
            category = object.category
            confidence = object.confidence
            location = object.location
            distance = object.distance
            speed = object.speed
            bbox_3d = " ".join(
                map(
                    str,
                    [
                        object.bbox_3d.location.x,
                        object.bbox_3d.location.y,
                        object.bbox_3d.location.z,
                        object.bbox_3d.extent.x,
                        object.bbox_3d.extent.y,
                        object.bbox_3d.extent.z,
                        object.bbox_3d.rotation.pitch,
                        object.bbox_3d.rotation.yaw,
                        object.bbox_3d.rotation.roll,
                    ],
                )
            )
            file.write(
                f"{' '.join(map(str, [*bbox_xyxy, category, confidence, location.x, location.y, location.z, distance, speed, bbox_3d]))}\n"
            )
    return path


def load_objects(path):
    objects = []
    with open(path, "r") as file:
        for line in file.readlines():
            bbox_xyxy = [
                int(round(float(item))) for item in line.strip().split(" ")[:4]
            ]
            category = line.strip().split(" ")[4]
            score = float(line.strip().split(" ")[5])
            location = (
                float(line.strip().split(" ")[6]),
                float(line.strip().split(" ")[7]),
                float(line.strip().split(" ")[8]),
            )
            distance = float(line.strip().split(" ")[9])
            speed = float(line.strip().split(" ")[10])
            bbox_3d_list = line.strip().split(" ")[11:]
            bbox_3d_list = list(map(float, bbox_3d_list))
            if len(bbox_3d_list) == 9:
                bbox_3d = carla.BoundingBox(
                    carla.Location(*bbox_3d_list[:3]),
                    carla.Vector3D(*bbox_3d_list[3:6]),
                )
                bbox_3d.rotation = carla.Rotation(*bbox_3d_list[6:])
            else:
                bbox_3d = None
            object = ObjectInfo(
                bbox_xyxy, category, score, location, distance, speed, bbox_3d
            )
            objects.append(object)
    return objects


def save_image(image, path):
    dir = os.path.dirname(path)
    make_dir(dir)
    # Reorder channels from BGRA to RGBA
    rgba_image = image[..., [2, 1, 0]]
    img_save = Image.fromarray(rgba_image.astype("uint8"))
    img_save.save(path)
    return path


def load_image(path):
    image = Image.open(path)
    # Convert image to numpy array
    image_np = np.array(image)
    bgr_image = image_np[..., [2, 1, 0]]
    return bgr_image


def save_predicted_objects(objects, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["predicted_objects"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return save_objects(objects, path)


def load_predicted_objects(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["predicted_objects"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_objects(path)


def save_predicted_captions(captions, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["predicted_captions"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        for caption in captions:
            file.write(caption + "\n")
    return path


def load_predicted_captions(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["predicted_captions"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    captions = []
    with open(path, "r") as file:
        for line in file.readlines():
            caption = line.strip()
            captions.append(caption)
    return captions


def save_ground_truth_objects(objects, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["ground_truth_objects"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return save_objects(objects, path)


def load_ground_truth_objects(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["ground_truth_objects"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_objects(path)


def save_scores(scores, epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["score"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        for name, sublist in scores.items():
            line = name + " " + " ".join(map(str, sublist))
            file.write(line + "\n")
    return path


def load_scores(epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["score"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    scores = {}
    with open(path, "r") as file:
        for line in file.readlines():
            tokens = line.strip().split()
            name = tokens[0]
            string_numbers = tokens[1:]
            scores_list = list(map(float, string_numbers))
            scores[name] = scores_list
    return scores


def save_ego_image(image, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["ego_image"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return save_image(image, path)


def load_ego_image(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["ego_image"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_image(path)


def save_birdseye_image(image, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["birdseye_image"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return save_image(image, path)


def load_birdseye_image(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["birdseye_image"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_image(path)


def save_depth_image_raw(image, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["depth_image_raw"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    image.convert(cc.Raw)
    image.save_to_disk(path)
    return path


def load_depth_image_raw(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["depth_image_raw"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_image(path)


def save_depth_image_log(image, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["depth_image_log"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    image.convert(cc.LogarithmicDepth)
    image.save_to_disk(path)
    return path


def load_depth_image_log(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["depth_image_log"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_image(path)


def save_vehicle_states(vehicle_states, epoch, trajectory, is_baseline=False):
    speeds, locations = vehicle_states
    path = fill_template(
        path_templates["vehicle_state"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        for speed, location in zip(speeds, locations):
            file.write(f"{speed} {location.x} {location.y} {location.z}\n")
    return path


def load_vehicle_states(epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["vehicle_state"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    speeds = []
    locations = []
    with open(path, "r") as file:
        for line in file.readlines():
            speed, x, y, z = line.strip().split(" ")
            speeds.append(float(speed))
            locations.append((float(x), float(y), float(z)))
    return speeds, locations


def save_vehicle_controls(vehicle_controls, epoch, trajectory, is_baseline=False):
    throttle_list = []
    steer_list = []
    brake_list = []
    hand_brake_list = []
    reverse_list = []
    manual_gear_shift_list = []
    gear_list = []
    for control in vehicle_controls:
        throttle_list.append(control.throttle)
        steer_list.append(control.steer)
        brake_list.append(control.brake)
        hand_brake_list.append(control.hand_brake)
        reverse_list.append(control.reverse)
        manual_gear_shift_list.append(control.manual_gear_shift)
        gear_list.append(control.gear)

    path = fill_template(
        path_templates["vehicle_control"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        file.write(" ".join(map(str, throttle_list)) + "\n")
        file.write(" ".join(map(str, steer_list)) + "\n")
        file.write(" ".join(map(str, brake_list)) + "\n")
        file.write(" ".join(map(str, hand_brake_list)) + "\n")
        file.write(" ".join(map(str, reverse_list)) + "\n")
        file.write(" ".join(map(str, manual_gear_shift_list)) + "\n")
        file.write(" ".join(map(str, gear_list)) + "\n")
    return path


def load_vehicle_controls(epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["vehicle_control"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    with open(path, "r") as file:
        lines = file.readlines()
        throttle_list = list(map(float, lines[0].strip().split(" ")))
        steer_list = list(map(float, lines[1].strip().split(" ")))
        brake_list = list(map(float, lines[2].strip().split(" ")))
        hand_brake_list = list(map(bool, lines[3].strip().split(" ")))
        reverse_list = list(map(bool, lines[4].strip().split(" ")))
        manual_gear_shift_list = list(map(bool, lines[5].strip().split(" ")))
        gear_list = list(map(int, lines[6].strip().split(" ")))
    return (
        throttle_list,
        steer_list,
        brake_list,
        hand_brake_list,
        reverse_list,
        manual_gear_shift_list,
        gear_list,
    )


def save_checkpoint(model, epoch):
    checkpoint_path = os.path.join(
        "/results",
        cfg.experiment_name,
        "checkpoints",
        f"{epoch}.pth",
    )
    make_dir(os.path.dirname(checkpoint_path))
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def get_checkpoint_path(epoch):
    checkpoint_path = os.path.join(
        "/results",
        cfg.experiment_name,
        "checkpoints",
        f"{epoch}.pth",
    )
    return checkpoint_path
