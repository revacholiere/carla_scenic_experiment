import carla
from praeception_models import get_train_transform
import numpy as np
import praeception_models as pm
import simulator as sim
from PIL import Image
import os
from config import cfg
from object_info import ObjectInfo


base = 10


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def save_predicted_captions(captions, i, seed):
    path =f"/home/ekin/Scenic/examples/ekin/experiment/captions/scenario{seed}_frame_{i}.txt"
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        for caption in captions:
            file.write(caption + "\n")
    return path





def collect(vehicle, vocabulary, world, vision_model, num_categories, camera_manager, behavior_agent, rulebook, device, i, seed):
    vision_model.to(device)
    camera_manager.step()
    ego_image_transform = get_train_transform()
    ego_image = camera_manager.ego_image_array

    
    transformed_ego_image = ego_image_transform(ego_image)
    device_ego_image = transformed_ego_image.to(device)
    predicted_detections, predicted_captions = pm.predict_detections(
            vision_model, device, device_ego_image.unsqueeze(0), vocabulary
        )

    if i % base == 0:
        rgba_image = ego_image[..., [2, 1, 0]]
        img_save = Image.fromarray(rgba_image.astype("uint8"))
        img_save.save(f'images/scenario{seed}_frame_{i}.png')
        save_predicted_captions(predicted_captions, i, seed)
    bboxes_xyxy, categories, confidences = predicted_detections
    #total_detections += len(categories)
    bboxes_xyxy = pm.resize_bboxes_xyxy(
        bboxes_xyxy,
        (cfg.common.img_width_resize, cfg.common.img_height_resize),
        (
            camera_manager.ego_camera_image_width,
            camera_manager.ego_camera_image_height,
        ),
    )
    locations, bboxes_3d = sim.compute_locations_and_bboxes_3d(
            camera_manager.depth_image_meters,
            bboxes_xyxy,
            camera_manager.depth_camera,
        )


    distances = [vehicle.get_location().distance(l) for l in locations]
    speeds = []
    for i in range(len(bboxes_3d)):
        if categories[i] in vocabulary.pedestrian_tokens:
            speeds.append(0)
        else:
            speeds.append(10)


    predicted_objects = [
            ObjectInfo(
                bbox_xyxy, category, confidence, location, distance, speed, bbox_3d
            )
            for bbox_xyxy, category, confidence, location, distance, speed, bbox_3d in zip(
                bboxes_xyxy,
                categories,
                confidences,
                locations,
                distances,
                speeds,
                bboxes_3d,
            )
        ]
    ground_truth_objects = sim.get_ground_truth_objects(
            vehicle,
            camera_manager.ego_camera,
        )

    for obj in predicted_objects:
            if obj.category in vocabulary.pedestrian_tokens:
                obj.category = 0
            elif obj.category in vocabulary.vehicle_tokens:
                obj.category = 1
    behavior_agent.update_object_information(predicted_objects)
    control = behavior_agent.run_step()
    #control_history.append(control)
    vehicle.apply_control(control)
    rulebook.step()
