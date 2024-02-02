param map = localPath('../../../assets/maps/CARLA/Town01.xodr')
param carla_map = 'Town01'
param time_step = 1.0/10
model scenic.simulators.carla.model
import carla
import myfile
from praeception_models import PretrainedImageCaptioningModel
from camera_manager import CameraManager
from config import cfg
from vocabulary import Vocabulary
import torch
import random

from agent_behaviors import BehaviorAgent
from object_info import ObjectInfo
from vocabulary import Vocabulary
from rulebook import RuleBook

torch.set_default_device('cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


EGO_MODEL = "vehicle.tesla.model3"
num_categories = len(Vocabulary())
vision_model = PretrainedImageCaptioningModel(
        cfg.common.hidden_size,
        cfg.common.num_heads,
        cfg.common.num_encoder_layers,
        cfg.common.num_decoder_layers,
        num_categories,
        cfg.common.image_size,
        cfg.common.patch_size,
        cfg.common.max_caption_length,)



behavior go():
    seed = random.randint(0,100)
    vocabulary = Vocabulary()
    i = 0
    carla_actor = self.carlaActor
    world = simulation().client.get_world()
    camera_manager = CameraManager(carla_actor)
    behavior_agent = BehaviorAgent(carla_actor)

    rulebook = RuleBook(carla_actor)
    vision_model.eval()

    camera_manager.start_collecting_images()
    world.tick()
    while True:    
        myfile.collect(carla_actor, vocabulary, world, vision_model, num_categories, camera_manager, behavior_agent, rulebook, device, i, seed)
        i += 1
        wait
        
    camera_manager.stop_collecting_images()
    camera_manager.destroy()
    

ego = new Car with behavior go(), with blueprint EGO_MODEL
front_car = new Car visible from ego, with behavior AutopilotBehavior()
side_front_car = new Car visible from ego, behind front_car, with behavior AutopilotBehavior()









