import logging
import math
import queue
import random
import sys
import time

import carla
import cv2
import numpy as np
from agents.tools.misc import get_speed
from shapely.geometry import Polygon

from config import cfg


logger = logging.getLogger(__name__)


class Rule:
    def __init__(
        self,
        name,
        type,
        value,
        weight,
        level,
        actor=None,
        min=0,
        max=math.inf,
        angle=360,
    ):
        self.name = name
        self.type = type
        self.value = value
        self.weight = weight
        self.level = level
        self.actor = actor
        self.min = min
        self.max = max
        self.angle = angle

    def __str__(self):
        return f"Rule({self.name}, {self.type}, {self.value}, {self.weight}, {self.level}, {self.actor}, {self.min}, {self.max}, {self.angle})"


class RuleBook(object):
    def __init__(self, vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._trajectory_log = []
        self._velocity_log = []
        self._transform_log = []

        self._rules = []
        self._scores = {}
        self._rules = RuleBook._build_rules()
        for rule in self._rules:
            self._scores[rule.name] = []
            logger.debug(f"Rule {rule} added to rulebook")
        self._scores["agg"] = []

    def _distance(self, actor):
        return actor.get_location().distance(self._vehicle.get_transform().location)

    def _within_angle(self, actor, angle):
        forward = self._vehicle.get_transform().get_forward_vector()
        logger.debug(f"forward: {forward}")
        actor_vector = actor.get_location() - self._vehicle.get_transform().location
        actor_vector = actor_vector / actor_vector.length()
        logger.debug(f"actor_vector: {actor_vector}")
        dot = forward.dot(actor_vector)
        cos = math.cos(angle * math.pi / 180)
        logger.debug(f"dot: {dot}, cos: {cos}")
        return dot > cos

    def _normalize(self, score):
        return 1 - math.exp(-score)

    def _detect_collision(self, target):
        target_bb = target.bounding_box
        target_vertices = target_bb.get_world_vertices(target.get_transform())
        target_list = [[v.x, v.y] for v in target_vertices]
        target_polygon = Polygon(target_list)

        self_vertices = self._trajectory_log[-1].get_world_vertices(
            self._transform_log[-1]
        )
        self_list = [[v.x, v.y] for v in self_vertices]

        self_vertices = self._trajectory_log[-2].get_world_vertices(
            self._transform_log[-2]
        )
        for v in self_vertices:
            self_list.append([v.x, v.y])

        self_polygon = Polygon(self_list)

        return self_polygon.intersects(target_polygon)

    def step(self):
        self._trajectory_log.append(self._vehicle.bounding_box)
        self._velocity_log.append(get_speed(self._vehicle))
        self._transform_log.append(self._vehicle.get_transform())
        agg_score = 0
        for rule in self._rules:
            if rule.type == "collision":
                score = self._normalize(self._collision_rule(rule))
            elif rule.type == "proximity":
                score = self._normalize(self._proximity_rule(rule))
            elif rule.type == "velocity":
                score = self._normalize(self._velocity_rule(rule))
            else:
                logger.warning(f"Unknown rule type: {rule.type}")
            self._scores[rule.name].append(score)
            agg_score += 2 ** (1 - rule.level) * score * rule.weight
        self._scores["agg"].append(agg_score)

    def _collision_rule(self, rule):
        score = 0
        if len(self._velocity_log) < 2:
            return score

        current_velocity = self._velocity_log[-1]
        actor_list = self._world.get_actors().filter(rule.actor)
        for actor in actor_list:
            if actor.id == self._vehicle.id:
                continue
            if self._detect_collision(actor):
                logger.debug(f"Collision detected with {actor.id}")
                score += score + rule.value * current_velocity
        return score

    def _proximity_rule(self, rule):
        score = 0
        if len(self._velocity_log) < 2:
            return score

        current_velocity = self._velocity_log[-1]
        previous_velocity = self._velocity_log[-2]
        acceleration = current_velocity - previous_velocity

        actor_list = self._world.get_actors().filter(rule.actor)
        for actor in actor_list:
            if actor.id == self._vehicle.id:
                continue
            if self._distance(actor) < rule.min or self._distance(actor) > rule.max:
                continue
            if self._within_angle(actor, rule.angle) and acceleration < 0:
                score += score + rule.value * current_velocity
        logger.debug(f"Proximity score[{rule.name}]: {score}")
        return score

    def _velocity_rule(self, rule):
        score = 0
        if len(self._velocity_log) < 2:
            return score

        current_velocity = self._velocity_log[-1]
        previous_velocity = self._velocity_log[-2]
        acceleration = current_velocity - previous_velocity

        if current_velocity < rule.min:
            score += score + rule.value * current_velocity
        elif current_velocity > rule.max:
            score += score + rule.value * current_velocity
        logger.debug(f"Velocity score[{rule.name}]: {score}")
        return score

    def get_violation_scores(self):
        return self._scores

    @classmethod
    def get_max_score(cls, force_rebuild=False):
        if force_rebuild or not hasattr(cls, "_rules"):
            rules = cls._build_rules()
        else:
            rules = cls._get_rules()
        max_score = 0
        for rule in rules:
            max_score += 2 ** (1 - rule.level) * rule.weight
        return max_score

    @classmethod
    def _build_rules(cls):
        rules = []
        for rule in cfg.common.rulebook:
            rules.append(
                Rule(
                    name=rule.name,
                    type=rule.type,
                    value=float(rule.value),
                    weight=float(rule.weight),
                    level=int(rule.level),
                    actor=rule.get("actor", None),
                    min=float(rule.get("min", 0)),
                    max=float(rule.get("max", math.inf)),
                    angle=float(rule.get("angle", 360)),
                )
            )
        RuleBook._cache_rules(rules)
        return rules

    @classmethod
    def _cache_rules(cls, rules):
        cls._rules = rules

    @classmethod
    def _get_rules(cls):
        return cls._rules
