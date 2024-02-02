class ObjectInfo:
    def __init__(self, bbox, category, confidence, location, distance, speed, bbox_3d):
        self._bbox = bbox
        self._category = category
        self._confidence = confidence

        # estimated world location of the obj
        self._location = location
        self._distance = distance
        self._speed = speed
        self._bbox3D = bbox_3d

    @property
    def bbox_xyxy(self):
        return self._bbox

    @bbox_xyxy.setter
    def bbox_xyxy(self, bbox):
        self._bbox = bbox

    @bbox_xyxy.deleter
    def bbox_xyxy(self):
        del self._bbox

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, category):
        self._category = category

    @category.deleter
    def category(self):
        del self._category

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        self._confidence = confidence

    @confidence.deleter
    def confidence(self):
        del self._confidence

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        self._location = location

    @location.deleter
    def location(self):
        del self._location

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance

    @distance.deleter
    def distance(self):
        del self._distance

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = speed

    @speed.deleter
    def speed(self):
        del self._speed

    @property
    def bbox_3d(self):
        return self._bbox3D

    @bbox_3d.setter
    def bbox_3d(self, bbox3D):
        self._bbox3D = bbox3D

    @bbox_3d.deleter
    def bbox_3d(self):
        del self._bbox3D
