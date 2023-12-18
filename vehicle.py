import cv2
import numpy as np


class Vehicle:

    def __init__(self, x_min, y_min, x_max, y_max):

        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_side = self.x_max - self.x_min
        self.y_side = self.y_max - self.y_min

    def intersect_with(self, rect):
        
        if not isinstance(rect, Vehicle):
            raise ValueError('Cannot compute intersection if "rect" is not a Vehicle')

        dx = min(self.x_max, rect.x_max) - max(self.x_min, rect.x_min)
        dy = min(self.y_max, rect.y_max) - max(self.y_min, rect.y_min)

        if dx >= 0 and dy >= 0:
            intersection = dx * dy
        else:
            intersection = 0.

        return intersection

    def resize_sides(self, ratio, bounds=None):
      
        off_x = abs(ratio * self.x_side - self.x_side) / 2
        off_y = abs(ratio * self.y_side - self.y_side) / 2

        # offset changes sign according if the resize is either positive or negative
        sign = np.sign(ratio - 1.)
        off_x = np.int32(off_x * sign)
        off_y = np.int32(off_y * sign)

        # update top-left and bottom-right coords
        new_x_min, new_y_min = self.x_min - off_x, self.y_min - off_y
        new_x_max, new_y_max = self.x_max + off_x, self.y_max + off_y

        if bounds:
            b_x_min, b_y_min, b_x_max, b_y_max = bounds
            new_x_min = max(new_x_min, b_x_min)
            new_y_min = max(new_y_min, b_y_min)
            new_x_max = min(new_x_max, b_x_max)
            new_y_max = min(new_y_max, b_y_max)

        return Vehicle(new_x_min, new_y_min, new_x_max, new_y_max)

    def draw(self, frame, color=255, thickness=1):
  
        cv2.rectangle(frame, (self.x_min, self.y_min), (self.x_max, self.y_max), color, thickness)

    def get_binary_mask(self, mask_shape):

        if mask_shape[0] < self.y_max or mask_shape[1] < self.x_max:
            raise ValueError('Mask shape is smaller than Vehicle size')
        mask = np.zeros(shape=mask_shape, dtype=np.uint8)
        mask = cv2.rectangle(mask, self.tl_corner, self.br_corner, color=255, thickness=cv2.FILLED)
        return mask

    def contains(self, x, y):

        if self.x_min < x < self.x_max and self.y_min < y < self.y_max:
            return True
        else:
            return False

    @property
    def center(self):
        center_x = self.x_min + self.x_side // 2
        center_y = self.y_min + self.y_side // 2
        return tuple(map(np.int32, (center_x, center_y)))

    @property
    def tl_corner(self):
     
        return tuple(map(np.int32, (self.x_min, self.y_min)))

    @property
    def br_corner(self):
     
        return tuple(map(np.int32, (self.x_max, self.y_max)))

    @property
    def coords(self):
      
        return tuple(map(np.int32, (self.x_min, self.y_min, self.x_max, self.y_max)))


    @property
    def area(self):
      
        return np.float32(self.x_side * self.y_side)