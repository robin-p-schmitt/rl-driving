import numpy as np
from globals_ import *


def get_angle(vec: vec2):
  return np.degrees(np.arctan2(vec.y, vec.x))