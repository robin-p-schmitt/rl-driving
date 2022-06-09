import numpy as np
from globals_ import *


def get_angle(vec: vec2):
  return np.degrees(np.arctan2(vec.y, vec.x))


def line_line_collision(x1, y1, x2, y2, x3, y3, x4, y4):

  u_a_nom = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3))
  u_b_nom = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3))
  denom = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

  if denom < 0:
    denom = -denom
    u_a_nom = -u_a_nom
    u_b_nom = -u_b_nom

  intersection = None
  if denom != 0:
    intersection = vec2(x=x1 + (u_a_nom / denom * (x2 - x1)), y=y1 + (u_a_nom / denom * (y2 - y1)))

  if 0.0 < u_a_nom < denom and 0.0 < u_b_nom < denom:
    return True, intersection

  return False, None
