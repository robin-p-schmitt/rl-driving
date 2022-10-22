import pyglet
from pyglet.gl import *
import pygame
import numpy as np
import math_util

import globals_
import os
import json

from svgpathtools import svg2paths

vec2 = pygame.math.Vector2


class Game:
  def __init__(self, track_idx):
    track_img = pyglet.image.load(os.path.join(globals_.image_path, "track%d.png" % track_idx))
    self.track_sprite = pyglet.sprite.Sprite(track_img, x=0, y=0)

    self.car = Car()
    self.walls = self.load_walls(track_idx=track_idx)
    self.state = self.get_state()
    self.state_size = len(self.state)
    self.action_size = 9

  def load_walls(self, track_idx):
    walls_json_file = os.path.join(globals_.image_path, "track%d_walls.json" % track_idx)
    with open(walls_json_file) as f:
      data = json.load(f)
      walls = [
        Wall(*position, self.car) for position in data["walls"]
      ]

    return walls

  def show_lidar(self):
    for wall in self.walls:
      for lidar in self.car.lidar:
        lidar.draw()
        coll, inters = math_util.line_line_collision(
          wall.x1, wall.y1, wall.x2, wall.y2, lidar.x, lidar.y, lidar.x2, lidar.y2)
        if coll and inters is not None:
          pyglet.shapes.Circle(*inters, radius=10).draw()

  def get_state(self):
    # first part of the state is the distance from any lidar beam to a wall
    state = [-1 for i in range(len(self.car.lidar))]
    for wall in self.walls:
      for i, lidar in enumerate(self.car.lidar):
        coll, inters = math_util.line_line_collision(
          wall.x1, wall.y1, wall.x2, wall.y2, lidar.x, lidar.y, lidar.x2, lidar.y2)
        if coll and inters is not None:
          dist = np.sqrt(np.square(lidar.x-inters.x) + np.square(lidar.y-inters.y))
          norm_dist = dist / self.car.lidar_range
          if state[i] == -1 or state[i] > norm_dist:
            state[i] = norm_dist

    # add current velocity to state
    norm_forw_velocity = max(0.0, self.car.velocity / self.car.max_velocity)
    norm_backw_velocity = max(0.0, self.car.velocity / self.car.max_rev_velocity)
    state += [norm_forw_velocity, norm_backw_velocity]

    return state

  def is_episode_finished(self):
    return self.car.is_dead

  def reset(self):
    self.car.reset()

  def make_action(self, action):
    self.car.is_accelerating = False
    self.car.is_reversing = False
    self.car.is_turning_right = False
    self.car.is_turning_left = False

    if action == 0:
      self.car.is_accelerating = True
    elif action == 1:
      self.car.is_reversing = True
    elif action == 2:
      self.car.is_turning_right = True
    elif action == 3:
      self.car.is_turning_left = True
    elif action == 4:
      self.car.is_accelerating = True
      self.car.is_turning_right = True
    elif action == 5:
      self.car.is_accelerating = True
      self.car.is_turning_left = True
    elif action == 6:
      self.car.is_reversing = True
      self.car.is_turning_right = True
    elif action == 7:
      self.car.is_reversing = True
      self.car.is_turning_left = True
    elif action == 8:
      pass

    self.update()

    base_reward = 1
    if abs(self.car.velocity) > 2:
      reward = base_reward * abs(self.car.velocity)
    elif abs(self.car.velocity) > 1:
      reward = -1
    else:
      reward = -2

    # reward += self.car.life_time * 0.1

    self.car.reward += reward

    return self.car.reward

  def render(self):
    glPushMatrix()

    self.track_sprite.draw()
    self.car.show()
    # self.show_lidar()

    glPopMatrix()

  def update(self):
    self.car.update()
    for wall in self.walls:
      wall.update()
    self.state = self.get_state()


class Car:
  def __init__(self):
    car_img = pyglet.image.load(os.path.join(globals_.image_path, "car.png"))
    self.car_sprite = pyglet.sprite.Sprite(car_img, x=0, y=0)

    self.width = self.car_sprite.width
    self.height = self.car_sprite.height

    self.init_x = (globals_.display_width / 2)
    self.init_y = 200

    self.turning_rate = 0.4
    self.friction = 0.94
    self.acceleration = 6.
    self.max_velocity = 12
    self.max_rev_velocity = -8
    self.lidar_range = 500

    self.is_accelerating = False
    self.is_reversing = False
    self.is_turning_left = False
    self.is_turning_right = False
    self.is_dead = False

    self.position = vec2(x=self.init_x, y=self.init_y)
    self.velocity = 0
    self.direction = vec2(x=0, y=1)
    self.angle = 0
    self.life_time = 0
    self.reward = 0

    self.lidar = self.get_lidar()

    self.car_sprite.update(x=self.position.x, y=self.position.y, rotation=90)

  def reset(self):
    self.is_accelerating = False
    self.is_reversing = False
    self.is_turning_left = False
    self.is_turning_right = False
    self.is_dead = False

    self.position = vec2(x=self.init_x, y=self.init_y)
    self.velocity = 0
    self.direction = vec2(x=0, y=1)
    self.angle = 0
    self.life_time = 0
    # print("TOTAL REWARD: ", self.reward)
    self.reward = 0

    self.car_sprite.update(x=self.position.x, y=self.position.y, rotation=90)
    self.lidar = self.get_lidar()

  def get_lidar(self):
    lidar_pos = self.position + self.direction * self.height / 2
    return [
      pyglet.shapes.Line(*lidar_pos, *self.position + self.direction.rotate((360 / 6) * i) * self.lidar_range)
      for i in range(6)
    ]

  def get_bounding_box(self):
    """
    Get current bounding box of car
    :return: List[[Vector2, Vector2]]
    """
    side_vector = self.direction.rotate(-90)
    corners = [[], []]
    multipliers = [
      [(1, -1), (1, 1)],
      [(-1, -1), (-1, 1)]]

    for row in [0, 1]:
      corners[row] = [[], []]
      for col in [0, 1]:
        vertical_move = multipliers[row][col][0] * self.direction * self.height / 2
        horizontal_move = multipliers[row][col][1] * side_vector * self.width / 2
        corners[row][col] = self.position + vertical_move + horizontal_move

    return [
      [corners[0][0], corners[0][1]],  # top
      [corners[0][0], corners[1][0]],  # left
      [corners[1][0], corners[1][1]],  # bottom
      [corners[1][1], corners[0][1]]   # right
    ]

  def show(self):
    side_vector = self.direction.rotate(-90)  # points in the orthogonal direction of the driving direction
    # the difference between the anchor point and the center position of the car
    diff_to_center = self.direction * self.height / 2 + side_vector * self.width / 2
    # update car but correct position such that the rotation is done around the center of the car
    self.car_sprite.update(
      x=self.position.x - diff_to_center.x,
      y=self.position.y - diff_to_center.y,
      rotation=-math_util.get_angle(self.direction) + 90)
    self.car_sprite.draw()

  def update(self):
    self.life_time += 1
    self.update_controls()
    self.move()

  def update_controls(self):
    multiplier = self.velocity

    if self.is_accelerating:
      self.velocity += self.acceleration
      self.velocity = min(self.velocity, self.max_velocity)
    elif self.is_reversing:
      self.velocity -= self.acceleration
      self.velocity = max(self.velocity, self.max_rev_velocity)
    else:
      self.velocity *= self.friction
      if abs(self.velocity) < .5:
        self.velocity = 0

    if self.is_turning_right:
      self.direction = self.direction.rotate(-self.turning_rate * multiplier)
    elif self.is_turning_left:
      self.direction = self.direction.rotate(self.turning_rate * multiplier)

  def move(self):
    self.position += self.direction * self.velocity
    self.lidar = self.get_lidar()


class Wall:
  def __init__(self, x1, y1, x2, y2, car: Car):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2
    self.line = pyglet.shapes.Line(x1, y1, x2, y2, width=2, color=[0, 0, 0])
    self.car = car

  def show(self):
    self.line.draw()

  def update(self):
    for corner1, corner2 in self.car.get_bounding_box():
      coll, _ = math_util.line_line_collision(
        self.x1, self.y1, self.x2, self.y2,
        corner1.x, corner1.y, corner2.x, corner2.y)
      if coll:
        self.car.is_dead = True
        return
