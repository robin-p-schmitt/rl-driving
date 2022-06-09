import pyglet
from pyglet.gl import *
import pygame
import numpy as np
import math_util

import globals_
import os

vec2 = pygame.math.Vector2


class Game:
  def __init__(self):
    track_img = pyglet.image.load(os.path.join(globals_.image_path, "track_straight.png"))
    self.track_sprite = pyglet.sprite.Sprite(track_img, x=0, y=0)

    self.car = Car()
    self.walls = self.add_walls()

  def add_walls(self):
    return [
      Wall(765, 0, 765, globals_.display_height, self.car), Wall(1035, 0, 1035, globals_.display_height, self.car)
    ]

  def render(self):
    glPushMatrix()

    self.track_sprite.draw()
    self.car.show()

    glPopMatrix()

  def update(self, dt):
    self.car.update(dt)
    for wall in self.walls:
      wall.update()


class Car:
  def __init__(self):
    car_img = pyglet.image.load(os.path.join(globals_.image_path, "car.png"))
    self.car_sprite = pyglet.sprite.Sprite(car_img, x=0, y=0)

    self.width = self.car_sprite.width
    self.height = self.car_sprite.height

    self.init_x = (globals_.display_width / 2)
    self.init_y = 200

    self.turning_rate = 1.
    self.max_velocitiy = 7
    self.max_rev_velocitiy = -4

    self.is_accelerating = False
    self.is_reversing = False
    self.is_turning_left = False
    self.is_turning_right = False
    self.is_dead = False

    self.position = vec2(x=self.init_x, y=self.init_y)
    self.acceleration = 4.
    self.velocity = 0
    self.direction = vec2(x=0, y=1)
    self.friction = 2
    self.angle = 0
    self.steering_angle = 0

    self.car_sprite.update(x=self.position.x, y=self.position.y, rotation=90)

  def reset(self):
    self.is_accelerating = False
    self.is_reversing = False
    self.is_turning_left = False
    self.is_turning_right = False
    self.is_dead = False

    self.position = vec2(x=self.init_x, y=self.init_y)
    self.acceleration = 4.
    self.velocity = 0
    self.direction = vec2(x=0, y=1)
    self.friction = 2
    self.angle = 0
    self.steering_angle = 0

    self.car_sprite.update(x=self.position.x, y=self.position.y, rotation=90)

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

  def update(self, dt):
    self.update_controls(dt)
    self.move()

  def update_controls(self, dt):
    multiplier = self.velocity / 4

    if self.is_accelerating:
      self.velocity += self.acceleration * dt
      self.velocity = min(self.velocity, self.max_velocitiy)
    elif self.is_reversing:
      self.velocity -= self.acceleration * dt
      self.velocity = max(self.velocity, self.max_rev_velocitiy)
    else:
      if self.velocity > 0:
        self.velocity -= self.friction * dt
      elif self.velocity < 0:
        self.velocity += self.friction * dt
      else:
        self.velocity = 0

    if self.is_turning_right:
      self.direction = self.direction.rotate(-self.turning_rate * multiplier)
    elif self.is_turning_left:
      self.direction = self.direction.rotate(self.turning_rate * multiplier)

  def move(self):
    self.position += self.direction * self.velocity


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
      coll = math_util.line_line_collision(
        self.x1, self.y1, self.x2, self.y2,
        corner1.x, corner1.y, corner2.x, corner2.y)
      if coll:
        self.car.is_dead = True
        return
