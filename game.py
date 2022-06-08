import pyglet
from pyglet.gl import *
import pygame
import numpy as np
import math_util

import globals
import os

vec2 = pygame.math.Vector2


class Game:
  def __init__(self):
    track_img = pyglet.image.load(os.path.join(globals.image_path, "track_straight.png"))
    self.track_sprite = pyglet.sprite.Sprite(track_img, x=0, y=0)

    self.car = Car()

  def render(self):
    glPushMatrix()

    self.track_sprite.draw()
    self.car.show()

    glPopMatrix()

  def update(self, dt):
    self.car.update(dt)


class Car:
  def __init__(self):
    car_img = pyglet.image.load(os.path.join(globals.image_path, "car.png"))
    self.car_sprite = pyglet.sprite.Sprite(car_img, x=0, y=0)

    self.init_x = (globals.display_width / 2)
    self.init_y = 200
    self.position = vec2(x=self.init_x, y=self.init_y)

    self.is_accelerating = False
    self.is_reversing = False
    self.is_turning_left = False
    self.is_turning_right = False

    self.turning_rate = 1.
    self.max_velocitiy = 7
    self.max_rev_velocitiy = -4

    self.acceleration = 4.
    self.velocity = 0
    self.direction = vec2(x=0, y=1)
    self.friction = 2
    self.angle = 0
    self.steering_angle = 0

    self.width = self.car_sprite.width
    self.height = self.car_sprite.height

    print(self.width)
    print(self.height)

    self.car_sprite.update(x=self.position.x, y=self.position.y, rotation=90)

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
