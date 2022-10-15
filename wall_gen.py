import pygame
import pyglet
from pyglet.gl import *
from pyglet.window import key

import globals_
import os
import json

frame_rate = 30.
vec2 = pygame.math.Vector2

class MyWindow(pyglet.window.Window):
  def __init__(self, track_idx, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.set_minimum_size(400, 300)

    background_color = [0, 0, 0, 255]
    background_color = [i/255 for i in background_color]
    pyglet.gl.glClearColor(*background_color)

    self.track_idx = track_idx
    self.is_drawing = False
    self.start_position = vec2(0, 0)
    self.end_position = vec2(0, 0)

    self.current_wall = pyglet.shapes.Line(*self.start_position, *self.end_position)
    self.walls = self.load_walls_from_json()

    track_img = pyglet.image.load(os.path.join(globals_.image_path, "track%d.png" % self.track_idx))
    self.track_sprite = pyglet.sprite.Sprite(track_img, x=0, y=0)

  def on_mouse_press(self, x, y, button, modifiers):
    self.is_drawing = True if not self.is_drawing else False
    self.start_position = vec2(x=x, y=y)

  def on_mouse_release(self, x, y, button, modifiers):
    if not self.is_drawing:
      self.walls.append(pyglet.shapes.Line(*self.current_wall.position))

  def on_mouse_motion(self, x, y, dx, dy):
    if self.is_drawing:
      self.current_wall = pyglet.shapes.Line(*self.start_position, *vec2(x, y))

  def on_key_release(self, symbol, modifiers):
    if symbol == key.S:
      self.save_walls_to_json()
    if symbol == key.D:
      self.walls = []

  def on_draw(self):
    glPushMatrix()

    self.track_sprite.draw()
    self.current_wall.draw()
    for wall in self.walls:
      wall.draw()

    glPopMatrix()

  def update(self, dt):
    pass

  def save_walls_to_json(self):
    with open(os.path.join(globals_.image_path, "track%d_walls.json" % self.track_idx), "w+") as f:
      json.dump({"walls": [wall.position for wall in self.walls]}, f)

  def load_walls_from_json(self):
    file_path = os.path.join(globals_.image_path, "track%d_walls.json" % self.track_idx)
    if os.path.exists(file_path):
      with open(file_path) as f:
        data = json.load(f)
        walls = [pyglet.shapes.Line(*position) for position in data["walls"]]
    else:
      walls = []

    return walls


if __name__ == "__main__":
  window = MyWindow(
    track_idx=1, width=globals_.display_width, height=globals_.display_height, caption="MAP GENERATOR", resizable=False)
  pyglet.clock.schedule_interval(window.update, 1/frame_rate)
  pyglet.app.run()
