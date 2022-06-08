import pyglet
from pyglet.gl import *

import globals
import os


class Game:
  def __init__(self):
    track_img = pyglet.image.load(os.path.join(globals.image_path, "track_straight.png"))
    self.track_sprite = pyglet.sprite.Sprite(track_img, x=0, y=0)

  def render(self):
    glPushMatrix()

    self.track_sprite.draw()

    glPopMatrix()

