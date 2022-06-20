import pyglet
from pyglet.window import key

from game import Game
from globals_ import display_width, display_height

from qlearning import QLearning

frame_rate = 30.


class MyWindow(pyglet.window.Window):
  def __init__(self, train, load_after_n_updates, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.set_minimum_size(400, 300)

    background_color = [0, 0, 0, 255]
    background_color = [i/255 for i in background_color]
    pyglet.gl.glClearColor(*background_color)

    self.game = Game(track_idx=1)
    self.car = self.game.car
    self.qlearning = QLearning(self.game, load_after_n_updates=load_after_n_updates)

    self.train = train

  def on_key_press(self, symbol, modifiers):
    pass
    # if symbol == key.UP:
    #   self.car.is_accelerating = True
    # if symbol == key.DOWN:
    #   self.car.is_reversing = True
    # if symbol == key.RIGHT:
    #   self.car.is_turning_right = True
    # if symbol == key.LEFT:
    #   self.car.is_turning_left = True

  def on_key_release(self, symbol, modifiers):
    pass
    # if symbol == key.UP:
    #   self.car.is_accelerating = False
    # if symbol == key.DOWN:
    #   self.car.is_reversing = False
    # if symbol == key.RIGHT:
    #   self.car.is_turning_right = False
    # if symbol == key.LEFT:
    #   self.car.is_turning_left = False

  def on_draw(self):
    self.game.render()

  def update(self, dt):
    if self.train:
      self.qlearning.train(dt)
    else:
      self.qlearning.test(dt)
    # self.game.update(dt)
    if self.game.is_episode_finished():
      self.game.reset()


if __name__ == "__main__":
  window = MyWindow(
    train=False, load_after_n_updates=45000,
    width=display_width, height=display_height, caption="RL Driving", resizable=False)
  pyglet.clock.schedule_interval(window.update, 1/frame_rate)
  pyglet.app.run()
