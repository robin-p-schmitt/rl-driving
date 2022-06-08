import pyglet
from game import Game
from globals import display_width, display_height

frame_rate = 30.


class MyWindow(pyglet.window.Window):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.set_minimum_size(400, 300)

    background_color = [0, 0, 0, 255]
    background_color = [i/255 for i in background_color]
    pyglet.gl.glClearColor(*background_color)

    self.game = Game()

  def on_draw(self):
    self.game.render()

  def update(self, dt):
    pass


if __name__ == "__main__":
  window = MyWindow(width=display_width, height=display_height, caption="RL Driving", resizable=False)
  pyglet.clock.schedule_interval(window.update, 1/frame_rate)
  pyglet.app.run()
