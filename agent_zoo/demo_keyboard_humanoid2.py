import gym, roboschool, sys, os
import numpy as np
import pyglet, pyglet.window as pw, pyglet.window.key as pwk
from pyglet import gl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

#
# This opens a third-party window (not test window), shows rendered chase camera, allows to control humanoid
# using keyboard (in a different way)
#

class PygletInteractiveWindow(pw.Window):
    def __init__(self, env):
        pw.Window.__init__(self, width=600, height=400, vsync=False, resizable=True)
        self.theta = 0
        self.still_open = True

        @self.event
        def on_close():
            self.still_open = False

        @self.event
        def on_resize(width, height):
            self.win_w = width
            self.win_h = height

        self.keys = {}
        self.human_pause = False
        self.human_done = False

    def imshow(self, arr):
        H, W, C = arr.shape
        assert C==3
        image = pyglet.image.ImageData(W, H, 'RGB', arr.tobytes(), pitch=W*-3)
        self.clear()
        self.switch_to()
        self.dispatch_events()
        texture = image.get_texture()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture.width  = W
        texture.height = H
        texture.blit(0, 0, width=self.win_w, height=self.win_h)
        self.flip()

    def on_key_press(self, key, modifiers):
        self.keys[key] = +1
        if key==pwk.ESCAPE: self.still_open = False

    def on_key_release(self, key, modifiers):
        self.keys[key] = 0

    def each_frame(self):
        self.theta += 0.05 * (self.keys.get(pwk.LEFT, 0) - self.keys.get(pwk.RIGHT, 0))


def random_action(action_space):
    a = action_space.sample()
    if isinstance(a, np.ndarray):
        a = a.astype(np.float32)
    return a


def demo_run():
    env = gym.make("RoboschoolHumanoidFlagrun-v1")
    #env = gym.make("RoboschoolAnt-v1")

    config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count = { "GPU": 0 } )
    sess = tf.InteractiveSession(config=config)

    from RoboschoolHumanoidFlagrunHarder_v1_2017jul import ZooPolicyTensorflow
    pi = ZooPolicyTensorflow("humanoid1", env.observation_space, env.action_space)

    control_me = PygletInteractiveWindow(env.unwrapped)

    env.reset()
    eu = env.unwrapped

    obs = env.reset()

    mycam = env.unwrapped.scene.cpp_world.new_camera_free_float(600, 400, "my_camera")

    while 1:
        a = pi.act(obs, env)
        #a = random_action(env.action_space)

        x, y, z = eu.body_xyz
        eu.walk_target_x = x + 2.0*np.cos(control_me.theta)   # 1.0 or less will trigger flag reposition by env itself
        eu.walk_target_y = y + 2.0*np.sin(control_me.theta)
        eu.flag = eu.scene.cpp_world.debug_sphere(eu.walk_target_x, eu.walk_target_y, 0.2, 0.2, 0xFF8080)
        eu.flag_timeout = 100500

        obs, r, done, _ = env.step(a)
        img = env.render("rgb_array")

        cam_x = x + 0.2*np.cos(control_me.theta)
        cam_y = y + 0.2*np.sin(control_me.theta)
        mycam.move_and_look_at(cam_x, cam_y, z, eu.walk_target_x, eu.walk_target_y, z * 0.9)
        rgb, depth, depth_mask, labeling, labeling_mask = mycam.render(False, False, False)
        rgb_array = np.fromstring(rgb, dtype=np.uint8).reshape((400, 600, 3))
        control_me.imshow(rgb_array)

        #control_me.imshow(img)
        control_me.each_frame()
        if control_me.still_open==False: break

if __name__=="__main__":
    demo_run()
