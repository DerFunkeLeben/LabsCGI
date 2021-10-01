from pyglet.gl import *
import pyglet
from pyglet import app, graphics, clock
from pyglet.window import Window, key
import numpy as np
d, d1, d2 = 10, 10, 10
wx, wy = 1.5 * d2, 1.5 * d2
n_rot = 0
width, height = int(30 * wx), int(30 * wy)
window = Window(visible=True, width=width, height=height, resizable=True)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)


def texInit():
    iWidth = iHeight = 64
    n = 3 * iWidth * iHeight
    img = np.zeros((iWidth, iHeight, 3), dtype='uint8')
    for i in range(iHeight):
        for j in range(iWidth):
            img[i, j, :] = (i & 8 ^ j & 8) * 255
    img = img.reshape(n)
    img = (GLubyte * n)(*img)
    p, r = GL_TEXTURE_2D, GL_RGB
    glTexParameterf(p, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(p, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(p, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(p, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glTexImage2D(p, 0, r, iWidth, iHeight, 0, r, GL_UNSIGNED_BYTE, img)
    glEnable(p)


texInit()

w4 = 5
zv = -5
v0, v1, v2, v3 = (-w4, w4, zv), (-w4, w4, 0), (0, w4, 0), (0, w4, zv)

q0, q1, q2, q3 = (0, w4, zv), (0, w4, 0), (w4, w4, 0), (w4, w4, zv)


def upd(a):
    global n_rot, translate_x, translate_y

    n_rot = (n_rot + 1) % 360


clock.schedule_interval(upd, 0.01)


@window.event
def on_draw():
    window.clear()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-wx, wx, -wy, wy, -20, 20)
    glRotatef(90, 1, 0, 0)
    glRotatef(n_rot, 0, 0, 1)
    graphics.draw(4, GL_QUADS, ('v3f', (v0 + v1 + v2 + v3)),
                  ('t2f', (0, 1, 0, 0, 1, 0, 1, 1)))

    gl.glColor3f(0, 1, 0)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex3f(0, w4, zv)
    gl.glVertex3f(0, w4, 0)
    gl.glVertex3f(w4, w4, 0)
    gl.glVertex3f(w4, w4, zv)
    gl.glEnd()


app.run()

clock.unschedule(upd)
