import numpy as np
import ctypes
from pyglet.gl import *
from pyglet import app
from pyglet.window import Window, key


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


w = 600
window = Window(visible=True, width=w, height=w, resizable=True, caption='Lab2')

settings = {
    'axis_default': True,
    'front_default': True,
    'back_default': True,
    'point_default': True,
    'shade_default': True,
    'rotate_default': True,
    }

coords = {
    'triangle': [Point(2 * w // 3, w // 2),
                 Point(2 * w // 3, 5 * w // 7),
                 Point(2 * w // 5, 3 * w // 5)],

    'rectangle': [Point(w // 4, 3 * w // 4),
                  Point(7 * w // 8, 3 * w // 4),
                  Point(7 * w // 8, w // 4),
                  Point(w // 4, w // 4)],

    'point': [Point(w // 30, w // 30)],
    'ellipse': [Point(3 * w // 5, 4 * w // 7), Point(w // 4,  w // 6)],  # center, radius
    'axisX': [Point(w // 30, 0), Point(w // 30, w)],
    'axisY': [Point(0, w // 30), Point(w, w // 30)],

}

colors = {
    'triangle': [[0, 0, 1], [0.7, 0.2, 1], [1, 0.2, 0.2]],
    'rectangle': [[0, 0, 0.5], [0.7, 0.7, 0], [0.2, 0.9, 0.1], [1, 0.7, 1]],
    'point': [[1, 0, 1]],
    'ellipse': [[1, 0, 0]],
    'axis': [[0, 0, 0]],
}


def draw_axis():
    if settings['axis_default']:
        glEnable(GL_LINE_STIPPLE)
    glLineStipple(1, 0x1C47)
    glLineWidth(2)

    glBegin(GL_LINES)
    glColor3f(*colors['axis'][0])

    glVertex2f(coords['axisX'][0].x, coords['axisX'][0].y)
    glVertex2f(coords['axisX'][1].x, coords['axisX'][1].y)
    glVertex2f(coords['axisY'][0].x, coords['axisY'][0].y)
    glVertex2f(coords['axisY'][1].x, coords['axisY'][1].y)

    glEnd()
    glDisable(GL_LINE_STIPPLE)


def draw_ellipse():
    num_segments = 1000
    theta = 2 * np.pi / num_segments
    x = 1
    y = 0
    center_x = coords['ellipse'][0].x
    center_y = coords['ellipse'][0].y
    radius_x = coords['ellipse'][1].x
    radius_y = coords['ellipse'][1].y

    glBegin(GL_LINE_LOOP)
    glColor3f(*colors['ellipse'][0])
    for ii in range(num_segments):
        glVertex2f(x * radius_x + center_x, y * radius_y + center_y)
        t = x
        x = np.cos(theta) * x - np.sin(theta) * y
        y = np.sin(theta) * t + np.cos(theta) * y
    glEnd()


def draw_point():
    if settings['point_default']:
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    glPointSize(15)
    glBegin(GL_POINTS)
    glColor3f(*colors['point'][0])
    glVertex2f(coords['point'][0].x, coords['point'][0].y)
    glEnd()
    glDisable(GL_POINT_SMOOTH)


def draw_rect():
    glPolygonMode(GL_BACK, GL_FILL if settings['back_default'] else GL_LINE)
    glPointSize(6)
    glBegin(GL_QUADS)
    for i in range(4):
        glColor3f(*colors['rectangle'][i])
        glVertex2f(coords['rectangle'][i].x, coords['rectangle'][i].y)
    glEnd()


def draw_triangle():
    glPolygonMode(GL_FRONT, GL_FILL if settings['front_default'] else GL_POINT)
    glPointSize(5)
    glBegin(GL_TRIANGLES)
    for i in range(3):
        glColor3f(*colors['triangle'][i])
        glVertex2f(coords['triangle'][i].x, coords['triangle'][i].y)
    glEnd()

    # vertices = np.array([coords['triangle'][0].x, coords['triangle'][0].y, 0,
    #                      coords['triangle'][1].x, coords['triangle'][1].y, 0,
    #                      coords['triangle'][2].x, coords['triangle'][2].y, 0])
    # color_tr = np.array(colors['triangle']).flatten()
    #
    # vertexPositionsGl = (GLfloat * len(vertices))(*vertices)
    # colorsGl = (GLfloat * len(color_tr))(*color_tr)
    #
    # positionBufferObject = GLuint()
    # glGenBuffers(1, positionBufferObject)
    # glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
    # glBufferData(GL_ARRAY_BUFFER, len(vertexPositionsGl) * 4, vertexPositionsGl, GL_STATIC_DRAW)
    #
    # glEnableVertexAttribArray(0)
    # glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0)
    # glColorPointer(3, GL_FLOAT, 0, colorsGl)
    #
    # glDrawArrays(GL_TRIANGLES, 0, 3)
    # glDisableVertexAttribArray(0)


@window.event
def on_draw():
    glClearColor(1, 1, 1, 0)
    window.clear()
    glShadeModel(GL_SMOOTH if settings['shade_default'] else GL_FLAT)

    glPushMatrix()
    glTranslatef(0, w * int(not settings['rotate_default']), 0.0)
    glRotatef(180 * int(not settings['rotate_default']), 1.0, 0.0, 0.0)

    draw_axis()
    draw_rect()
    draw_ellipse()
    draw_point()
    draw_triangle()

    glPopMatrix()


@window.event
def on_key_press(symbol, modifiers):
    global settings
    if symbol == key._1:
        settings['axis_default'] = not settings['axis_default']
    elif symbol == key._2:
        settings['shade_default'] = not settings['shade_default']
    elif symbol == key._3:
        settings['front_default'] = not settings['front_default']
    elif symbol == key._4:
        settings['back_default'] = not settings['back_default']
    elif symbol == key._5:
        settings['point_default'] = not settings['point_default']
    elif symbol == key._6:
        settings['rotate_default'] = not settings['rotate_default']
    elif symbol == key._7:
        for item in settings:
            settings[item] = True


app.run()
