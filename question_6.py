import pyglet
from pyglet import app, gl, graphics
from pyglet.window import Window, key
import numpy as np
from sys import exit
d = 12
wx, wy = 1.5 * d, 1.1 * d # Параметры области визуализации
width, height = int(20 * wx), int(20 * wy) # Размеры окна вывода
#
mtClr0 = [1, 1, 0, 0] # Цвет материала
light_position0 = [0, 40, 40, 0] # Позиция источника света
lghtClr0 = [0.75, 0, 0, 0] # Цвет источника света
mtClr = (gl.GLfloat * 4)()
light_position = (gl.GLfloat * 4)()
lghtClr = (gl.GLfloat * 4)()
for k in range(4): mtClr[k] = mtClr0[k]
for k in range(4): light_position[k] = light_position0[k]
for k in range(4): lghtClr[k] = lghtClr0[k]
#
window = Window(visible = True, width = width, height = height,
                resizable = True, caption = 'Материал')
gl.glClearColor(0.1, 0.1, 0.1, 1.0)
gl.glClear(gl.GL_COLOR_BUFFER_BIT)
gl.glEnable(gl.GL_LIGHTING) # Активизируем использование материалов
@window.event
def on_draw():
    window.clear()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-wx, wx, -wy, wy, -1, 1)
    gl.glShadeModel(gl.GL_SMOOTH) # GL_FLAT - без интерполяции цветов
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, mtClr)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, lghtClr)
    gl.glEnable(gl.GL_LIGHT0) # Включаем в уравнение освещенности источник GL_LIGHT0
    gl.glColor3f(1, 0, 0)
    gl.glBegin(gl.GL_QUADS)
    gl.glNormal3f(0, 0, 1)
    gl.glVertex3f(-d, -d, 0)
    gl.glNormal3f(0, 0, 1)
    gl.glVertex3f(d, -d, 0)
    gl.glNormal3f(0, 0, 1)
    gl.glVertex3f(d, d, 0)
    gl.glNormal3f(0, 0, -1)
    gl.glVertex3f(-d, d, 0)
    gl.glEnd()
@window.event
def on_key_press(symbol, modifiers):
    if symbol == key._1:
        gl.glDisable(gl.GL_LIGHTING)
    elif symbol == key._2:
        gl.glEnable(gl.GL_LIGHTING)
app.run()