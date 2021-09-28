from pyglet.gl import *
import pyglet
from pyglet import app, graphics, clock
from pyglet.window import Window
import numpy as np

w = 600
h = w/10
k = w//5*4
n_rot = 0
translate_x = translate_y = 1
translate_step = 5
p = gl.GL_TEXTURE_2D
textureIDs = (gl.GLuint * 6)() 

verts = ((h, -h, -h),
         (h, h, -h),
         (-h, h, -h),
         (-h, -h, -h),
         (h, -h, h),
         (h, h, h),
         (-h, -h, h),
         (-h, h, h))

faces = ((0, 1, 2, 3), 
         (3, 2, 6, 7),
          (6, 7, 5, 4),
         (4, 5, 1, 0),
         (1, 5, 7, 2),
         (4, 0, 3, 6))

clrs = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
        (0, 1, 1), (1, 1, 1), (1, 0, 0), (0, 1, 0),
        (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 1, 1))

quad = (0, 0, 0,
        k, 0, 0,
        k, k, 0,
        0, k, 0) 


tc = 1 
t_coords = ((0, 0), (0, tc), (tc, tc), (tc, 0)) 
def cube_draw():
    k = -1
    for face in faces:
        k += 1
        m = -1
        v4, c4, t4 = (), (), ()
        gl.glBindTexture(p, textureIDs[k])
        for v in face:
            m += 1
            c4 += clrs[k + m]
            t4 += t_coords[m]
            v4 += verts[v]

        graphics.draw(4, gl.GL_QUADS, ('v3f', v4), ('c3f', c4), ('t2f', t4))

def cube(a):
    global n_rot, translate_x, translate_y

    n_rot = (n_rot + 1) % 360

    if translate_x == 1 and translate_y <= k:
        translate_y += translate_step
    elif translate_y > k and translate_x <= k:
        translate_x += translate_step
    elif translate_x > k and translate_y >= 0:
        translate_y -= translate_step
    elif translate_y < 0 and translate_x >= 0:
        translate_x -= translate_step
    else:
        translate_x = translate_y = 1

    
window = Window(visible = True, width = w, height = w,
                resizable = True, caption = 'ЛР 3')
glClearColor(0.1, 0.1, 0.1, 1.0)
glClear(GL_COLOR_BUFFER_BIT)
glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LESS)
glLineWidth(2)
clock.schedule_interval(cube, 0.01)

@window.event
def on_draw():
    window.clear()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glRotatef(n_rot, 1, 0, 0)
    glRotatef(15, 0, 1, 0)
    glRotatef(15, 0, 0, 1)
    glOrtho(-w, w, -w, w, -w, w)
    glPolygonMode(GL_FRONT_AND_BACK , GL_LINE)
    graphics.draw(4, GL_QUADS, ('v3f', quad), ('c3f', 4*[0, 0, 1]))
    glPolygonMode(GL_FRONT_AND_BACK , GL_FILL)
    glTranslatef(translate_x, translate_y, 0)
    cube_draw()

app.run()
clock.unschedule(cube)





