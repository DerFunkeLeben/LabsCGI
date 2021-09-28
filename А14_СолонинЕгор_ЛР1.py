import numpy as np
from pyglet.gl import *
from pyglet.window import Window
from pyglet import app
w = 200
vp = np.full((w, w, 3), 255, dtype='uint8')


arr = np.random.uniform(0,100, size=(20,20))
print(arr)

# рамка
vp[-5:, :] = vp[:5, :] = [0, 0, 0]
vp[:, -5:] = vp[:, :5] = [0, 0, 0]

# синий квадрат
k = w // 20
vp[4*k:16*k, 4*k:16*k] = [0, 102, 204]

# серые прямоугольники
vp[12*k:15*k, 7*k:13*k] = [224, 224, 224]
vp[5*k:10*k, 6*k:14*k] = [224, 224, 224]

# надписи
k = w // 40
vp[16*k:17*k, 15*k:25*k] = [0, 102, 204]
vp[12*k:13*k, 15*k:25*k] = [0, 102, 204]
vp[25*k:29*k, 22*k:24*k] = [0, 102, 204]

# точка bottom right
vp[9*k:10*k, 30*k:31*k] = [224, 224, 224]

# вырез треугольника
for i in range(140, 160):
    vp[300-i:160, i:i+1] = [255, 255, 255]

vp = vp.flatten()
vp = (GLubyte * (w * w * 3))(*vp)
window = Window(visible=True, width=w, height=w, caption='disc')


@window.event
def on_draw():
    window.clear()
    glDrawPixels(w, w, GL_RGB, GL_UNSIGNED_BYTE, vp)


app.run()
