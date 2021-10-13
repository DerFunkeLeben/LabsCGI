from pyglet.gl import *
import pyglet
from pyglet import app, graphics, clock
from pyglet.window import Window, key
import numpy as np

width, height = 1000, 1000
w = 10
n_rot = 0
l = 3
window = Window(visible=True, width=width, height=height, resizable=True)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
glClearColor(0.4, 1.0, 0.8, 1.0)
glLineWidth(3)

settings = {
    'display_bases': True,
    'display_normals': True,
    'depth_test': True,
    'cull_face': True,
    'static_source_active': True,
    'lightning_active': True,
    'normalize_manual': True,
    'normalize_auto': False,
    'diffuse': True,
    'lightning_rotate': False,
    'lightning_x': 2 * l,
    'lightning_y': l,
    'lightning_z': 0,
    'prism_x': 0,
    'prism_y': 0,
    'prism_z': 0,
}


def show_axis():
    glBegin(GL_LINES)
    glColor3f(0, 0, 0)

    glVertex3f(0, 0, 0)
    glVertex3f(w, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, w, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, w)

    glEnd()

    glEnable(GL_LINE_STIPPLE)
    glLineStipple(1, 0x1C47)
    glLineWidth(2)

    glBegin(GL_LINES)

    glVertex3f(-w, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, -w, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, -w)
    glVertex3f(0, 0, 0)

    glEnd()
    glDisable(GL_LINE_STIPPLE)


def get_normals_edges(coords):
    normals = []

    for i in range(len(coords)):
        bl = [coords[i][0], coords[i][1], coords[i][2]]
        tl = [coords[i][0], coords[i][1] + 2 * l, coords[i][2]]
        tr = [coords[0][0], coords[0][1] + 2 * l, coords[0][2]] if i == len(
            coords)-1 else [coords[i+1][0], coords[i+1][1] + 2 * l, coords[i+1][2]]

        v1 = list(map(lambda x, y: x - y, tl, bl))
        v2 = list(map(lambda x, y: x - y, tr, tl))
        normal = np.cross(v1, v2)
        normal = normal / \
            np.linalg.norm(normal) if settings['normalize_manual'] else normal
        normals.append(normal)
    return normals


def get_normals_vertexes_per_base(normals_l, normal_base):
    n_top_all = []
    for i in range(len(normals_l)):
        n1 = list(map(lambda x, y, z: x + y + z, normal_base, normals_l[i], normals_l[0])) if i == len(
            normals_l)-1 else list(map(lambda x, y, z: x + y + z, normal_base, normals_l[i], normals_l[i+1]))

        n1 = n1 / np.linalg.norm(n1) if settings['normalize_manual'] else n1
        n_top_all.append(n1)
    return n_top_all


def get_all_normals_vertexes(normals_l, coords):
    normal_top = [0, 1, 0]
    normal_btm = [0, -1, 0]

    n_tops = get_normals_vertexes_per_base(normals_l, normal_top)
    n_btms = get_normals_vertexes_per_base(normals_l, normal_btm)

    return n_tops, n_btms


def draw_normals(coords):
    normals = get_normals_edges(coords)
    n_top, n_btm = get_all_normals_vertexes(normals, coords)
    coords_0 = coords.pop(0)
    coords_new = coords
    coords_new.append(coords_0)
    glColor3f(1, 1, 1)
    if settings['display_normals']:
        glBegin(GL_LINES)
        for i in range(len(normals)):
            glVertex3f(coords_new[i][0], coords_new[i][1], coords_new[i][2])
            glVertex3f(coords_new[i][0]+n_btm[i][0], coords_new[i]
                       [1]+n_btm[i][1], coords_new[i][2]+n_btm[i][2])

            glVertex3f(coords_new[i][0], coords_new[i]
                       [1] + 2*l, coords_new[i][2])
            glVertex3f(coords_new[i][0]+n_top[i][0], coords_new[i]
                       [1] + 2*l + n_top[i][1], coords_new[i][2]+n_top[i][2])
        glEnd()
    return coords_new, n_top, n_btm


def draw_prism():
    glPushMatrix()
    glTranslatef(settings['prism_x'],
                 settings['prism_y'], settings['prism_z'],)

    coords = [[0, 0, 0],
              [l, 0, 0],
              [3/2*l, 0, np.sqrt(3)/2*l],
              [l, 0, np.sqrt(3)*l],
              [0, 0, np.sqrt(3)*l],
              [-1/2*l, 0, np.sqrt(3)/2*l]]
    glFrontFace(GL_CW)
    glColor3f(1, 0, 0)

    glEnable(GL_NORMALIZE) if settings['normalize_auto'] else glDisable(
        GL_NORMALIZE)

    if settings['display_bases']:
        glBegin(GL_LINE_LOOP)
        for coord in coords:
            glVertex3f(coord[0], coord[1], coord[2])
        glEnd()

        glBegin(GL_LINE_LOOP)
        for coord in coords:
            glVertex3f(coord[0], coord[1] + l * 2, coord[2])
        glEnd()

    coords_new, n_top, n_btm = draw_normals(coords)

    glBegin(GL_QUADS)
    for i in range(len(coords_new)):
        glNormal3f(*n_btm[i])
        glVertex3f(*coords_new[i])

        glNormal3f(*n_top[i])
        glVertex3f(coords_new[i][0], coords_new[i]
                   [1] + 2 * l, coords_new[i][2])

        if i != len(coords_new) - 1:
            glNormal3f(*n_top[i+1])
            glVertex3f(coords_new[i+1][0], coords_new[i+1]
                       [1] + 2 * l, coords_new[i+1][2])

            glNormal3f(*n_btm[i+1])
            glVertex3f(coords_new[i+1][0], coords_new[i+1]
                       [1], coords_new[i+1][2])
        else:
            glNormal3f(*n_top[0])
            glVertex3f(coords_new[0][0], coords_new[0]
                       [1] + 2 * l, coords_new[0][2])

            glNormal3f(*n_btm[0])
            glVertex3f(coords_new[0][0], coords_new[0][1], coords_new[0][2])

    glEnd()
    glPopMatrix()


def upd(a):
    global n_rot
    n_rot = (n_rot + 1) % 360 if settings['lightning_rotate'] else n_rot


def draw_light_src():
    l0 = [settings['lightning_x'],
          settings['lightning_y'], settings['lightning_z']]
    l1 = [0, 3*l, 2*l]
    mtClr0 = [1, 0.4, 0.2]
    lghtClr0 = [1, 1, 0]
    lghtClr1 = [1, 0, 1]
    radius = np.sqrt(l0[0] ** 2 + l0[2] ** 2)
    glPointSize(10)
    glEnable(GL_POINT_SMOOTH)

    glBegin(GL_POINTS)

    glColor3f(*lghtClr0)
    glVertex3f(radius * np.cos(n_rot), l0[1], radius * np.sin(n_rot))

    glColor3f(*lghtClr1)
    glVertex3f(*l1)

    glEnd()

    glEnable(GL_LIGHTING) if settings['lightning_active'] else glDisable(
        GL_LIGHTING)

    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)
              (radius * np.cos(n_rot), l0[1], radius * np.sin(n_rot), 0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat * 4)(*lghtClr0, 0))

    glLightfv(GL_LIGHT1, GL_POSITION, (GLfloat * 4)(*l1, 0))
    glLightfv(GL_LIGHT1, GL_SPECULAR, (GLfloat * 4)(*lghtClr1, 0))

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(*mtClr0, 0))
    if settings['diffuse']:
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(*mtClr0, 0))

    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1) if settings['static_source_active'] else glDisable(
        GL_LIGHT1)


clock.schedule_interval(upd, 0.08)


@window.event
def on_draw():
    window.clear()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-w, w, -w, w, -w, w)
    glRotatef(-230, 0, 1, 0)
    glRotatef(20, 1, 0, 0)

    glDisable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST) if settings['depth_test'] else glDisable(
        GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE) if settings['cull_face'] else glDisable(
        GL_CULL_FACE)

    show_axis()
    draw_light_src()
    draw_prism()


@window.event
def on_key_press(symbol, modifiers):
    global settings
    if symbol == key._1:
        settings['display_bases'] = not settings['display_bases']
    elif symbol == key._3:
        settings['display_normals'] = not settings['display_normals']
    elif symbol == key._6:
        settings['depth_test'] = not settings['depth_test']
    elif symbol == key._7:
        settings['cull_face'] = not settings['cull_face']
    elif symbol == key._8:
        settings['static_source_active'] = not settings['static_source_active']
    elif symbol == key._9:
        settings['normalize_manual'] = not settings['normalize_manual']
    elif symbol == key._0:
        settings['normalize_auto'] = not settings['normalize_auto']
    elif symbol == key.MINUS:
        settings['lightning_active'] = not settings['lightning_active']
    elif symbol == key.EQUAL:
        settings['diffuse'] = not settings['diffuse']
    elif symbol == key.BACKSPACE:
        settings['lightning_rotate'] = not settings['lightning_rotate']

    elif symbol == key.NUM_1:
        settings['lightning_x'] -= l
    elif symbol == key.NUM_4:
        settings['lightning_x'] += l
    elif symbol == key.NUM_2:
        settings['lightning_y'] -= l
    elif symbol == key.NUM_5:
        settings['lightning_y'] += l
    elif symbol == key.NUM_3:
        settings['lightning_z'] -= l
    elif symbol == key.NUM_6:
        settings['lightning_z'] += l

    elif symbol == key.LEFT:
        settings['prism_x'] -= l
    elif symbol == key.RIGHT:
        settings['prism_x'] += l
    elif symbol == key.DOWN:
        settings['prism_y'] -= l
    elif symbol == key.UP:
        settings['prism_y'] += l
    elif symbol == key.PAGEDOWN:
        settings['prism_z'] -= l
    elif symbol == key.PAGEUP:
        settings['prism_z'] += l


app.run()

clock.unschedule(upd)
