# pylint: disable=no-member
""" scatter using MarkersVisual """

import numpy as np
import sys

from vispy import app, visuals, scene


# build your visuals, that's all
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
# The real-things : plot using scene
# build canvas
canvas = scene.SceneCanvas(keys='interactive', show=True)
# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# plot ! note the parent parameter
p1 = Scatter3D(parent=view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)

def run(pos, colors = None, dotsize = 10, cam_dist = 400):
    """TODO: Docstring for run.

    :pos: TODO
    :colors: TODO
    :returns: TODO

    """
    p1.set_data(pos, face_color=colors, symbol='o', size=dotsize, edge_width=1, edge_color='black', scaling = False)

    # camera property
    view.camera.distance = cam_dist
    view.camera.fov = 45

    # run
    app.run()

import load_setup
exp_b = load_setup.read_setup('meshdata/all.h5')
pos = exp_b.PArr[0].pos
for i in range(1, len(exp_b.PArr)):
    P = exp_b.PArr[i]
    pos = np.append(pos, P.pos, axis = 0)
# print(pos)
colors = np.zeros((len(pos), 3))
pos_norm = np.sqrt(np.sum(pos**2))
colors[:,0] = pos_norm/np.amax(pos_norm)
run(pos, colors, dotsize = 5, cam_dist = 100e-3)


