import os
import numpy as np
import mujoco
import mujoco.viewer


def deg(d):
    return np.deg2rad(d)


root = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.normpath(os.path.join(root, "..", "assets"))

model = mujoco.MjModel.from_xml_path(os.path.join(assets_dir, "scene.xml"))
data = mujoco.MjData(model)

#            pan     lift    elbow   wrist1  wrist2  wrist3
HOME_QPOS = [deg(0), deg(-90), deg(-90), deg(-90), deg(90), deg(0)]
data.qpos[:6] = HOME_QPOS
data.ctrl[:6] = HOME_QPOS
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:6] = HOME_QPOS
        mujoco.mj_step(model, data)
        viewer.sync()
