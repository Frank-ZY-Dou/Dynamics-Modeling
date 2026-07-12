"""Render rollouts dumped by the evaluation scripts (--dump_rollout) to mp4.

Each *_rollout.npz becomes one side-by-side H.264 video: the model rollout on
the left (white arm, predicted contact force as a red arrow at the gripper)
and the recorded ground truth on the right (green arm, measured force). The
two panels are rendered separately with the same fixed camera, so each scene
contains exactly one robot; there are no text overlays.

The platform is inferred from the rollout layout: OpenManipulator-X dumps
(evaluate_actuator.py) store gt_q with 7 columns, SO-101 dumps
(evaluate_actuator_so101.py) with 6. Force-only dumps from infer_actuator.py
carry no sim_q; both panels then replay the recorded motion and differ only in
the force arrow. Encoding uses the system ffmpeg (libx264).

Usage:
    python render_rollout.py --rollout_dir outputs/force_sensor_rollouts --output_dir outputs/videos
"""
import argparse
import copy
import glob
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

OMX_MODEL_DIR = "robot"
OMX_SCENE_XML = "scene.xml"
OMX_ROBOT_XML = "open_manipulator_x.xml"
SO101_MODEL_DIR = "robot_so101"
SO101_SCENE_XML = "so101_torque_scene.xml"
SO101_ROOT_BODY = "base"

PRED_COLOR = "1 1 1"  # model arm (left panel)
GT_COLOR = "0 1 0"    # ground-truth arm (right panel)
ARROW_RGBA = (1.0, 0.1, 0.1, 1.0)
FLOOR_MARGIN = 0.005  # keep arrow tips at least 5 mm above the floor plane

# Standard project viewpoints, per platform
CAMERAS = {
    'omx': {'lookat': [0.25, 0.0, 0.15], 'distance': 1.10, 'azimuth': 90.0, 'elevation': -5.2},
    'so101': {'lookat': [0.4, 0.0, 0.12], 'distance': 1.05, 'azimuth': 90.0, 'elevation': -15.0},
}


def modify_robot_appearance(element, alpha, color="original", suffix="", disable_collision=False):
    if 'name' in element.attrib:
        element.attrib['name'] = element.attrib['name'] + suffix
    if element.tag == 'geom':
        if color != "original":
            if 'material' in element.attrib:
                del element.attrib['material']
            element.attrib['rgba'] = f"{color} {alpha}"
        elif alpha < 1.0:
            if 'rgba' in element.attrib:
                rgba = element.attrib['rgba'].split()
                if len(rgba) >= 3:
                    element.attrib['rgba'] = f"{rgba[0]} {rgba[1]} {rgba[2]} {alpha}"
            else:
                if 'material' in element.attrib:
                    del element.attrib['material']
                element.attrib['rgba'] = f"0.5 0.5 0.5 {alpha}"
        if disable_collision:
            element.attrib['contype'] = "0"
            element.attrib['conaffinity'] = "0"
    for child in element:
        modify_robot_appearance(child, alpha, color, suffix, disable_collision)


def build_omx_scene(color, offwidth=1920, offheight=1080):
    """Build an OMX scene with a single recolored robot at the scene origin."""
    scene_root = ET.parse(os.path.join(OMX_MODEL_DIR, OMX_SCENE_XML)).getroot()
    robot_root = ET.parse(os.path.join(OMX_MODEL_DIR, OMX_ROBOT_XML)).getroot()

    # Offscreen framebuffer large enough for the requested resolution
    visual = scene_root.find('visual')
    if visual is None:
        visual = ET.SubElement(scene_root, 'visual')
    g = visual.find('global')
    if g is None:
        g = ET.SubElement(visual, 'global')
    g.set('offwidth', str(offwidth))
    g.set('offheight', str(offheight))

    # Drop the include; we inject the robot bodies explicitly
    for inc in scene_root.findall('include'):
        scene_root.remove(inc)

    scene_worldbody = scene_root.find('worldbody')
    if scene_worldbody is None:
        scene_worldbody = ET.SubElement(scene_root, 'worldbody')

    robot_asset = robot_root.find('asset')
    if robot_asset is not None:
        scene_asset = scene_root.find('asset')
        if scene_asset is None:
            scene_asset = ET.SubElement(scene_root, 'asset')
        for child in robot_asset:
            scene_asset.append(copy.deepcopy(child))
    for tag in ('default', 'compiler', 'option'):
        node = robot_root.find(tag)
        if node is not None:
            scene_root.insert(0, copy.deepcopy(node))

    robot_worldbody = robot_root.find('worldbody')
    arm_root = ET.Element('body', {'name': 'arm_root', 'pos': '0 0 0'})
    for child in robot_worldbody:
        arm_root.append(copy.deepcopy(child))
    modify_robot_appearance(arm_root, 1.0, color=color)
    scene_worldbody.append(arm_root)

    # No actuators needed: qpos is set directly for rendering
    for act in scene_root.findall('actuator'):
        scene_root.remove(act)
    xml_content = ET.tostring(scene_root, encoding='unicode')

    cwd = os.getcwd()
    try:
        os.chdir(OMX_MODEL_DIR)
        mj_model = mujoco.MjModel.from_xml_string(xml_content)
    finally:
        os.chdir(cwd)
    return mj_model


def build_so101_scene(color, offwidth=1920, offheight=1080):
    """Recolor the single SO-101 robot in its torque scene."""
    scene_root = ET.parse(os.path.join(SO101_MODEL_DIR, SO101_SCENE_XML)).getroot()

    visual = scene_root.find('visual')
    if visual is None:
        visual = ET.SubElement(scene_root, 'visual')
    g = visual.find('global')
    if g is None:
        g = ET.SubElement(visual, 'global')
    g.set('offwidth', str(offwidth))
    g.set('offheight', str(offheight))

    scene_worldbody = scene_root.find('worldbody')
    robot_body = None
    for body in scene_worldbody.findall('body'):
        if body.attrib.get('name') == SO101_ROOT_BODY:
            robot_body = body
            break
    if robot_body is None:
        raise RuntimeError(f"Robot root body '{SO101_ROOT_BODY}' not found in {SO101_SCENE_XML}")
    modify_robot_appearance(robot_body, 1.0, color=color)

    for act in scene_root.findall('actuator'):
        scene_root.remove(act)
    xml_content = ET.tostring(scene_root, encoding='unicode')

    cwd = os.getcwd()
    try:
        os.chdir(SO101_MODEL_DIR)
        mj_model = mujoco.MjModel.from_xml_string(xml_content)
    finally:
        os.chdir(cwd)
    return mj_model


def load_rollout(npz_path):
    """Load a rollout dump; the gt_q width identifies the platform."""
    data = np.load(npz_path)
    gt_q = np.asarray(data['gt_q'])
    if gt_q.shape[1] == 7:
        robot = 'omx'
        # (T, 7): arm rad x4, motor5, aperture x2 -> arm rad x4 + gripper m
        gt = np.concatenate([gt_q[:, :4], gt_q[:, 5:6]], axis=1)
    elif gt_q.shape[1] == 6:
        robot = 'so101'
        gt = gt_q
    else:
        raise ValueError(f'{npz_path}: unexpected gt_q width {gt_q.shape[1]} (expected 7 for OMX, 6 for SO-101)')
    # Force-only dumps have no simulated motion: replay the recorded one
    pred = np.asarray(data['sim_q']) if 'sim_q' in data else gt
    force_pred = np.asarray(data['force_pred']) if 'force_pred' in data else None
    force_gt = np.asarray(data['force_gt']) if 'force_gt' in data else None
    return robot, pred, gt, force_pred, force_gt


def build_panels(robot, width, height):
    """One model/data/renderer per panel: prediction (white) and GT (green)."""
    offwidth, offheight = max(width, 1920), max(height, 1080)
    build = build_omx_scene if robot == 'omx' else build_so101_scene
    models = [build(color, offwidth, offheight) for color in (PRED_COLOR, GT_COLOR)]
    datas = [mujoco.MjData(m) for m in models]
    renderers = [mujoco.Renderer(m, height=height, width=width) for m in models]
    return models, datas, renderers


def set_arm_qpos(robot, mj_model, mj_data, q):
    if robot == 'omx':
        mj_data.qpos[:4] = q[:4]
        mj_data.qpos[4] = q[4]
        if mj_model.nq >= 6:
            mj_data.qpos[5] = q[4]
    else:
        mj_data.qpos[:6] = q[:6]


def add_force_arrow(scene, anchor, force, scale, rgba):
    """Tail-anchored force arrow; drawn length clamped above the floor."""
    mag = float(np.linalg.norm(force))
    if mag < 0.05 or scene.ngeom >= scene.maxgeom:
        return
    origin = np.asarray(anchor, dtype=np.float64)
    vec = np.asarray(force, dtype=np.float64) * scale
    tip = origin + vec
    if tip[2] < FLOOR_MARGIN and vec[2] < 0:
        frac = min(max((origin[2] - FLOOR_MARGIN) / (-vec[2]), 0.0), 1.0)
        vec = vec * frac
        tip = origin + vec
        if float(np.linalg.norm(vec)) < 0.01:
            return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.005,
                         origin.astype(np.float64), tip.astype(np.float64))
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def open_video_writer(out_path, width, height, fps):
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on PATH (required for H.264 encoding)')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:',
           '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
           out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_task(rollout, panels, cam_params, out_path, width, height, fps, frame_skip, arrow_scale):
    robot, pred_q, gt_q, force_pred, force_gt = rollout
    models, datas, renderers = panels
    T = min(len(pred_q), len(gt_q))

    # Force arrows are tail-anchored at the gripper
    if robot == 'omx':
        anchor_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'end_effector_target') for m in models]
    else:
        anchor_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe') for m in models]

    cam = mujoco.MjvCamera()
    cam.lookat[:] = cam_params['lookat']
    cam.distance = cam_params['distance']
    cam.azimuth = cam_params['azimuth']
    cam.elevation = cam_params['elevation']

    # Shift arrow anchors slightly toward the camera so the robot does not occlude them
    az = np.deg2rad(cam_params['azimuth'])
    el = np.deg2rad(cam_params['elevation'])
    toward_cam = np.array([-np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), -np.sin(el)])
    arrow_offset = toward_cam * 0.02

    writer = open_video_writer(out_path, 2 * width, height, fps)
    try:
        for t in range(0, T, frame_skip):
            frames = []
            for model, data, renderer, anchor_id, q, force in zip(
                    models, datas, renderers, anchor_ids,
                    (pred_q, gt_q), (force_pred, force_gt)):
                set_arm_qpos(robot, model, data, q[t])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                if force is not None and anchor_id >= 0:
                    anchor = data.xpos[anchor_id] if robot == 'omx' else data.site_xpos[anchor_id]
                    add_force_arrow(renderer.scene, anchor + arrow_offset, force[t],
                                    arrow_scale, ARROW_RGBA)
                frames.append(renderer.render())
            writer.stdin.write(np.hstack(frames).tobytes())
    finally:
        writer.stdin.close()
        ret = writer.wait()
    if ret != 0:
        raise RuntimeError(f'ffmpeg exited with status {ret} for {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Render rollout dumps to side-by-side prediction/GT videos')
    parser.add_argument('--rollout_dir', type=str, required=True, help='Directory with *_rollout.npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output mp4 files')
    parser.add_argument('--width', type=int, default=960, help='Panel width (the video is twice as wide)')
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--frame_skip', type=int, default=2, help='Render every N-th step (data is ~60Hz)')
    parser.add_argument('--cam_lookat', type=str, default=None, help='x,y,z; default depends on the robot')
    parser.add_argument('--cam_distance', type=float, default=None)
    parser.add_argument('--cam_azimuth', type=float, default=None)
    parser.add_argument('--cam_elevation', type=float, default=None)
    parser.add_argument('--arrow_scale', type=float, default=0.022, help='Arrow length per Newton (m)')
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.rollout_dir, '*_rollout.npz')))
    if not npz_files:
        raise FileNotFoundError(f'No *_rollout.npz files in {args.rollout_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    panel_cache = {}
    for npz_path in npz_files:
        rollout = load_rollout(npz_path)
        robot = rollout[0]
        if robot not in panel_cache:
            panel_cache[robot] = build_panels(robot, args.width, args.height)

        cam_params = dict(CAMERAS[robot])
        if args.cam_lookat is not None:
            cam_params['lookat'] = [float(x.strip()) for x in args.cam_lookat.split(',')]
        if args.cam_distance is not None:
            cam_params['distance'] = args.cam_distance
        if args.cam_azimuth is not None:
            cam_params['azimuth'] = args.cam_azimuth
        if args.cam_elevation is not None:
            cam_params['elevation'] = args.cam_elevation

        task = os.path.basename(npz_path).replace('_rollout.npz', '')
        out_path = os.path.join(args.output_dir, f'{task}.mp4')
        print(f'{task}...', end=' ', flush=True)
        render_task(rollout, panel_cache[robot], cam_params, out_path,
                    args.width, args.height, args.fps, args.frame_skip, args.arrow_scale)
        print(f'-> {out_path}')


if __name__ == '__main__':
    main()
