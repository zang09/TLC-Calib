#
# Copyright (C) 2026, SNU
# SNU VGI lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact haebeom.jung@snu.ac.kr
#

import torch
import viser
import nerfview
import numpy as np
from typing import Tuple

from scene import Scene
from scene.cameras import Camera
from gaussian_renderer import render
from scipy.spatial.transform import Rotation


class ViewerRenderer:
    def __init__(self, scene, pipe, dataset, handles: dict, server):
        self.scene = scene
        self.pipe = pipe
        self.dataset = dataset

        server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self.server = server
        self.gaussian_scale_slider = handles["gaussian_scale"]
        self.anchor_pc_handle = handles["anchor_pc"]
        self.offset_pc_handle = handles["offset_pc"]
        self.camera_handles_and_objects = handles["cameras"]
        self.camera_group_handle = handles["camera_group"]

        self.last_image = None

    @torch.no_grad()
    def __call__(self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        try:
            W, H = img_wh
            c2w_np = camera_state.c2w
            K_np = camera_state.get_K(img_wh)

            w2c_np = np.linalg.inv(c2w_np).astype(np.float32)
            R_np, T_np = w2c_np[:3, :3], w2c_np[:3, 3]

            fx, fy, cx, cy = float(K_np[0, 0]), float(K_np[1, 1]), float(K_np[0, 2]), float(K_np[1, 2])

            viewpoint_cam = Camera(
                cam_id=0, uid=0, cam2lidar=np.identity(4,dtype=np.float32), \
                lidar_R=np.identity(3,dtype=np.float32), lidar_t=np.zeros(3,dtype=np.float32), \
                GT_cam_R=R_np, GT_cam_t=T_np, cam_R=R_np, cam_t=T_np, \
                fx=fx, fy=fy, cx=cx, cy=cy, \
                image=torch.zeros((3, H, W), device="cuda"), \
                alpha_mask=None, image_name="viewer", timestamp=0.0
            )

            bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            scale_factor = self.gaussian_scale_slider.value

            render_pkg = render(viewpoint_cam, self.scene.gaussians, self.pipe, background, scale_factor)

            if self.anchor_pc_handle.visible:
                self.anchor_pc_handle.points = self.scene.gaussians.get_anchor_pcd()
            if self.offset_pc_handle.visible:
                self.offset_pc_handle.points = self.scene.gaussians.get_offset_pcd()

            if self.camera_group_handle.visible:
                for opt_handle, _, cam in self.camera_handles_and_objects:
                    R = cam.cam_R.cpu().numpy()
                    T = cam.cam_t.cpu().numpy()
                    w2c = np.eye(4)
                    w2c[:3, :3] = R
                    w2c[:3, 3] = T.flatten()
                    c2w = np.linalg.inv(w2c)

                    position = c2w[:3, 3]
                    quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                    opt_handle.position = position
                    opt_handle.wxyz = quat_wxyz

            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            self.last_image = image.permute(1, 2, 0).cpu().numpy()
            return self.last_image

        except Exception as e:
            print(f"[viser] Viewer Renderer Error: {e}")
            return self.last_image


def setup_viewer(server: viser.ViserServer, scene: Scene, camera_frame_step: int = 3):
    print("[viser] Loading initial point cloud for viewer...")
    try:
        all_points = scene.init_pointcloud.points
        center = np.mean(all_points, axis=0)
        # scene_extent = np.max(np.linalg.norm(all_points - center, axis=1))
    except:
        center = np.zeros(3)
        # scene_extent = 1.0

    with server.gui.add_folder("Camera Views"):
        orthogonal_button = server.gui.add_button("Orthogonal View")

    with server.gui.add_folder("Visibility Controls"):
        toggle_initial = server.gui.add_checkbox("Initial Points", initial_value=False)
        toggle_anchors = server.gui.add_checkbox("Anchors", initial_value=False)
        toggle_offsets = server.gui.add_checkbox("Auxiliary", initial_value=False)
        toggle_opt_cameras = server.gui.add_checkbox("Opt Cams", initial_value=False)
        toggle_gt_cameras = server.gui.add_checkbox("Ref Cams", initial_value=False)
        image_dropdown = server.gui.add_dropdown("Content", ("None", "Image"), initial_value="None")

    with server.gui.add_folder("Scale Controls"):
        point_size_slider = server.gui.add_slider("Point Size", min=0.01, max=0.1, step=0.01, initial_value=0.01)
        gaussian_scale_slider = server.gui.add_slider("Gaussian Scale", min=0.1, max=2.0, step=0.05, initial_value=1.0)

    initial_pc_handle = server.scene.add_point_cloud(
        name="/initial_points", points=scene.init_pointcloud.points,
        colors=(50, 150, 255), point_size=0.01, visible=toggle_initial.value
    )
    anchor_pc_handle = server.scene.add_point_cloud(
        name="/anchors", points=np.zeros((0, 3)), colors=(50, 200, 50), point_size=0.01, visible=toggle_anchors.value
    )
    offset_pc_handle = server.scene.add_point_cloud(
        name="/offsets", points=np.zeros((0, 3)), colors=(250, 50, 50), point_size=0.01, visible=toggle_offsets.value
    )

    opt_camera_group_handle = server.scene.add_transform_controls(
        name="/opt_cameras",
        disable_axes=True,
        disable_sliders=True,
        disable_rotations=True,
        visible=toggle_opt_cameras.value,
    )

    gt_camera_group_handle = server.scene.add_transform_controls(
        name="/gt_cameras",
        disable_axes=True,
        disable_sliders=True,
        disable_rotations=True,
        visible=toggle_gt_cameras.value,
    )

    UP_DIRECTION = np.array([0, 0, 1])
    def create_click_handler(camera, is_gt):
        def handler(event: viser.SceneNodePointerEvent):
            client = event.client
            if client is None: return

            if is_gt:
                R, T = camera.GT_cam_R, camera.GT_cam_t
            else:
                R, T = camera.cam_R.cpu().numpy(), camera.cam_t.cpu().numpy()

            w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = T.flatten()
            c2w = np.linalg.inv(w2c)

            position = c2w[:3, 3]
            quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            # OpenCV standard: camera looks along +Z axis
            look_at_target = c2w[:3, :3] @ np.array([0, 0, 1.0]) + position

            client.camera.position = position
            client.camera.wxyz = quat_wxyz
            client.camera.look_at = look_at_target
            client.camera.up_direction = UP_DIRECTION

            print(f"Jumped to camera: {event.target.name}")
        return handler

    camera_handles_and_objects = []
    camera_images = []

    all_cameras = scene.get_train_cameras()

    frame_step = max(1, camera_frame_step)
    cameras_by_frame = {}
    for cam in all_cameras:
        frame_key = round(float(getattr(cam, "timestamp", cam.uid)), 9)
        cameras_by_frame.setdefault(frame_key, []).append(cam)

    selected_cameras = []
    for frame_key in sorted(cameras_by_frame)[::frame_step]:
        selected_cameras.extend(sorted(cameras_by_frame[frame_key], key=lambda cam: cam.cam_id))
    all_cameras = selected_cameras

    for i, cam in enumerate(all_cameras):
        R = cam.cam_R.cpu().numpy()
        T = cam.cam_t.cpu().numpy()
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.flatten()
        c2w = np.linalg.inv(w2c)

        position = c2w[:3, 3]

        rotation_matrix = c2w[:3, :3]
        quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        opt_frustum_handle = server.scene.add_camera_frustum(
            name=f"/opt_cameras/frustum_{i}",
            fov=cam.FoVy, aspect=cam.image_width / cam.image_height, scale=0.3,
            wxyz=quat_wxyz,
            position=position,
            color=(200, 200, 200),
        )
        opt_frustum_handle.on_click(create_click_handler(cam, is_gt=False))

        # GT Cam
        gt_R = cam.GT_cam_R
        gt_T = cam.GT_cam_t
        gt_w2c = np.eye(4)
        gt_w2c[:3, :3] = gt_R
        gt_w2c[:3, 3] = gt_T.flatten()
        gt_c2w = np.linalg.inv(gt_w2c)

        gt_position = gt_c2w[:3, 3]
        gt_quat_xyzw = Rotation.from_matrix(gt_c2w[:3, :3]).as_quat()
        gt_quat_wxyz = np.array([gt_quat_xyzw[3], gt_quat_xyzw[0], gt_quat_xyzw[1], gt_quat_xyzw[2]])

        gt_frustum_handle = server.scene.add_camera_frustum(
            name=f"/gt_cameras/frustum_{i}",
            fov=cam.FoVy, aspect=cam.image_width / cam.image_height, scale=0.3,
            wxyz=gt_quat_wxyz,
            position=gt_position,
            color=(255, 255, 0), # yellow
        )
        gt_frustum_handle.on_click(create_click_handler(cam, is_gt=True))

        camera_handles_and_objects.append((opt_frustum_handle, gt_frustum_handle, cam))
        image_tensor = cam.original_image.permute(1, 2, 0)
        camera_images.append(image_tensor.cpu().numpy())

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        nonlocal UP_DIRECTION
        if not all_cameras:
            return

        cam = all_cameras[0]

        gt_R, gt_T = cam.GT_cam_R, cam.GT_cam_t
        gt_w2c = np.eye(4)
        gt_w2c[:3, :3] = gt_R
        gt_w2c[:3, 3] = gt_T.flatten()
        gt_c2w = np.linalg.inv(gt_w2c)

        position = gt_c2w[:3, 3]
        look_at_target = gt_c2w[:3, :3] @ np.array([0, 0, 1.0]) + position
        quat_xyzw = Rotation.from_matrix(gt_c2w[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        client.camera.position = position
        client.camera.wxyz = quat_wxyz
        client.camera.look_at = look_at_target
        UP_DIRECTION = np.asarray(client.camera.up_direction)

    @toggle_initial.on_update
    def _(_):
        initial_pc_handle.visible = toggle_initial.value

    @toggle_anchors.on_update
    def _(_):
        anchor_pc_handle.visible = toggle_anchors.value

    @toggle_offsets.on_update
    def _(_):
        offset_pc_handle.visible = toggle_offsets.value

    @toggle_opt_cameras.on_update
    def _(_): opt_camera_group_handle.visible = toggle_opt_cameras.value

    @toggle_gt_cameras.on_update
    def _(_): gt_camera_group_handle.visible = toggle_gt_cameras.value

    @image_dropdown.on_update
    def _(_):
        show_image = image_dropdown.value == "Image"
        for (opt_handle, gt_handle, _), image_np in zip(camera_handles_and_objects, camera_images):
            opt_handle.image = image_np if show_image else None
            gt_handle.image = image_np if show_image else None

    @point_size_slider.on_update
    def _(_):
        size = point_size_slider.value
        initial_pc_handle.point_size = size
        anchor_pc_handle.point_size = size
        offset_pc_handle.point_size = size

    @orthogonal_button.on_click
    def _(_):
        for client in server.get_clients().values():
            current_pos = client.camera.position
            current_wxyz = client.camera.wxyz
            current_xyzw = np.array([current_wxyz[1], current_wxyz[2], current_wxyz[3], current_wxyz[0]])

            new_pos = (current_pos[0], current_pos[1], 50.0)

            look_at_target = (new_pos[0], new_pos[1], center[2])

            R_current = Rotation.from_quat(current_xyzw).as_matrix()
            current_right_vec = R_current[:, 0]
            new_right_vec = np.array([current_right_vec[0], current_right_vec[1], 0.0])

            if np.linalg.norm(new_right_vec) < 1e-5:
                new_right_vec = np.array([1.0, 0.0, 0.0])
            else:
                new_right_vec /= np.linalg.norm(new_right_vec)

            new_forward_vec = np.array([0.0, 0.0, -1.0])
            new_up_vec = np.cross(new_forward_vec, new_right_vec)
            R_new = np.stack([new_right_vec, new_up_vec, new_forward_vec], axis=1)

            new_quat_xyzw = Rotation.from_matrix(R_new).as_quat()
            new_wxyz = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]])

            client.camera.position = new_pos
            client.camera.wxyz = new_wxyz
            client.camera.look_at = look_at_target

    return {
        "gaussian_scale": gaussian_scale_slider,
        "anchor_pc": anchor_pc_handle,
        "offset_pc": offset_pc_handle,
        "cameras": camera_handles_and_objects,
        "camera_group": opt_camera_group_handle
    }
