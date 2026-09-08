import numpy as np
from typing import Tuple, Optional
from frame.schema import Frame
from camera.pan_tilt import PanTiltMechanism
from camera.viewport import CameraViewport
from config.schemas import SimConfig

class VirtualCamera:
    """
    FR-05 Virtual Camera & FR-09 Frame Generator.
    Renders a 640x480 pixel frame from the environment canvas based on current
    pan-tilt angles and FOV, outputting a standardized Frame object.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        self.pan_tilt = PanTiltMechanism(
            initial_pan_deg=0.0,
            initial_tilt_deg=0.0,
            max_pan_speed_deg_s=config.max_pan_speed_deg_s,
            max_tilt_speed_deg_s=config.max_tilt_speed_deg_s
        )
        self.viewport = CameraViewport(
            res_x=config.camera_res_x,
            res_y=config.camera_res_y,
            fov_h_deg=config.camera_fov_h_deg,
            fov_v_deg=config.camera_fov_v_deg,
            world_width=config.screen_width,
            world_height=config.screen_height
        )
        self._frame_count = 0

    def update_actuator(self, pan_cmd_deg_s: float, tilt_cmd_deg_s: float, dt: float) -> Tuple[float, float]:
        """Apply control command to pan-tilt mechanism."""
        return self.pan_tilt.update(pan_cmd_deg_s, tilt_cmd_deg_s, dt)

    def capture_frame(self, canvas: np.ndarray, timestamp: float,
                      ground_truth_world_pos: Optional[Tuple[float, float]] = None) -> Frame:
        """
        Crops environment canvas and packages into a Frame object.
        Pads with zeros if crop extends beyond environment boundaries.
        """
        x1, y1, x2, y2 = self.viewport.get_crop_rect(self.pan_tilt.pan_deg, self.pan_tilt.tilt_deg)
        h_env, w_env = canvas.shape[:2]

        # Target 640x480 frame buffer
        frame_buf = np.zeros((self.config.camera_res_y, self.config.camera_res_x), dtype=np.uint8)

        # Compute overlap region on canvas
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(w_env, x2)
        crop_y2 = min(h_env, y2)

        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
            # Destination slice indices inside frame_buf
            dst_x1 = crop_x1 - x1
            dst_y1 = crop_y1 - y1
            dst_x2 = dst_x1 + (crop_x2 - crop_x1)
            dst_y2 = dst_y1 + (crop_y2 - crop_y1)

            frame_buf[dst_y1:dst_y2, dst_x1:dst_x2] = canvas[crop_y1:crop_y2, crop_x1:crop_x2]

        gt_x, gt_y = ground_truth_world_pos if ground_truth_world_pos else (None, None)

        frame = Frame(
            image=frame_buf,
            timestamp=timestamp,
            frame_index=self._frame_count,
            camera_pan=self.pan_tilt.pan_deg,
            camera_tilt=self.pan_tilt.tilt_deg,
            ground_truth_x=gt_x,
            ground_truth_y=gt_y
        )
        self._frame_count += 1
        return frame
