from typing import Tuple

class CameraViewport:
    """
    FR-06 Camera FOV & Viewport Mapper.
    Maps pointing direction angles (pan_deg, tilt_deg) and angular FOV parameters
    to world pixel crop coordinates over the 2D environment canvas.
    """
    def __init__(self, res_x: int = 640, res_y: int = 480, fov_h_deg: float = 4.0, fov_v_deg: float = 3.0,
                 world_width: int = 2000, world_height: int = 2000):
        self.res_x = res_x
        self.res_y = res_y
        self.fov_h_deg = fov_h_deg
        self.fov_v_deg = fov_v_deg
        self.world_width = world_width
        self.world_height = world_height

        # Scale constants
        self.px_per_deg_h = self.res_x / self.fov_h_deg
        self.px_per_deg_v = self.res_y / self.fov_v_deg

        # World center baseline (pan=0, tilt=0 points to center of environment)
        self.world_center_x = world_width / 2.0
        self.world_center_y = world_height / 2.0

    def get_world_center(self, pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
        """Calculates camera boresight center in world coordinates."""
        world_x = self.world_center_x + pan_deg * self.px_per_deg_h
        world_y = self.world_center_y + tilt_deg * self.px_per_deg_v
        return world_x, world_y

    def get_crop_rect(self, pan_deg: float, tilt_deg: float) -> Tuple[int, int, int, int]:
        """
        Returns (x1, y1, x2, y2) world pixel crop rectangle for the camera frame.
        """
        cx, cy = self.get_world_center(pan_deg, tilt_deg)
        x1 = int(round(cx - self.res_x / 2.0))
        y1 = int(round(cy - self.res_y / 2.0))
        x2 = x1 + self.res_x
        y2 = y1 + self.res_y
        return x1, y1, x2, y2

    def world_to_camera_coords(self, world_x: float, world_y: float,
                               pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
        """
        Transforms world (x, y) point to camera frame pixel coordinates (0..640, 0..480).
        Camera center is at (res_x/2, res_y/2).
        """
        cam_cx, cam_cy = self.get_world_center(pan_deg, tilt_deg)
        pixel_x = (world_x - cam_cx) + self.res_x / 2.0
        pixel_y = (world_y - cam_cy) + self.res_y / 2.0
        return pixel_x, pixel_y
