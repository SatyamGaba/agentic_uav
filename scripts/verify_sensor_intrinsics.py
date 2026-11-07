import math
import numpy as np
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

cfg = CameraConfig()
print(f"CameraConfig width={cfg.image_width}, height={cfg.image_height}, hfov={cfg.hfov_deg}, camera_height={cfg.camera_height}")
print(f"CameraConfig fx={cfg.fx}, fy={cfg.fy}, cx={cfg.cx}, cy={cfg.cy}")

# Verify against pinhole formula
# W = 640, H = 480
# HFOV = 90 deg -> tan(HFOV/2) = tan(45 deg) = 1.0
# fx = (W / 2) / tan(HFOV/2) = 320 / 1.0 = 320.0
# Square pixels assumption in Habitat-Sim -> fy = fx = 320.0
# cx = W / 2 = 320.0, cy = H / 2 = 240.0
# Vertical FOV: 2 * atan((H/2) / fy) = 2 * atan(240 / 320) = 2 * atan(0.75) = 2 * 0.6435 rad = 1.287 rad = 73.74 deg

vfov_deg = math.degrees(2.0 * math.atan((cfg.image_height / 2.0) / cfg.fy))
print(f"Calculated VFOV: {vfov_deg:.2f}°")

# Nearest visible floor point:
# Floor is at y = -1.25 in camera frame (camera at y=1.25m, pitch=0)
# Bottom image row y_pixel = 480 -> y_cam_norm = (480 - 240) / 320 = 240 / 320 = 0.75
# In pinhole optical frame:
# x_c = (u_px - cx) * z_c / fx
# y_c = (v_px - cy) * z_c / fy
# Since camera is level (pitch = 0):
# Agent frame: Y_agent = -y_c + H_cam
# On floor: Y_agent = 0 -> -y_c + H_cam = 0 -> y_c = H_cam = 1.25
# Since y_c = (v_px - cy) * z_c / fy:
# 1.25 = (v_px - 240) * z_c / 320
# z_c = 1.25 * 320 / (v_px - 240)
# At the very bottom edge of image (v_px = 480):
# z_c_min = 1.25 * 320 / 240 = 1.25 / 0.75 = 1.6667 meters!

z_floor_bottom = cfg.camera_height * cfg.fy / (cfg.image_height - cfg.cy)
print(f"Nearest visible floor point at bottom edge of image (v=480): {z_floor_bottom:.4f} meters")

# At v = 360 (quarter down from center):
# z_floor_360 = 1.25 * 320 / 120 = 3.33 meters!
# At v = 240 (horizon / center):
# z_floor_240 = infinity!
print(f"Floor at row 360 (v=0.75): {cfg.camera_height * cfg.fy / (360 - cfg.cy):.2f} meters")
print(f"Floor at row 300 (v=0.625): {cfg.camera_height * cfg.fy / (300 - cfg.cy):.2f} meters")

