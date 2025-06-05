import math
import carla
import time

def print_grid(center_x=0.0, center_y=0.0, center_yaw_deg=0.0, z=0.12):
    """
    Draws a 20x20 meter grid (1m spacing) around the ego vehicle, aligned with its yaw.
    Colors are assigned in square rings around the center, cycling through a fixed list of colors.
    """
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    max_dist = 20  # Grid goes from -20 to 20 meters
    sample_fac = 1
    num_points = max_dist * sample_fac

    # Fixed list of colors to cycle through
    ring_colors = [
        carla.Color(255, 0, 0),     # Red
        carla.Color(255, 255, 0),   # Yellow
        carla.Color(0, 255, 0),     # Green
        carla.Color(0, 255, 255),   # Cyan
        carla.Color(0, 0, 255),     # Blue
        carla.Color(255, 0, 255),   # Magenta
    ]
    num_colors = len(ring_colors)

    yaw_rad = math.radians(center_yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    for x in range(-num_points, num_points + 1):
        for y in range(-num_points, num_points + 1):
            dx = x / sample_fac
            dy = y / sample_fac

            # Determine square ring index (based on max offset in grid)
            ring_index = int(max(abs(dx), abs(dy)))
            color = ring_colors[ring_index % num_colors]  # Wrap around color list

            # Rotate the point around the ego's yaw
            x_rot = dx * cos_yaw - dy * sin_yaw
            y_rot = dx * sin_yaw + dy * cos_yaw

            loc = carla.Location(
                x=center_x + x_rot,
                y=center_y + y_rot,
                z=z
            )
            world.debug.draw_point(loc, size=0.05, color=color, life_time=-1)
            time.sleep(0.001)