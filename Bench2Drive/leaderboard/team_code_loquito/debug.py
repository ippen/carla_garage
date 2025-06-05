import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import carla
import time


# def generate_color_gradient(num):
#     """
#     Return a list of carla.Color objects, from center (red) to edge (purple),
#     using predefined vibrant colors that work well in CARLA.
#     """
#     predefined_colors = [
#         carla.Color(255, 0, 0),       # Red
#         carla.Color(255, 255, 0),     # Yellow
#         carla.Color(0, 255, 0),       # Green
#         carla.Color(0, 255, 255),     # Cyan
#         carla.Color(0, 0, 255),       # Blue
#         carla.Color(255, 0, 255),     # Magenta
#         carla.Color(255, 255, 255),   # White
#     ]

#     # Stretch or trim to fit requested number
#     gradient = []
#     for i in range(num):
#         idx = int(i / num * len(predefined_colors))
#         idx = min(idx, len(predefined_colors) - 1)
#         gradient.append(predefined_colors[idx])

#     return gradient


# def print_grid(center_x=0.0, center_y=0.0, center_yaw_deg=0.0, z=0.12):
#     """
#     Draws a 20x20 meter grid (1m spacing) around the ego vehicle, aligned with its yaw.
#     Uses predefined vivid colors based on distance from center.
#     """
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(10.0)
#     world = client.get_world()
#     max_dist = 20
#     sample_fac = 1

#     max_distance = math.sqrt(max_dist**2 + max_dist**2)
#     num_levels = int(max_distance) + 1
#     gradient_colors = generate_color_gradient(num_levels)

#     yaw_rad = math.radians(center_yaw_deg)
#     cos_yaw = math.cos(yaw_rad)
#     sin_yaw = math.sin(yaw_rad)
#     num_points = max_dist * sample_fac

#     for x in range(-num_points, num_points+1):
#         for y in range(-num_points, num_points+1):
#             dx = x / sample_fac
#             dy = y / sample_fac
#             distance = int(math.sqrt(dx**2 + dy**2))
#             color = gradient_colors[min(distance, len(gradient_colors) - 1)]

#             # Rotate (dx, dy) around origin using ego yaw
#             x_rot = dx * cos_yaw - dy * sin_yaw
#             y_rot = dx * sin_yaw + dy * cos_yaw

#             loc = carla.Location(
#                 x=center_x + x_rot,
#                 y=center_y + y_rot,
#                 z=z
#             )
#             world.debug.draw_point(loc, size=0.05, color=color, life_time=-1)
#             time.sleep(0.001)

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