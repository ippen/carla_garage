import numpy as np
import carla

class Navigator:
    def __init__(self, dense_route_world_coord, debug=False):
        self.dense_route_world_coord = dense_route_world_coord
        self.locations = np.array([(rp[0].location.x, rp[0].location.y) for rp in self.dense_route_world_coord])
        self.commands = np.array([rp[1] for rp in self.dense_route_world_coord])

        self.current_index = 0
        self.debug = debug
        if debug:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.display()

        print(f"Navigator initialized with {len(self.dense_route_world_coord)} waypoints.")
    
    def get_next_target_waypoint(self, current_x, current_y, distance=20.0):
        """
        Get the next target waypoint that is on the route exactly distance meters away from the current position.
        
        Args:
            current_x: float, current x coordinate of the vehicle
            current_y: float, current y coordinate of the vehicle
            distance: float, distance in meters from the current position
        """
        def extrapolate_point(point, direction, distance):
            px, py = point
            dx, dy = direction
            s = distance
            a = dx**2 + dy**2
            b = 2 * (px * dx + py * dy)
            c = px**2 + py**2 - s**2
            D = b**2 - 4 * a * c
            
            t1 = 0
            if D >= 0:
                t1 = (-b + np.sqrt(D)) / (2 * a + 1e-6)
            t1 = max(0, t1)  # Ensure positive factor
            
            extr_point = point + t1 * direction
            return extr_point
        
        possible_locations = self.locations[self.current_index:]
        distances = np.linalg.norm(possible_locations - np.array([current_x, current_y]), axis=1)
        next_index = np.argmax(distances > distance)
        
        self.current_index += next_index

        prev_index = max(0, self.current_index - 1)
        prev_loc = self.locations[prev_index]
        # For numerical stability of direction vector get the point that is at least 0.2 meters away from the previous point
        next_loc = self.locations[self.current_index]
        for loc in self.locations[self.current_index:]:
            if np.linalg.norm(loc - prev_loc) > 0.2:
                next_loc = loc
                break
        direction = next_loc - prev_loc
        direction /= np.linalg.norm(direction)
        exact_loc = extrapolate_point(prev_loc - np.array([current_x, current_y]), direction, distance)
        exact_loc += np.array([current_x, current_y])

        if self.debug:
            print(f"Target waypoint: {exact_loc}")
            loc = carla.Location(x=exact_loc[0], y=exact_loc[1], z=0.6)
            
            self.world.debug.draw_point(loc, size=0.05, life_time=1000.0)
        
        return exact_loc
    
    def display(self):
        for loc, _ in self.dense_route_world_coord:
            new_loc = carla.Transform(loc.location + carla.Location(z=0.5), loc.rotation)
            loc.location.z += 0.5
            print("Waypoint: ", loc.location)
            self.world.debug.draw_point(new_loc.location, size=0.1, life_time=-1.0, color=carla.Color(0, 0, 255))

    def to_dict(self):
        """
        Return the route as a dictionary with waypoints and their commands.
        Each waypoint includes index (idx), x, y, and command.
        """
        route_data = []
        for i, ((x, y), command) in enumerate(zip(self.locations, self.commands)):
            route_data.append({
                "idx": int(i),
                "x": float(x),
                "y": float(y)
            })
        
        return route_data