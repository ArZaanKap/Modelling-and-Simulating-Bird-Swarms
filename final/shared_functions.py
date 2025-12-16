"""Shared functions for Boids and Cucker-Smale simulations."""

import numpy as np


def create_obstacles(obstacle_type='spheres', num_obstacles=3, boundary_limit=50, 
                     min_radius=5.0, max_radius=10.0, seed=None):
    """Create random obstacles in the environment."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random positions within boundary (with margin for obstacle size)
    margin = max_radius + 5.0
    centers = np.random.uniform(-boundary_limit + margin, 
                                 boundary_limit - margin, 
                                 (num_obstacles, 3))
    
    # Generate random radii
    radii = np.random.uniform(min_radius, max_radius, num_obstacles)
    
    obstacles = {
        'centers': centers,
        'radii': radii,
        'type': obstacle_type
    }
    
    return obstacles


def create_predefined_obstacles(preset='center_column'):
    """Create predefined obstacle configurations (center_column, wall, scattered, tunnel, ring)."""
    if preset == 'center_column':
        # Single large obstacle in the center
        obstacles = {
            'centers': np.array([[0.0, 0.0, 0.0]]),
            'radii': np.array([12.0]),
            'type': 'spheres'
        }
    
    elif preset == 'wall':
        # Vertical wall of obstacles blocking path
        centers = np.array([
            [20.0, -20.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.0, 20.0, 0.0],
            [20.0, -20.0, 20.0],
            [20.0, 0.0, 20.0],
            [20.0, 20.0, 20.0],
            [20.0, -20.0, -20.0],
            [20.0, 0.0, -20.0],
            [20.0, 20.0, -20.0],
        ])
        obstacles = {
            'centers': centers,
            'radii': np.ones(len(centers)) * 8.0,
            'type': 'spheres'
        }
    
    elif preset == 'scattered':
        # Random scattered obstacles
        centers = np.array([
            [-25.0, 15.0, 10.0],
            [15.0, -20.0, -5.0],
            [30.0, 10.0, 15.0],
            [-10.0, -25.0, 20.0],
            [0.0, 30.0, -15.0],
        ])
        obstacles = {
            'centers': centers,
            'radii': np.array([7.0, 8.0, 6.0, 9.0, 7.0]),
            'type': 'spheres'
        }
    
    elif preset == 'tunnel':
        # Two parallel walls forming a tunnel
        top_wall = [[x, 0.0, 25.0] for x in range(-30, 40, 15)]
        bottom_wall = [[x, 0.0, -25.0] for x in range(-30, 40, 15)]
        centers = np.array(top_wall + bottom_wall)
        obstacles = {
            'centers': centers,
            'radii': np.ones(len(centers)) * 8.0,
            'type': 'spheres'
        }
    
    elif preset == 'ring':
        # Obstacles arranged in a ring
        num_in_ring = 8
        ring_radius = 30.0
        angles = np.linspace(0, 2*np.pi, num_in_ring, endpoint=False)
        centers = np.zeros((num_in_ring, 3))
        centers[:, 0] = ring_radius * np.cos(angles)
        centers[:, 1] = ring_radius * np.sin(angles)
        centers[:, 2] = 0.0
        obstacles = {
            'centers': centers,
            'radii': np.ones(num_in_ring) * 6.0,
            'type': 'spheres'
        }
    
    else:
        # Default: no obstacles
        obstacles = {
            'centers': np.empty((0, 3)),
            'radii': np.empty(0),
            'type': 'spheres'
        }
    
    return obstacles


def compute_obstacle_avoidance_force(positions, velocities, obstacles, 
                                     avoidance_strength=50.0, 
                                     detection_range=15.0,
                                     use_predictive=True):
    """Compute obstacle avoidance forces using distance-based repulsion and predictive avoidance."""
    num_agents = positions.shape[0]
    avoidance_forces = np.zeros((num_agents, 3))
    
    if obstacles['centers'].size == 0:
        return avoidance_forces
    
    obstacle_centers = obstacles['centers']
    obstacle_radii = obstacles['radii']
    num_obstacles = len(obstacle_radii)
    
    diff = positions[:, np.newaxis, :] - obstacle_centers[np.newaxis, :, :]
    dist_to_center = np.linalg.norm(diff, axis=2)
    dist_to_surface = dist_to_center - obstacle_radii[np.newaxis, :]
    direction_away = diff / (dist_to_center[:, :, np.newaxis] + 1e-8)
    
    # Distance-based repulsion
    effective_range = obstacle_radii[np.newaxis, :] + detection_range
    within_range = dist_to_center < effective_range
    
    safety_margin = 3.0
    adjusted_dist = dist_to_surface - safety_margin
    proximity = np.clip(1.0 - adjusted_dist / detection_range, 0.0, 5.0)
    force_magnitude = avoidance_strength * (proximity ** 3.0) * within_range
    
    # Extra force in critical zone
    too_close = dist_to_surface < safety_margin
    critical_force = np.where(too_close, avoidance_strength * 10.0 * (1.0 - dist_to_surface / safety_margin), 0.0)
    force_magnitude += critical_force
    
    # Predictive avoidance
    if use_predictive:
        look_ahead_time = 1.0
        future_positions = positions + velocities * look_ahead_time
        future_diff = future_positions[:, np.newaxis, :] - obstacle_centers[np.newaxis, :, :]
        future_dist = np.linalg.norm(future_diff, axis=2)
        future_dist_to_surface = future_dist - obstacle_radii[np.newaxis, :]
        heading_toward = future_dist_to_surface < dist_to_surface
        predictive_boost = np.where(heading_toward, 1.5, 1.0)
        force_magnitude *= predictive_boost
    
    # Emergency force if inside obstacle
    inside_obstacle = dist_to_surface < 0
    penetration_depth = np.maximum(-dist_to_surface, 0.0)
    emergency_force = np.where(inside_obstacle, 
                              avoidance_strength * 20.0 * (1.0 + penetration_depth), 
                              0.0)
    force_magnitude += emergency_force
    
    weighted_forces = force_magnitude[:, :, np.newaxis] * direction_away
    avoidance_forces = np.sum(weighted_forces, axis=1)
    
    return avoidance_forces


def draw_obstacle_pointcloud(ax, center, radius, num_points=150, color='crimson', alpha=0.6):
    """Draw a point cloud sphere for an obstacle using Fibonacci lattice."""
    # Generate points uniformly distributed on sphere surface using Fibonacci lattice
    indices = np.arange(num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)  # Polar angle
    theta = np.pi * (1 + 5**0.5) * indices  # Azimuthal angle (golden ratio)
    
    # Convert to Cartesian coordinates
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    
    # Draw as scatter points
    return ax.scatter(x, y, z, c=color, s=8, alpha=alpha, edgecolors='none')


def draw_all_obstacles(ax, obstacles, num_points=120, color='crimson', alpha=0.5):
    """Draw all obstacles using point cloud visualization."""
    for i in range(len(obstacles['radii'])):
        draw_obstacle_pointcloud(ax, obstacles['centers'][i], 
                                obstacles['radii'][i],
                                num_points=num_points, color=color, alpha=alpha)


def spawn_target_point(boundary_limit=50, margin=10, obstacles=None, min_obstacle_distance=5.0):
    """Spawn a new target point within bounds and not inside obstacles."""
    max_attempts = 100
    valid_range = boundary_limit - margin
    
    for _ in range(max_attempts):
        # Generate random point within bounds
        target = np.random.uniform(-valid_range, valid_range, 3)
        
        # Check if inside or too close to any obstacle
        if obstacles is not None and obstacles['centers'].size > 0:
            distances = np.linalg.norm(obstacles['centers'] - target, axis=1)
            min_allowed = obstacles['radii'] + min_obstacle_distance
            
            if np.all(distances > min_allowed):
                return target
        else:
            return target
    
    # Fallback: return center if no valid point found
    return np.zeros(3)


def compute_target_attraction_force(positions, target_point, attraction_strength=5.0, 
                                    arrival_radius=10.0):
    """Compute attraction force toward a target point with soft arrival."""
    # Direction to target for each agent
    to_target = target_point - positions  # (N, 3)
    distances = np.linalg.norm(to_target, axis=1, keepdims=True)  # (N, 1)
    
    # Normalize direction
    direction = to_target / (distances + 1e-8)
    
    # Soft arrival: reduce force when close to target
    # Full force outside arrival_radius, linear decrease inside
    arrival_factor = np.clip(distances / arrival_radius, 0.0, 1.0)
    
    # Attraction force
    attraction_force = attraction_strength * arrival_factor * direction
    
    return attraction_force


def check_target_reached(positions, target_point, reach_threshold=15.0, 
                         required_fraction=0.6):
    """Check if agents have collectively reached the target point."""
    distances = np.linalg.norm(positions - target_point, axis=1)
    agents_at_target = np.sum(distances < reach_threshold)
    fraction_at_target = agents_at_target / len(positions)
    
    reached = fraction_at_target >= required_fraction
    
    return reached, fraction_at_target


def draw_target_point(ax, target_point, size=200, color='red', marker='*', alpha=0.9):
    """Draw a target point marker."""
    return ax.scatter([target_point[0]], [target_point[1]], [target_point[2]], 
                     c=color, s=size, marker=marker, alpha=alpha, 
                     edgecolors='darkred', linewidths=2)


def check_obstacle_collisions(positions, obstacles, collision_radius=1.0):
    """Check which agents are colliding with obstacles."""
    num_agents = positions.shape[0]
    
    if obstacles['centers'].size == 0:
        return (np.zeros(num_agents, dtype=bool), 
                np.full(num_agents, -1, dtype=int),
                np.zeros(num_agents))
    
    obstacle_centers = obstacles['centers']
    obstacle_radii = obstacles['radii']
    
    # Distance from each agent to each obstacle center
    diff = positions[:, np.newaxis, :] - obstacle_centers[np.newaxis, :, :]
    dist_to_center = np.linalg.norm(diff, axis=2)  # (N, M)
    
    # Distance to surface (accounting for collision radius)
    total_radii = obstacle_radii[np.newaxis, :] + collision_radius
    dist_to_surface = dist_to_center - total_radii  # (N, M)
    
    # Find minimum distance to any obstacle for each agent
    min_dist_to_surface = np.min(dist_to_surface, axis=1)  # (N,)
    closest_obstacle_idx = np.argmin(dist_to_surface, axis=1)  # (N,)
    
    # Collision if distance to surface is negative
    collision_mask = min_dist_to_surface < 0
    penetration_depth = np.maximum(-min_dist_to_surface, 0)
    
    # Set closest obstacle to -1 if no collision
    closest_obstacle_idx = np.where(collision_mask, closest_obstacle_idx, -1)
    
    return collision_mask, closest_obstacle_idx, penetration_depth


def check_bird_collisions(positions, collision_radius=0.5):
    """Check for collisions between birds (inter-agent collisions)."""
    num_agents = positions.shape[0]
    
    if num_agents < 2:
        return 0, [], 0
    
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Collision threshold (sum of radii)
    collision_threshold = 2 * collision_radius
    
    # Find collision pairs (only upper triangle to avoid double counting)
    collision_matrix = (distances < collision_threshold) & (distances > 0)  # Exclude self
    
    # Get unique collision pairs from upper triangle
    collision_pairs = []
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if collision_matrix[i, j]:
                collision_pairs.append((i, j))
    
    # Count unique birds in collision
    birds_in_collision_set = set()
    for i, j in collision_pairs:
        birds_in_collision_set.add(i)
        birds_in_collision_set.add(j)
    
    return len(collision_pairs), collision_pairs, len(birds_in_collision_set)


def update_wind_ou_process(wind_state, wind_theta, wind_sigma, wind_strength, dt):
    """Ornstein-Uhlenbeck process for realistic wind dynamics."""
    # OU process discretization: X(t+dt) = X(t) + drift + diffusion
    drift = -wind_theta * wind_state * dt
    diffusion = wind_sigma * np.sqrt(dt) * np.random.randn(3)
    
    new_wind_state = wind_state + drift + diffusion
    
    # Return wind force (scaled by strength parameter)
    wind_force = wind_strength * new_wind_state
    
    return wind_force, new_wind_state


def apply_sensor_noise(positions, velocities, sensor_noise):
    """Add Gaussian noise to perceived positions/velocities."""
    # Add Gaussian noise to positions (each coordinate independently)
    noisy_positions = positions + np.random.randn(*positions.shape) * sensor_noise
    
    # Add Gaussian noise to velocities
    noisy_velocities = velocities + np.random.randn(*velocities.shape) * sensor_noise
    
    return noisy_positions, noisy_velocities


def apply_dropout(data, dropout_probability):
    """Probabilistic dropout for perception failures."""
    # For each interaction, randomly drop with probability p
    dropout_mask = np.random.rand(*data.shape) > dropout_probability
    
    # Apply dropout (AND operation for boolean, multiplication for numeric)
    return data * dropout_mask


def compute_stochastic_speed_variations(num_agents, variation_amplitude=0.3):
    """Generate random speed variations for each agent."""
    return variation_amplitude * np.random.randn(num_agents, 1)


def compute_turn_penalty(current_velocities, previous_velocities, max_penalty=2.5):
    """Calculate speed reduction during turns (aerodynamic constraint)."""
    # Normalize velocity directions
    current_speed = np.linalg.norm(current_velocities, axis=1, keepdims=True)
    previous_speed = np.linalg.norm(previous_velocities, axis=1, keepdims=True)
    
    v_hat = current_velocities / (current_speed + 1e-8)
    v_prev_hat = previous_velocities / (previous_speed + 1e-8)
    
    # Calculate angle between current and previous velocity
    cos_angle = np.sum(v_hat * v_prev_hat, axis=1, keepdims=True)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    # Turn magnitude: 0 (no turn) to 2 (180° turn)
    turn_magnitude = 1 - cos_angle
    
    # Speed reduction during turns
    turn_penalty = -max_penalty * turn_magnitude
    
    return turn_penalty


def limit_speed(velocities, min_speed, max_speed):
    """Limit agent speeds to realistic range."""
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds_clamped = np.clip(speeds, min_speed, max_speed)
    factor = speeds_clamped / (speeds + 1e-8)
    return velocities * factor


def compute_speed_regulation_force(current_velocities, target_speed_base, alpha_speed,
                                   stochastic_variation=None, turn_penalty=None,
                                   min_speed=10.0, max_speed=25.0):
    """Compute speed regulation force with stochastic variations and turn coupling."""
    num_agents = current_velocities.shape[0]
    current_speed = np.linalg.norm(current_velocities, axis=1, keepdims=True)
    v_hat = current_velocities / (current_speed + 1e-8)
    
    # Compute stochastic variations if not provided
    if stochastic_variation is None:
        stochastic_variation = compute_stochastic_speed_variations(num_agents)
    
    # Use turn penalty if provided, otherwise zero
    if turn_penalty is None:
        turn_penalty = 0
    
    # Combined target speed
    target_speed = target_speed_base + stochastic_variation + turn_penalty
    target_speed = np.clip(target_speed, min_speed, max_speed)
    
    # Speed regulation force
    speed_error = target_speed - current_speed
    speed_force = alpha_speed * speed_error * v_hat
    
    return speed_force

