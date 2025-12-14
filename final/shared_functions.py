"""
Shared Functions for Boids and Cucker-Smale Simulations
--------------------------------------------------------
Common functions used across multiple flocking simulation models.

Contains:
- Ornstein-Uhlenbeck wind process (correlated environmental noise)
- Sensor noise (perception uncertainty)
- Interaction/neighbor dropout (attention limits)
- Speed regulation with stochastic variations and turn coupling
- Speed limiting utilities
- Obstacle creation and collision avoidance
"""

import numpy as np


# ==================== OBSTACLE FUNCTIONS ====================

def create_obstacles(obstacle_type='spheres', num_obstacles=3, boundary_limit=50, 
                     min_radius=5.0, max_radius=10.0, seed=None):
    """
    Create obstacles in the environment.
    
    Parameters:
    -----------
    obstacle_type : str
        Type of obstacles: 'spheres', 'cylinders', or 'custom'
    num_obstacles : int
        Number of obstacles to create
    boundary_limit : float
        Boundary limit to keep obstacles within simulation space
    min_radius : float
        Minimum obstacle radius
    max_radius : float
        Maximum obstacle radius
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    obstacles : dict
        Dictionary containing obstacle information:
        - 'centers': ndarray (M, 3) - obstacle center positions
        - 'radii': ndarray (M,) - obstacle radii
        - 'type': str - obstacle type
    """
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
    """
    Create predefined obstacle configurations.
    
    Parameters:
    -----------
    preset : str
        Preset configuration name:
        - 'center_column': Single large obstacle in center
        - 'wall': Row of obstacles forming a wall
        - 'scattered': Multiple scattered obstacles
        - 'tunnel': Two parallel walls forming a tunnel
        - 'ring': Obstacles arranged in a ring
        
    Returns:
    --------
    obstacles : dict
        Dictionary containing obstacle information
    """
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
    """
    Compute obstacle avoidance forces for all agents.
    
    Uses a combination of:
    1. Distance-based repulsion (stronger when closer)
    2. Predictive avoidance (look ahead based on velocity)
    
    Parameters:
    -----------
    positions : ndarray (N, 3)
        Current positions of all agents
    velocities : ndarray (N, 3)
        Current velocities of all agents
    obstacles : dict
        Obstacle dictionary with 'centers' and 'radii'
    avoidance_strength : float
        Base strength of avoidance force
    detection_range : float
        Additional detection range beyond obstacle radius
    use_predictive : bool
        Whether to use predictive avoidance (look ahead)
        
    Returns:
    --------
    avoidance_forces : ndarray (N, 3)
        Obstacle avoidance force for each agent
    """
    num_agents = positions.shape[0]
    avoidance_forces = np.zeros((num_agents, 3))
    
    # If no obstacles, return zero forces
    if obstacles['centers'].size == 0:
        return avoidance_forces
    
    obstacle_centers = obstacles['centers']  # (M, 3)
    obstacle_radii = obstacles['radii']      # (M,)
    num_obstacles = len(obstacle_radii)
    
    # Compute distances from each agent to each obstacle center
    # positions: (N, 3), obstacle_centers: (M, 3)
    # diff: (N, M, 3)
    diff = positions[:, np.newaxis, :] - obstacle_centers[np.newaxis, :, :]
    
    # Distance from agent to obstacle center
    dist_to_center = np.linalg.norm(diff, axis=2)  # (N, M)
    
    # Distance from agent to obstacle surface (negative if inside)
    dist_to_surface = dist_to_center - obstacle_radii[np.newaxis, :]  # (N, M)
    
    # Unit direction vectors pointing away from obstacles
    direction_away = diff / (dist_to_center[:, :, np.newaxis] + 1e-8)  # (N, M, 3)
    
    # ===== DISTANCE-BASED REPULSION =====
    # Force increases as agent gets closer to obstacle surface
    # Only apply force when within detection range
    effective_range = obstacle_radii[np.newaxis, :] + detection_range  # (1, M)
    
    # Mask for agents within detection range of each obstacle
    within_range = dist_to_center < effective_range  # (N, M)
    
    # Safety margin: minimum distance boids should maintain from obstacle surface
    safety_margin = 3.0  # Boids will try to stay at least 3 units away from obstacle surface
    
    # Normalized proximity (1.0 at safety margin, 0.0 at detection range boundary)
    # Use adjusted distance that accounts for safety margin
    adjusted_dist = dist_to_surface - safety_margin
    proximity = np.clip(1.0 - adjusted_dist / detection_range, 0.0, 5.0)  # (N, M)
    
    # Apply aggressive exponential falloff - force increases very rapidly when close
    # Use higher power (3.0) for much stronger repulsion when approaching safety margin
    force_magnitude = avoidance_strength * (proximity ** 3.0) * within_range  # (N, M)
    
    # Add extra force when within safety margin (critical zone)
    too_close = dist_to_surface < safety_margin
    critical_force = np.where(too_close, avoidance_strength * 10.0 * (1.0 - dist_to_surface / safety_margin), 0.0)
    force_magnitude += critical_force
    
    # ===== PREDICTIVE AVOIDANCE (LOOK AHEAD) =====
    if use_predictive:
        # Project position forward based on current velocity
        look_ahead_time = 1.0  # seconds
        future_positions = positions + velocities * look_ahead_time  # (N, 3)
        
        # Check if future position would be inside or too close to obstacle
        future_diff = future_positions[:, np.newaxis, :] - obstacle_centers[np.newaxis, :, :]
        future_dist = np.linalg.norm(future_diff, axis=2)  # (N, M)
        future_dist_to_surface = future_dist - obstacle_radii[np.newaxis, :]
        
        # Add extra force if heading toward obstacle
        heading_toward = future_dist_to_surface < dist_to_surface  # (N, M)
        predictive_boost = np.where(heading_toward, 1.5, 1.0)  # Boost force if heading toward
        
        force_magnitude *= predictive_boost
    
    # ===== EMERGENCY FORCE IF INSIDE OBSTACLE =====
    # If agent is inside obstacle, apply very strong outward force
    inside_obstacle = dist_to_surface < 0  # (N, M)
    # Much stronger emergency force - scales with how deep inside the obstacle
    penetration_depth = np.maximum(-dist_to_surface, 0.0)  # How far inside
    emergency_force = np.where(inside_obstacle, 
                              avoidance_strength * 20.0 * (1.0 + penetration_depth), 
                              0.0)  # (N, M)
    force_magnitude += emergency_force
    
    # ===== COMBINE FORCES FROM ALL OBSTACLES =====
    # Weight forces by magnitude and sum across obstacles
    weighted_forces = force_magnitude[:, :, np.newaxis] * direction_away  # (N, M, 3)
    avoidance_forces = np.sum(weighted_forces, axis=1)  # (N, 3)
    
    return avoidance_forces


# ==================== VISUALIZATION FUNCTIONS ====================

def draw_obstacle_pointcloud(ax, center, radius, num_points=150, color='crimson', alpha=0.6):
    """
    Draw a point cloud sphere for an obstacle (fast and visually appealing).
    
    Uses Fibonacci lattice for uniform point distribution on sphere surface.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        The axis to draw on
    center : array-like (3,)
        Center position of obstacle
    radius : float
        Radius of obstacle
    num_points : int
        Number of points to draw on sphere surface
    color : str
        Color of points
    alpha : float
        Transparency (0-1)
        
    Returns:
    --------
    scatter : PathCollection
        The scatter plot object (for potential updates)
    """
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
    """
    Draw all obstacles using point cloud visualization.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        The axis to draw on
    obstacles : dict
        Obstacle dictionary with 'centers' and 'radii'
    num_points : int
        Number of points per obstacle
    color : str
        Color of obstacles
    alpha : float
        Transparency
    """
    for i in range(len(obstacles['radii'])):
        draw_obstacle_pointcloud(ax, obstacles['centers'][i], 
                                obstacles['radii'][i],
                                num_points=num_points, color=color, alpha=alpha)


# ==================== TARGET POINT FUNCTIONS ====================

def spawn_target_point(boundary_limit=50, margin=10, obstacles=None, min_obstacle_distance=5.0):
    """
    Spawn a new target point within world bounds and not inside obstacles.
    
    Parameters:
    -----------
    boundary_limit : float
        World boundary limit
    margin : float
        Margin from boundaries
    obstacles : dict or None
        Obstacle dictionary with 'centers' and 'radii'
    min_obstacle_distance : float
        Minimum distance from obstacle surface
        
    Returns:
    --------
    target : ndarray (3,)
        Target point position
    """
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
    """
    Compute attraction force toward a target point for all agents.
    
    Uses a soft arrival behavior - force decreases as agents approach target.
    
    Parameters:
    -----------
    positions : ndarray (N, 3)
        Current positions of all agents
    target_point : ndarray (3,)
        Target point position
    attraction_strength : float
        Base attraction strength
    arrival_radius : float
        Radius within which force starts to decrease (soft landing)
        
    Returns:
    --------
    attraction_force : ndarray (N, 3)
        Attraction force for each agent
    """
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
    """
    Check if agents have collectively reached the target point.
    
    Parameters:
    -----------
    positions : ndarray (N, 3)
        Current positions of all agents
    target_point : ndarray (3,)
        Target point position
    reach_threshold : float
        Distance threshold to consider an agent as "reached"
    required_fraction : float
        Fraction of agents that must reach target (0-1)
        
    Returns:
    --------
    reached : bool
        True if enough agents have reached the target
    fraction_at_target : float
        Fraction of agents currently at target
    """
    distances = np.linalg.norm(positions - target_point, axis=1)
    agents_at_target = np.sum(distances < reach_threshold)
    fraction_at_target = agents_at_target / len(positions)
    
    reached = fraction_at_target >= required_fraction
    
    return reached, fraction_at_target


def draw_target_point(ax, target_point, size=200, color='red', marker='*', alpha=0.9):
    """
    Draw a target point marker.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        The axis to draw on
    target_point : ndarray (3,)
        Target position
    size : float
        Marker size
    color : str
        Marker color
    marker : str
        Marker style
    alpha : float
        Transparency
        
    Returns:
    --------
    scatter : PathCollection
        The scatter plot object (for updates)
    """
    return ax.scatter([target_point[0]], [target_point[1]], [target_point[2]], 
                     c=color, s=size, marker=marker, alpha=alpha, 
                     edgecolors='darkred', linewidths=2)


def check_obstacle_collisions(positions, obstacles, collision_radius=1.0):
    """
    Check which agents are colliding with obstacles.
    
    Parameters:
    -----------
    positions : ndarray (N, 3)
        Current positions of all agents
    obstacles : dict
        Obstacle dictionary with 'centers' and 'radii'
    collision_radius : float
        Agent collision radius (added to obstacle radius)
        
    Returns:
    --------
    collision_mask : ndarray (N,)
        Boolean mask indicating which agents are colliding
    closest_obstacle_idx : ndarray (N,)
        Index of closest obstacle for each agent (-1 if no collision)
    penetration_depth : ndarray (N,)
        How far inside the obstacle each agent is (0 if no collision)
    """
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


def update_wind_ou_process(wind_state, wind_theta, wind_sigma, wind_strength, dt):
    """
    Ornstein-Uhlenbeck process for realistic wind dynamics.
    
    Models slowly-varying environmental wind (more realistic than white noise).
    OU process: dX = -θ(X - μ)dt + σ*dW
    
    Properties:
    - Mean reversion: wind returns to calm (μ=0)
    - Correlated over time: realistic gusts
    - Continuous but varying
    
    Research: OU processes model turbulent air flow in meteorology
    
    Parameters:
    -----------
    wind_state : ndarray (3,)
        Current wind velocity (OU state)
    wind_theta : float
        Mean reversion rate (how quickly wind changes)
    wind_sigma : float
        Wind noise intensity
    wind_strength : float
        Wind force amplitude
    dt : float
        Time step
        
    Returns:
    --------
    wind_force : ndarray (3,)
        3D wind velocity vector (same for all birds)
    new_wind_state : ndarray (3,)
        Updated wind state for next iteration
    """
    # OU process discretization: X(t+dt) = X(t) + drift + diffusion
    drift = -wind_theta * wind_state * dt
    diffusion = wind_sigma * np.sqrt(dt) * np.random.randn(3)
    
    new_wind_state = wind_state + drift + diffusion
    
    # Return wind force (scaled by strength parameter)
    wind_force = wind_strength * new_wind_state
    
    return wind_force, new_wind_state


def apply_sensor_noise(positions, velocities, sensor_noise):
    """
    Add sensor noise to perceived neighbor positions/velocities.
    
    Models:
    - Visual perception uncertainty (depth, angle errors)
    - Attention limitations (can't perfectly track all birds)
    - Environmental factors (glare, fog, distance)
    
    Research: Birds have ~1-2° angular resolution, translates to position uncertainty
    
    Parameters:
    -----------
    positions : ndarray (N, 3)
        True positions of all birds
    velocities : ndarray (N, 3)
        True velocities of all birds
    sensor_noise : float
        Noise amplitude for perception uncertainty
        
    Returns:
    --------
    noisy_positions : ndarray (N, 3)
        Perceived positions with Gaussian noise
    noisy_velocities : ndarray (N, 3)
        Perceived velocities with Gaussian noise
    """
    # Add Gaussian noise to positions (each coordinate independently)
    noisy_positions = positions + np.random.randn(*positions.shape) * sensor_noise
    
    # Add Gaussian noise to velocities
    noisy_velocities = velocities + np.random.randn(*velocities.shape) * sensor_noise
    
    return noisy_positions, noisy_velocities


def apply_dropout(data, dropout_probability):
    """
    Probabilistic dropout for perception failures (works with masks or weights).
    
    Models:
    - Occlusion (birds blocking view of others)
    - Attention limits (can't process all neighbors simultaneously)
    - Momentary distractions
    
    Research: Birds don't perfectly track all neighbors, especially in dense flocks
    
    Parameters:
    -----------
    data : ndarray (N, N) or (N, M)
        Boolean mask or weight matrix representing interactions
    dropout_probability : float
        Probability of dropping each interaction (0 to 1)
        
    Returns:
    --------
    dropped_data : ndarray (N, N) or (N, M)
        Data with some interactions randomly dropped
    """
    # For each interaction, randomly drop with probability p
    dropout_mask = np.random.rand(*data.shape) > dropout_probability
    
    # Apply dropout (AND operation for boolean, multiplication for numeric)
    return data * dropout_mask


def compute_stochastic_speed_variations(num_agents, variation_amplitude=0.3):
    """
    Generate random speed variations for each agent.
    
    Models individual differences, energy variations, fatigue.
    Research: Attanasi et al. (2014) - speed variations in starling flocks
    
    Parameters:
    -----------
    num_agents : int
        Number of agents in simulation
    variation_amplitude : float
        Standard deviation of speed variations
        
    Returns:
    --------
    speed_variations : ndarray (N, 1)
        Random speed adjustments for each agent
    """
    return variation_amplitude * np.random.randn(num_agents, 1)


def compute_turn_penalty(current_velocities, previous_velocities, max_penalty=2.5):
    """
    Calculate speed reduction during turns (aerodynamic constraint).
    
    Birds slow down during sharp turns.
    Research: Storms et al. (2019) - birds slow down during collective turns
    Physical basis: turning requires centripetal force, easier at lower speeds
    
    Parameters:
    -----------
    current_velocities : ndarray (N, 3)
        Current velocity vectors
    previous_velocities : ndarray (N, 3)
        Previous velocity vectors
    max_penalty : float
        Maximum speed reduction for 180° turn
        
    Returns:
    --------
    turn_penalty : ndarray (N, 1)
        Speed reduction based on turn magnitude
    """
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
    """
    Limit agent speeds to realistic range.
    
    Parameters:
    -----------
    velocities : ndarray (N, 3)
        Current velocity vectors
    min_speed : float
        Minimum allowed speed
    max_speed : float
        Maximum allowed speed
        
    Returns:
    --------
    limited_velocities : ndarray (N, 3)
        Velocities with speeds clamped to [min_speed, max_speed]
    """
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds_clamped = np.clip(speeds, min_speed, max_speed)
    factor = speeds_clamped / (speeds + 1e-8)
    return velocities * factor


def compute_speed_regulation_force(current_velocities, target_speed_base, alpha_speed,
                                   stochastic_variation=None, turn_penalty=None,
                                   min_speed=10.0, max_speed=25.0):
    """
    Compute speed regulation force with stochastic variations and turn coupling.
    
    Combines:
    1. Base target speed (behavioral preference)
    2. Stochastic speed variations (individual differences)
    3. Turn penalty (aerodynamic constraint)
    
    F_speed = α * (v_target - |v|) * v_hat
    where v_target = v0 + stochastic_variation + turn_penalty
    
    Parameters:
    -----------
    current_velocities : ndarray (N, 3)
        Current velocity vectors
    target_speed_base : float
        Base desired cruise speed
    alpha_speed : float
        Speed regulation strength
    stochastic_variation : ndarray (N, 1) or None
        Random speed variations (if None, computed automatically)
    turn_penalty : ndarray (N, 1) or None
        Speed reduction from turning (if None, no penalty)
    min_speed : float
        Minimum allowed target speed
    max_speed : float
        Maximum allowed target speed
        
    Returns:
    --------
    speed_force : ndarray (N, 3)
        Speed regulation forces
    """
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

