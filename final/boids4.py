"""
Research-Based 3D Boids Simulation - Physics & Perception
----------------------------------------------------------
All components grounded in research and physics:

SOCIAL FORCES:
- Topological neighbor selection (k-nearest, not metric distance)
- Elastic potential well cohesion (linear force with distance)
- 300° field of view (realistic bird vision)
- Omnidirectional separation, FOV-based alignment/cohesion

PERCEPTION REALISM:
- Sensor noise (imperfect perception of neighbor pos/vel)
- Neighbor dropout (attention limits, occlusion)

PHYSICS & SPEED DYNAMICS:
- Aerodynamic drag (quadratic with speed)
- Stall prevention (thrust below minimum flight speed)
- Stochastic speed variations (random target speed fluctuations)
- Speed-turn coupling (birds slow during sharp turns)
- Ornstein-Uhlenbeck wind process (correlated environmental noise)

STOCHASTICITY:
- Euler-Maruyama integration (velocity direction noise)
- Stochastic speed variations (individual differences)
- Sensor noise (perception uncertainty)
- Neighbor dropout (probabilistic)
- OU wind (correlated environmental)

Based on:
- Starling flock research (Ballerini et al.)
- Speed variations (Attanasi et al. 2014)
- Turn-speed coupling (Storms et al. 2019)
- Avian vision studies (perception limits)
- Aerodynamics (drag, stall speed)
- Meteorological models (OU process for wind)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from shared_functions import (update_wind_ou_process, apply_sensor_noise, 
                              apply_dropout, compute_turn_penalty, limit_speed,
                              compute_speed_regulation_force, 
                              create_predefined_obstacles, compute_obstacle_avoidance_force,
                              draw_all_obstacles, spawn_target_point, 
                              compute_target_attraction_force, check_target_reached,
                              draw_target_point, check_obstacle_collisions)


class BoidSwarm:
    def __init__(self, num_boids=50, ts=0.1, sigma=0.08, k_neighbors=7, 
                 sensor_noise=0.5, neighbor_dropout=0.1, wind_strength=0.3,
                 obstacles=None, obstacle_avoidance_strength=150.0):
        self.num_boids = num_boids
        self.ts = ts
        self.sigma = sigma  # Stochastic noise amplitude
        self.k_neighbors = k_neighbors  # Number of topological neighbors
        
        # Perception noise parameters - INCREASED for realistic variability
        # Real flock has polarization CV=28%, so we need significant noise
        self.sensor_noise = sensor_noise  # Noise in perceiving neighbor pos/vel
        self.neighbor_dropout = neighbor_dropout  # Probability of missing a neighbor
        
        # NEW: Ornstein-Uhlenbeck wind process parameters
        self.wind_strength = wind_strength  # Wind force amplitude
        self.wind_theta = 0.5  # Mean reversion rate (how quickly wind changes)
        self.wind_sigma = 0.8  # Wind noise intensity
        self.wind_state = np.zeros(3)  # Current wind velocity (OU state)
        
        # NEW: Obstacle parameters
        self.obstacles = obstacles if obstacles is not None else {'centers': np.empty((0, 3)), 'radii': np.empty(0), 'type': 'spheres'}
        self.obstacle_avoidance_strength = obstacle_avoidance_strength
        self.obstacle_detection_range = 25.0  # Increased detection range for earlier avoidance
        
        # NEW: Target point parameters
        self.target_point = None
        self.target_attraction_strength = 10.0  # How strongly boids are attracted to target (increased)
        self.target_reach_threshold = 20.0     # Distance to consider "reached"
        self.target_required_fraction = 0.5    # Fraction of boids needed to reach target
        self.boundary_limit = 50               # For spawning targets within bounds
        
        # Initialize positions in a cluster (like birds starting together)
        self.positions = np.random.randn(num_boids, 3) * 5
        
        # Start with mostly aligned velocities for faster flock formation
        # Base direction: positive x-axis (all birds flying roughly same direction)
        base_direction = np.array([1.0, 0.0, 0.0])
        
        # Add small random perturbations (±15° cone around base direction)
        perturbations = np.random.randn(num_boids, 3) * 0.2
        self.velocities = base_direction + perturbations
        
        # Normalize to consistent speed (matched to real bird data)
        initial_speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = self.velocities / (initial_speeds + 1e-8) * 8.37
        
        # Track visual history for smoother appearance
        self.prev_positions = self.positions.copy()
        
        # NEW: Track previous velocity for turn angle calculations
        self.v_prev = self.velocities.copy()
        
        # Aerodynamic parameters (physics-based)
        self.v_min_flight = 4.5  # Minimum speed to maintain flight (stall speed)
        self.drag_coefficient = 0.002  # Very low drag to reach ~7 m/s target
        
        # Speed regulation parameters (behavioral + aerodynamic)
        self.v0 = 8.0  # Higher target to compensate for forces
        self.alpha_speed = 0.15  # Low regulation = more natural speed variation

    def limit_speed(self, min_speed=4.5, max_speed=10.0):
        """Limit boid speeds to realistic range matching real bird data (wider range for CV~19%)"""
        self.velocities = limit_speed(self.velocities, min_speed, max_speed)

    def boundaries(self, limit=50, margin=10, turn_factor=1.0):
        """Smooth boundary avoidance with progressive turning"""
        for i in range(3):  # x, y, z
            too_low = self.positions[:, i] < -limit + margin
            too_high = self.positions[:, i] > limit - margin
            
            # Increase turn force closer to boundary
            distance_low = -limit + margin - self.positions[too_low, i]
            distance_high = self.positions[too_high, i] - (limit - margin)
            
            if np.any(too_low):
                self.velocities[too_low, i] += turn_factor * (1 + distance_low * 0.1)
            if np.any(too_high):
                self.velocities[too_high, i] -= turn_factor * (1 + distance_high * 0.1)

    def apply_field_of_view(self, diff, perception_mask):
        """
        Birds have ~300° field of view (research-based).
        Only consider neighbors within this forward-weighted cone.
        """
        # Normalize velocity directions
        velocity_dirs = self.velocities / (np.linalg.norm(self.velocities, axis=1, keepdims=True) + 1e-8)
        
        # Calculate angle between velocity and direction to neighbor
        neighbor_dirs = -diff / (np.linalg.norm(diff, axis=2, keepdims=True) + 1e-8)
        
        # Dot product gives cosine of angle
        cos_angles = np.sum(velocity_dirs[:, np.newaxis, :] * neighbor_dirs, axis=2)
        
        # Field of view: cos(150°) = -0.866, so we see from -150° to +150° (300° total)
        # This matches real bird vision better than 270°
        fov_mask = cos_angles > -0.866
        
        # Combine with perception mask
        return perception_mask & fov_mask

    def update_wind_ou_process(self):
        """
        NEW: Ornstein-Uhlenbeck process for realistic wind dynamics.
        Uses shared function for wind dynamics computation.
        """
        wind_force, self.wind_state = update_wind_ou_process(
            self.wind_state, self.wind_theta, self.wind_sigma, 
            self.wind_strength, self.ts
        )
        return wind_force
    
    def apply_sensor_noise(self, positions, velocities):
        """
        NEW: Add sensor noise to perceived neighbor positions/velocities.
        Uses shared function for sensor noise computation.
        """
        return apply_sensor_noise(positions, velocities, self.sensor_noise)
    
    def apply_neighbor_dropout(self, mask):
        """
        NEW: Probabilistic neighbor dropout (perception failures).
        Uses shared function for dropout computation.
        """
        return apply_dropout(mask, self.neighbor_dropout)

    def get_topological_neighbors(self, dist, k):
        """
        RESEARCH-BASED: Get k-nearest neighbors for each bird (topological interaction).
        Real starling flocks use ~6-7 nearest neighbors regardless of distance.
        This is more biologically accurate than metric distance thresholds.
        """
        # For each bird, find indices of k nearest neighbors
        # argsort returns indices that would sort the array
        # [:, 1:k+1] excludes self (index 0) and takes next k birds
        k_nearest_indices = np.argsort(dist, axis=1)[:, 1:k+1]
        
        # Create boolean mask for k-nearest neighbors
        topological_mask = np.zeros_like(dist, dtype=bool)
        for i in range(self.num_boids):
            topological_mask[i, k_nearest_indices[i]] = True
        
        return topological_mask

    def boids_algorithm(self):
        """
        RESEARCH-ENHANCED boids algorithm with realistic noise & speed dynamics:
        1. Topological neighbor selection (not metric distance)
        2. Sensor noise (imperfect perception of neighbors)
        3. Neighbor dropout (attention/occlusion limits)
        4. Ornstein-Uhlenbeck wind process (correlated environmental noise)
        5. Omnidirectional separation (no FOV)
        6. FOV-based alignment/cohesion with 300° vision
        7. Elastic potential well cohesion
        8. Aerodynamic drag + stall prevention
        9. Stochastic speed variations (random target speed)
        10. Speed-turn coupling (birds slow during sharp turns)
        11. Euler-Maruyama stochastic integration
        """
        # ==================== SENSOR NOISE (PERCEPTION UNCERTAINTY) ====================
        # NEW: Birds perceive neighbors with noise (imperfect vision)
        noisy_positions, noisy_velocities = self.apply_sensor_noise(self.positions, self.velocities)
        
        # OPTIMIZED: Calculate pairwise differences using NOISY perceived positions
        diff = noisy_positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        dist_sq = np.sum(diff * diff, axis=2)  # Faster than norm
        dist = np.sqrt(dist_sq + 1e-8)  # Add epsilon for numerical stability

        # ==================== SEPARATION (OMNIDIRECTIONAL, CLOSE RANGE) ====================
        # Birds avoid collisions from ALL directions - use metric distance for close-range
        SEPARATION_RADIUS = 10.0  # Tuned to achieve ~4.3m mean neighbor distance
        sep_mask = (dist < SEPARATION_RADIUS) & (dist > 0)
        
        # NEW: Apply neighbor dropout to separation (perception failures/occlusion)
        sep_mask = self.apply_neighbor_dropout(sep_mask)
        
        # Stronger force when closer (inverse square law)
        dist_safe = dist.copy()
        dist_safe[dist_safe < 0.1] = 0.1
        
        # Inverse distance weighting for separation
        sep_weights = np.where(sep_mask, 1.0 / (dist_safe ** 2), 0)
        sep_vectors = np.sum(-diff * sep_weights[:, :, np.newaxis], axis=1)

        # ==================== TOPOLOGICAL NEIGHBORS (RESEARCH-BASED) ====================
        # Get k-nearest neighbors for alignment and cohesion
        # This is how REAL birds interact - not based on fixed radius!
        topological_mask = self.get_topological_neighbors(dist, self.k_neighbors)
        
        # NEW: Apply neighbor dropout to topological neighbors
        topological_mask = self.apply_neighbor_dropout(topological_mask)
        
        # Apply 300° field of view constraint to topological neighbors
        topological_mask_fov = self.apply_field_of_view(diff, topological_mask)
        
        neighbor_count = np.sum(topological_mask_fov, axis=1, keepdims=True)
        neighbor_count[neighbor_count == 0] = 1

        # ==================== ALIGNMENT (FOV-CONSTRAINED TOPOLOGICAL NEIGHBORS) ====================
        # NEW: Use NOISY perceived velocities for alignment
        alignment_vectors = (
            noisy_velocities[np.newaxis, :, :] * topological_mask_fov[:, :, np.newaxis]
        )
        alignment_avg = np.sum(alignment_vectors, axis=1) / neighbor_count
        alignment = alignment_avg - self.velocities

        # ==================== COHESION (ELASTIC POTENTIAL WELL) ====================
        # RESEARCH-BASED: Birds behave as if in an elastic potential well
        # Force increases LINEARLY with distance from flock center
        
        # NEW: Calculate local flock center from NOISY perceived positions
        cohesion_target = (
            np.sum(noisy_positions[np.newaxis, :, :] * topological_mask_fov[:, :, np.newaxis], axis=1)
            / neighbor_count
        )
        
        # Direction to flock center
        to_center = cohesion_target - self.positions
        distance_to_center = np.linalg.norm(to_center, axis=1, keepdims=True)
        
        # Elastic force: F = k * distance (linear, like a spring)
        # Normalized direction × distance for linear scaling
        k_elastic = 0.15  # Spring constant (reduced for proper spacing)
        cohesion = k_elastic * distance_to_center * (to_center / (distance_to_center + 1e-8))

        # ==================== AERODYNAMIC FORCES (PHYSICS-BASED) ====================
        # Birds experience aerodynamic drag and need minimum speed for lift
        
        current_speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        v_hat = self.velocities / (current_speed + 1e-8)
        
        # 1. DRAG FORCE: F_drag = -C_d * v^2 * v̂
        # Quadratic drag (realistic for bird flight at these speeds)
        drag_force = -self.drag_coefficient * (current_speed ** 2) * v_hat
        
        # 2. THRUST/LIFT FORCE: Birds must maintain minimum speed to stay aloft
        # Below stall speed, birds apply extra thrust to avoid stalling
        # This models wing flapping effort to maintain flight
        speed_deficit = self.v_min_flight - current_speed
        stall_prevention = np.maximum(speed_deficit, 0)  # Only when below min speed
        
        # Thrust force proportional to how far below stall speed
        thrust_force = 2.0 * stall_prevention * v_hat
        
        # ==================== SPEED REGULATION (STOCHASTIC + TURN COUPLING) ====================
        # RESEARCH-BASED: Birds have varying target speeds and slow during turns
        # Uses shared functions for modular computation
        
        # 1. Compute turn penalty using shared function
        turn_penalty = compute_turn_penalty(self.velocities, self.v_prev, max_penalty=2.5)
        
        # 2. Compute speed regulation force (includes stochastic variations)
        speed_regulation_force = compute_speed_regulation_force(
            self.velocities, self.v0, self.alpha_speed,
            stochastic_variation=None,  # Auto-computed
            turn_penalty=turn_penalty,
            min_speed=10.0, max_speed=25.0
        )

        # ==================== ENVIRONMENTAL WIND (OU PROCESS) ====================
        # NEW: Add correlated wind force (more realistic than white noise)
        wind_force = self.update_wind_ou_process()
        
        # Broadcast wind to all birds (same wind affects everyone)
        wind_force_broadcasted = np.tile(wind_force, (self.num_boids, 1))

        # ==================== OBSTACLE AVOIDANCE ====================
        # Compute forces to avoid obstacles in the environment (strong prevention)
        obstacle_avoidance = compute_obstacle_avoidance_force(
            self.positions, self.velocities, self.obstacles,
            avoidance_strength=self.obstacle_avoidance_strength,
            detection_range=self.obstacle_detection_range,  # Longer range for earlier detection
            use_predictive=True
        )

        # ==================== TARGET ATTRACTION ====================
        # Attract boids toward target point (if exists)
        if self.target_point is not None:
            target_attraction = compute_target_attraction_force(
                self.positions, self.target_point,
                attraction_strength=self.target_attraction_strength,
                arrival_radius=self.target_reach_threshold
            )
        else:
            target_attraction = np.zeros_like(self.positions)

        # ==================== COMBINE FORCES ====================
        # TUNED TO MATCH TIME-AVERAGED REAL DATA:
        # - Speed: 7.01 m/s, Polar: 0.71, Neighbor Dist: 9.0m
        # - Real flock has CV=28% polarization (significant fluctuation)
        # Stored as attributes for parameter reporting
        self.w_sep = 15.0      # Separation for ~9.0m spacing
        self.w_ali = 6.0       # VERY LOW alignment for ~0.71 polarization
        self.w_coh = 4.0       # LOW cohesion allows more spacing
        # Aerodynamic forces applied directly (physics, not weighted)
        
        # Social forces (weighted)
        social_forces = (self.w_sep * sep_vectors + 
                        self.w_ali * alignment + 
                        self.w_coh * cohesion)
        
        # Physical forces (not weighted - these are physics + behavioral speed regulation!)
        physical_forces = (drag_force + 
                          thrust_force + 
                          speed_regulation_force +
                          wind_force_broadcasted +
                          obstacle_avoidance +
                          target_attraction)
        
        # Total steering force
        steering = social_forces + physical_forces

        # ==================== UPDATE WITH EULER-MARUYAMA ====================
        # Smooth acceleration (limit turning rate) - relaxed for better flocking
        max_force = 8.0  # Lower value allows more natural variation
        steering_norm = np.linalg.norm(steering, axis=1, keepdims=True)
        steering = np.where(steering_norm > max_force,
                          steering * max_force / (steering_norm + 1e-8),
                          steering)
        
        # Euler-Maruyama integration: deterministic + stochastic term
        stochastic_noise = self.sigma * np.sqrt(self.ts) * np.random.randn(self.num_boids, 3)
        
        # Apply steering (deterministic) and noise (stochastic)
        self.velocities += steering * self.ts + stochastic_noise
        
        # Enforce speed limits
        self.limit_speed()
        
        # Handle boundaries
        self.boundaries()
        
        # Update positions and track previous velocity (for turn angle calculation)
        self.prev_positions = self.positions.copy()
        self.v_prev = self.velocities.copy()  # NEW: Store for next iteration's turn coupling
        self.positions += self.velocities * self.ts


# ==================== MAIN BLOCK ====================
# Only run visualization when executing this file directly
# (not when importing BoidSwarm class for comparison scripts)
if __name__ == '__main__':
    # ==================== SIMULATION SETUP ====================
    ts = 0.08  # Time step
    sim_time = 60
    N = 90  # Number of boids
    K_NEIGHBORS = 7 #24  # Increased to 16 for single cohesive flock (prevents fragmentation)
                      # Research suggests 6-7 for real starlings, but more needed for 40 birds

    # Create obstacles (choose a preset or use create_obstacles for random)
    # Options: 'center_column', 'wall', 'scattered', 'tunnel', 'ring'
    OBSTACLE_PRESET = 'scattered'  # Change this to try different obstacle configurations
    obstacles = create_predefined_obstacles(OBSTACLE_PRESET)

    swarm = BoidSwarm(N, ts, sigma=0.08, k_neighbors=K_NEIGHBORS,
                      sensor_noise=0.5,      # NEW: Perception uncertainty
                      neighbor_dropout=0.1,  # NEW: 10% chance to miss neighbor
                      wind_strength=0.3,     # NEW: OU process wind strength
                      obstacles=obstacles,   # NEW: Obstacles in environment
                      obstacle_avoidance_strength=150.0)  # NEW: Strong avoidance force to prevent collisions

    # ==================== VISUALIZATION SETUP ====================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Research-Based Boids 3D - Physics & Perception", fontsize=14, fontweight='bold')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Add info text with research-based features highlighted
    info_text = ax.text2D(0.02, 0.98, 
                         f"Research-Based Boids: {N} | Topological: {K_NEIGHBORS}-NN\n"
                         f"Init: Pre-aligned (±15° cone) | Spacing: 20 units (w_sep=12)\n"
                         f"Cohesion: Strong (w={35.0}, k={0.5}) for single flock\n"
                         f"Perception: Sensor Noise + Dropout | Wind: OU Process\n"
                         f"Speed Dynamics: Drag + Stall + Stochastic + Turn Coupling\n"
                         f"Speed: 10-25 units (relaxed) | Max Force: 7.0 (less harsh)\n"
                         f"Color: Speed (blue=slow, yellow=fast)",
                         transform=ax.transAxes, 
                         color="black", 
                         fontsize=7.2,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

    timer_text = ax.text2D(0.98, 0.98, "", 
                          transform=ax.transAxes, 
                          color="black", 
                          fontsize=12,
                          horizontalalignment='right',
                          verticalalignment='top')

    # Main boid scatter plot
    scat = ax.scatter(swarm.positions[:, 0], 
                     swarm.positions[:, 1], 
                     swarm.positions[:, 2],
                     c="darkblue", 
                     s=30,
                     alpha=0.8,
                     edgecolors='navy')

    # Draw all obstacles using shared function
    draw_all_obstacles(ax, swarm.obstacles, num_points=120, color='crimson', alpha=0.5)

    # Initialize target point
    swarm.target_point = spawn_target_point(
        boundary_limit=50, margin=15, 
        obstacles=swarm.obstacles, min_obstacle_distance=10.0
    )
    target_scatter = [draw_target_point(ax, swarm.target_point, size=300, color='lime', marker='*')]

    # Velocity arrows (optional)
    SHOW_ARROWS = False

    if SHOW_ARROWS:
        quiver_list = [ax.quiver(swarm.positions[:, 0],
                                 swarm.positions[:, 1],
                                 swarm.positions[:, 2],
                                 swarm.velocities[:, 0],
                                 swarm.velocities[:, 1],
                                 swarm.velocities[:, 2],
                                 length=2.0,
                                 normalize=True,
                                 color='red',
                                 alpha=0.5,
                                 arrow_length_ratio=0.3)]
    else:
        quiver_list = [None]


    def update(frame):
        """Animation update function"""
        global target_scatter
        
        # Check if target is reached and respawn if needed
        if swarm.target_point is not None:
            reached, fraction = check_target_reached(
                swarm.positions, swarm.target_point,
                reach_threshold=swarm.target_reach_threshold,
                required_fraction=swarm.target_required_fraction
            )
            if reached:
                # Remove old target marker
                target_scatter[0].remove()
                # Spawn new target
                swarm.target_point = spawn_target_point(
                    boundary_limit=50, margin=15,
                    obstacles=swarm.obstacles, min_obstacle_distance=10.0
                )
                # Draw new target
                target_scatter[0] = draw_target_point(ax, swarm.target_point, size=300, color='lime', marker='*')
        
        swarm.boids_algorithm()
        
        # Update boid positions
        scat._offsets3d = (swarm.positions[:, 0],
                           swarm.positions[:, 1],
                           swarm.positions[:, 2])
        
        # Check for obstacle collisions and flash boids red if inside obstacles
        collision_mask, _, _ = check_obstacle_collisions(
            swarm.positions, swarm.obstacles, collision_radius=1.0
        )
        
        # Visual feedback: color by speed (blue=slow, yellow=fast)
        speeds = np.linalg.norm(swarm.velocities, axis=1)
        colors = plt.cm.viridis((speeds - 10.0) / (25.0 - 10.0))  # Normalize to new speed range
        
        # Flash boids red if they're inside obstacles (flashing effect every 3 frames)
        if np.any(collision_mask):
            flash_on = (frame // 3) % 2 == 0  # Flash every 3 frames
            red_color = np.array([1.0, 0.0, 0.0, 1.0]) if flash_on else np.array([1.0, 0.3, 0.3, 1.0])
            # Convert colors to array if needed
            if not isinstance(colors, np.ndarray):
                colors = np.array(colors)
            colors[collision_mask] = red_color  # Bright red or dim red
        
        scat.set_color(colors)
        
        # Update velocity vectors (if enabled)
        if SHOW_ARROWS:
            global quiver_list
            if quiver_list[0] is not None:
                quiver_list[0].remove()
            
            quiver_list[0] = ax.quiver(swarm.positions[:, 0],
                                       swarm.positions[:, 1],
                                       swarm.positions[:, 2],
                                       swarm.velocities[:, 0],
                                       swarm.velocities[:, 1],
                                       swarm.velocities[:, 2],
                                       length=2.0,
                                       normalize=True,
                                       color='red',
                                       alpha=0.5,
                                       arrow_length_ratio=0.3)
        
        # Set fixed boundaries
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        
        # Update timer
        curr_time = frame * ts
        timer_text.set_text(f"t = {curr_time:.2f}s")
        
        if SHOW_ARROWS:
            return scat, quiver_list[0], timer_text, info_text
        else:
            return scat, timer_text, info_text


    # Create animation
    anim = FuncAnimation(fig, update, 
                        frames=int(sim_time / ts),
                        interval=30,
                        blit=False, 
                        repeat=True)

    plt.tight_layout()
    plt.show()

