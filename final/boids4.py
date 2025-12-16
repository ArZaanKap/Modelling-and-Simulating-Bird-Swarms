"""Research-based 3D Boids with topological neighbors, perception noise, and physics."""

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
        self.sigma = sigma
        self.k_neighbors = k_neighbors
        
        self.sensor_noise = sensor_noise
        self.neighbor_dropout = neighbor_dropout
        
        # Ornstein-Uhlenbeck wind process
        self.wind_strength = wind_strength
        self.wind_theta = 0.5
        self.wind_sigma = 0.8
        self.wind_state = np.zeros(3)
        
        # Obstacles
        self.obstacles = obstacles if obstacles is not None else {'centers': np.empty((0, 3)), 'radii': np.empty(0), 'type': 'spheres'}
        self.obstacle_avoidance_strength = obstacle_avoidance_strength
        self.obstacle_detection_range = 25.0
        
        # Target point
        self.target_point = None
        self.target_attraction_strength = 10.0
        self.target_reach_threshold = 20.0
        self.target_required_fraction = 0.5
        self.boundary_limit = 50
        
        # Initialize positions and velocities
        self.positions = np.random.randn(num_boids, 3) * 5
        base_direction = np.array([1.0, 0.0, 0.0])
        perturbations = np.random.randn(num_boids, 3) * 0.2
        self.velocities = base_direction + perturbations
        initial_speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = self.velocities / (initial_speeds + 1e-8) * 8.37
        
        self.prev_positions = self.positions.copy()
        self.v_prev = self.velocities.copy()
        
        # Aerodynamics
        self.v_min_flight = 4.5
        self.drag_coefficient = 0.003
        
        # Speed regulation
        self.v0 = 10.0
        self.alpha_speed = 0.6

    def limit_speed(self, min_speed=4.0, max_speed=11.0):
        """Limit boid speeds to realistic range."""
        self.velocities = limit_speed(self.velocities, min_speed, max_speed)

    def boundaries(self, limit=50, margin=10, turn_factor=1.0):
        """Smooth boundary avoidance with progressive turning."""
        for i in range(3):
            too_low = self.positions[:, i] < -limit + margin
            too_high = self.positions[:, i] > limit - margin
            
            distance_low = -limit + margin - self.positions[too_low, i]
            distance_high = self.positions[too_high, i] - (limit - margin)
            
            if np.any(too_low):
                self.velocities[too_low, i] += turn_factor * (1 + distance_low * 0.1)
            if np.any(too_high):
                self.velocities[too_high, i] -= turn_factor * (1 + distance_high * 0.1)

    def apply_field_of_view(self, diff, perception_mask):
        """Apply 300° field of view constraint."""
        velocity_dirs = self.velocities / (np.linalg.norm(self.velocities, axis=1, keepdims=True) + 1e-8)
        neighbor_dirs = -diff / (np.linalg.norm(diff, axis=2, keepdims=True) + 1e-8)
        cos_angles = np.sum(velocity_dirs[:, np.newaxis, :] * neighbor_dirs, axis=2)
        fov_mask = cos_angles > -0.866
        return perception_mask & fov_mask

    def update_wind_ou_process(self):
        """Ornstein-Uhlenbeck process for wind dynamics."""
        wind_force, self.wind_state = update_wind_ou_process(
            self.wind_state, self.wind_theta, self.wind_sigma, 
            self.wind_strength, self.ts
        )
        return wind_force
    
    def apply_sensor_noise(self, positions, velocities):
        """Add sensor noise to perceived positions/velocities."""
        return apply_sensor_noise(positions, velocities, self.sensor_noise)
    
    def apply_neighbor_dropout(self, mask):
        """Probabilistic neighbor dropout."""
        return apply_dropout(mask, self.neighbor_dropout)

    def get_topological_neighbors(self, dist, k):
        """Get k-nearest neighbors for each bird (topological interaction)."""
        k_nearest_indices = np.argsort(dist, axis=1)[:, 1:k+1]
        topological_mask = np.zeros_like(dist, dtype=bool)
        for i in range(self.num_boids):
            topological_mask[i, k_nearest_indices[i]] = True
        return topological_mask

    def boids_algorithm(self):
        """Research-enhanced boids with perception noise, topological neighbors, and physics."""
        noisy_positions, noisy_velocities = self.apply_sensor_noise(self.positions, self.velocities)
        
        diff = noisy_positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist = np.sqrt(dist_sq + 1e-8)

        # Separation (omnidirectional)
        SEPARATION_RADIUS = 7.0
        sep_mask = (dist < SEPARATION_RADIUS) & (dist > 0)
        sep_mask = self.apply_neighbor_dropout(sep_mask)
        
        dist_safe = dist.copy()
        dist_safe[dist_safe < 0.1] = 0.1
        sep_weights = np.where(sep_mask, 1.0 / (dist_safe ** 2), 0)
        sep_vectors = np.sum(-diff * sep_weights[:, :, np.newaxis], axis=1)

        # Topological neighbors
        topological_mask = self.get_topological_neighbors(dist, self.k_neighbors)
        topological_mask = self.apply_neighbor_dropout(topological_mask)
        topological_mask_fov = self.apply_field_of_view(diff, topological_mask)
        
        neighbor_count = np.sum(topological_mask_fov, axis=1, keepdims=True)
        neighbor_count[neighbor_count == 0] = 1

        # Alignment
        alignment_vectors = (
            noisy_velocities[np.newaxis, :, :] * topological_mask_fov[:, :, np.newaxis]
        )
        alignment_avg = np.sum(alignment_vectors, axis=1) / neighbor_count
        alignment = alignment_avg - self.velocities

        # Cohesion (elastic potential well)
        cohesion_target = (
            np.sum(noisy_positions[np.newaxis, :, :] * topological_mask_fov[:, :, np.newaxis], axis=1)
            / neighbor_count
        )
        to_center = cohesion_target - self.positions
        distance_to_center = np.linalg.norm(to_center, axis=1, keepdims=True)
        k_elastic = 0.15
        cohesion = k_elastic * distance_to_center * (to_center / (distance_to_center + 1e-8))

        # Aerodynamics
        current_speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        v_hat = self.velocities / (current_speed + 1e-8)
        drag_force = -self.drag_coefficient * (current_speed ** 2) * v_hat
        speed_deficit = self.v_min_flight - current_speed
        stall_prevention = np.maximum(speed_deficit, 0)
        thrust_force = 2.0 * stall_prevention * v_hat

        # Speed regulation
        turn_penalty = compute_turn_penalty(self.velocities, self.v_prev, max_penalty=2.5)
        speed_regulation_force = compute_speed_regulation_force(
            self.velocities, self.v0, self.alpha_speed,
            stochastic_variation=None,
            turn_penalty=turn_penalty,
            min_speed=4.0, max_speed=11.0
        )

        # Wind (OU process)
        wind_force = self.update_wind_ou_process()
        wind_force_broadcasted = np.tile(wind_force, (self.num_boids, 1))

        # Obstacle avoidance
        obstacle_avoidance = compute_obstacle_avoidance_force(
            self.positions, self.velocities, self.obstacles,
            avoidance_strength=self.obstacle_avoidance_strength,
            detection_range=self.obstacle_detection_range,
            use_predictive=True
        )

        # Target attraction
        if self.target_point is not None:
            target_attraction = compute_target_attraction_force(
                self.positions, self.target_point,
                attraction_strength=self.target_attraction_strength,
                arrival_radius=self.target_reach_threshold
            )
        else:
            target_attraction = np.zeros_like(self.positions)

        # Combine forces
        self.w_sep = 20.0
        self.w_ali = 1.4
        self.w_coh = 18.0
        
        social_forces = (self.w_sep * sep_vectors + 
                        self.w_ali * alignment + 
                        self.w_coh * cohesion)
        
        physical_forces = (drag_force + 
                          thrust_force + 
                          speed_regulation_force +
                          wind_force_broadcasted +
                          obstacle_avoidance +
                          target_attraction)
        
        steering = social_forces + physical_forces

        # Euler-Maruyama update
        max_force = 8.0
        steering_norm = np.linalg.norm(steering, axis=1, keepdims=True)
        steering = np.where(steering_norm > max_force,
                          steering * max_force / (steering_norm + 1e-8),
                          steering)
        
        stochastic_noise = self.sigma * np.sqrt(self.ts) * np.random.randn(self.num_boids, 3)
        self.velocities += steering * self.ts + stochastic_noise
        
        self.limit_speed()
        self.boundaries()
        
        self.prev_positions = self.positions.copy()
        self.v_prev = self.velocities.copy()
        self.positions += self.velocities * self.ts


if __name__ == '__main__':
    ts = 0.08
    sim_time = 60
    N = 90
    K_NEIGHBORS = 7

    OBSTACLE_PRESET = 'scattered'
    obstacles = create_predefined_obstacles(OBSTACLE_PRESET)

    swarm = BoidSwarm(N, ts, sigma=0.08, k_neighbors=K_NEIGHBORS,
                      sensor_noise=0.5,
                      neighbor_dropout=0.1,
                      wind_strength=0.3,
                      obstacles=obstacles,
                      obstacle_avoidance_strength=150.0)

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

    scat = ax.scatter(swarm.positions[:, 0], 
                     swarm.positions[:, 1], 
                     swarm.positions[:, 2],
                     c="darkblue", 
                     s=30,
                     alpha=0.8,
                     edgecolors='navy')

    draw_all_obstacles(ax, swarm.obstacles, num_points=120, color='crimson', alpha=0.5)

    swarm.target_point = spawn_target_point(
        boundary_limit=50, margin=15, 
        obstacles=swarm.obstacles, min_obstacle_distance=10.0
    )
    target_scatter = [draw_target_point(ax, swarm.target_point, size=300, color='lime', marker='*')]

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
        """Animation update function."""
        global target_scatter
        
        if swarm.target_point is not None:
            reached, fraction = check_target_reached(
                swarm.positions, swarm.target_point,
                reach_threshold=swarm.target_reach_threshold,
                required_fraction=swarm.target_required_fraction
            )
            if reached:
                target_scatter[0].remove()
                swarm.target_point = spawn_target_point(
                    boundary_limit=50, margin=15,
                    obstacles=swarm.obstacles, min_obstacle_distance=10.0
                )
                target_scatter[0] = draw_target_point(ax, swarm.target_point, size=300, color='lime', marker='*')
        
        swarm.boids_algorithm()
        
        scat._offsets3d = (swarm.positions[:, 0],
                           swarm.positions[:, 1],
                           swarm.positions[:, 2])
        
        collision_mask, _, _ = check_obstacle_collisions(
            swarm.positions, swarm.obstacles, collision_radius=1.0
        )
        
        speeds = np.linalg.norm(swarm.velocities, axis=1)
        colors = plt.cm.viridis((speeds - 10.0) / (25.0 - 10.0))
        
        if np.any(collision_mask):
            flash_on = (frame // 3) % 2 == 0
            red_color = np.array([1.0, 0.0, 0.0, 1.0]) if flash_on else np.array([1.0, 0.3, 0.3, 1.0])
            if not isinstance(colors, np.ndarray):
                colors = np.array(colors)
            colors[collision_mask] = red_color
        
        scat.set_color(colors)
        
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
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        
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

