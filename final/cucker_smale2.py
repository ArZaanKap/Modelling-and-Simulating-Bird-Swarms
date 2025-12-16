"""Cucker-Smale flocking model with Morse potential and stochastic dynamics."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from shared_functions import (update_wind_ou_process, apply_sensor_noise, 
                              apply_dropout, compute_turn_penalty, 
                              compute_stochastic_speed_variations, limit_speed,
                              create_predefined_obstacles, compute_obstacle_avoidance_force,
                              draw_all_obstacles, spawn_target_point,
                              compute_target_attraction_force, check_target_reached,
                              draw_target_point, check_obstacle_collisions)


class CuckerSmaleSwarm:
    def __init__(self, num_boids=50, dt=0.08, sigma=0.2, 
                 sensor_noise=0.5, interaction_dropout=0.1, wind_strength=0.3,
                 obstacles=None, obstacle_avoidance_strength=150.0):
        """Initialize Cucker-Smale swarm with attraction-repulsion dynamics."""
        self.N = num_boids
        self.dt = dt
        self.sigma = sigma
        
        self.sensor_noise = sensor_noise
        self.interaction_dropout = interaction_dropout
        
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
        
        # Initialize positions and velocities
        self.x = np.random.uniform(-20, 20, (num_boids, 3))
        self.v = np.random.randn(num_boids, 3)
        self.v[:, 0] += 10
        speeds = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v = self.v / speeds * 15.0
        
        # Morse potential parameters
        self.C_a = 5.0
        self.l_a = 18.0
        self.C_r = 6.0
        self.l_r = 5.5
        
        # Cucker-Smale alignment
        self.K = 0.015
        self.beta = 1.3
        
        # Aerodynamics
        self.v_min_flight = 4.5
        self.drag_coefficient = 0.003
        
        # Speed regulation
        self.v0 = 8.0
        self.alpha_speed = 0.06
        self.min_speed = 4.0
        self.max_speed = 11.0
        
        # Boundaries
        self.boundary_limit = 50
        self.boundary_margin = 10
        self.boundary_strength = 2.0

    def morse_potential_force(self, noisy_positions):
        """Compute attraction-repulsion forces from Morse potential."""
        diff = noisy_positions[:, np.newaxis, :] - self.x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2, keepdims=True)
        dist = np.sqrt(dist_sq + 1e-8)
        
        r_hat = diff / dist
        
        exp_attract = np.exp(-dist / self.l_a)
        exp_repel = np.exp(-dist / self.l_r)
        
        dU_dr = (self.C_a / self.l_a) * exp_attract - (self.C_r / self.l_r) * exp_repel
        
        mask = (dist[:, :, 0] > 0.1).astype(np.float32)
        forces_pairwise = -dU_dr * r_hat * mask[:, :, np.newaxis]
        forces = np.sum(forces_pairwise, axis=1)
        
        return forces

    def cucker_smale_alignment(self, noisy_positions, noisy_velocities):
        """Compute Cucker-Smale velocity alignment force."""
        diff_x = noisy_positions[:, np.newaxis, :] - self.x[np.newaxis, :, :]
        dist_sq = np.sum(diff_x * diff_x, axis=2)
        
        phi = np.power(1.0 + dist_sq, -self.beta)
        np.fill_diagonal(phi, 0)
        phi = self.apply_interaction_dropout(phi)
        
        diff_v = noisy_velocities[np.newaxis, :, :] - self.v[:, np.newaxis, :]
        weighted_alignment = phi[:, :, np.newaxis] * diff_v
        alignment = (self.K / self.N) * np.sum(weighted_alignment, axis=1)
        
        return alignment

    def speed_regulation(self):
        """Stochastic speed regulation with turn coupling."""
        stochastic_variation = compute_stochastic_speed_variations(self.N, variation_amplitude=2.0)
        
        if hasattr(self, 'v_prev'):
            turn_penalty = compute_turn_penalty(self.v, self.v_prev, max_penalty=3.5)
        else:
            turn_penalty = 0
            self.v_prev = self.v.copy()
        
        current_speed = np.linalg.norm(self.v, axis=1, keepdims=True)
        v_hat = self.v / (current_speed + 1e-8)
        
        target_speed = self.v0 + stochastic_variation + turn_penalty
        target_speed = np.clip(target_speed, self.min_speed, self.max_speed)
        
        speed_error = target_speed - current_speed
        speed_force = self.alpha_speed * speed_error * v_hat
        
        self.v_prev = self.v.copy()
        
        return speed_force

    def boundary_force(self):
        """Soft boundary constraint to keep birds in simulation area."""
        forces = np.zeros_like(self.x)
        limit = self.boundary_limit - self.boundary_margin
        
        too_low = self.x < -limit
        too_high = self.x > limit
        
        forces[too_low] += self.boundary_strength * (-limit - self.x[too_low])
        forces[too_high] -= self.boundary_strength * (self.x[too_high] - limit)
        
        return forces

    def update_wind_ou_process(self):
        """Ornstein-Uhlenbeck process for wind dynamics."""
        wind_force, self.wind_state = update_wind_ou_process(
            self.wind_state, self.wind_theta, self.wind_sigma, 
            self.wind_strength, self.dt
        )
        return wind_force
    
    def apply_sensor_noise(self, positions, velocities):
        """Add sensor noise to perceived positions/velocities."""
        return apply_sensor_noise(positions, velocities, self.sensor_noise)
    
    def apply_interaction_dropout(self, weights):
        """Probabilistic interaction dropout."""
        return apply_dropout(weights, self.interaction_dropout)

    def step(self):
        """Single time step using Euler-Maruyama integration."""
        noisy_positions, noisy_velocities = self.apply_sensor_noise(self.x, self.v)
        
        F_morse = self.morse_potential_force(noisy_positions)
        F_align = self.cucker_smale_alignment(noisy_positions, noisy_velocities)
        F_speed = self.speed_regulation()
        F_boundary = self.boundary_force()
        
        current_speed = np.linalg.norm(self.v, axis=1, keepdims=True)
        v_hat = self.v / (current_speed + 1e-8)
        
        F_drag = -self.drag_coefficient * (current_speed ** 2) * v_hat
        
        speed_deficit = self.v_min_flight - current_speed
        stall_prevention = np.maximum(speed_deficit, 0)
        F_thrust = 2.0 * stall_prevention * v_hat
        
        wind_force = self.update_wind_ou_process()
        F_wind = np.tile(wind_force, (self.N, 1))
        
        F_obstacle = compute_obstacle_avoidance_force(
            self.x, self.v, self.obstacles,
            avoidance_strength=self.obstacle_avoidance_strength,
            detection_range=self.obstacle_detection_range,
            use_predictive=True
        )
        
        if self.target_point is not None:
            F_target = compute_target_attraction_force(
                self.x, self.target_point,
                attraction_strength=self.target_attraction_strength,
                arrival_radius=self.target_reach_threshold
            )
        else:
            F_target = np.zeros_like(self.x)
        
        F_total = F_morse + F_align + F_speed + F_boundary + F_wind + F_obstacle + F_target + F_drag + F_thrust
        
        noise = self.sigma * np.sqrt(self.dt) * np.random.randn(self.N, 3)
        
        self.v += F_total * self.dt + noise
        self.v = limit_speed(self.v, self.min_speed, self.max_speed)
        self.x += self.v * self.dt


if __name__ == '__main__':
    dt = 0.08
    sim_time = 60
    N = 50

    OBSTACLE_PRESET = 'scattered'
    obstacles = create_predefined_obstacles(OBSTACLE_PRESET)

    swarm = CuckerSmaleSwarm(num_boids=N, dt=dt, sigma=0.08,
                             sensor_noise=0.5,
                             interaction_dropout=0.1,
                             wind_strength=0.3,
                             obstacles=obstacles,
                             obstacle_avoidance_strength=150.0)

    positions_history = []

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Cucker-Smale Model with Attraction-Repulsion", 
                 fontsize=14, fontweight='bold')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Info text
    info_text = ax.text2D(0.02, 0.98,
                         f"Cucker-Smale Model (Enhanced Stochasticity)\n"
                         f"Birds: {N} | Spacing: ~7-9 units\n"
                         f"Base Speed: {swarm.v0:.1f} ± stochastic variation\n"
                         f"Speed Range: {swarm.min_speed:.1f}-{swarm.max_speed:.1f} units\n"
                         f"Stochasticity: 6 layers (Wiener + Speed Var + Turn + OU Wind + Sensor + Dropout)\n"
                         f"Perception: Sensor Noise + 10% Interaction Dropout\n"
                         f"Color: Blue (slow) → Yellow (fast)",
                         transform=ax.transAxes,
                         fontsize=8.5,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    timer_text = ax.text2D(0.98, 0.98, "",
                          transform=ax.transAxes,
                          fontsize=12,
                          horizontalalignment='right',
                          verticalalignment='top')

    scat = ax.scatter(swarm.x[:, 0], swarm.x[:, 1], swarm.x[:, 2],
                     c='darkblue', s=30, alpha=0.8, edgecolors='navy')

    draw_all_obstacles(ax, swarm.obstacles, num_points=120, color='crimson', alpha=0.5)

    swarm.target_point = spawn_target_point(
        boundary_limit=50, margin=15, 
        obstacles=swarm.obstacles, min_obstacle_distance=10.0
    )
    target_scatter = [draw_target_point(ax, swarm.target_point, size=300, color='lime', marker='*')]

    SHOW_ARROWS = False
    if SHOW_ARROWS:
        quiver_list = [ax.quiver(swarm.x[:, 0], swarm.x[:, 1], swarm.x[:, 2],
                                 swarm.v[:, 0], swarm.v[:, 1], swarm.v[:, 2],
                                 length=2.0, normalize=True,
                                 color='red', alpha=0.5, arrow_length_ratio=0.3)]
    else:
        quiver_list = [None]


    def update(frame):
        """Animation update function."""
        global target_scatter
        
        if swarm.target_point is not None:
            reached, fraction = check_target_reached(
                swarm.x, swarm.target_point,
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
        
        swarm.step()
        scat._offsets3d = (swarm.x[:, 0], swarm.x[:, 1], swarm.x[:, 2])
        
        collision_mask, _, _ = check_obstacle_collisions(
            swarm.x, swarm.obstacles, collision_radius=1.0
        )
        
        speeds = np.linalg.norm(swarm.v, axis=1)
        colors = plt.cm.viridis((speeds - swarm.min_speed) / (swarm.max_speed - swarm.min_speed))
        
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
            quiver_list[0] = ax.quiver(swarm.x[:, 0], swarm.x[:, 1], swarm.x[:, 2],
                                       swarm.v[:, 0], swarm.v[:, 1], swarm.v[:, 2],
                                       length=2.0, normalize=True,
                                       color='red', alpha=0.5, arrow_length_ratio=0.3)
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        
        curr_time = frame * dt
        timer_text.set_text(f"t = {curr_time:.2f}s")
        
        if SHOW_ARROWS:
            return scat, quiver_list[0], timer_text, info_text
        else:
            return scat, timer_text, info_text


    anim = FuncAnimation(fig, update,
                        frames=int(sim_time / dt),
                        interval=30,  # 33fps
                        blit=False,
                        repeat=True)

    plt.tight_layout()
    plt.show()

