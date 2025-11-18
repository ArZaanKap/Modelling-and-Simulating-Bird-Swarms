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


class BoidSwarm:
    def __init__(self, num_boids=50, ts=0.1, sigma=0.15, k_neighbors=7, 
                 sensor_noise=0.5, neighbor_dropout=0.1, wind_strength=0.3):
        self.num_boids = num_boids
        self.ts = ts
        self.sigma = sigma  # Stochastic noise amplitude for Euler-Maruyama
        self.k_neighbors = k_neighbors  # Number of topological neighbors
        
        # NEW: Perception noise parameters (sensor uncertainty)
        self.sensor_noise = sensor_noise  # Noise in perceiving neighbor pos/vel
        self.neighbor_dropout = neighbor_dropout  # Probability of missing a neighbor
        
        # NEW: Ornstein-Uhlenbeck wind process parameters
        self.wind_strength = wind_strength  # Wind force amplitude
        self.wind_theta = 0.5  # Mean reversion rate (how quickly wind changes)
        self.wind_sigma = 0.8  # Wind noise intensity
        self.wind_state = np.zeros(3)  # Current wind velocity (OU state)
        
        # Initialize positions in a cluster (like birds starting together)
        self.positions = np.random.randn(num_boids, 3) * 5
        
        # Start with mostly aligned velocities for faster flock formation
        # Base direction: positive x-axis (all birds flying roughly same direction)
        base_direction = np.array([1.0, 0.0, 0.0])
        
        # Add small random perturbations (±15° cone around base direction)
        perturbations = np.random.randn(num_boids, 3) * 0.2
        self.velocities = base_direction + perturbations
        
        # Normalize to consistent speed
        initial_speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = self.velocities / (initial_speeds + 1e-8) * 15.0
        
        # Track visual history for smoother appearance
        self.prev_positions = self.positions.copy()
        
        # NEW: Track previous velocity for turn angle calculations
        self.v_prev = self.velocities.copy()
        
        # Aerodynamic parameters (physics-based)
        self.v_min_flight = 10.0  # Minimum speed to maintain flight (stall speed)
        self.drag_coefficient = 0.015  # Aerodynamic drag coefficient
        
        # Speed regulation parameters (behavioral + aerodynamic)
        self.v0 = 15.0  # Desired cruise speed
        self.alpha_speed = 0.6  # Speed regulation strength

    def limit_speed(self, min_speed=10.0, max_speed=25.0):
        """Limit boid speeds to realistic range (relaxed for better dynamics)"""
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        speed_clamped = np.clip(speed, min_speed, max_speed)
        factor = speed_clamped / (speed + 1e-8)
        self.velocities *= factor

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
        
        Models slowly-varying environmental wind (more realistic than white noise).
        OU process: dX = -θ(X - μ)dt + σ*dW
        
        Properties:
        - Mean reversion: wind returns to calm (μ=0)
        - Correlated over time: realistic gusts
        - Continuous but varying
        
        Research: OU processes model turbulent air flow in meteorology
        """
        # OU process discretization: X(t+dt) = X(t) + drift + diffusion
        drift = -self.wind_theta * self.wind_state * self.ts
        diffusion = self.wind_sigma * np.sqrt(self.ts) * np.random.randn(3)
        
        self.wind_state += drift + diffusion
        
        # Return wind force (scaled by strength parameter)
        return self.wind_strength * self.wind_state
    
    def apply_sensor_noise(self, positions, velocities):
        """
        NEW: Add sensor noise to perceived neighbor positions/velocities.
        
        Models:
        - Visual perception uncertainty (depth, angle errors)
        - Attention limitations (can't perfectly track all birds)
        - Environmental factors (glare, fog, distance)
        
        Research: Birds have ~1-2° angular resolution, translates to position uncertainty
        """
        # Add Gaussian noise to positions (each coordinate independently)
        noisy_positions = positions + np.random.randn(*positions.shape) * self.sensor_noise
        
        # Add Gaussian noise to velocities
        noisy_velocities = velocities + np.random.randn(*velocities.shape) * self.sensor_noise
        
        return noisy_positions, noisy_velocities
    
    def apply_neighbor_dropout(self, mask):
        """
        NEW: Probabilistic neighbor dropout (perception failures).
        
        Models:
        - Occlusion (birds blocking view of others)
        - Attention limits (can't process all neighbors simultaneously)
        - Momentary distractions
        
        Research: Birds don't perfectly track all neighbors, especially in dense flocks
        
        Parameters:
        -----------
        mask : ndarray (N, N)
            Boolean mask of which neighbors are considered
            
        Returns:
        --------
        dropped_mask : ndarray (N, N)
            Mask with some neighbors randomly dropped
        """
        # For each bird-neighbor pair, randomly drop with probability p
        dropout_mask = np.random.rand(*mask.shape) > self.neighbor_dropout
        
        # Combine with existing mask (AND operation)
        return mask & dropout_mask

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
        SEPARATION_RADIUS = 20.0  # Significantly increased for much more personal space
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
        k_elastic = 0.1  #0.5  # Spring constant (tuned for single flock cohesion)
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
        
        # 1. STOCHASTIC SPEED VARIATIONS (Attanasi et al. 2014)
        # Each bird's target speed varies randomly (individual differences, energy, fatigue)
        stochastic_speed_variation = 0.3 * np.random.randn(self.num_boids, 1)
        
        # 2. SPEED-TURN COUPLING (Storms et al. 2019)
        # Birds slow down during sharp turns (aerodynamic constraint)
        v_prev_hat = self.v_prev / (np.linalg.norm(self.v_prev, axis=1, keepdims=True) + 1e-8)
        cos_angle = np.sum(v_hat * v_prev_hat, axis=1, keepdims=True)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        # Turn magnitude: 0 (no turn) to 2 (180° turn)
        turn_magnitude = 1 - cos_angle
        
        # Speed reduction during turns: up to -2.5 units for sharp turns
        turn_penalty = -2.5 * turn_magnitude
        
        # 3. COMBINED TARGET SPEED
        target_speed = self.v0 + stochastic_speed_variation + turn_penalty
        target_speed = np.clip(target_speed, 10.0, 25.0)  # Relaxed upper limit
        
        # 4. SPEED REGULATION FORCE
        # Drive toward varying target speed (behavioral preference + aerodynamic constraint)
        speed_error = target_speed - current_speed
        speed_regulation_force = self.alpha_speed * speed_error * v_hat

        # ==================== ENVIRONMENTAL WIND (OU PROCESS) ====================
        # NEW: Add correlated wind force (more realistic than white noise)
        wind_force = self.update_wind_ou_process()
        
        # Broadcast wind to all birds (same wind affects everyone)
        wind_force_broadcasted = np.tile(wind_force, (self.num_boids, 1))

        # ==================== COMBINE FORCES ====================
        # Balanced weights for realistic flocking
        w_sep = 15.0      # Avoid collisions (increased for much more spacing)
        w_ali = 35.0       # Match neighbor velocities (increased for better alignment)
        w_coh = 10.0      # Stay with group (increased from 25 to prevent fragmentation)
        # Aerodynamic forces applied directly (physics, not weighted)
        
        # Social forces (weighted)
        social_forces = (w_sep * sep_vectors + 
                        w_ali * alignment + 
                        w_coh * cohesion)
        
        # Physical forces (not weighted - these are physics + behavioral speed regulation!)
        physical_forces = (drag_force + 
                          thrust_force + 
                          speed_regulation_force +
                          wind_force_broadcasted)
        
        # Total steering force
        steering = social_forces + physical_forces

        # ==================== UPDATE WITH EULER-MARUYAMA ====================
        # Smooth acceleration (limit turning rate) - relaxed for better flocking
        max_force = 7.0  # Increased from 4.5 - less harsh clipping allows stronger responses
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


# ==================== SIMULATION SETUP ====================
ts = 0.08  # Time step
sim_time = 60
N = 40  # Number of boids
K_NEIGHBORS = 7 #24  # Increased to 16 for single cohesive flock (prevents fragmentation)
                  # Research suggests 6-7 for real starlings, but more needed for 40 birds

swarm = BoidSwarm(N, ts, sigma=0.08, k_neighbors=K_NEIGHBORS,
                  sensor_noise=0.5,      # NEW: Perception uncertainty
                  neighbor_dropout=0.1,  # NEW: 10% chance to miss neighbor
                  wind_strength=0.3)     # NEW: OU process wind strength

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
    swarm.boids_algorithm()
    
    # Update boid positions
    scat._offsets3d = (swarm.positions[:, 0],
                       swarm.positions[:, 1],
                       swarm.positions[:, 2])
    
    # Visual feedback: color by speed (blue=slow, yellow=fast)
    speeds = np.linalg.norm(swarm.velocities, axis=1)
    colors = plt.cm.viridis((speeds - 10.0) / (25.0 - 10.0))  # Normalize to new speed range
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

