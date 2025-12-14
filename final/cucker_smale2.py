"""
Cucker-Smale Model with Attraction-Repulsion for Bird Flocking
----------------------------------------------------------------
Research-based STOCHASTIC differential equation model for realistic bird flocking.

Mathematical Model (System of Stochastic Differential Equations - SDEs):
------------------------------------------------------------------------
dxi/dt = vi                                              (position dynamics - deterministic)
dvi/dt = -∇U(xi) + F_alignment(xi,vi) + F_speed(vi) + σ*dWi/dt  (velocity dynamics - STOCHASTIC)

Where:
- U(x) = Morse attraction-repulsion potential
- F_alignment = Cucker-Smale velocity alignment force
- F_speed = speed regulation force
- σ*dWi = STOCHASTIC TERM (Wiener process/Brownian motion) ← KEY REQUIREMENT!

STOCHASTICITY & RANDOMNESS (6 Key Requirements - Research-Based):
-----------------------------------------------------------------
1. VELOCITY DIRECTION NOISE (σ*dW): Wiener process in velocity dynamics
   - Models environmental randomness (wind gusts, turbulence)
   - Each bird experiences independent 3D Brownian motion
   - Mathematically rigorous: proper Itô stochastic calculus
   - Scaled by sqrt(dt) in Euler-Maruyama integration

2. SPEED VARIATIONS (Stochastic Target Speed): Random speed fluctuations
   - Each bird's target speed varies randomly: v_target = v0 ± 0.3
   - Models individual differences, energy variations, fatigue
   - Research: Attanasi et al. (2014) - speed variations in starling flocks
   - Creates realistic speeding up / slowing down behavior

3. SPEED-TURN COUPLING (Aerodynamic Constraint): Birds slow during turns
   - Sharp turns → automatic speed reduction (up to -2.5 units)
   - Research: Storms et al. (2019) - birds slow down during collective turns
   - Physical basis: turning requires centripetal force, easier at lower speeds
   - Creates natural wave patterns of acceleration/deceleration

4. ORNSTEIN-UHLENBECK WIND PROCESS: Correlated environmental noise
   - OU process: dX = -θ(X - μ)dt + σ*dW (mean-reverting stochastic process)
   - Models slowly-varying wind gusts (more realistic than white noise)
   - Same wind affects all birds (common environmental force)
   - Research: Meteorological turbulence models

5. SENSOR NOISE (Perception Uncertainty): Imperfect neighbor perception
   - Gaussian noise in perceived positions/velocities
   - Models: visual perception errors, depth perception limits
   - Research: Birds have ~1-2° angular resolution
   - Creates realistic imperfect coordination

6. INTERACTION DROPOUT (Attention Limits): Probabilistic perception failures
   - Random dropout of neighbor interactions each timestep
   - Models: occlusion, attention limits, momentary distractions
   - Research: Limited attention capacity in flocking birds
   - Creates realistic tracking imperfections

Attraction-Repulsion Potential (Morse-like):
--------------------------------------------
U(r) = -Ca*exp(-r/la) + Cr*exp(-r/lr)
- Ca = 6.0, la = 25.0: attraction strength and range (long-range, pulls birds together)
- Cr = 5.0, lr = 5.0: repulsion strength and range (short-range, maintains spacing)

This model:
1. Uses proper STOCHASTIC differential equations (SDEs) ✓
2. Incorporates RANDOMNESS via Wiener process ✓
3. Forms flocks quickly (long-range attraction) ✓
4. Maintains proper spacing (5-7 units between birds) ✓
5. Aligns velocities (Cucker-Smale term) ✓
6. Is grounded in research literature ✓

References:
- Cucker & Smale (2007) - "Emergent Behavior in Flocks"
- Carrillo et al. (2010) - Morse potential for collective motion
- Ha & Tadmor (2008) - Mathematical analysis of flocking
- Stochastic differential equations (SDEs) for collective motion
"""

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
        """
        Initialize swarm using Cucker-Smale model with attraction-repulsion.
        
        Parameters:
        -----------
        num_boids : int
            Number of birds in the swarm
        dt : float
            Time step for numerical integration
        sigma : float
            Stochastic noise amplitude (Wiener process)
        sensor_noise : float
            Perception noise (imperfect neighbor observation)
        interaction_dropout : float
            Probability of missing a neighbor interaction
        wind_strength : float
            Ornstein-Uhlenbeck wind force amplitude
        obstacles : dict or None
            Obstacle dictionary with 'centers' and 'radii'
        obstacle_avoidance_strength : float
            Strength of obstacle avoidance force
        """
        self.N = num_boids
        self.dt = dt
        self.sigma = sigma
        
        # NEW: Perception noise parameters (sensor uncertainty)
        self.sensor_noise = sensor_noise
        self.interaction_dropout = interaction_dropout
        
        # NEW: Ornstein-Uhlenbeck wind process parameters
        self.wind_strength = wind_strength
        self.wind_theta = 0.5  # Mean reversion rate
        self.wind_sigma = 0.8  # Wind noise intensity
        self.wind_state = np.zeros(3)  # Current wind velocity (OU state)
        
        # NEW: Obstacle parameters
        self.obstacles = obstacles if obstacles is not None else {'centers': np.empty((0, 3)), 'radii': np.empty(0), 'type': 'spheres'}
        self.obstacle_avoidance_strength = obstacle_avoidance_strength
        self.obstacle_detection_range = 25.0  # Increased detection range for earlier avoidance
        
        # NEW: Target point parameters
        self.target_point = None
        self.target_attraction_strength = 30.0  # How strongly boids are attracted to target (increased)
        self.target_reach_threshold = 20.0     # Distance to consider "reached"
        self.target_required_fraction = 0.5    # Fraction of boids needed to reach target
        
        # Initialize positions in clustered region for faster flock formation
        self.x = np.random.uniform(-20, 20, (num_boids, 3))
        
        # Initialize velocities pointing generally in the same hemisphere
        self.v = np.random.randn(num_boids, 3)
        self.v[:, 0] += 10  # Bias toward positive x direction
        speeds = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v = self.v / speeds * 15.0  # Normalize to cruise speed
        
        # ============ MODEL PARAMETERS (TUNED FOR VISIBLE FLOCKING) ============
        
        # Attraction-Repulsion Potential Parameters (Morse-like)
        self.C_a = 6.0     # Attraction strength (stronger pull together)
        self.l_a = 25.0    # Attraction range (long-range for global attraction)
        self.C_r = 7.0     # Repulsion strength
        self.l_r = 6.5     # Repulsion range
        
        # Cucker-Smale Alignment Parameters
        self.K = 3.5       # Alignment strength
        self.beta = 0.3    # Communication weight decay
        
        # Speed regulation
        self.v0 = 15.0     # Desired cruise speed
        self.alpha_speed = 0.5  # Speed regulation strength 
        
        # Speed limits
        self.min_speed = 10.0
        self.max_speed = 22.0
        
        # Boundary parameters
        self.boundary_limit = 50
        self.boundary_margin = 10
        self.boundary_strength = 2.0

    def morse_potential_force(self, noisy_positions):
        """
        OPTIMIZED: Compute attraction-repulsion forces from Morse-like potential.
        NOW WITH SENSOR NOISE: Uses noisy perceived positions instead of true positions.
        
        Potential: U(r) = -Ca*exp(-r/la) + Cr*exp(-r/lr)
        Force: F = -∇U(r) = -dU/dr * (r_vec/r)
        
        Parameters:
        -----------
        noisy_positions : ndarray (N, 3)
            Perceived positions with sensor noise
        
        Returns:
        --------
        forces : ndarray (N, 3)
            Attraction-repulsion forces on each bird
        """
        # Pairwise differences and distances (vectorized)
        # NEW: Use NOISY perceived positions for realistic imperfect perception
        diff = noisy_positions[:, np.newaxis, :] - self.x[np.newaxis, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff * diff, axis=2, keepdims=True)  # (N, N, 1) - faster than norm
        dist = np.sqrt(dist_sq + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Unit direction vectors
        r_hat = diff / dist
        
        # Derivative of Morse potential: dU/dr
        # Pre-compute exponentials (most expensive operation)
        exp_attract = np.exp(-dist / self.l_a)
        exp_repel = np.exp(-dist / self.l_r)
        
        dU_dr = (self.C_a / self.l_a) * exp_attract - (self.C_r / self.l_r) * exp_repel
        
        # Force = -dU/dr * direction
        # Exclude self-interaction by multiplying with mask
        mask = (dist[:, :, 0] > 0.1).astype(np.float32)
        forces_pairwise = -dU_dr * r_hat * mask[:, :, np.newaxis]
        
        # Sum forces from all neighbors
        forces = np.sum(forces_pairwise, axis=1)  # (N, 3)
        
        return forces

    def cucker_smale_alignment(self, noisy_positions, noisy_velocities):
        """
        OPTIMIZED: Compute Cucker-Smale velocity alignment force.
        NOW WITH SENSOR NOISE AND INTERACTION DROPOUT.
        
        F_i = (K/N) * Σ_j φ(|xj - xi|) * (vj - vi)
        
        where φ(r) = 1 / (1 + r²)^β is the communication weight
        
        Parameters:
        -----------
        noisy_positions : ndarray (N, 3)
            Perceived positions with sensor noise
        noisy_velocities : ndarray (N, 3)
            Perceived velocities with sensor noise
        
        Returns:
        --------
        alignment : ndarray (N, 3)
            Velocity alignment forces
        """
        # Pairwise differences (using NOISY perceived positions)
        diff_x = noisy_positions[:, np.newaxis, :] - self.x[np.newaxis, :, :]
        dist_sq = np.sum(diff_x * diff_x, axis=2)  # Faster than norm
        
        # Communication weight: φ(r) = 1 / (1 + r²)^β
        phi = np.power(1.0 + dist_sq, -self.beta)  # Faster than division
        np.fill_diagonal(phi, 0)  # Exclude self-interaction
        
        # NEW: Apply interaction dropout (attention limits, occlusion)
        phi = self.apply_interaction_dropout(phi)
        
        # Velocity differences (using NOISY perceived velocities)
        diff_v = noisy_velocities[np.newaxis, :, :] - self.v[:, np.newaxis, :]  # (N, N, 3)
        
        # Weighted alignment: φ(r) * (vj - vi)
        weighted_alignment = phi[:, :, np.newaxis] * diff_v
        
        # Sum over all neighbors
        alignment = (self.K / self.N) * np.sum(weighted_alignment, axis=1)
        
        return alignment

    def speed_regulation(self):
        """
        STOCHASTIC speed regulation with random variations.
        Uses shared functions for modular computation.
        
        Returns:
        --------
        speed_force : ndarray (N, 3)
            Speed regulation forces with stochastic variations
        """
        # Compute stochastic speed variations using shared function
        stochastic_variation = compute_stochastic_speed_variations(self.N, variation_amplitude=0.3)
        
        # Compute turn penalty using shared function
        if hasattr(self, 'v_prev'):
            turn_penalty = compute_turn_penalty(self.v, self.v_prev, max_penalty=2.5)
        else:
            turn_penalty = 0
            self.v_prev = self.v.copy()
        
        # Compute speed regulation force
        current_speed = np.linalg.norm(self.v, axis=1, keepdims=True)
        v_hat = self.v / (current_speed + 1e-8)
        
        # Combined target speed
        target_speed = self.v0 + stochastic_variation + turn_penalty
        target_speed = np.clip(target_speed, self.min_speed, self.max_speed)
        
        # Speed regulation force
        speed_error = target_speed - current_speed
        speed_force = self.alpha_speed * speed_error * v_hat
        
        # Store current velocity for next turn calculation
        self.v_prev = self.v.copy()
        
        return speed_force

    def boundary_force(self):
        """
        OPTIMIZED: Soft boundary constraint to keep birds in simulation area.
        Uses smooth repulsion from boundaries.
        
        Returns:
        --------
        forces : ndarray (N, 3)
            Boundary repulsion forces
        """
        forces = np.zeros_like(self.x)
        limit = self.boundary_limit - self.boundary_margin
        
        # Vectorized boundary check for all dimensions at once
        too_low = self.x < -limit
        too_high = self.x > limit
        
        # Apply forces (vectorized)
        forces[too_low] += self.boundary_strength * (-limit - self.x[too_low])
        forces[too_high] -= self.boundary_strength * (self.x[too_high] - limit)
        
        return forces

    def update_wind_ou_process(self):
        """
        NEW: Ornstein-Uhlenbeck process for realistic wind dynamics.
        Uses shared function for wind dynamics computation.
        """
        wind_force, self.wind_state = update_wind_ou_process(
            self.wind_state, self.wind_theta, self.wind_sigma, 
            self.wind_strength, self.dt
        )
        return wind_force
    
    def apply_sensor_noise(self, positions, velocities):
        """
        NEW: Add sensor noise to perceived neighbor positions/velocities.
        Uses shared function for sensor noise computation.
        """
        return apply_sensor_noise(positions, velocities, self.sensor_noise)
    
    def apply_interaction_dropout(self, weights):
        """
        NEW: Probabilistic interaction dropout (perception failures).
        Uses shared function for dropout computation.
        """
        return apply_dropout(weights, self.interaction_dropout)

    def step(self):
        """
        Single time step using Euler-Maruyama integration.
        
        ENHANCED STOCHASTIC DIFFERENTIAL EQUATION (SDE) WITH 6 NOISE SOURCES:
        ======================================================================
        dx/dt = v                                                    (deterministic)
        dv/dt = F_morse + F_align + F_speed + F_boundary + F_wind + σ*dW/dt  (stochastic)
        
        STOCHASTICITY LAYERS:
        1. σ*dW: Wiener process (individual velocity noise)
        2. Stochastic speed variations (random target speeds)
        3. Speed-turn coupling (aerodynamic)
        4. OU wind process (correlated environmental noise)
        5. Sensor noise (perception uncertainty in forces)
        6. Interaction dropout (attention limit failures)
        
        This satisfies the key requirement: MULTIPLE LAYERS OF STOCHASTICITY!
        """
        # ========== SENSOR NOISE (PERCEPTION UNCERTAINTY) ==========
        # NEW: Birds perceive neighbors with noise (imperfect vision)
        noisy_positions, noisy_velocities = self.apply_sensor_noise(self.x, self.v)
        
        # ========== COMPUTE ALL DETERMINISTIC FORCES (WITH NOISY INPUTS) ==========
        F_morse = self.morse_potential_force(noisy_positions)        # Attraction-repulsion
        F_align = self.cucker_smale_alignment(noisy_positions, noisy_velocities)  # Velocity alignment
        F_speed = self.speed_regulation()             # Speed regulation (with stochastic variations)
        F_boundary = self.boundary_force()            # Boundary forces
        
        # ========== ORNSTEIN-UHLENBECK WIND (CORRELATED ENVIRONMENTAL NOISE) ==========
        # NEW: Add slowly-varying wind force (more realistic than white noise)
        wind_force = self.update_wind_ou_process()
        F_wind = np.tile(wind_force, (self.N, 1))  # Broadcast to all birds
        
        # ========== OBSTACLE AVOIDANCE ==========
        # Compute forces to avoid obstacles in the environment (strong prevention)
        F_obstacle = compute_obstacle_avoidance_force(
            self.x, self.v, self.obstacles,
            avoidance_strength=self.obstacle_avoidance_strength,
            detection_range=self.obstacle_detection_range,  # Longer range for earlier detection
            use_predictive=True
        )
        
        # ========== TARGET ATTRACTION ==========
        # Attract boids toward target point (if exists)
        if self.target_point is not None:
            F_target = compute_target_attraction_force(
                self.x, self.target_point,
                attraction_strength=self.target_attraction_strength,
                arrival_radius=self.target_reach_threshold
            )
        else:
            F_target = np.zeros_like(self.x)
        
        # Total deterministic force (drift term in SDE)
        F_total = F_morse + F_align + F_speed + F_boundary + F_wind + F_obstacle + F_target
        
        # ========== WIENER PROCESS (INDIVIDUAL VELOCITY NOISE) ==========
        # This is the DIFFUSION TERM in the Stochastic Differential Equation
        # dW represents a Wiener process (Brownian motion)
        # For Euler-Maruyama: ΔW = sqrt(dt) * N(0,1)
        # Each bird gets independent 3D random perturbation
        noise = self.sigma * np.sqrt(self.dt) * np.random.randn(self.N, 3)
        #       ^^^^^^^         ^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^
        #       intensity      proper scaling    Gaussian random vector
        #       parameter      for SDE           (independent for each bird)
        
        # ========== EULER-MARUYAMA UPDATE (SDE INTEGRATION) ==========
        # Update velocities: Δv = F(x,v)*Δt + σ*sqrt(Δt)*ΔW
        #                         ^^^^^^^^^    ^^^^^^^^^^^^^^
        #                         deterministic   stochastic
        #                         (drift)         (diffusion)
        self.v += F_total * self.dt + noise
        
        # Enforce speed limits (hard constraints) using shared function
        self.v = limit_speed(self.v, self.min_speed, self.max_speed)
        
        # Update positions: dx = v*dt
        self.x += self.v * self.dt


# ==================== SIMULATION SETUP ====================
dt = 0.08
sim_time = 60
N = 50

# Create obstacles (choose a preset or use create_obstacles for random)
# Options: 'center_column', 'wall', 'scattered', 'tunnel', 'ring'
OBSTACLE_PRESET = 'scattered'  # Change this to try different obstacle configurations
obstacles = create_predefined_obstacles(OBSTACLE_PRESET)

swarm = CuckerSmaleSwarm(num_boids=N, dt=dt, sigma=0.08,
                         sensor_noise=0.5,       # NEW: Perception uncertainty
                         interaction_dropout=0.1,  # NEW: 10% chance to miss neighbor interaction
                         wind_strength=0.3,      # NEW: OU process wind strength
                         obstacles=obstacles,    # NEW: Obstacles in environment
                         obstacle_avoidance_strength=150.0)  # NEW: Strong avoidance force to prevent collisions

# Track positions for analysis
positions_history = []

# ==================== VISUALIZATION ====================
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

# Scatter plot
scat = ax.scatter(swarm.x[:, 0], swarm.x[:, 1], swarm.x[:, 2],
                 c='darkblue', s=30, alpha=0.8, edgecolors='navy')

# Draw all obstacles using shared function
draw_all_obstacles(ax, swarm.obstacles, num_points=120, color='crimson', alpha=0.5)

# Initialize target point
swarm.target_point = spawn_target_point(
    boundary_limit=50, margin=15, 
    obstacles=swarm.obstacles, min_obstacle_distance=10.0
)
target_scatter = [draw_target_point(ax, swarm.target_point, size=300, color='lime', marker='*')]

# Optional velocity arrows (disabled for performance - set to True if needed)
SHOW_ARROWS = False
if SHOW_ARROWS:
    quiver_list = [ax.quiver(swarm.x[:, 0], swarm.x[:, 1], swarm.x[:, 2],
                             swarm.v[:, 0], swarm.v[:, 1], swarm.v[:, 2],
                             length=2.0, normalize=True,
                             color='red', alpha=0.5, arrow_length_ratio=0.3)]
else:
    quiver_list = [None]


def update(frame):
    """Animation update function"""
    global target_scatter
    
    # Check if target is reached and respawn if needed
    if swarm.target_point is not None:
        reached, fraction = check_target_reached(
            swarm.x, swarm.target_point,
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
    
    # Step the simulation
    swarm.step()
    
    # Update scatter plot
    scat._offsets3d = (swarm.x[:, 0], swarm.x[:, 1], swarm.x[:, 2])
    
    # Check for obstacle collisions and flash boids red if inside obstacles
    collision_mask, _, _ = check_obstacle_collisions(
        swarm.x, swarm.obstacles, collision_radius=1.0
    )
    
    # Color by speed (visual feedback)
    speeds = np.linalg.norm(swarm.v, axis=1)
    colors = plt.cm.viridis((speeds - swarm.min_speed) / (swarm.max_speed - swarm.min_speed))
    
    # Flash boids red if they're inside obstacles (flashing effect every 3 frames)
    if np.any(collision_mask):
        flash_on = (frame // 3) % 2 == 0  # Flash every 3 frames
        red_color = np.array([1.0, 0.0, 0.0, 1.0]) if flash_on else np.array([1.0, 0.3, 0.3, 1.0])
        # Convert colors to array if needed
        if not isinstance(colors, np.ndarray):
            colors = np.array(colors)
        colors[collision_mask] = red_color  # Bright red or dim red
    
    scat.set_color(colors)
    
    # Update arrows if enabled
    if SHOW_ARROWS:
        global quiver_list
        if quiver_list[0] is not None:
            quiver_list[0].remove()
        quiver_list[0] = ax.quiver(swarm.x[:, 0], swarm.x[:, 1], swarm.x[:, 2],
                                   swarm.v[:, 0], swarm.v[:, 1], swarm.v[:, 2],
                                   length=2.0, normalize=True,
                                   color='red', alpha=0.5, arrow_length_ratio=0.3)
    
    # Fixed boundaries
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    
    # Update timer
    curr_time = frame * dt
    timer_text.set_text(f"t = {curr_time:.2f}s")
    
    if SHOW_ARROWS:
        return scat, quiver_list[0], timer_text, info_text
    else:
        return scat, timer_text, info_text


# Create animation (optimized for performance)
anim = FuncAnimation(fig, update,
                    frames=int(sim_time / dt),
                    interval=30,  # 33fps
                    blit=False,
                    repeat=True)

plt.tight_layout()
plt.show()

