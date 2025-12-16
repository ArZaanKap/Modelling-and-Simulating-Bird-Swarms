"""
Model Comparison Script
=======================
Compares Boids and Cucker-Smale models with real bird data (Mobbing Flocks 6 and 7).
Generates key metrics, charts, and graphs for your report.

Usage:
    python model_comparison.py

Outputs:
    - plots/comparison_metrics.png     : Bar charts of key metrics
    - plots/comparison_3d.png          : 3D flock visualizations
    - plots/comparison_distributions.png : Distribution analysis
    - results/comparison_summary.csv   : Summary table
    - results/analysis_report.txt      : Detailed analysis with parameter suggestions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Create output directories
SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR / 'plots'
RESULTS_DIR = SCRIPT_DIR / 'results'
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from boids4 import BoidSwarm
from cucker_smale2 import CuckerSmaleSwarm


# ============================================================================
# DATA LOADING (Simplified from load_bird_data.py)
# ============================================================================

def get_data_path():
    """Get path to the data directory."""
    script_dir = Path(__file__).parent
    return script_dir / '../dataset/code_upload_2/code_upload_2/data'


def load_flock(flock_number):
    """Load mobbing flock data."""
    data_path = get_data_path()
    
    if flock_number == 4:
        filename = 'mob_04_p1.mat'
    elif flock_number == 5:
        filename = 'mob_05_p1.mat'
    else:
        filename = f'mob_{flock_number:02d}.mat'
    
    mat_data = loadmat(str(data_path / filename))
    tracks = mat_data.get('tracks_filt', list(mat_data.values())[-1])
    
    return {
        'positions': tracks[:, 1:4],
        'velocities': tracks[:, 5:8],
        'times': tracks[:, 4],
        'bird_ids': tracks[:, 0].astype(int),
        'num_birds': len(np.unique(tracks[:, 0])),
        'time_range': (tracks[:, 4].min(), tracks[:, 4].max()),
        'filename': filename
    }


def get_flock_at_time(data, time):
    """Get flock state at a specific time."""
    time_idx = np.argmin(np.abs(data['times'] - time))
    actual_time = data['times'][time_idx]
    mask = data['times'] == actual_time
    return data['positions'][mask], data['velocities'][mask]


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(positions, velocities):
    """
    Compute key flocking metrics.
    
    Returns:
    --------
    metrics : dict
        - mean_speed: Average speed (m/s)
        - std_speed: Speed standard deviation
        - polarization: Order parameter (0-1)
        - mean_neighbor_dist: Mean distance to 7 nearest neighbors
        - dist_to_75pct: Mean distance to reach 75% of flock members
        - density: Birds per cubic meter
    """
    # Speed statistics
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    
    # Polarization (alignment order parameter)
    v_normalized = velocities / (speeds[:, np.newaxis] + 1e-8)
    polarization = np.linalg.norm(np.mean(v_normalized, axis=0))
    
    # Neighbor distances (7-nearest neighbors)
    if len(positions) > 1:
        dist_matrix = squareform(pdist(positions))
        k = min(7, len(positions) - 1)
        neighbor_dists = []
        for i in range(len(positions)):
            sorted_dists = np.sort(dist_matrix[i])[1:k+1]
            neighbor_dists.extend(sorted_dists)
        mean_neighbor_dist = np.mean(neighbor_dists)
        
        # Distance to reach 75% of flock
        # For each bird, find distance needed to include 75% of other birds
        n_birds = len(positions)
        n_75pct = int(np.ceil(0.75 * (n_birds - 1)))  # 75% of other birds
        dist_to_75pct_list = []
        for i in range(n_birds):
            sorted_dists = np.sort(dist_matrix[i])[1:]  # Exclude self
            if len(sorted_dists) >= n_75pct:
                dist_to_75pct_list.append(sorted_dists[n_75pct - 1])
        dist_to_75pct = np.mean(dist_to_75pct_list) if dist_to_75pct_list else 0
    else:
        mean_neighbor_dist = 0
        dist_to_75pct = 0
    
    # Density (birds per volume)
    center = np.mean(positions, axis=0)
    distances_from_center = np.linalg.norm(positions - center, axis=1)
    radius = np.percentile(distances_from_center, 95)
    volume = (4/3) * np.pi * (radius ** 3) if radius > 0 else 1
    density = len(positions) / volume
    
    return {
        'mean_speed': mean_speed,
        'std_speed': std_speed,
        'polarization': polarization,
        'mean_neighbor_dist': mean_neighbor_dist,
        'dist_to_75pct': dist_to_75pct,
        'density': density,
        'num_agents': len(positions)
    }


# ============================================================================
# SIMULATION RUNNERS
# ============================================================================

def run_boids_simulation(num_boids=50, num_steps=400, dt=0.08):
    """
    Run Boids simulation and return final state metrics.
    
    NOTE: Runs WITHOUT obstacles or targets for fair comparison with real data.
    Uses INCREASED noise parameters for realistic variability (matching real CV).
    
    Returns: positions_history, velocities_history, swarm (for parameter extraction)
    """
    # Explicitly set empty obstacles and no target for fair comparison
    empty_obstacles = {'centers': np.empty((0, 3)), 'radii': np.empty(0), 'type': 'spheres'}
    
    # Noise parameters tuned for realistic variability
    # Real flock has: Speed CV=18.7%, Polar CV=28.3%, NeighDist CV=58.5%
    swarm = BoidSwarm(
        num_boids=num_boids, ts=dt, sigma=0.15, k_neighbors=7,
        sensor_noise=1.0, neighbor_dropout=0.15, wind_strength=0.5,
        obstacles=empty_obstacles,  # No obstacles for fair test
        obstacle_avoidance_strength=0.0  # Disable obstacle avoidance
    )
    # Ensure no target point is set
    swarm.target_point = None
    
    positions_history = []
    velocities_history = []
    
    for _ in range(num_steps):
        swarm.boids_algorithm()
        positions_history.append(swarm.positions.copy())
        velocities_history.append(swarm.velocities.copy())
    
    return positions_history, velocities_history, swarm


def run_cucker_smale_simulation(num_boids=50, num_steps=400, dt=0.08):
    """
    Run Cucker-Smale simulation and return final state metrics.
    
    NOTE: Runs WITHOUT obstacles or targets for fair comparison with real data.
    Uses INCREASED noise parameters for realistic variability (matching real CV).
    
    Returns: positions_history, velocities_history, swarm (for parameter extraction)
    """
    # Explicitly set empty obstacles and no target for fair comparison
    empty_obstacles = {'centers': np.empty((0, 3)), 'radii': np.empty(0), 'type': 'spheres'}
    
    # Noise parameters tuned for realistic variability
    # Real flock has: Speed CV=18.7%, Polar CV=28.3%, NeighDist CV=58.5%
    swarm = CuckerSmaleSwarm(
        num_boids=num_boids, dt=dt, sigma=0.15,
        sensor_noise=1.0, interaction_dropout=0.15, wind_strength=0.5,
        obstacles=empty_obstacles,  # No obstacles for fair test
        obstacle_avoidance_strength=0.0  # Disable obstacle avoidance
    )
    # Ensure no target point is set
    swarm.target_point = None
    
    positions_history = []
    velocities_history = []
    
    for _ in range(num_steps):
        swarm.step()
        positions_history.append(swarm.x.copy())
        velocities_history.append(swarm.v.copy())
    
    return positions_history, velocities_history, swarm


# ============================================================================
# COMPARISON AND VISUALIZATION
# ============================================================================

def compare_all(flock_number=6):
    """
    Run full comparison between models and real data for a single flock.
    
    Uses TIME-AVERAGED metrics for real data (not a single snapshot) for
    proper comparison with simulation averages.
    
    Parameters:
    -----------
    flock_number : int
        Which mobbing flock to compare against (default: 6)
    """
    print("="*70)
    print(f"MODEL COMPARISON: Boids & Cucker-Smale vs Real Flock {flock_number}")
    print("="*70)
    
    results = {}
    
    # =========================================================================
    # LOAD REAL DATA AND COMPUTE TIME-AVERAGED METRICS
    # =========================================================================
    print(f"\nLoading Mobbing Flock {flock_number} (computing time-averaged metrics)...")
    data = load_flock(flock_number)
    
    # Get unique time points
    unique_times = np.sort(np.unique(data['times']))
    
    # Compute metrics at each time frame
    real_metrics_list = []
    for t in unique_times:
        pos, vel = get_flock_at_time(data, t)
        if len(pos) >= 3:
            real_metrics_list.append(compute_metrics(pos, vel))
    
    # Time-average all metrics
    real_metrics = {
        key: np.mean([m[key] for m in real_metrics_list])
        for key in real_metrics_list[0].keys()
    }
    # Also store std for each metric
    real_metrics_std = {
        key: np.std([m[key] for m in real_metrics_list])
        for key in real_metrics_list[0].keys()
    }
    
    # Get a representative snapshot for 3D visualization (middle time)
    mid_time = (data['time_range'][0] + data['time_range'][1]) / 2
    positions, velocities = get_flock_at_time(data, mid_time)
    
    results['real_flock'] = {
        'metrics': real_metrics,
        'metrics_std': real_metrics_std,
        'positions': positions,  # For visualization only
        'velocities': velocities,
        'num_birds': data['num_birds'],
        'flock_number': flock_number,
        'num_frames': len(real_metrics_list),
        'duration': unique_times[-1] - unique_times[0]
    }
    print(f"  ✓ {int(real_metrics['num_agents'])} birds over {len(real_metrics_list)} frames ({results['real_flock']['duration']:.1f}s)")
    print(f"  ✓ Time-averaged: Speed {real_metrics['mean_speed']:.2f}±{real_metrics_std['mean_speed']:.2f} m/s, "
          f"Polar {real_metrics['polarization']:.3f}±{real_metrics_std['polarization']:.3f}")
    
    # Use same number of agents as real flock
    num_boids = int(real_metrics['num_agents'])
    print(f"\nRunning simulations with {num_boids} agents (matching real flock)...")
    
    # Boids simulation (average of 3 runs)
    print("  Running Boids model (3 runs)...")
    boids_metrics_list = []
    boids_final_pos = None
    boids_final_vel = None
    boids_swarm = None
    for run in range(3):
        pos_hist, vel_hist, swarm = run_boids_simulation(num_boids=num_boids)
        # Use last 50% for steady state
        start_idx = len(pos_hist) // 2
        for i in range(start_idx, len(pos_hist)):
            boids_metrics_list.append(compute_metrics(pos_hist[i], vel_hist[i]))
        boids_final_pos = pos_hist[-1]
        boids_final_vel = vel_hist[-1]
        boids_swarm = swarm  # Keep last swarm for parameter extraction
    
    # Average metrics
    boids_metrics = {
        key: np.mean([m[key] for m in boids_metrics_list])
        for key in boids_metrics_list[0].keys()
    }
    results['boids'] = {
        'metrics': boids_metrics,
        'positions': boids_final_pos,
        'velocities': boids_final_vel,
        'swarm': boids_swarm  # Store swarm for parameter access
    }
    print(f"    ✓ Mean speed: {boids_metrics['mean_speed']:.2f} m/s, Polarization: {boids_metrics['polarization']:.3f}")
    
    # Cucker-Smale simulation (average of 3 runs)
    print("  Running Cucker-Smale model (3 runs)...")
    cs_metrics_list = []
    cs_final_pos = None
    cs_final_vel = None
    cs_swarm = None
    for run in range(3):
        pos_hist, vel_hist, swarm = run_cucker_smale_simulation(num_boids=num_boids)
        start_idx = len(pos_hist) // 2
        for i in range(start_idx, len(pos_hist)):
            cs_metrics_list.append(compute_metrics(pos_hist[i], vel_hist[i]))
        cs_final_pos = pos_hist[-1]
        cs_final_vel = vel_hist[-1]
        cs_swarm = swarm  # Keep last swarm for parameter extraction
    
    cs_metrics = {
        key: np.mean([m[key] for m in cs_metrics_list])
        for key in cs_metrics_list[0].keys()
    }
    results['cucker_smale'] = {
        'metrics': cs_metrics,
        'positions': cs_final_pos,
        'velocities': cs_final_vel,
        'swarm': cs_swarm  # Store swarm for parameter access
    }
    print(f"    ✓ Mean speed: {cs_metrics['mean_speed']:.2f} m/s, Polarization: {cs_metrics['polarization']:.3f}")
    
    return results


def print_comparison_table(results):
    """Print formatted comparison table with time-averaged metrics."""
    print("\n" + "="*90)
    print("COMPARISON TABLE (Time-Averaged Metrics)")
    print("="*90)
    
    flock_num = results['real_flock']['flock_number']
    duration = results['real_flock'].get('duration', 0)
    num_frames = results['real_flock'].get('num_frames', 1)
    
    print(f"Real data averaged over {num_frames} frames ({duration:.1f}s)")
    print(f"Simulation data averaged over last 50% of 400 steps (3 runs)")
    print()
    
    headers = ['Source', 'N', 'Speed (m/s)', 'Speed Std', 'Polarization', 'Neighbor Dist', 'Density']
    row_format = "{:<20} {:>5} {:>12} {:>10} {:>12} {:>14} {:>12}"
    
    print(row_format.format(*headers))
    print("-"*90)
    
    rows = []
    for key in ['real_flock', 'boids', 'cucker_smale']:
        m = results[key]['metrics']
        name = {
            'real_flock': f'Real Flock {flock_num}',
            'boids': 'Boids Model',
            'cucker_smale': 'Cucker-Smale'
        }[key]
        
        row = [
            name,
            int(m['num_agents']),
            f"{m['mean_speed']:.2f}",
            f"{m['std_speed']:.2f}",
            f"{m['polarization']:.3f}",
            f"{m['mean_neighbor_dist']:.2f}",
            f"{m['density']:.6f}"
        ]
        rows.append(row)
        print(row_format.format(*row))
    
    print("-"*90)
    
    # Calculate errors vs real data
    real_m = results['real_flock']['metrics']
    
    print(f"\nERROR ANALYSIS (vs Real Flock {flock_num} Time-Averaged):")
    print("-"*60)
    
    for model in ['boids', 'cucker_smale']:
        m = results[model]['metrics']
        speed_err = abs(m['mean_speed'] - real_m['mean_speed']) / real_m['mean_speed'] * 100
        pol_err = abs(m['polarization'] - real_m['polarization']) / real_m['polarization'] * 100
        dist_err = abs(m['mean_neighbor_dist'] - real_m['mean_neighbor_dist']) / real_m['mean_neighbor_dist'] * 100
        
        name = 'Boids' if model == 'boids' else 'Cucker-Smale'
        print(f"\n{name}:")
        print(f"  Speed Error:     {speed_err:>6.1f}%")
        print(f"  Polarization Error: {pol_err:>6.1f}%")
        print(f"  Neighbor Dist Error: {dist_err:>6.1f}%")
    
    return rows


def plot_comparison_bar_charts(results):
    """Generate individual bar chart comparisons of key metrics."""
    print("\nGenerating individual metric bar charts...")
    
    flock_num = results['real_flock']['flock_number']
    
    # Prepare data
    labels = [f'Real Flock {flock_num}', 'Boids', 'Cucker-Smale']
    keys = ['real_flock', 'boids', 'cucker_smale']
    colors = ['#2E86AB', '#28A745', '#DC3545']
    
    metrics_to_plot = [
        ('mean_speed', 'Mean Speed (m/s)', 'mean_speed'),
        ('polarization', 'Polarization (Order Parameter)', 'polarization'),
        ('mean_neighbor_dist', 'Mean Neighbor Distance (m)', 'neighbor_distance'),
        ('std_speed', 'Speed Variability (Std Dev)', 'speed_std')
    ]
    
    # Create individual plots for each metric
    for metric, title, filename in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        values = [results[k]['metrics'][metric] for k in keys]
        
        bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_ylabel(title.split('(')[0].strip())
        ax.set_title(f'{title}\nComparison with Real Bird Data (Flock {flock_num})', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}' if metric != 'polarization' else f'{val:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'{filename}_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {PLOTS_DIR / f'{filename}_comparison.png'}")
        plt.close()
    
    # Also create a combined 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model Comparison Overview\n(Mobbing Flock {flock_num})', 
                 fontsize=14, fontweight='bold')
    
    for idx, (metric, title, _) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        values = [results[k]['metrics'][metric] for k in keys]
        
        bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_ylabel(title.split('(')[0].strip())
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}' if metric != 'polarization' else f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xticklabels(labels, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'metrics_overview.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'metrics_overview.png'}")
    plt.close()
    
    return fig


def plot_3d_comparison(results):
    """Generate individual 3D comparison visualizations for each source."""
    print("\nGenerating individual 3D plots...")
    
    flock_num = results['real_flock']['flock_number']
    
    plots = [
        ('real_flock', f'Real Flock {flock_num}', '#2E86AB', 'real_flock_3d'),
        ('boids', 'Boids Model', '#28A745', 'boids_3d'),
        ('cucker_smale', 'Cucker-Smale Model', '#DC3545', 'cucker_smale_3d')
    ]
    
    # Individual 3D plots
    for key, title, color, filename in plots:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        pos = results[key]['positions']
        vel = results[key]['velocities']
        speeds = np.linalg.norm(vel, axis=1)
        
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c=speeds, cmap='viridis', s=50, alpha=0.7)
        
        # Add velocity arrows (subsampled)
        step = max(1, len(pos) // 20)
        scale = 0.4
        ax.quiver(pos[::step, 0], pos[::step, 1], pos[::step, 2],
                 vel[::step, 0]*scale, vel[::step, 1]*scale, vel[::step, 2]*scale,
                 color='red', alpha=0.6, arrow_length_ratio=0.3)
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title(f'{title}\n({len(pos)} agents)', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(sc, ax=ax, label='Speed (m/s)', shrink=0.7, pad=0.1)
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {PLOTS_DIR / f'{filename}.png'}")
        plt.close()
    
    # Combined 3D comparison
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('3D Flock Structure Comparison', fontsize=14, fontweight='bold')
    
    for idx, (key, title, color, _) in enumerate(plots):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        pos = results[key]['positions']
        vel = results[key]['velocities']
        speeds = np.linalg.norm(vel, axis=1)
        
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c=speeds, cmap='viridis', s=30, alpha=0.7)
        
        step = max(1, len(pos) // 15)
        scale = 0.3
        ax.quiver(pos[::step, 0], pos[::step, 1], pos[::step, 2],
                 vel[::step, 0]*scale, vel[::step, 1]*scale, vel[::step, 2]*scale,
                 color='red', alpha=0.5, arrow_length_ratio=0.3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{title}\n({len(pos)} agents)')
        
        plt.colorbar(sc, ax=ax, label='Speed (m/s)', shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'flock_3d_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'flock_3d_comparison.png'}")
    plt.close()
    
    return fig


def plot_metric_distributions(results):
    """Plot individual distribution charts for key metrics."""
    print("\nGenerating individual distribution plots...")
    
    flock_num = results['real_flock']['flock_number']
    real_m = results['real_flock']['metrics']
    
    # 1. Speed Distribution Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, color in [('real_flock', f'Real Flock {flock_num}', '#2E86AB'),
                               ('boids', 'Boids Model', '#28A745'),
                               ('cucker_smale', 'Cucker-Smale Model', '#DC3545')]:
        vel = results[key]['velocities']
        speeds = np.linalg.norm(vel, axis=1)
        ax.hist(speeds, bins=20, alpha=0.5, label=label, color=color, density=True, edgecolor='black')
    ax.set_xlabel('Speed (m/s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Speed Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'speed_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'speed_distribution.png'}")
    plt.close()
    
    # 2. Neighbor Distance Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, color in [('real_flock', f'Real Flock {flock_num}', '#2E86AB'),
                               ('boids', 'Boids Model', '#28A745'),
                               ('cucker_smale', 'Cucker-Smale Model', '#DC3545')]:
        pos = results[key]['positions']
        dist_matrix = squareform(pdist(pos))
        k = min(7, len(pos) - 1)
        neighbor_dists = []
        for i in range(len(pos)):
            sorted_dists = np.sort(dist_matrix[i])[1:k+1]
            neighbor_dists.extend(sorted_dists)
        ax.hist(neighbor_dists, bins=30, alpha=0.5, label=label, color=color, density=True, edgecolor='black')
    ax.set_xlabel('Neighbor Distance (m)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('7-Nearest Neighbor Distance Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'neighbor_distance_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'neighbor_distance_distribution.png'}")
    plt.close()
    
    # 3. Model Error Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_for_error = ['mean_speed', 'std_speed', 'polarization', 'mean_neighbor_dist']
    metric_labels = ['Mean Speed', 'Speed Std', 'Polarization', 'Neighbor Dist']
    x = np.arange(len(metrics_for_error))
    width = 0.35
    
    boids_errors = []
    cs_errors = []
    for metric in metrics_for_error:
        boids_errors.append(abs(results['boids']['metrics'][metric] - real_m[metric]) / real_m[metric] * 100)
        cs_errors.append(abs(results['cucker_smale']['metrics'][metric] - real_m[metric]) / real_m[metric] * 100)
    
    bars1 = ax.bar(x - width/2, boids_errors, width, label='Boids Model', color='#28A745', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, cs_errors, width, label='Cucker-Smale Model', color='#DC3545', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Model Errors vs Real Bird Data', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, boids_errors):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, cs_errors):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'error_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'error_comparison.png'}")
    plt.close()
    
    # 4. Radar/Spider Chart for Model Comparison
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    categories = ['Speed Match', 'Speed Var Match', 'Polarization', 'Spacing Match']
    # Convert errors to "match scores" (100 - error, capped at 0-100)
    boids_scores = [max(0, 100 - e) for e in boids_errors]
    cs_scores = [max(0, 100 - e) for e in cs_errors]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    boids_scores_plot = boids_scores + [boids_scores[0]]
    cs_scores_plot = cs_scores + [cs_scores[0]]
    angles += angles[:1]
    
    ax.plot(angles, boids_scores_plot, 'o-', linewidth=2, label='Boids', color='#28A745')
    ax.fill(angles, boids_scores_plot, alpha=0.25, color='#28A745')
    ax.plot(angles, cs_scores_plot, 's-', linewidth=2, label='Cucker-Smale', color='#DC3545')
    ax.fill(angles, cs_scores_plot, alpha=0.25, color='#DC3545')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title('Model Performance Radar\n(Higher = Better Match to Real Data)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'performance_radar.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'performance_radar.png'}")
    plt.close()
    
    # 5. Summary Statistics Table as Image
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    boids_m = results['boids']['metrics']
    cs_m = results['cucker_smale']['metrics']
    
    table_data = [
        ['Metric', f'Real Flock {flock_num}', 'Boids Model', 'Cucker-Smale', 'Boids Error', 'C-S Error'],
        ['Mean Speed (m/s)', f'{real_m["mean_speed"]:.2f}', f'{boids_m["mean_speed"]:.2f}', f'{cs_m["mean_speed"]:.2f}', 
         f'{boids_errors[0]:.1f}%', f'{cs_errors[0]:.1f}%'],
        ['Speed Std (m/s)', f'{real_m["std_speed"]:.2f}', f'{boids_m["std_speed"]:.2f}', f'{cs_m["std_speed"]:.2f}',
         f'{boids_errors[1]:.1f}%', f'{cs_errors[1]:.1f}%'],
        ['Polarization', f'{real_m["polarization"]:.3f}', f'{boids_m["polarization"]:.3f}', f'{cs_m["polarization"]:.3f}',
         f'{boids_errors[2]:.1f}%', f'{cs_errors[2]:.1f}%'],
        ['Neighbor Dist (m)', f'{real_m["mean_neighbor_dist"]:.2f}', f'{boids_m["mean_neighbor_dist"]:.2f}', f'{cs_m["mean_neighbor_dist"]:.2f}',
         f'{boids_errors[3]:.1f}%', f'{cs_errors[3]:.1f}%'],
        ['Avg Error', '-', '-', '-', f'{np.mean(boids_errors):.1f}%', f'{np.mean(cs_errors):.1f}%']
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.18, 0.15, 0.15, 0.15, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style last row (averages)
    for i in range(6):
        table[(5, i)].set_facecolor('#E2EFDA')
        table[(5, i)].set_text_props(fontweight='bold')
    
    ax.set_title('Model Comparison Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'summary_table.png'}")
    plt.close()
    
    # 6. Velocity Direction Alignment Visualization (2D projection)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Velocity Direction Alignment (Top-Down View)', fontsize=14, fontweight='bold')
    
    for idx, (key, title, color) in enumerate([('real_flock', f'Real Flock {flock_num}', '#2E86AB'),
                                                ('boids', 'Boids Model', '#28A745'),
                                                ('cucker_smale', 'Cucker-Smale Model', '#DC3545')]):
        ax = axes[idx]
        pos = results[key]['positions']
        vel = results[key]['velocities']
        
        # Normalize velocities for direction arrows
        speeds = np.linalg.norm(vel, axis=1, keepdims=True)
        vel_norm = vel / (speeds + 1e-8)
        
        ax.scatter(pos[:, 0], pos[:, 1], c=color, s=30, alpha=0.6)
        ax.quiver(pos[:, 0], pos[:, 1], vel_norm[:, 0], vel_norm[:, 1],
                 color=color, alpha=0.7, scale=25)
        
        pol = results[key]['metrics']['polarization']
        ax.set_title(f'{title}\nPolarization: {pol:.3f}', fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'velocity_alignment.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'velocity_alignment.png'}")
    plt.close()
    
    return fig


def plot_time_series(num_boids=97, num_steps=400, dt=0.08, flock_number=6):
    """
    Run simulations and plot time series of key metrics to show convergence.
    Now includes time-varying metrics from real flock data for fair comparison.
    """
    print("\nGenerating time series convergence plots...")
    
    # =========================================================================
    # LOAD REAL FLOCK DATA AND COMPUTE TIME-VARYING METRICS
    # =========================================================================
    print("  Loading real flock data for time-varying metrics...")
    real_data = load_flock(flock_number)
    
    # Get unique time points from real data
    unique_times = np.sort(np.unique(real_data['times']))
    
    real_speeds = []
    real_polarizations = []
    real_neighbor_dists = []
    real_times_normalized = []
    
    for t in unique_times:
        pos, vel = get_flock_at_time(real_data, t)
        if len(pos) >= 3:  # Need at least a few birds
            metrics = compute_metrics(pos, vel)
            real_speeds.append(metrics['mean_speed'])
            real_polarizations.append(metrics['polarization'])
            real_neighbor_dists.append(metrics['mean_neighbor_dist'])
            real_times_normalized.append(t - unique_times[0])  # Normalize to start at 0
    
    real_times_normalized = np.array(real_times_normalized)
    real_speeds = np.array(real_speeds)
    real_polarizations = np.array(real_polarizations)
    real_neighbor_dists = np.array(real_neighbor_dists)
    
    print(f"  Real data: {len(real_times_normalized)} time points over {real_times_normalized[-1]:.2f}s")
    print(f"  Real speed range: {real_speeds.min():.2f} - {real_speeds.max():.2f} m/s (mean: {real_speeds.mean():.2f})")
    print(f"  Real polarization range: {real_polarizations.min():.3f} - {real_polarizations.max():.3f}")
    print(f"  Real neighbor dist range: {real_neighbor_dists.min():.2f} - {real_neighbor_dists.max():.2f} m")
    
    # =========================================================================
    # RUN SIMULATIONS (with increased noise for realistic variability)
    # =========================================================================
    empty_obstacles = {'centers': np.empty((0, 3)), 'radii': np.empty(0), 'type': 'spheres'}
    
    # Boids (noise tuned for realistic CV)
    boids_swarm = BoidSwarm(
        num_boids=num_boids, ts=dt, sigma=0.15, k_neighbors=7,
        sensor_noise=1.0, neighbor_dropout=0.15, wind_strength=0.5,
        obstacles=empty_obstacles, obstacle_avoidance_strength=0.0
    )
    boids_swarm.target_point = None
    
    boids_speeds = []
    boids_polarizations = []
    boids_neighbor_dists = []
    
    for step in range(num_steps):
        boids_swarm.boids_algorithm()
        metrics = compute_metrics(boids_swarm.positions, boids_swarm.velocities)
        boids_speeds.append(metrics['mean_speed'])
        boids_polarizations.append(metrics['polarization'])
        boids_neighbor_dists.append(metrics['mean_neighbor_dist'])
    
    # Cucker-Smale (noise tuned for realistic CV)
    cs_swarm = CuckerSmaleSwarm(
        num_boids=num_boids, dt=dt, sigma=0.15,
        sensor_noise=1.0, interaction_dropout=0.15, wind_strength=0.5,
        obstacles=empty_obstacles, obstacle_avoidance_strength=0.0
    )
    cs_swarm.target_point = None
    
    cs_speeds = []
    cs_polarizations = []
    cs_neighbor_dists = []
    
    for step in range(num_steps):
        cs_swarm.step()
        metrics = compute_metrics(cs_swarm.x, cs_swarm.v)
        cs_speeds.append(metrics['mean_speed'])
        cs_polarizations.append(metrics['polarization'])
        cs_neighbor_dists.append(metrics['mean_neighbor_dist'])
    
    sim_time = np.arange(num_steps) * dt
    
    # =========================================================================
    # PLOT 1: SPEED TIME SERIES
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(real_times_normalized, real_speeds, label='Real Flock Data', color='#2E86AB', linewidth=2.5, alpha=0.9)
    ax.plot(sim_time, boids_speeds, label='Boids Model', color='#28A745', linewidth=2, alpha=0.8)
    ax.plot(sim_time, cs_speeds, label='Cucker-Smale Model', color='#DC3545', linewidth=2, alpha=0.8)
    
    # Add shaded region for real data range
    ax.axhline(y=np.mean(real_speeds), color='#2E86AB', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between([0, max(sim_time[-1], real_times_normalized[-1])], 
                    np.mean(real_speeds) - np.std(real_speeds),
                    np.mean(real_speeds) + np.std(real_speeds),
                    color='#2E86AB', alpha=0.1, label=f'Real ±1σ ({np.std(real_speeds):.2f} m/s)')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mean Speed (m/s)', fontsize=12)
    ax.set_title('Speed Dynamics: Real Data vs Simulations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'speed_convergence.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'speed_convergence.png'}")
    plt.close()
    
    # =========================================================================
    # PLOT 2: POLARIZATION TIME SERIES
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(real_times_normalized, real_polarizations, label='Real Flock Data', color='#2E86AB', linewidth=2.5, alpha=0.9)
    ax.plot(sim_time, boids_polarizations, label='Boids Model', color='#28A745', linewidth=2, alpha=0.8)
    ax.plot(sim_time, cs_polarizations, label='Cucker-Smale Model', color='#DC3545', linewidth=2, alpha=0.8)
    
    ax.axhline(y=np.mean(real_polarizations), color='#2E86AB', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between([0, max(sim_time[-1], real_times_normalized[-1])], 
                    np.mean(real_polarizations) - np.std(real_polarizations),
                    np.mean(real_polarizations) + np.std(real_polarizations),
                    color='#2E86AB', alpha=0.1, label=f'Real ±1σ ({np.std(real_polarizations):.3f})')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Polarization', fontsize=12)
    ax.set_title('Polarization (Alignment) Dynamics: Real Data vs Simulations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'polarization_convergence.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'polarization_convergence.png'}")
    plt.close()
    
    # =========================================================================
    # PLOT 3: NEIGHBOR DISTANCE TIME SERIES
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(real_times_normalized, real_neighbor_dists, label='Real Flock Data', color='#2E86AB', linewidth=2.5, alpha=0.9)
    ax.plot(sim_time, boids_neighbor_dists, label='Boids Model', color='#28A745', linewidth=2, alpha=0.8)
    ax.plot(sim_time, cs_neighbor_dists, label='Cucker-Smale Model', color='#DC3545', linewidth=2, alpha=0.8)
    
    ax.axhline(y=np.mean(real_neighbor_dists), color='#2E86AB', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between([0, max(sim_time[-1], real_times_normalized[-1])], 
                    np.mean(real_neighbor_dists) - np.std(real_neighbor_dists),
                    np.mean(real_neighbor_dists) + np.std(real_neighbor_dists),
                    color='#2E86AB', alpha=0.1, label=f'Real ±1σ ({np.std(real_neighbor_dists):.2f} m)')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mean Neighbor Distance (m)', fontsize=12)
    ax.set_title('Neighbor Distance Dynamics: Real Data vs Simulations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'neighbor_distance_convergence.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'neighbor_distance_convergence.png'}")
    plt.close()
    
    # =========================================================================
    # PLOT 4: COMBINED OVERVIEW
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Time-Varying Metrics: Real Data vs Simulations', fontsize=14, fontweight='bold')
    
    # Speed
    axes[0].plot(real_times_normalized, real_speeds, label='Real', color='#2E86AB', linewidth=2)
    axes[0].plot(sim_time, boids_speeds, label='Boids', color='#28A745', linewidth=1.5, alpha=0.8)
    axes[0].plot(sim_time, cs_speeds, label='C-S', color='#DC3545', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Mean Speed (m/s)')
    axes[0].set_title('Speed')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Polarization
    axes[1].plot(real_times_normalized, real_polarizations, label='Real', color='#2E86AB', linewidth=2)
    axes[1].plot(sim_time, boids_polarizations, label='Boids', color='#28A745', linewidth=1.5, alpha=0.8)
    axes[1].plot(sim_time, cs_polarizations, label='C-S', color='#DC3545', linewidth=1.5, alpha=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Polarization')
    axes[1].set_title('Polarization')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    # Neighbor Distance
    axes[2].plot(real_times_normalized, real_neighbor_dists, label='Real', color='#2E86AB', linewidth=2)
    axes[2].plot(sim_time, boids_neighbor_dists, label='Boids', color='#28A745', linewidth=1.5, alpha=0.8)
    axes[2].plot(sim_time, cs_neighbor_dists, label='C-S', color='#DC3545', linewidth=1.5, alpha=0.8)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Neighbor Distance (m)')
    axes[2].set_title('Neighbor Distance')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'convergence_overview.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'convergence_overview.png'}")
    plt.close()
    
    # =========================================================================
    # PLOT 5: VARIABILITY COMPARISON (NEW)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate coefficient of variation (CV) for each metric and source
    metrics_names = ['Speed', 'Polarization', 'Neighbor Dist']
    x_pos = np.arange(len(metrics_names))
    width = 0.25
    
    # CV = std/mean * 100 (as percentage)
    real_cv = [
        np.std(real_speeds) / np.mean(real_speeds) * 100,
        np.std(real_polarizations) / np.mean(real_polarizations) * 100,
        np.std(real_neighbor_dists) / np.mean(real_neighbor_dists) * 100
    ]
    boids_cv = [
        np.std(boids_speeds) / np.mean(boids_speeds) * 100,
        np.std(boids_polarizations) / np.mean(boids_polarizations) * 100,
        np.std(boids_neighbor_dists) / np.mean(boids_neighbor_dists) * 100
    ]
    cs_cv = [
        np.std(cs_speeds) / np.mean(cs_speeds) * 100,
        np.std(cs_polarizations) / np.mean(cs_polarizations) * 100,
        np.std(cs_neighbor_dists) / np.mean(cs_neighbor_dists) * 100
    ]
    
    bars1 = ax.bar(x_pos - width, real_cv, width, label='Real Flock', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x_pos, boids_cv, width, label='Boids Model', color='#28A745', alpha=0.8)
    bars3 = ax.bar(x_pos + width, cs_cv, width, label='Cucker-Smale', color='#DC3545', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax.set_title('Temporal Variability Comparison\n(Lower = More Stable)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'variability_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'variability_comparison.png'}")
    plt.close()
    
    # Return summary statistics for the report
    return {
        'real': {
            'speed_mean': np.mean(real_speeds), 'speed_std': np.std(real_speeds),
            'polarization_mean': np.mean(real_polarizations), 'polarization_std': np.std(real_polarizations),
            'neighbor_dist_mean': np.mean(real_neighbor_dists), 'neighbor_dist_std': np.std(real_neighbor_dists),
            'duration': real_times_normalized[-1], 'num_frames': len(real_times_normalized)
        },
        'boids': {
            'speed_mean': np.mean(boids_speeds), 'speed_std': np.std(boids_speeds),
            'polarization_mean': np.mean(boids_polarizations), 'polarization_std': np.std(boids_polarizations),
            'neighbor_dist_mean': np.mean(boids_neighbor_dists), 'neighbor_dist_std': np.std(boids_neighbor_dists)
        },
        'cucker_smale': {
            'speed_mean': np.mean(cs_speeds), 'speed_std': np.std(cs_speeds),
            'polarization_mean': np.mean(cs_polarizations), 'polarization_std': np.std(cs_polarizations),
            'neighbor_dist_mean': np.mean(cs_neighbor_dists), 'neighbor_dist_std': np.std(cs_neighbor_dists)
        }
    }


def save_summary_csv(results):
    """Save comparison summary to CSV."""
    import csv
    
    flock_num = results['real_flock']['flock_number']
    
    rows = []
    for key in ['real_flock', 'boids', 'cucker_smale']:
        m = results[key]['metrics']
        name = {
            'real_flock': f'Real Flock {flock_num}',
            'boids': 'Boids Model',
            'cucker_smale': 'Cucker-Smale Model'
        }[key]
        rows.append({
            'Source': name,
            'Num_Agents': int(m['num_agents']),
            'Mean_Speed_ms': round(m['mean_speed'], 2),
            'Speed_Std_ms': round(m['std_speed'], 2),
            'Polarization': round(m['polarization'], 3),
            'Mean_Neighbor_Dist_m': round(m['mean_neighbor_dist'], 2),
            'Density_per_m3': round(m['density'], 6)
        })
    
    csv_path = RESULTS_DIR / 'comparison_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  ✓ Saved: {csv_path}")


def generate_analysis_report(results, time_series_stats=None):
    """
    Generate a focused analysis report based on time-averaged metrics.
    All comparisons use time-averaged data for proper statistical comparison.
    """
    print("\nGenerating analysis report...")
    
    flock_num = results['real_flock']['flock_number']
    num_frames = results['real_flock'].get('num_frames', 1)
    duration = results['real_flock'].get('duration', 0)
    
    # Extract swarm objects for dynamic parameter access
    boids_swarm = results['boids'].get('swarm')
    cs_swarm = results['cucker_smale'].get('swarm')
    
    # Boids parameters
    bp = {
        'v0': getattr(boids_swarm, 'v0', 7.5) if boids_swarm else 7.5,
        'drag': getattr(boids_swarm, 'drag_coefficient', 0.005) if boids_swarm else 0.005,
        'sigma': getattr(boids_swarm, 'sigma', 0.15) if boids_swarm else 0.15,
        'k_neighbors': getattr(boids_swarm, 'k_neighbors', 7) if boids_swarm else 7,
        'w_sep': getattr(boids_swarm, 'w_sep', 20.0) if boids_swarm else 20.0,
        'w_ali': getattr(boids_swarm, 'w_ali', 15.0) if boids_swarm else 15.0,
        'w_coh': getattr(boids_swarm, 'w_coh', 5.0) if boids_swarm else 5.0,
        'sensor_noise': getattr(boids_swarm, 'sensor_noise', 1.0) if boids_swarm else 1.0,
        'wind_strength': getattr(boids_swarm, 'wind_strength', 0.5) if boids_swarm else 0.5,
        'neighbor_dropout': getattr(boids_swarm, 'neighbor_dropout', 0.15) if boids_swarm else 0.15,
    }
    
    # Cucker-Smale parameters
    csp = {
        'v0': getattr(cs_swarm, 'v0', 7.0) if cs_swarm else 7.0,
        'K': getattr(cs_swarm, 'K', 1.5) if cs_swarm else 1.5,
        'beta': getattr(cs_swarm, 'beta', 0.5) if cs_swarm else 0.5,
        'C_a': getattr(cs_swarm, 'C_a', 8.0) if cs_swarm else 8.0,
        'C_r': getattr(cs_swarm, 'C_r', 6.0) if cs_swarm else 6.0,
        'l_r': getattr(cs_swarm, 'l_r', 5.0) if cs_swarm else 5.0,
        'sigma': getattr(cs_swarm, 'sigma', 0.15) if cs_swarm else 0.15,
        'sensor_noise': getattr(cs_swarm, 'sensor_noise', 1.0) if cs_swarm else 1.0,
        'wind_strength': getattr(cs_swarm, 'wind_strength', 0.5) if cs_swarm else 0.5,
        'interaction_dropout': getattr(cs_swarm, 'interaction_dropout', 0.15) if cs_swarm else 0.15,
    }
    
    # Get time-averaged metrics
    real_m = results['real_flock']['metrics']
    real_std = results['real_flock'].get('metrics_std', {})
    boids_m = results['boids']['metrics']
    cs_m = results['cucker_smale']['metrics']
    
    # Calculate errors vs time-averaged real data
    boids_errors = {
        'speed': (boids_m['mean_speed'] - real_m['mean_speed']) / real_m['mean_speed'] * 100,
        'speed_std': (boids_m['std_speed'] - real_m['std_speed']) / real_m['std_speed'] * 100,
        'polarization': (boids_m['polarization'] - real_m['polarization']) / real_m['polarization'] * 100,
        'neighbor_dist': (boids_m['mean_neighbor_dist'] - real_m['mean_neighbor_dist']) / real_m['mean_neighbor_dist'] * 100,
    }
    
    cs_errors = {
        'speed': (cs_m['mean_speed'] - real_m['mean_speed']) / real_m['mean_speed'] * 100,
        'speed_std': (cs_m['std_speed'] - real_m['std_speed']) / real_m['std_speed'] * 100,
        'polarization': (cs_m['polarization'] - real_m['polarization']) / real_m['polarization'] * 100,
        'neighbor_dist': (cs_m['mean_neighbor_dist'] - real_m['mean_neighbor_dist']) / real_m['mean_neighbor_dist'] * 100,
    }
    
    boids_avg_error = np.mean([abs(e) for e in boids_errors.values()])
    cs_avg_error = np.mean([abs(e) for e in cs_errors.values()])
    
    # Generate clean, focused report
    report = f"""
================================================================================
MODEL COMPARISON ANALYSIS REPORT (Time-Averaged)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

All metrics are TIME-AVERAGED for proper statistical comparison:
- Real flock: {num_frames} frames over {duration:.1f} seconds
- Simulations: Last 50% of 400 timesteps, averaged over 3 runs

================================================================================
1. TIME-AVERAGED METRICS COMPARISON
================================================================================

                          Real Flock {flock_num}    Boids Model       Cucker-Smale
                          ---------------    -----------       ------------
Mean Speed (m/s)          {real_m['mean_speed']:>8.2f}           {boids_m['mean_speed']:>8.2f}            {cs_m['mean_speed']:>8.2f}
Speed Std (m/s)           {real_m['std_speed']:>8.2f}           {boids_m['std_speed']:>8.2f}            {cs_m['std_speed']:>8.2f}
Polarization              {real_m['polarization']:>8.3f}           {boids_m['polarization']:>8.3f}            {cs_m['polarization']:>8.3f}
Neighbor Distance (m)     {real_m['mean_neighbor_dist']:>8.2f}           {boids_m['mean_neighbor_dist']:>8.2f}            {cs_m['mean_neighbor_dist']:>8.2f}
Dist to 75% of Flock (m)  {real_m.get('dist_to_75pct', 0):>8.2f}           {boids_m.get('dist_to_75pct', 0):>8.2f}            {cs_m.get('dist_to_75pct', 0):>8.2f}
Number of Agents          {int(real_m['num_agents']):>8d}           {int(boids_m['num_agents']):>8d}            {int(cs_m['num_agents']):>8d}

================================================================================
2. ERROR ANALYSIS (% Difference from Real Data)
================================================================================

Metric                    Boids Error       Cucker-Smale Error
                          -----------       ------------------
Mean Speed                {boids_errors['speed']:>+8.1f}%          {cs_errors['speed']:>+8.1f}%
Speed Std                 {boids_errors['speed_std']:>+8.1f}%          {cs_errors['speed_std']:>+8.1f}%
Polarization              {boids_errors['polarization']:>+8.1f}%          {cs_errors['polarization']:>+8.1f}%
Neighbor Distance         {boids_errors['neighbor_dist']:>+8.1f}%          {cs_errors['neighbor_dist']:>+8.1f}%

Average Absolute Error:   {boids_avg_error:>8.1f}%          {cs_avg_error:>8.1f}%

BEST MODEL: {'BOIDS' if boids_avg_error < cs_avg_error else 'CUCKER-SMALE'} (lower average error)

================================================================================
3. MODEL PARAMETERS
================================================================================

BOIDS MODEL:
  Dynamics:
    - Target speed v0 = {bp['v0']}
    - Drag coefficient = {bp['drag']}
    - Topological neighbors k = {bp['k_neighbors']}
    - Weights: separation={bp['w_sep']}, alignment={bp['w_ali']}, cohesion={bp['w_coh']}
  
  Stochasticity (MATCHED):
    - Sigma (Euler-Maruyama) = {bp['sigma']}
    - Sensor noise = {bp['sensor_noise']}
    - Neighbor dropout = {bp['neighbor_dropout']}
    - Wind strength = {bp['wind_strength']}

CUCKER-SMALE MODEL:
  Dynamics:
    - Target speed v0 = {csp['v0']}
    - Communication strength K = {csp['K']}
    - Decay rate beta = {csp['beta']}
    - Morse potential: C_a={csp['C_a']}, C_r={csp['C_r']}, l_r={csp['l_r']}
  
  Stochasticity (MATCHED):
    - Sigma (Euler-Maruyama) = {csp['sigma']}
    - Sensor noise = {csp['sensor_noise']}
    - Interaction dropout = {csp['interaction_dropout']}
    - Wind strength = {csp['wind_strength']}

================================================================================
4. KEY INSIGHTS & DISCUSSION POINTS
================================================================================

BOIDS MODEL ANALYSIS:
"""
    
    # Add specific insights for Boids
    if abs(boids_errors['speed']) < 10:
        report += "  [+] Speed well-matched to real data\n"
    else:
        report += f"  [-] Speed differs by {boids_errors['speed']:+.1f}% from real data\n"
    
    if abs(boids_errors['polarization']) < 10:
        report += "  [+] Polarization well-matched to real data\n"
    else:
        sign = "too high" if boids_errors['polarization'] > 0 else "too low"
        report += f"  [-] Polarization {sign} ({boids_errors['polarization']:+.1f}%)\n"
    
    if abs(boids_errors['neighbor_dist']) < 20:
        report += "  [+] Neighbor distance well-matched to real data\n"
    else:
        sign = "too large" if boids_errors['neighbor_dist'] > 0 else "too small"
        report += f"  [-] Neighbor distance {sign} ({boids_errors['neighbor_dist']:+.1f}%)\n"

    report += """
CUCKER-SMALE MODEL ANALYSIS:
"""
    
    # Add specific insights for Cucker-Smale
    if abs(cs_errors['speed']) < 10:
        report += "  [+] Speed well-matched to real data\n"
    else:
        report += f"  [-] Speed differs by {cs_errors['speed']:+.1f}% from real data\n"
    
    if abs(cs_errors['polarization']) < 10:
        report += "  [+] Polarization well-matched to real data\n"
    else:
        sign = "too high" if cs_errors['polarization'] > 0 else "too low"
        report += f"  [-] Polarization {sign} ({cs_errors['polarization']:+.1f}%)\n"
    
    if abs(cs_errors['neighbor_dist']) < 20:
        report += "  [+] Neighbor distance well-matched to real data\n"
    else:
        sign = "too large" if cs_errors['neighbor_dist'] > 0 else "too small"
        report += f"  [-] Neighbor distance {sign} ({cs_errors['neighbor_dist']:+.1f}%)\n"

    report += f"""
OVERALL COMPARISON:
  - Both models use MATCHED stochasticity (sigma={bp['sigma']}, noise={bp['sensor_noise']}, dropout={bp['neighbor_dropout']})
  - Simulations run without obstacles/targets for fair comparison
  - Real flock shows natural variation: polarization fluctuates over time
  - Real mean neighbor distance ({real_m['mean_neighbor_dist']:.1f}m) reflects actual spacing

================================================================================
5. TEMPORAL VARIABILITY (Coefficient of Variation)
================================================================================
"""
    
    if time_series_stats:
        ts = time_series_stats
        report += f"""
                          Real Data         Boids             Cucker-Smale
Speed CV:                 {ts['real']['speed_std']/ts['real']['speed_mean']*100:>6.1f}%           {ts['boids']['speed_std']/ts['boids']['speed_mean']*100:>6.1f}%            {ts['cucker_smale']['speed_std']/ts['cucker_smale']['speed_mean']*100:>6.1f}%
Polarization CV:          {ts['real']['polarization_std']/ts['real']['polarization_mean']*100:>6.1f}%           {ts['boids']['polarization_std']/ts['boids']['polarization_mean']*100:>6.1f}%            {ts['cucker_smale']['polarization_std']/ts['cucker_smale']['polarization_mean']*100:>6.1f}%
Neighbor Dist CV:         {ts['real']['neighbor_dist_std']/ts['real']['neighbor_dist_mean']*100:>6.1f}%           {ts['boids']['neighbor_dist_std']/ts['boids']['neighbor_dist_mean']*100:>6.1f}%            {ts['cucker_smale']['neighbor_dist_std']/ts['cucker_smale']['neighbor_dist_mean']*100:>6.1f}%

Interpretation:
  - Real data shows {ts['real']['polarization_std']/ts['real']['polarization_mean']*100:.0f}% polarization CV -> flock has natural alignment fluctuations
  - Higher CV = more dynamic/natural behavior
  - Lower CV = more rigid/artificial behavior
"""
    else:
        report += "  (Time series data not available)\n"
    
    report += """
================================================================================
END OF REPORT
================================================================================
"""
    
    # Save report
    report_path = RESULTS_DIR / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  ✓ Saved: {report_path}")
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY (Time-Averaged Comparison)")
    print("="*60)
    print(f"Real Flock: Speed={real_m['mean_speed']:.2f} m/s, Polar={real_m['polarization']:.3f}, NeighDist={real_m['mean_neighbor_dist']:.2f}m")
    print(f"Boids:      Speed={boids_m['mean_speed']:.2f} m/s, Polar={boids_m['polarization']:.3f}, NeighDist={boids_m['mean_neighbor_dist']:.2f}m (err: {boids_avg_error:.1f}%)")
    print(f"C-S:        Speed={cs_m['mean_speed']:.2f} m/s, Polar={cs_m['polarization']:.3f}, NeighDist={cs_m['mean_neighbor_dist']:.2f}m (err: {cs_avg_error:.1f}%)")
    print(f"\nBest model: {'BOIDS' if boids_avg_error < cs_avg_error else 'CUCKER-SMALE'}")
    
    return report


# ============================================================================
# MAIN
# ============================================================================

def main(flock_number=6):
    """Run the full comparison pipeline."""
    print("\n" + "="*70)
    print("MODEL COMPARISON SCRIPT")
    print(f"Comparing Boids & Cucker-Smale with Mobbing Flock {flock_number}")
    print("="*70)
    
    # Run comparison
    results = compare_all(flock_number=flock_number)
    
    # Print table
    print_comparison_table(results)
    
    # Generate plots
    print("\n" + "-"*50)
    print("GENERATING FIGURES FOR REPORT")
    print("-"*50)
    
    # Individual metric bar charts
    plot_comparison_bar_charts(results)
    
    # Individual 3D visualizations
    plot_3d_comparison(results)
    
    # Distributions and analysis charts
    plot_metric_distributions(results)
    
    # Time series convergence plots (now with real data time-varying metrics!)
    num_agents = results['real_flock']['metrics']['num_agents']
    flock_num = results['real_flock']['flock_number']
    time_series_stats = plot_time_series(num_boids=int(num_agents), flock_number=flock_num)
    
    # Print time-varying statistics
    print("\n" + "-"*50)
    print("TIME-VARYING METRICS SUMMARY")
    print("-"*50)
    ts = time_series_stats
    print(f"\nReal Flock (over {ts['real']['duration']:.2f}s, {ts['real']['num_frames']} frames):")
    print(f"  Speed:         {ts['real']['speed_mean']:.2f} +/- {ts['real']['speed_std']:.2f} m/s")
    print(f"  Polarization:  {ts['real']['polarization_mean']:.3f} +/- {ts['real']['polarization_std']:.3f}")
    print(f"  Neighbor Dist: {ts['real']['neighbor_dist_mean']:.2f} +/- {ts['real']['neighbor_dist_std']:.2f} m")
    
    print(f"\nBoids Model (simulation):")
    print(f"  Speed:         {ts['boids']['speed_mean']:.2f} +/- {ts['boids']['speed_std']:.2f} m/s")
    print(f"  Polarization:  {ts['boids']['polarization_mean']:.3f} +/- {ts['boids']['polarization_std']:.3f}")
    print(f"  Neighbor Dist: {ts['boids']['neighbor_dist_mean']:.2f} +/- {ts['boids']['neighbor_dist_std']:.2f} m")
    
    print(f"\nCucker-Smale Model (simulation):")
    print(f"  Speed:         {ts['cucker_smale']['speed_mean']:.2f} +/- {ts['cucker_smale']['speed_std']:.2f} m/s")
    print(f"  Polarization:  {ts['cucker_smale']['polarization_mean']:.3f} +/- {ts['cucker_smale']['polarization_std']:.3f}")
    print(f"  Neighbor Dist: {ts['cucker_smale']['neighbor_dist_mean']:.2f} +/- {ts['cucker_smale']['neighbor_dist_std']:.2f} m")
    
    # Save data files
    save_summary_csv(results)
    generate_analysis_report(results, time_series_stats)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nGenerated files in plots/:")
    print("  📊 Individual metric charts:")
    print("      mean_speed_comparison.png, polarization_comparison.png")
    print("      neighbor_distance_comparison.png, speed_std_comparison.png")
    print("  📈 Distribution plots:")
    print("      speed_distribution.png, neighbor_distance_distribution.png")
    print("  📉 Error analysis:")
    print("      error_comparison.png, performance_radar.png")
    print("  🎯 3D visualizations:")
    print("      real_flock_3d.png, boids_3d.png, cucker_smale_3d.png")
    print("  ⏱️ Time series (with real data time-varying!):")
    print("      speed_convergence.png, polarization_convergence.png")
    print("      neighbor_distance_convergence.png, convergence_overview.png")
    print("      variability_comparison.png (NEW: compares temporal stability)")
    print("  📋 Summary:")
    print("      summary_table.png, velocity_alignment.png, metrics_overview.png")
    print("\nGenerated files in results/:")
    print(f"  📄 comparison_summary.csv")
    print(f"  📝 analysis_report.txt (now with time-varying stats!)")


if __name__ == '__main__':
    main()
