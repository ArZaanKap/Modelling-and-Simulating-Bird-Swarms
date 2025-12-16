"""Obstacle & Target Navigation Comparison - Boids vs Cucker-Smale"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from boids4 import BoidSwarm
from cucker_smale2 import CuckerSmaleSwarm
from shared_functions import create_predefined_obstacles, check_obstacle_collisions

PLOTS_DIR = Path(__file__).parent / 'plots' / 'obstacle_navigation'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_cohesion_metrics(positions):
    """Compute flock cohesion metrics with robust split detection."""
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    std_dist = np.std(distances)
    
    sorted_dists = np.sort(distances)
    if len(sorted_dists) > 5:
        gaps = np.diff(sorted_dists)
        max_gap = np.max(gaps)
        median_gap = np.median(gaps)
        has_cluster_gap = max_gap > 5 * median_gap
    else:
        has_cluster_gap = False
    
    is_split = (max_dist > 2.5 * mean_dist) or (std_dist > mean_dist) or has_cluster_gap
    
    return {
        'mean_dist_to_center': mean_dist,
        'max_dist_to_center': max_dist,
        'std_dist': std_dist,
        'flock_radius': np.percentile(distances, 90),
        'is_split': is_split,
        'has_cluster_gap': has_cluster_gap
    }


def compute_polarization(velocities):
    """Compute alignment order parameter."""
    speeds = np.linalg.norm(velocities, axis=1)
    v_norm = velocities / (speeds[:, np.newaxis] + 1e-8)
    return np.linalg.norm(np.mean(v_norm, axis=0))


def run_obstacle_scenario(model_type='boids', num_boids=50, num_steps=500, dt=0.08):
    """Run simulation with obstacles and target point, returning time series metrics."""
    obstacles = create_predefined_obstacles('scattered')
    
    if model_type == 'boids':
        swarm = BoidSwarm(
            num_boids=num_boids, ts=dt, sigma=0.15, k_neighbors=7,
            sensor_noise=1.0, neighbor_dropout=0.15, wind_strength=0.5,
            obstacles=obstacles, obstacle_avoidance_strength=150.0
        )
        swarm.target_point = np.array([40.0, 0.0, 0.0])
        get_pos = lambda: swarm.positions
        get_vel = lambda: swarm.velocities
        step_fn = lambda: swarm.boids_algorithm()
    else:
        swarm = CuckerSmaleSwarm(
            num_boids=num_boids, dt=dt, sigma=0.15,
            sensor_noise=1.0, interaction_dropout=0.15, wind_strength=0.5,
            obstacles=obstacles, obstacle_avoidance_strength=150.0
        )
        swarm.target_point = np.array([40.0, 0.0, 0.0])
        get_pos = lambda: swarm.x
        get_vel = lambda: swarm.v
        step_fn = lambda: swarm.step()
    
    times, cohesion, polarization = [], [], []
    dist_to_target, obstacle_collisions, split_events = [], [], []
    
    for step in range(num_steps):
        step_fn()
        pos = get_pos()
        vel = get_vel()
        
        times.append(step * dt)
        
        coh = compute_cohesion_metrics(pos)
        cohesion.append(coh['mean_dist_to_center'])
        split_events.append(1 if coh['is_split'] else 0)
        polarization.append(compute_polarization(vel))
        
        center = np.mean(pos, axis=0)
        dist_to_target.append(np.linalg.norm(center - swarm.target_point))
        
        obs_collision_mask, _, _ = check_obstacle_collisions(pos, obstacles)
        obstacle_collisions.append(int(np.sum(obs_collision_mask)))
    
    return {
        'times': np.array(times),
        'cohesion': np.array(cohesion),
        'polarization': np.array(polarization),
        'dist_to_target': np.array(dist_to_target),
        'obstacle_collisions': np.array(obstacle_collisions),
        'split_events': np.array(split_events),
        'total_obstacle_collisions': np.sum(obstacle_collisions),
        'split_ratio': np.mean(split_events),
        'final_dist_to_target': dist_to_target[-1],
        'target_reached': dist_to_target[-1] < 20.0
    }


def compute_reformation_time(split_events, dt):
    """Compute reformation statistics from split events."""
    splits = []
    in_split = False
    split_start = 0
    
    for i, is_split in enumerate(split_events):
        if is_split and not in_split:
            in_split = True
            split_start = i
        elif not is_split and in_split:
            in_split = False
            duration = (i - split_start) * dt
            splits.append(duration)
    
    if in_split:
        duration = (len(split_events) - split_start) * dt
        splits.append(duration)
    
    if len(splits) == 0:
        return {'max_split_duration': 0.0, 'mean_split_duration': 0.0,
                'num_splits': 0, 'total_split_time': 0.0}
    
    return {
        'max_split_duration': max(splits),
        'mean_split_duration': np.mean(splits),
        'num_splits': len(splits),
        'total_split_time': sum(splits)
    }


def run_comparison(num_runs=3, num_boids=50, num_steps=600):
    """Run multiple trials and compare models."""
    print("="*60)
    print("OBSTACLE & TARGET NAVIGATION COMPARISON")
    print("="*60)
    
    dt = 0.08
    boids_results, cs_results = [], []
    
    print(f"\nRunning {num_runs} trials for each model...")
    for run in range(num_runs):
        print(f"  Trial {run+1}/{num_runs}...")
        boids_results.append(run_obstacle_scenario('boids', num_boids, num_steps, dt))
        cs_results.append(run_obstacle_scenario('cucker_smale', num_boids, num_steps, dt))
    
    def get_reform_stats(results_list):
        stats = [compute_reformation_time(r['split_events'], dt) for r in results_list]
        return {
            'max_split_duration': np.mean([s['max_split_duration'] for s in stats]),
            'mean_split_duration': np.mean([s['mean_split_duration'] for s in stats]),
            'num_splits': np.mean([s['num_splits'] for s in stats]),
            'total_split_time': np.mean([s['total_split_time'] for s in stats])
        }
    
    boids_reform = get_reform_stats(boids_results)
    cs_reform = get_reform_stats(cs_results)
    
    total_time = num_steps * dt  # Total simulation time
    
    metrics = {
        'Boids': {
            'total_obstacle_collisions': np.mean([r['total_obstacle_collisions'] for r in boids_results]),
            'split_ratio': np.mean([r['split_ratio'] for r in boids_results]) * 100,
            'mean_cohesion': np.mean([np.mean(r['cohesion']) for r in boids_results]),
            'mean_polarization': np.mean([np.mean(r['polarization']) for r in boids_results]),
            'target_success': np.mean([r['target_reached'] for r in boids_results]) * 100,
            'final_dist': np.mean([r['final_dist_to_target'] for r in boids_results]),
            'max_split_duration': boids_reform['max_split_duration'],
            'mean_split_duration': boids_reform['mean_split_duration'],
            'num_splits': boids_reform['num_splits'],
            'total_split_time': boids_reform['total_split_time']
        },
        'Cucker-Smale': {
            'total_obstacle_collisions': np.mean([r['total_obstacle_collisions'] for r in cs_results]),
            'split_ratio': np.mean([r['split_ratio'] for r in cs_results]) * 100,
            'mean_cohesion': np.mean([np.mean(r['cohesion']) for r in cs_results]),
            'mean_polarization': np.mean([np.mean(r['polarization']) for r in cs_results]),
            'target_success': np.mean([r['target_reached'] for r in cs_results]) * 100,
            'final_dist': np.mean([r['final_dist_to_target'] for r in cs_results]),
            'max_split_duration': cs_reform['max_split_duration'],
            'mean_split_duration': cs_reform['mean_split_duration'],
            'num_splits': cs_reform['num_splits'],
            'total_split_time': cs_reform['total_split_time']
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Simulation: {num_steps} steps × {dt}s = {total_time:.1f}s total")
    print(f"\n{'Metric':<25} {'Boids':>15} {'Cucker-Smale':>15}")
    print("-"*60)
    print(f"{'Obstacle collisions':<25} {metrics['Boids']['total_obstacle_collisions']:>15.0f} {metrics['Cucker-Smale']['total_obstacle_collisions']:>15.0f}")
    print(f"{'Split time (%)':<25} {metrics['Boids']['split_ratio']:>14.1f}% {metrics['Cucker-Smale']['split_ratio']:>14.1f}%")
    print(f"{'Num split events':<25} {metrics['Boids']['num_splits']:>15.1f} {metrics['Cucker-Smale']['num_splits']:>15.1f}")
    print(f"{'Max split duration (s)':<25} {metrics['Boids']['max_split_duration']:>15.2f} {metrics['Cucker-Smale']['max_split_duration']:>15.2f}")
    print(f"{'Mean split duration (s)':<25} {metrics['Boids']['mean_split_duration']:>15.2f} {metrics['Cucker-Smale']['mean_split_duration']:>15.2f}")
    print(f"{'Total split time (s)':<25} {metrics['Boids']['total_split_time']:>15.2f} {metrics['Cucker-Smale']['total_split_time']:>15.2f}")
    print(f"{'Mean cohesion (m)':<25} {metrics['Boids']['mean_cohesion']:>15.2f} {metrics['Cucker-Smale']['mean_cohesion']:>15.2f}")
    print(f"{'Mean polarization':<25} {metrics['Boids']['mean_polarization']:>15.3f} {metrics['Cucker-Smale']['mean_polarization']:>15.3f}")
    print(f"{'Target success (%)':<25} {metrics['Boids']['target_success']:>14.0f}% {metrics['Cucker-Smale']['target_success']:>14.0f}%")
    print(f"{'Final dist to target (m)':<25} {metrics['Boids']['final_dist']:>15.1f} {metrics['Cucker-Smale']['final_dist']:>15.1f}")
    
    generate_plots(boids_results[-1], cs_results[-1], metrics)
    
    return metrics, boids_results, cs_results


def generate_plots(boids_run, cs_run, metrics):
    """Generate comparison plots."""
    print("\nGenerating plots...")
    
    times = boids_run['times']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, boids_run['dist_to_target'], 'g-', label='Boids', linewidth=2)
    ax.plot(times, cs_run['dist_to_target'], 'r-', label='Cucker-Smale', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to Target (m)')
    ax.set_title('Distance to Target Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'distance_to_target.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'distance_to_target.png'}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, boids_run['cohesion'], 'g-', label='Boids', linewidth=2)
    ax.plot(times, cs_run['cohesion'], 'r-', label='Cucker-Smale', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Distance to Flock Center (m)')
    ax.set_title('Flock Cohesion (lower = tighter)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'flock_cohesion.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'flock_cohesion.png'}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, boids_run['polarization'], 'g-', label='Boids', linewidth=2)
    ax.plot(times, cs_run['polarization'], 'r-', label='Cucker-Smale', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Polarization')
    ax.set_title('Alignment During Navigation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'alignment_navigation.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'alignment_navigation.png'}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['Obstacle\nColl.', 'Split %', 'Cohesion\n(m)', 'Polarization']
    boids_vals = [
        metrics['Boids']['total_obstacle_collisions'],
        metrics['Boids']['split_ratio'],
        metrics['Boids']['mean_cohesion'],
        metrics['Boids']['mean_polarization']
    ]
    cs_vals = [
        metrics['Cucker-Smale']['total_obstacle_collisions'],
        metrics['Cucker-Smale']['split_ratio'],
        metrics['Cucker-Smale']['mean_cohesion'],
        metrics['Cucker-Smale']['mean_polarization']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x - width/2, boids_vals, width, label='Boids', color='#28A745')
    ax.bar(x + width/2, cs_vals, width, label='Cucker-Smale', color='#DC3545')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'performance_summary.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'performance_summary.png'}")
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Obstacle Collision Accumulation Over Time', fontsize=14, fontweight='bold')
    
    ax.plot(times, np.cumsum(boids_run['obstacle_collisions']), 'g-', label='Boids', linewidth=2)
    ax.plot(times, np.cumsum(cs_run['obstacle_collisions']), 'r-', label='Cucker-Smale', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Obstacle Collisions')
    ax.set_title('Obstacle Collisions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'collision_accumulation.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'collision_accumulation.png'}")
    plt.close()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Event Timeline: Splits & Obstacle Collisions Over Time', fontsize=14, fontweight='bold')
    
    ax = axes[0]
    ax.fill_between(times, 0, boids_run['split_events'], alpha=0.4, color='orange', 
                    label='Split detected', step='mid')
    ax.bar(times, boids_run['obstacle_collisions'], width=times[1]-times[0], alpha=0.7, 
           color='red', label='Obstacle collisions')
    ax.set_ylabel('Events')
    ax.set_title('Boids Model')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    ax = axes[1]
    ax.fill_between(times, 0, cs_run['split_events'], alpha=0.4, color='orange', 
                    label='Split detected', step='mid')
    ax.bar(times, cs_run['obstacle_collisions'], width=times[1]-times[0], alpha=0.7, 
           color='red', label='Obstacle collisions')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Events')
    ax.set_title('Cucker-Smale Model')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'event_timeline.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'event_timeline.png'}")
    plt.close()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Flock Cohesion with Split Events', fontsize=14, fontweight='bold')
    
    for idx, (run, name, color) in enumerate([(boids_run, 'Boids', '#28A745'), 
                                               (cs_run, 'Cucker-Smale', '#DC3545')]):
        ax = axes[idx]
        ax.plot(times, run['cohesion'], color=color, linewidth=2, label='Mean dist to center')
        
        split_mask = np.array(run['split_events'], dtype=bool)
        ax.fill_between(times, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 50, 
                        where=split_mask, alpha=0.2, color='orange', label='Split period')
        
        obs_collision_times = times[run['obstacle_collisions'] > 0]
        if len(obs_collision_times) > 0:
            ax.scatter(obs_collision_times, np.ones(len(obs_collision_times)) * np.max(run['cohesion']) * 0.95,
                      marker='x', color='red', s=50, label='Obstacle collision', zorder=5)
        
        ax.set_ylabel('Mean Dist to Center (m)')
        ax.set_title(f'{name} Model')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cohesion_with_events.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR / 'cohesion_with_events.png'}")
    plt.close()


if __name__ == '__main__':
    metrics, boids_results, cs_results = run_comparison(num_runs=3, num_boids=50, num_steps=2000)
    print("\n✓ Comparison complete!")
