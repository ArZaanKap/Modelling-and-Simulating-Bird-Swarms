"""
Flock Visualization Tool
========================
Visualize real bird flocking data from the dataset.
Default: Mobbing Flock 6

Usage:
    python visualize_flocks.py              # Visualize mobbing flock 6 (default)
    python visualize_flocks.py 7            # Visualize mobbing flock 7
    python visualize_flocks.py all          # Visualize all available flocks
    python visualize_flocks.py 6 animate    # Live animation of flock 6
    python visualize_flocks.py animate      # Live animation of default flock 6
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
from pathlib import Path
import sys


# ============================================================================
# DATA LOADING
# ============================================================================

def get_data_path():
    """Get path to the data directory."""
    script_dir = Path(__file__).parent
    data_path = script_dir / '../dataset/code_upload_2/code_upload_2/data'
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    return data_path


def load_flock(flock_number):
    """
    Load a mobbing flock by number.
    
    Parameters:
    -----------
    flock_number : int
        Flock number (1-10, with 4 and 5 having parts p1/p2)
    
    Returns:
    --------
    data : dict
        Flock data with positions, velocities, times, etc.
    """
    data_path = get_data_path()
    
    # Handle special cases for flocks 4 and 5
    if flock_number == 4:
        filename = 'mob_04_p1.mat'
    elif flock_number == 5:
        filename = 'mob_05_p1.mat'
    else:
        filename = f'mob_{flock_number:02d}.mat'
    
    filepath = data_path / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Flock file not found: {filepath}")
    
    mat_data = loadmat(str(filepath))
    
    # Extract tracks_filt
    if 'tracks_filt' in mat_data:
        tracks = mat_data['tracks_filt']
    else:
        possible_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        tracks = mat_data[possible_keys[0]] if possible_keys else None
        if tracks is None:
            raise KeyError(f"Could not find data in {filename}")
    
    # Parse data
    bird_ids = tracks[:, 0].astype(int)
    positions = tracks[:, 1:4]
    times = tracks[:, 4]
    velocities = tracks[:, 5:8]
    
    return {
        'positions': positions,
        'velocities': velocities,
        'times': times,
        'bird_ids': bird_ids,
        'num_birds': len(np.unique(bird_ids)),
        'time_range': (times.min(), times.max()),
        'filename': filename
    }


def get_flock_at_time(data, time):
    """Get positions and velocities at a specific time."""
    time_idx = np.argmin(np.abs(data['times'] - time))
    actual_time = data['times'][time_idx]
    mask = data['times'] == actual_time
    return data['positions'][mask], data['velocities'][mask], data['bird_ids'][mask]


def get_unique_timesteps(data):
    """Get all unique timesteps in the dataset."""
    return np.unique(data['times'])


def list_available_flocks():
    """List all available flock files."""
    data_path = get_data_path()
    flocks = []
    for i in range(1, 11):
        if i == 4:
            filename = 'mob_04_p1.mat'
        elif i == 5:
            filename = 'mob_05_p1.mat'
        else:
            filename = f'mob_{i:02d}.mat'
        if (data_path / filename).exists():
            flocks.append(i)
    return flocks


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_flock_3d(flock_number=6, save=True):
    """
    Create a 3D visualization of a flock.
    
    Parameters:
    -----------
    flock_number : int
        Mobbing flock number (default: 6)
    save : bool
        Whether to save the plot
    """
    print(f"\n{'='*60}")
    print(f"VISUALIZING MOBBING FLOCK {flock_number}")
    print(f"{'='*60}")
    
    data = load_flock(flock_number)
    
    # Get data at middle of time range
    mid_time = (data['time_range'][0] + data['time_range'][1]) / 2
    positions, velocities, bird_ids = get_flock_at_time(data, mid_time)
    
    speeds = np.linalg.norm(velocities, axis=1)
    
    print(f"File: {data['filename']}")
    print(f"Number of birds: {data['num_birds']}")
    print(f"Time range: {data['time_range'][0]:.2f}s - {data['time_range'][1]:.2f}s")
    print(f"Snapshot time: {mid_time:.2f}s")
    print(f"Mean speed: {np.mean(speeds):.2f} m/s")
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Mobbing Flock {flock_number} - Real Bird Data\n'
                 f'{len(positions)} birds at t={mid_time:.2f}s', 
                 fontsize=14, fontweight='bold')
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    sc = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     c=speeds, cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Flock Structure (colored by speed)')
    plt.colorbar(sc, ax=ax1, label='Speed (m/s)', shrink=0.6)
    
    # 3D with velocity arrows
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                c='blue', s=30, alpha=0.6)
    # Subsample arrows for clarity
    step = max(1, len(positions) // 20)
    scale = 0.5
    ax2.quiver(positions[::step, 0], positions[::step, 1], positions[::step, 2],
               velocities[::step, 0]*scale, velocities[::step, 1]*scale, velocities[::step, 2]*scale,
               color='red', alpha=0.7, arrow_length_ratio=0.3)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Velocity Directions')
    
    # XY projection
    ax3 = fig.add_subplot(223)
    ax3.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='viridis', s=40, alpha=0.7)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Top View (XY Plane)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # XZ projection
    ax4 = fig.add_subplot(224)
    ax4.scatter(positions[:, 0], positions[:, 2], c=speeds, cmap='viridis', s=40, alpha=0.7)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Side View (XZ Plane)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_file = f'flock_{flock_number}_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_file}")
    
    plt.show()
    return fig


def visualize_flock_statistics(flock_number=6, save=True):
    """
    Visualize statistical distributions for a flock.
    """
    print(f"\nComputing statistics for flock {flock_number}...")
    
    data = load_flock(flock_number)
    mid_time = (data['time_range'][0] + data['time_range'][1]) / 2
    positions, velocities, _ = get_flock_at_time(data, mid_time)
    
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Compute neighbor distances
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(positions))
    
    # Get 7-nearest neighbor distances for each bird
    k = min(7, len(positions) - 1)
    neighbor_dists = []
    for i in range(len(positions)):
        dists = np.sort(dist_matrix[i])[1:k+1]  # Exclude self
        neighbor_dists.extend(dists)
    neighbor_dists = np.array(neighbor_dists)
    
    # Compute polarization
    speed_norms = velocities / (speeds[:, np.newaxis] + 1e-8)
    polarization = np.linalg.norm(np.mean(speed_norms, axis=0))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Mobbing Flock {flock_number} - Statistics\n'
                 f'{len(positions)} birds | Polarization: {polarization:.3f}',
                 fontsize=14, fontweight='bold')
    
    # Speed distribution
    axes[0, 0].hist(speeds, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0, 0].axvline(np.mean(speeds), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(speeds):.2f} m/s')
    axes[0, 0].set_xlabel('Speed (m/s)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Speed Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Neighbor distance distribution
    axes[0, 1].hist(neighbor_dists, bins=30, color='forestgreen', edgecolor='white', alpha=0.8)
    axes[0, 1].axvline(np.mean(neighbor_dists), color='red', linestyle='--',
                       label=f'Mean: {np.mean(neighbor_dists):.2f} m')
    axes[0, 1].set_xlabel('Distance to Neighbors (m)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'{k}-Nearest Neighbor Distances')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Velocity components
    axes[1, 0].hist(velocities[:, 0], bins=20, alpha=0.7, label='Vx', color='red')
    axes[1, 0].hist(velocities[:, 1], bins=20, alpha=0.7, label='Vy', color='green')
    axes[1, 0].hist(velocities[:, 2], bins=20, alpha=0.7, label='Vz', color='blue')
    axes[1, 0].set_xlabel('Velocity (m/s)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Velocity Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics text
    axes[1, 1].axis('off')
    stats_text = f"""
    FLOCK STATISTICS SUMMARY
    ========================
    
    Number of Birds: {len(positions)}
    
    SPEED:
      Mean: {np.mean(speeds):.2f} m/s
      Std:  {np.std(speeds):.2f} m/s
      Min:  {np.min(speeds):.2f} m/s
      Max:  {np.max(speeds):.2f} m/s
    
    NEIGHBOR DISTANCE (7-NN):
      Mean: {np.mean(neighbor_dists):.2f} m
      Std:  {np.std(neighbor_dists):.2f} m
    
    ORDER PARAMETER:
      Polarization: {polarization:.3f}
      (1.0 = perfectly aligned)
    
    SPATIAL EXTENT:
      X range: {positions[:,0].max() - positions[:,0].min():.1f} m
      Y range: {positions[:,1].max() - positions[:,1].min():.1f} m
      Z range: {positions[:,2].max() - positions[:,2].min():.1f} m
    """
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save:
        output_file = f'flock_{flock_number}_statistics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
    
    plt.show()
    return fig


def visualize_all_flocks(save=True):
    """Visualize all available flocks in a grid."""
    available = list_available_flocks()
    print(f"\nVisualizing all {len(available)} available flocks...")
    
    n_flocks = len(available)
    cols = 3
    rows = (n_flocks + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 4*rows))
    fig.suptitle('All Mobbing Flocks Overview', fontsize=14, fontweight='bold')
    
    for idx, flock_num in enumerate(available):
        try:
            data = load_flock(flock_num)
            mid_time = (data['time_range'][0] + data['time_range'][1]) / 2
            positions, velocities, _ = get_flock_at_time(data, mid_time)
            speeds = np.linalg.norm(velocities, axis=1)
            
            ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=speeds, cmap='viridis', s=20, alpha=0.7)
            ax.set_title(f'Flock {flock_num} ({len(positions)} birds)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        except Exception as e:
            print(f"  Warning: Could not load flock {flock_num}: {e}")
    
    plt.tight_layout()
    
    if save:
        output_file = 'all_flocks_overview.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_file}")
    
    plt.show()
    return fig


def animate_flock(flock_number=6, speed_factor=1.0):
    """
    Create a live animation of a flock over time.
    
    Parameters:
    -----------
    flock_number : int
        Mobbing flock number (default: 6)
    speed_factor : float
        Animation speed multiplier (default: 1.0)
    """
    print(f"\n{'='*60}")
    print(f"ANIMATING MOBBING FLOCK {flock_number}")
    print(f"{'='*60}")
    
    # Load flock data
    data = load_flock(flock_number)
    timesteps = get_unique_timesteps(data)
    
    print(f"File: {data['filename']}")
    print(f"Number of birds: {data['num_birds']}")
    print(f"Time range: {data['time_range'][0]:.3f}s - {data['time_range'][1]:.3f}s")
    print(f"Number of frames: {len(timesteps)}")
    print(f"\nStarting animation... (close window to stop)")
    
    # Set up the figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get initial data
    init_pos, init_vel, _ = get_flock_at_time(data, timesteps[0])
    init_speeds = np.linalg.norm(init_vel, axis=1)
    
    # Create initial scatter plot
    scat = ax.scatter(init_pos[:, 0], init_pos[:, 1], init_pos[:, 2],
                      c=init_speeds, cmap='viridis', s=50, alpha=0.8,
                      vmin=np.percentile(np.linalg.norm(data['velocities'], axis=1), 5),
                      vmax=np.percentile(np.linalg.norm(data['velocities'], axis=1), 95))
    
    # Create velocity arrows (subsampled)
    step = max(1, len(init_pos) // 15)
    scale = 0.02  # Scale for velocity arrows
    quiver = ax.quiver(init_pos[::step, 0], init_pos[::step, 1], init_pos[::step, 2],
                       init_vel[::step, 0]*scale, init_vel[::step, 1]*scale, init_vel[::step, 2]*scale,
                       color='red', alpha=0.6, arrow_length_ratio=0.3)
    
    # Calculate data bounds for consistent axis limits
    all_positions = data['positions']
    x_range = [all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1]
    y_range = [all_positions[:, 1].min() - 1, all_positions[:, 1].max() + 1]
    z_range = [all_positions[:, 2].min() - 1, all_positions[:, 2].max() + 1]
    
    # Make axes equal aspect ratio
    max_range = max(x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0])
    mid_x = (x_range[0] + x_range[1]) / 2
    mid_y = (y_range[0] + y_range[1]) / 2
    mid_z = (z_range[0] + z_range[1]) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(scat, ax=ax, shrink=0.6, label='Speed (m/s)')
    
    # Add info text
    info_text = ax.text2D(0.02, 0.98,
                         f"Mobbing Flock {flock_number}\n"
                         f"Birds: {data['num_birds']}\n"
                         f"Color: Speed (m/s)",
                         transform=ax.transAxes,
                         fontsize=10,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Timer text
    timer_text = ax.text2D(0.98, 0.98, "",
                          transform=ax.transAxes,
                          fontsize=12,
                          horizontalalignment='right',
                          verticalalignment='top',
                          fontweight='bold')
    
    ax.set_title(f'Live Animation: Mobbing Flock {flock_number}',
                 fontsize=14, fontweight='bold')
    
    def update(frame_idx):
        """Animation update function."""
        nonlocal quiver
        
        # Get data at this timestep
        t = timesteps[frame_idx]
        positions, velocities, _ = get_flock_at_time(data, t)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Update scatter plot
        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        scat.set_array(speeds)
        
        # Update velocity arrows
        quiver.remove()
        step = max(1, len(positions) // 15)
        quiver = ax.quiver(positions[::step, 0], positions[::step, 1], positions[::step, 2],
                           velocities[::step, 0]*scale, velocities[::step, 1]*scale, velocities[::step, 2]*scale,
                           color='red', alpha=0.6, arrow_length_ratio=0.3)
        
        # Update timer
        timer_text.set_text(f"t = {t:.3f}s\nFrame {frame_idx+1}/{len(timesteps)}")
        
        return scat, timer_text
    
    # Calculate interval based on actual time differences and speed factor
    if len(timesteps) > 1:
        avg_dt = np.mean(np.diff(timesteps))
        interval = max(20, int(avg_dt * 1000 / speed_factor))  # Convert to ms
    else:
        interval = 50
    
    # Create animation
    anim = FuncAnimation(fig, update,
                        frames=len(timesteps),
                        interval=interval,
                        blit=False,
                        repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    print("="*60)
    print("FLOCK VISUALIZATION TOOL")
    print("="*60)
    
    # Parse command line arguments
    args = [arg.lower() for arg in sys.argv[1:]]
    
    # Check for 'animate' flag
    animate_mode = 'animate' in args
    if animate_mode:
        args.remove('animate')
    
    # Determine flock number
    flock_num = 6  # default
    show_all = False
    
    for arg in args:
        if arg == 'all':
            show_all = True
        else:
            try:
                flock_num = int(arg)
            except ValueError:
                print(f"Invalid argument: {arg}")
                print("Usage: python visualize_flocks.py [flock_number|all] [animate]")
                return
    
    if show_all:
        visualize_all_flocks()
    elif animate_mode:
        # Live animation mode
        animate_flock(flock_num)
    else:
        # Default: static visualizations
        print(f"\nNo 'animate' specified. Showing static visualizations for Flock {flock_num}")
        print("(Use 'python visualize_flocks.py 6 animate' for live animation)")
        visualize_flock_3d(flock_num)
        visualize_flock_statistics(flock_num)
    
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
