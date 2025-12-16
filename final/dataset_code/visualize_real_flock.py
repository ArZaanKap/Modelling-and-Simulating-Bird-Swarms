"""
Animate Real Flock Data
-----------------------
Interactive 3D animation of real bird flock data from mobbing events.
Shows position and velocity of each bird over time.

Usage:
    python visualize_real_flock.py           # Default: Flock 6
    python visualize_real_flock.py 7         # Specify flock number
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
from pathlib import Path
import sys


def get_data_path():
    """Get path to the bird flock data files."""
    return Path(__file__).parent.parent / 'dataset' / 'code_upload_2' / 'code_upload_2' / 'data'


def load_flock(flock_number):
    """Load mobbing flock data and organize by time."""
    data_path = get_data_path()
    
    if flock_number == 4:
        filename = 'mob_04_p1.mat'
    elif flock_number == 5:
        filename = 'mob_05_p1.mat'
    else:
        filename = f'mob_{flock_number:02d}.mat'
    
    print(f"Loading {filename}...")
    mat_data = loadmat(str(data_path / filename))
    tracks = mat_data.get('tracks_filt', list(mat_data.values())[-1])
    
    # Organize data by time frame
    unique_times = np.sort(np.unique(tracks[:, 4]))
    
    frames = []
    for t in unique_times:
        mask = tracks[:, 4] == t
        frames.append({
            'time': t,
            'positions': tracks[mask, 1:4],
            'velocities': tracks[mask, 5:8],
            'bird_ids': tracks[mask, 0].astype(int)
        })
    
    return {
        'frames': frames,
        'unique_times': unique_times,
        'num_frames': len(frames),
        'time_range': (unique_times[0], unique_times[-1]),
        'duration': unique_times[-1] - unique_times[0],
        'filename': filename
    }


def compute_metrics(positions, velocities):
    """Compute flocking metrics for a single frame."""
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    
    # Polarization
    v_normalized = velocities / (speeds[:, np.newaxis] + 1e-8)
    polarization = np.linalg.norm(np.mean(v_normalized, axis=0))
    
    return {
        'mean_speed': mean_speed,
        'polarization': polarization,
        'num_birds': len(positions)
    }


def animate_flock(flock_number=6, speed=1.0, skip_frames=1):
    """
    Create an interactive 3D animation of the real flock data.
    
    Parameters:
    -----------
    flock_number : int
        Which mobbing flock to visualize (1-10)
    speed : float
        Animation speed multiplier (1.0 = real-time, 2.0 = 2x speed)
    skip_frames : int
        Skip every N frames (use for very long recordings)
    """
    # Load data
    data = load_flock(flock_number)
    frames = data['frames'][::skip_frames]  # Subsample if needed
    
    print(f"\nFlock {flock_number} loaded:")
    print(f"  Duration: {data['duration']:.2f} seconds")
    print(f"  Frames: {data['num_frames']} (showing {len(frames)} after skip)")
    print(f"  Birds per frame: {len(frames[0]['positions'])}")
    
    # Calculate bounds for consistent axes
    all_positions = np.vstack([f['positions'] for f in frames])
    x_min, y_min, z_min = all_positions.min(axis=0) - 5
    x_max, y_max, z_max = all_positions.max(axis=0) + 5
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], [], c='blue', s=30, alpha=0.8)
    quiver = None  # Will hold velocity arrows
    
    # Title and labels
    title = ax.set_title(f'Real Flock {flock_number} - Frame 0', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Set consistent axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Make axes equal aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Info text
    info_text = fig.text(0.02, 0.95, '', fontsize=10, family='monospace',
                         verticalalignment='top', transform=fig.transFigure,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        """Initialize animation."""
        return scatter,
    
    def update(frame_idx):
        """Update function for each animation frame."""
        nonlocal quiver
        
        frame = frames[frame_idx]
        pos = frame['positions']
        vel = frame['velocities']
        
        # Update scatter positions
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # Color by speed
        speeds = np.linalg.norm(vel, axis=1)
        colors = plt.cm.viridis((speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-8))
        scatter.set_facecolors(colors)
        
        # Remove old quiver and draw new velocity arrows
        if quiver is not None:
            quiver.remove()
        
        # Normalize velocities for display (scale to reasonable arrow length)
        vel_norm = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-8) * 2
        quiver = ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                          vel_norm[:, 0], vel_norm[:, 1], vel_norm[:, 2],
                          color='red', alpha=0.5, arrow_length_ratio=0.3, linewidth=0.5)
        
        # Compute metrics
        metrics = compute_metrics(pos, vel)
        
        # Update title and info
        time = frame['time'] - frames[0]['time']
        title.set_text(f'Real Flock {flock_number} - Time: {time:.2f}s (Frame {frame_idx+1}/{len(frames)})')
        
        info_text.set_text(
            f"Birds: {metrics['num_birds']}\n"
            f"Speed: {metrics['mean_speed']:.2f} m/s\n"
            f"Polarization: {metrics['polarization']:.3f}\n"
            f"Speed range: {speeds.min():.1f}-{speeds.max():.1f} m/s"
        )
        
        return scatter, quiver
    
    # Calculate interval (milliseconds between frames)
    # Original data has ~55 fps, so interval ~18ms at real-time
    dt = (data['unique_times'][skip_frames] - data['unique_times'][0]) if len(data['unique_times']) > skip_frames else 0.018
    interval = int(dt * 1000 / speed)  # Convert to ms, adjust by speed
    interval = max(10, interval)  # Minimum 10ms for smooth playback
    
    print(f"\nAnimation interval: {interval}ms (speed: {speed}x)")
    print("Controls: Close window to stop")
    
    # Create animation
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(frames), interval=interval, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def plot_trajectory(flock_number=6, bird_id=None):
    """
    Plot 3D trajectory of the flock centroid or a specific bird over time.
    
    Parameters:
    -----------
    flock_number : int
        Which mobbing flock to visualize
    bird_id : int or None
        If specified, plot trajectory of that specific bird
    """
    data = load_flock(flock_number)
    frames = data['frames']
    
    # Extract trajectories
    times = []
    centroids = []
    bird_trajectory = []
    
    for frame in frames:
        times.append(frame['time'] - frames[0]['time'])
        centroids.append(np.mean(frame['positions'], axis=0))
        
        if bird_id is not None and bird_id in frame['bird_ids']:
            idx = np.where(frame['bird_ids'] == bird_id)[0][0]
            bird_trajectory.append(frame['positions'][idx])
    
    times = np.array(times)
    centroids = np.array(centroids)
    
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
             'b-', linewidth=1.5, alpha=0.7, label='Flock centroid')
    ax1.scatter(*centroids[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(*centroids[-1], c='red', s=100, marker='x', label='End')
    
    if bird_id is not None and len(bird_trajectory) > 0:
        bird_trajectory = np.array(bird_trajectory)
        ax1.plot(bird_trajectory[:, 0], bird_trajectory[:, 1], bird_trajectory[:, 2],
                'r-', linewidth=1, alpha=0.5, label=f'Bird {bird_id}')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Flock {flock_number} Trajectory ({data["duration"]:.1f}s)')
    ax1.legend()
    
    # Time series of position components
    ax2 = fig.add_subplot(122)
    ax2.plot(times, centroids[:, 0], 'r-', label='X', alpha=0.7)
    ax2.plot(times, centroids[:, 1], 'g-', label='Y', alpha=0.7)
    ax2.plot(times, centroids[:, 2], 'b-', label='Z', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Centroid Position Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/flock_{flock_number}_trajectory.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/flock_{flock_number}_trajectory.png")
    plt.show()


if __name__ == '__main__':
    # Parse command line arguments
    flock_number = 6
    if len(sys.argv) > 1:
        try:
            flock_number = int(sys.argv[1])
        except ValueError:
            print(f"Invalid flock number: {sys.argv[1]}")
            print("Usage: python visualize_real_flock.py [flock_number]")
            sys.exit(1)
    
    print("="*60)
    print(f"REAL FLOCK {flock_number} VISUALIZATION")
    print("="*60)
    
    # Create animation
    # Use skip_frames to speed up for long recordings (flock 6 has 8600+ frames)
    animate_flock(flock_number=flock_number, speed=2.0, skip_frames=5)
