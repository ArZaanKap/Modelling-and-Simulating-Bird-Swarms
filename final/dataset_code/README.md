# Dataset Code - Analysis Scripts

This folder contains **2 clean scripts** for analyzing real bird data and comparing with your models.

## Scripts

### 1. `visualize_flocks.py` - Flock Visualization

Visualize real bird flocking data from the dataset with static images or **live animation**.

```bash
# Default: Mobbing Flock 6 (static visualization)
python visualize_flocks.py

# Visualize specific flock (e.g., flock 7)
python visualize_flocks.py 7

# Visualize all available flocks
python visualize_flocks.py all

# LIVE ANIMATION of flock 6 (default)
python visualize_flocks.py animate

# LIVE ANIMATION of specific flock (e.g., flock 7)
python visualize_flocks.py 7 animate
```

**Outputs:**
- `flock_X_visualization.png` - 3D views and projections
- `flock_X_statistics.png` - Statistical distributions
- `all_flocks_overview.png` - Grid of all flocks (with 'all' option)
- **Live Animation** - Real-time playback of bird trajectories (with 'animate' option)

---

### 2. `model_comparison.py` - Model Comparison (Main Script)

Compares both models (Boids & Cucker-Smale) with real data (Flocks 6 & 7).

**Note:** Models are run WITHOUT obstacles or targets for a fair comparison with real data.
When running `boids4.py` or `cucker_smale2.py` directly, obstacles and targets ARE enabled.

```bash
python model_comparison.py
```

**Outputs:**
- `comparison_metrics.png` - Bar charts of key metrics
- `comparison_3d.png` - 3D flock visualizations
- `comparison_distributions.png` - Distribution analysis + error plots
- `comparison_summary.csv` - Data table for report

---

## Key Metrics Compared

| Metric | Description |
|--------|-------------|
| **Mean Speed** | Average velocity magnitude (m/s) |
| **Polarization** | Order parameter (0-1), measures alignment |
| **Neighbor Distance** | Mean distance to 7 nearest neighbors (m) |
| **Density** | Birds per cubic meter |

---

## Quick Start for Your Report

```bash
cd dataset_code
python model_comparison.py      # Generate comparison figures
python visualize_flocks.py      # Generate flock 6 visualization
python visualize_flocks.py 7    # Generate flock 7 visualization
```

Then use the generated PNG files in Section 3 of your report!
