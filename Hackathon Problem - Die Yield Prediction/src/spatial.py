import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

def compute_spatial_features(df_subset, full_df=None):
    """Compute spatial features: radial distance, 3x3/5x5 pre-test fail density, edge distance."""
    if full_df is None:
        full_df = df_subset

    spatial_list = []
    
    for wid, group in df_subset.groupby('wafer_id'):
        rows = group['die_row'].values
        cols = group['die_col'].values
        
        r_max = max(rows.max(), 1)
        c_max = max(cols.max(), 1)
        cy, cx = r_max / 2.0, c_max / 2.0
        
        # Normalized radial distance
        rad = np.sqrt((rows - cy)**2 + (cols - cx)**2)
        max_r = rad.max() if rad.max() > 0 else 1.0
        norm_rad = rad / max_r
        
        # Query full wafer pre-test map
        full_wafer = full_df[full_df['wafer_id'] == wid]
        grid_rows = full_wafer['die_row'].max() + 1
        grid_cols = full_wafer['die_col'].max() + 1
        
        old_fail_grid = np.zeros((grid_rows, grid_cols), dtype=float)
        valid_grid = np.zeros((grid_rows, grid_cols), dtype=float)
        
        old_fail_grid[full_wafer['die_row'].values, full_wafer['die_col'].values] = full_wafer['old_label'].values
        valid_grid[full_wafer['die_row'].values, full_wafer['die_col'].values] = 1.0
        
        # 3x3 fail density
        fail_3 = uniform_filter(old_fail_grid, size=3, mode='constant', cval=0.0)
        val_3 = uniform_filter(valid_grid, size=3, mode='constant', cval=0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            dens_3 = np.where(val_3 > 0, fail_3 / val_3, 0.0)
            
        # 5x5 fail density
        fail_5 = uniform_filter(old_fail_grid, size=5, mode='constant', cval=0.0)
        val_5 = uniform_filter(valid_grid, size=5, mode='constant', cval=0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            dens_5 = np.where(val_5 > 0, fail_5 / val_5, 0.0)
            
        d3 = dens_3[rows, cols]
        d5 = dens_5[rows, cols]
        
        # Edge distance
        edge_dist = np.minimum(np.minimum(rows, r_max - rows), np.minimum(cols, c_max - cols))
        norm_edge_dist = edge_dist / (max(r_max, c_max) / 2.0)
        
        sub_sp = pd.DataFrame({
            'radial_dist': norm_rad,
            'density_3x3': d3,
            'density_5x5': d5,
            'edge_dist': norm_edge_dist,
            'die_row': rows,
            'die_col': cols
        }, index=group.index)
        spatial_list.append(sub_sp)
        
    return pd.concat(spatial_list).sort_index()
