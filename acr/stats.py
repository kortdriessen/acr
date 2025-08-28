import numpy as np
import math
import os
import pandas as pd
import acr.utils

def calculate_wilcoxon_z(W_statistic, N, ties=None, continuity_correction=True):
    """
    Computes the Z-statistic for a Wilcoxon signed-rank test.

    Parameters:
    - W_statistic (float): The sum of ranks for one group (e.g., sum of positive ranks).
                           If your W is min(W_pos, W_neg), you should use the sum of positive ranks
                           or sum of negative ranks here. For instance, if W_pos + W_neg = N(N+1)/2,
                           and your W_statistic is W_pos.
    - N (int): The number of pairs with non-zero differences.
    - ties (list of int, optional): A list where each element is the count of
                                    observations in a group of tied absolute ranks.
                                    Example: if there are two differences tied for one rank,
                                    and three differences tied for another rank, ties = [2, 3].
                                    Defaults to None (no ties).
    - continuity_correction (bool): Whether to apply the continuity correction.
                                    Defaults to True.

    Returns:
    - float: The calculated Z-statistic.
    - None: If N is too small or sigma_W is zero.
    """

    if N <= 0:
        print("N must be greater than 0.")
        return None

    mu_W = N * (N + 1) / 4

    # Calculate variance
    variance_W_no_ties = N * (N + 1) * (2 * N + 1) / 24

    tie_correction_term = 0
    if ties:
        for t_j in ties:
            tie_correction_term += (t_j**3 - t_j)
    
    variance_W = variance_W_no_ties - (tie_correction_term / 48)

    if variance_W <= 0: # Avoid division by zero or sqrt of negative
        print("Variance is zero or negative. Cannot compute Z-statistic (check N and ties).")
        return None
        
    sigma_W = math.sqrt(variance_W)

    if sigma_W == 0:
        print("Standard deviation (sigma_W) is zero. Cannot compute Z-statistic.")
        return None
    # Calculate Z-statistic with continuity correction
    if continuity_correction:
        if W_statistic > mu_W:
            z_val = (W_statistic - mu_W - 0.5) / sigma_W
        elif W_statistic < mu_W:
            z_val = (W_statistic - mu_W + 0.5) / sigma_W
        else: # W_statistic == mu_W
            z_val = 0.0
    else:
        z_val = (W_statistic - mu_W) / sigma_W
        
    return z_val

def calculate_wilx_r(W_statistic, N, ties=None, continuity_correction=True):
    z = calculate_wilcoxon_z(W_statistic, N, ties, continuity_correction)
    return np.abs(z)/np.sqrt(N)



def write_stats_result(test_name, test_type, test_statistic, p_value, effect_size_method, effect_size, notes=''):
    stat_path = os.path.join(acr.utils.materials_root, "stats_summary.xlsx")

    # convert any array-like effect_size to a string for Excel compatibility
    if isinstance(effect_size, (np.ndarray, list, tuple)):
        effect_size = ",".join(map(str, effect_size))

    # if the file exists, load it; otherwise start a fresh DataFrame
    if os.path.exists(stat_path):
        df = pd.read_excel(stat_path)
    else:
        df = pd.DataFrame(columns=[
            "test_name",
            "test_type",
            "test_statistic",
            "p_value",
            "effect_size_method",
            "effect_size",
            "notes"
        ])

    # if test_name already exists, update the row
    if test_name in df['test_name'].values:
        df.loc[df['test_name'] == test_name, 'test_type'] = test_type
        df.loc[df['test_name'] == test_name, 'test_statistic'] = test_statistic
        df.loc[df['test_name'] == test_name, 'p_value'] = p_value
        df.loc[df['test_name'] == test_name, 'effect_size_method'] = effect_size_method
        df.loc[df['test_name'] == test_name, 'effect_size'] = effect_size
        df.loc[df['test_name'] == test_name, 'notes'] = notes
    
    else:
        # build the new row
        new_row = {
            "test_name": test_name,
            "test_type": test_type,
            "test_statistic": test_statistic,
            "p_value": p_value,
            "effect_size_method": effect_size_method,
            "effect_size": effect_size,
            "notes": notes,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # save
    df.to_excel(stat_path, index=False)