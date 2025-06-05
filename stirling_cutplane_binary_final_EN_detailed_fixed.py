
import numpy as np
import math
from scipy.stats import linregress
from gurobipy import Model, GRB

# Parameters
lambda_val = 100 / 60  # Arrival rate (customers per minute)
mu_val = 1 / 15        # Service rate per server (customers per minute)
target_wait_time = 10  # Target average waiting time (in minutes)

# Log-domain Stirling approximation
def log_stirling(n):
    if n == 0:
        return 0.0
    return 0.5 * np.log(2 * np.pi * n) + n * np.log(n) - n

# Compute Wq(s) using log-domain Stirling
def compute_Wq_stirling_log(s, lam, mu):
    if s == 0 or lam / (s * mu) >= 1:
        return float('inf')
    a = lam / mu
rho0 = lam / mu  # Offered load (independent of s)
rho = lam / (s * mu)
    log_terms = np.array([n * np.log(a) - log_stirling(n) for n in range(s)])
    sum_terms = np.sum(np.exp(log_terms))
    log_last_term = s * np.log(a) - log_stirling(s) + np.log(1 / (1 - rho))
    last_term = np.exp(log_last_term)
    P0 = 1 / (sum_terms + last_term)
    log_pw_numerator = s * np.log(a) - log_stirling(s)
    Pw = np.exp(log_pw_numerator) * P0 / (1 - rho)
    Lq = Pw * rho / (1 - rho)
    Wq = Lq / lam
    return Wq

# Construct s*(rho) by searching feasible s under each rho
def compute_s_star_curve(rho_values, lam, mu, T):
    s_star_vals = []
    for rho in rho_values:
        s = int(np.ceil(lam / (mu * rho)))
        while compute_Wq_stirling_log(s, lam, mu) > T:
            s += 1
        s_star_vals.append(s)
    return s_star_vals

# Fit piecewise linear segments for s*(rho)
def fit_segments(rho_vals, s_vals, threshold=1.0):
    slopes = np.diff(s_vals) / np.diff(rho_vals)
    segments = []
    i = 0
    while i < len(rho_vals) - 2:
        x = rho_vals[i:i+3]
        y = s_vals[i:i+3]
        slope, intercept, _, _, _ = linregress(x, y)
        segments.append((x[0], x[-1], slope, intercept))
        i += 2
    return segments

# Solve the cut-plane MILP model using Gurobi
def solve_cut_plane(segments, lam, mu):
    model = Model("CutPlane")
    model.setParam('OutputFlag', 0)
    s_var = model.addVar(vtype=GRB.INTEGER, name="s", lb=1)
    for rho_lo, rho_hi, alpha, beta in segments:
        rho_mid = (rho_lo + rho_hi) / 2
        cut_val = alpha * rho_mid + beta
        model.addConstr(s_var >= cut_val)
    model.setObjective(s_var, GRB.MINIMIZE)
    model.optimize()
    return int(s_var.X)

# Perform binary search to refine the feasible solution
def binary_refine(s_upper, lam, mu, T):
    low = 1
    high = s_upper
    best_s = s_upper
    while low <= high:
        mid = (low + high) // 2
        Wq = compute_Wq_stirling_log(mid, lam, mu)
        if Wq <= T:
            best_s = mid
            high = mid - 1
        else:
            low = mid + 1
    return best_s, compute_Wq_stirling_log(best_s, lam, mu)

# Main execution
if __name__ == "__main__":
    rho_vals = np.round(np.arange(0.70, 0.951, 0.01), 3)
    s_star_vals = compute_s_star_curve(rho_vals, lambda_val, mu_val, target_wait_time)
    segments = fit_segments(rho_vals, s_star_vals)
    s_cut = solve_cut_plane(segments, lambda_val, mu_val)
    s_final, Wq_final = binary_refine(s_cut, lambda_val, mu_val, target_wait_time)

    print("Cut-plane solution (upper bound):", s_cut)
    print("Refined optimal number of servers:", s_final)
    print("Verified average waiting time (min):", round(Wq_final, 4))
