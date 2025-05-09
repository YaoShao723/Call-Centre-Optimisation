
import math

# Stirling approximation for factorial
def stirling_approx(n):
    if n == 0:
        return 1.0
    return math.sqrt(2 * math.pi * n) * (n / math.e) ** n

# Erlang-C formula for average waiting time Wq using Stirling approximation
def Wq_stirling(lambda_val, mu_val, s):
    rho = lambda_val / (s * mu_val)
    if rho >= 1:
        return float('inf')
    a = lambda_val / mu_val
    sum_terms = sum((a ** n) / stirling_approx(n) for n in range(s))
    last_term = (a ** s) / stirling_approx(s) * (1 / (1 - rho))
    P0 = 1 / (sum_terms + last_term)
    Pw = ((a ** s) / stirling_approx(s)) * P0 / (1 - rho)
    Lq = Pw * rho / (1 - rho)
    return Lq / lambda_val  # Wq

# Binary search to find the smallest number of servers s such that Wq <= target
def binary_search_staff(lambda_val, mu_val, target_wq_minutes):
    target_wq = target_wq_minutes  # corrected: already in minutes  # convert minutes to hours
    s_min = int(math.floor(lambda_val / mu_val)) + 1
    s_max = 200
    result = -1

    while s_min <= s_max:
        mid = (s_min + s_max) // 2
        wq = Wq_stirling(lambda_val, mu_val, mid)
        if wq <= target_wq:
            result = mid
            s_max = mid - 1
        else:
            s_min = mid + 1

    return result

# Example usage
if __name__ == "__main__":
    lambda_val = 100 / 60  # arrivals per minute
    mu_val = 1 / 15        # service rate per minute per server
    target_wq_minutes = 10  # target wait time in minutes

    optimal_s = binary_search_staff(lambda_val, mu_val, target_wq_minutes)
    print("Minimum number of servers needed:", optimal_s)
