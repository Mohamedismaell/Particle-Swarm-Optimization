import numpy as np
from .objective import objective_function


def execute_pso(devices, cloudlets, points, num_particles=30, iterations=100, weight=0.5, c1=1.5, c2=1.5):
    if not devices or not cloudlets:
        return None, None

    num_cloudlets = len(cloudlets)
    num_points = len(points)

    dim = num_cloudlets
    upper_bound = num_points + 2

    positions = np.random.uniform(0, upper_bound, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    personal_best_positions = positions.copy()
    personal_best_scores = np.array([float('inf')] * num_particles)

    global_best_position = positions[0].copy()
    global_best_score = float('inf')

    print("\nStarting Optimization (Smart PSO)...")

    for t in range(iterations):
        for i in range(num_particles):
            fitness = objective_function(
                positions[i], cloudlets, devices, points)

            if fitness < personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = positions[i].copy()

            if fitness < global_best_score:
                global_best_score = fitness
                global_best_position = positions[i].copy()

        r1, r2 = np.random.rand(
            dim), np.random.rand(dim)

        velocities = weight * velocities + c1 * r1 * (personal_best_positions - positions) + \
            c2 * r2 * (global_best_position - positions)

        positions = positions + velocities
        positions = np.clip(positions, 0, upper_bound)

        if t % 10 == 0:

            print(
                f"Iter {t}: Best Fitness = {global_best_score:.2f}", flush=True)
    return global_best_position, global_best_score
