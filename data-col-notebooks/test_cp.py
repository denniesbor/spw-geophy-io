import numpy as np
import cupy as cp
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky, solve
import time
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CPU implementations
def solve_eqn_cpu(Y, injection_currents):
    """CPU version of equation solver."""
    logger.info("CPU: Solving for nodal voltages...")
    start = time.perf_counter()

    Y_reg = Y.T @ Y + 1e-20 * eye(Y.shape[1])
    V_nodal = spsolve(Y_reg, Y.T @ injection_currents)

    elapsed = time.perf_counter() - start
    logger.info(f"CPU: Nodal voltages solved in {elapsed:.4f} seconds")
    return V_nodal, elapsed


def get_and_solve_cholesky_cpu(Y, Je):
    """CPU version of Cholesky solver."""
    logger.info("CPU: Solving using Cholesky...")
    start = time.perf_counter()

    regularization = 1e-6
    Y_reg = Y + np.eye(Y.shape[0]) * regularization
    L = cholesky(Y_reg, lower=True)
    P = solve(L, Je)
    V_n = solve(L.T, P)

    elapsed = time.perf_counter() - start
    logger.info(f"CPU: Cholesky solution found in {elapsed:.4f} seconds")
    return V_n, elapsed


# GPU implementations
def solve_eqn_gpu(Y, injection_currents):
    """GPU version of equation solver."""
    logger.info("GPU: Solving for nodal voltages...")
    start = time.perf_counter()

    # Transfer to GPU
    Y_gpu = cp.asarray(Y)
    injection_currents_gpu = cp.asarray(injection_currents)

    # Solve on GPU
    Y_reg = Y_gpu.T @ Y_gpu + 1e-20 * cp.eye(Y_gpu.shape[1])
    V_nodal = cp.linalg.solve(Y_reg, Y_gpu.T @ injection_currents_gpu)

    # Transfer back to CPU
    result = cp.asnumpy(V_nodal)

    elapsed = time.perf_counter() - start
    logger.info(f"GPU: Nodal voltages solved in {elapsed:.4f} seconds")
    return result, elapsed


def get_and_solve_cholesky_gpu(Y, Je):
    """GPU version of Cholesky solver."""
    logger.info("GPU: Solving using Cholesky...")
    start = time.perf_counter()

    # Transfer to GPU
    Y_gpu = cp.asarray(Y)
    Je_gpu = cp.asarray(Je)

    regularization = 1e-6
    Y_reg = Y_gpu + cp.eye(Y_gpu.shape[0]) * regularization
    L = cp.linalg.cholesky(Y_reg, lower=True)
    P = cp.linalg.solve(L, Je_gpu)
    V_n = cp.linalg.solve(L.T, P)

    # Transfer back to CPU
    result = cp.asnumpy(V_n)

    elapsed = time.perf_counter() - start
    logger.info(f"GPU: Cholesky solution found in {elapsed:.4f} seconds")
    return result, elapsed


def compare_solvers(sizes=[1000, 2000, 5000, 10000]):
    """Compare CPU and GPU implementations across different matrix sizes."""
    results = {
        "size": [],
        "cpu_basic": [],
        "cpu_cholesky": [],
        "gpu_basic": [],
        "gpu_cholesky": [],
        "max_diff_basic": [],
        "max_diff_cholesky": [],
    }

    for size in sizes:
        logger.info(f"\nTesting with matrix size {size}x{size}")

        # Generate test data
        Y = np.random.rand(size, size)
        Je = np.random.rand(size)

        # Clear GPU memory before each test
        cp.get_default_memory_pool().free_all_blocks()

        # Run all solvers
        try:
            # Basic solver
            cpu_result_basic, cpu_time_basic = solve_eqn_cpu(Y, Je)
            gpu_result_basic, gpu_time_basic = solve_eqn_gpu(Y, Je)

            # Cholesky solver
            cpu_result_chol, cpu_time_chol = get_and_solve_cholesky_cpu(Y, Je)
            gpu_result_chol, gpu_time_chol = get_and_solve_cholesky_gpu(Y, Je)

            # Compare results
            max_diff_basic = np.max(np.abs(cpu_result_basic - gpu_result_basic))
            max_diff_chol = np.max(np.abs(cpu_result_chol - gpu_result_chol))

            # Store results
            results["size"].append(size)
            results["cpu_basic"].append(cpu_time_basic)
            results["cpu_cholesky"].append(cpu_time_chol)
            results["gpu_basic"].append(gpu_time_basic)
            results["gpu_cholesky"].append(gpu_time_chol)
            results["max_diff_basic"].append(max_diff_basic)
            results["max_diff_cholesky"].append(max_diff_chol)

            logger.info(f"Maximum difference (basic): {max_diff_basic:.2e}")
            logger.info(f"Maximum difference (Cholesky): {max_diff_chol:.2e}")

        except Exception as e:
            logger.error(f"Error with size {size}: {str(e)}")
            continue

    return results


def plot_results(results):
    """Plot performance comparison results."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(results["size"], results["cpu_basic"], "b-", label="CPU Basic")
    plt.plot(results["size"], results["gpu_basic"], "r--", label="GPU Basic")
    plt.plot(results["size"], results["cpu_cholesky"], "g-", label="CPU Cholesky")
    plt.plot(results["size"], results["gpu_cholesky"], "m--", label="GPU Cholesky")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (seconds)")
    plt.title("Solver Performance Comparison")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(results["size"], results["max_diff_basic"], "b-", label="Basic Solver")
    plt.semilogy(results["size"], results["max_diff_cholesky"], "r--", label="Cholesky")
    plt.xlabel("Matrix Size")
    plt.ylabel("Maximum Difference")
    plt.title("Numerical Difference Between CPU and GPU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run comparison
    sizes = [1000, 2000, 5000, 10000]  # Adjust based on your GPU memory
    results = compare_solvers(sizes)

    # Plot results
    plot_results(results)

    # Print summary
    print("\nPerformance Summary:")
    for i, size in enumerate(results["size"]):
        print(f"\nMatrix size: {size}x{size}")
        print(
            f"Basic Solver - CPU: {results['cpu_basic'][i]:.4f}s, GPU: {results['gpu_basic'][i]:.4f}s"
        )
        print(
            f"Cholesky Solver - CPU: {results['cpu_cholesky'][i]:.4f}s, GPU: {results['gpu_cholesky'][i]:.4f}s"
        )
        print(
            f"Maximum differences - Basic: {results['max_diff_basic'][i]:.2e}, Cholesky: {results['max_diff_cholesky'][i]:.2e}"
        )
