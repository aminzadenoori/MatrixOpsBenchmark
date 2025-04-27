import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
from pathlib import Path

cp = None
# Try to import CuPy; fall back to NumPy if not available
try:
    import cupy as cp
    xp = cp  # Use CuPy for GPU if available
    print("Using CuPy (GPU)")
except ImportError:
    xp = np  # Fall back to NumPy (CPU)
    print("Using NumPy (CPU)")

# Matrix sizes to test
matrix_sizes = [1000, 2000, 3000, 4000, 5000]

# Reference results (F64 on CPU for dense and sparse), matrices, and times
reference_results = {}
reference_matrices = {}
reference_times = {
    "dense": {
        "elemwise": [], "exp": [], "mean": [], "matprod": [], "inv": [], "svd": []
    },
    "sparse": {
        "elemwise": [], "exp": [], "mean": [], "matprod": []
    }
}

# Timing results for plotting (exclude CPU_F64 for dense since it's the reference)
timings = {
    "elemwise": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "exp": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "mean": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "matprod": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "inv": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "svd": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
}

# Sparse timing results (include CPU_F64 since F64 sparse is experimented)
sparse_timings = {
    "elemwise": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": [], "CPU_F64": []},
    "exp": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": [], "CPU_F64": []},
    "mean": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": [], "CPU_F64": []},
    "matprod": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": [], "CPU_F64": []},
}

# Error results for plotting (relative errors vs CPU F64 reference)
errors = {
    "elemwise": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "exp": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "mean": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "matprod": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "inv": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
    "svd": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": []},
}

# Sparse error results
sparse_errors = {
    "elemwise": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": [], "CPU_F64": []},
    "exp": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": [], "CPU_F64": []},
    "mean": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": [], "CPU_F64": []},
    "matprod": {"GPU_F32": [], "GPU_F64": [], "CPU_F32": [], "CPU_F64": []},
}

# Function to convert between CuPy and NumPy arrays
def to_numpy(array):
    if xp == cp:
        return array.get()  # Transfer CuPy array to NumPy
    return array

# Function to run experiments for a given precision and matrix type
def run_experiments(dtype, label, sparse=False, size_idx=0):
    print(f"\nRunning experiments with {label} precision {'(sparse)' if sparse else '(non-sparse)'}...")
    timings_dict = sparse_timings if sparse else timings
    errors_dict = sparse_errors if sparse else errors
    N = matrix_sizes[size_idx]
    print(f"\nMatrix size: {N}x{N}")
    
    # Retrieve reference matrices
    A_ref, B_ref = reference_matrices[f"matrices_{N}_{'sparse' if sparse else 'dense'}"]
    
    # Convert matrices to the appropriate dtype and device
    if sparse:
        A = A_ref.astype(dtype)
        B = B_ref.astype(dtype)
        if xp == cp:
            A = cp.sparse.csr_matrix(A)
            B = cp.sparse.csr_matrix(B)
    else:
        A = xp.array(A_ref.astype(dtype))
        B = xp.array(B_ref.astype(dtype))

    # 1. Element-wise product
    start = time.time()
    if sparse:
        C = A.multiply(B)
    else:
        C = A * B
    if xp == cp:
        cp.cuda.Stream.null.synchronize()
    elemwise_time = time.time() - start
    print(f"Element-wise product time: {elemwise_time:.4f} seconds")
    C_ref = reference_results[f"elemwise_{N}_{'sparse' if sparse else 'dense'}"]
    C_np = to_numpy(C.todense() if sparse else C)
    error = np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref) if np.linalg.norm(C_ref) > 0 else 0
    print(f"Relative error vs CPU F64: {error:.6e}")
    timings_dict["elemwise"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(elemwise_time)
    errors_dict["elemwise"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(error)

    # 2. Exponential of each element
    start = time.time()
    if sparse:
        expA = A.copy()
        expA.data = xp.exp(expA.data)
    else:
        expA = xp.exp(A)
    if xp == cp:
        cp.cuda.Stream.null.synchronize()
    exp_time = time.time() - start
    print(f"Exponential time: {exp_time:.4f} seconds")
    expA_ref = reference_results[f"exp_{N}_{'sparse' if sparse else 'dense'}"]
    expA_np = to_numpy(expA.todense() if sparse else expA)
    error = np.linalg.norm(expA_np - expA_ref) / np.linalg.norm(expA_ref) if np.linalg.norm(expA_ref) > 0 else 0
    print(f"Relative error vs CPU F64: {error:.6e}")
    timings_dict["exp"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(exp_time)
    errors_dict["exp"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(error)

    # 3. Means of the rows
    start = time.time()
    if sparse:
        meanRows = xp.array(A.mean(axis=1)).flatten()
    else:
        meanRows = xp.mean(A, axis=1)
    if xp == cp:
        cp.cuda.Stream.null.synchronize()
    mean_time = time.time() - start
    print(f"Means of rows time: {mean_time:.4f} seconds")
    meanRows_ref = reference_results[f"mean_{N}_{'sparse' if sparse else 'dense'}"]
    meanRows_np = to_numpy(meanRows)
    error = np.linalg.norm(meanRows_np - meanRows_ref) / np.linalg.norm(meanRows_ref) if np.linalg.norm(meanRows_ref) > 0 else 0
    print(f"Relative error vs CPU F64: {error:.6e}")
    timings_dict["mean"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(mean_time)
    errors_dict["mean"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(error)

    # 4. Matrix product (C = A * B)
    start = time.time()
    C = A @ B
    if xp == cp:
        cp.cuda.Stream.null.synchronize()
    matprod_time = time.time() - start
    print(f"Matrix product time: {matprod_time:.4f} seconds")
    C_ref = reference_results[f"matprod_{N}_{'sparse' if sparse else 'dense'}"]
    C_np = to_numpy(C.todense() if sparse else C)
    error = np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref) if np.linalg.norm(C_ref) > 0 else 0
    print(f"Relative error vs CPU F64: {error:.6e}")
    timings_dict["matprod"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(matprod_time)
    errors_dict["matprod"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(error)

    # 5. Inverse of a matrix (only for non-sparse)
    if not sparse:
        start = time.time()
        invA = xp.linalg.inv(A)
        if xp == cp:
            cp.cuda.Stream.null.synchronize()
        inv_time = time.time() - start
        print(f"Matrix inverse time: {inv_time:.4f} seconds")
        invA_ref = reference_results[f"inv_{N}_dense"]
        invA_np = to_numpy(invA)
        error = np.linalg.norm(invA_np - invA_ref) / np.linalg.norm(invA_ref) if np.linalg.norm(invA_ref) > 0 else 0
        print(f"Relative error vs CPU F64: {error:.6e}")
        timings["inv"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(inv_time)
        errors["inv"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(error)
    else:
        print("Matrix inverse skipped for sparse matrices.")

    # 6. Singular Value Decomposition (SVD) (only for non-sparse)
    if not sparse:
        start = time.time()
        U, S, VT = xp.linalg.svd(A, full_matrices=True)
        if xp == cp:
            cp.cuda.Stream.null.synchronize()
        svd_time = time.time() - start
        print(f"SVD time: {svd_time:.4f} seconds")
        S_ref = reference_results[f"svd_S_{N}_dense"]
        S_np = to_numpy(S)
        error = np.linalg.norm(S_np - S_ref) / np.linalg.norm(S_ref) if np.linalg.norm(S_ref) > 0 else 0
        print(f"SVD (S) relative error vs CPU F64: {error:.6e}")
        timings["svd"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(svd_time)
        errors["svd"][f"{'GPU' if xp == cp else 'CPU'}_{label}"].append(error)
    else:
        print("SVD skipped for sparse matrices.")

    # CPU F32 execution for comparison (only if running on GPU)
    if xp == cp:
        print("\nRunning CPU F32 for comparison...")
        A_np = A_ref.astype(np.float32)
        B_np = B_ref.astype(np.float32)
        if sparse:
            A_np = sp.csr_matrix(A_np)
            B_np = sp.csr_matrix(B_np)

        # 1. Element-wise product
        start = time.time()
        if sparse:
            C_np = A_np.multiply(B_np)
        else:
            C_np = A_np * B_np
        cpu_elemwise_time = time.time() - start
        print(f"CPU F32 Element-wise product time: {cpu_elemwise_time:.4f} seconds")
        print(f"Speedup (GPU vs CPU): {cpu_elemwise_time / elemwise_time:.2f}x")
        C_ref = reference_results[f"elemwise_{N}_{'sparse' if sparse else 'dense'}"]
        C_np = C_np.todense() if sparse else C_np
        error = np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref) if np.linalg.norm(C_ref) > 0 else 0
        print(f"CPU F32 Relative error vs CPU F64: {error:.6e}")
        timings_dict["elemwise"]["CPU_F32"].append(cpu_elemwise_time)
        errors_dict["elemwise"]["CPU_F32"].append(error)

        # 2. Exponential
        start = time.time()
        if sparse:
            expA_np = A_np.copy()
            expA_np.data = np.exp(expA_np.data)
        else:
            expA_np = np.exp(A_np)
        cpu_exp_time = time.time() - start
        print(f"CPU F32 Exponential time: {cpu_exp_time:.4f} seconds")
        print(f"Speedup (GPU vs CPU): {cpu_exp_time / exp_time:.2f}x")
        expA_ref = reference_results[f"exp_{N}_{'sparse' if sparse else 'dense'}"]
        expA_np = expA_np.todense() if sparse else expA_np
        error = np.linalg.norm(expA_np - expA_ref) / np.linalg.norm(expA_ref) if np.linalg.norm(expA_ref) > 0 else 0
        print(f"CPU F32 Relative error vs CPU F64: {error:.6e}")
        timings_dict["exp"]["CPU_F32"].append(cpu_exp_time)
        errors_dict["exp"]["CPU_F32"].append(error)

        # 3. Means of the rows
        start = time.time()
        if sparse:
            meanRows_np = np.array(A_np.mean(axis=1)).flatten()
        else:
            meanRows_np = np.mean(A_np, axis=1)
        cpu_mean_time = time.time() - start
        print(f"CPU F32 Means of rows time: {cpu_mean_time:.4f} seconds")
        print(f"Speedup (GPU vs CPU): {cpu_mean_time / mean_time:.2f}x")
        meanRows_ref = reference_results[f"mean_{N}_{'sparse' if sparse else 'dense'}"]
        error = np.linalg.norm(meanRows_np - meanRows_ref) / np.linalg.norm(meanRows_ref) if np.linalg.norm(meanRows_ref) > 0 else 0
        print(f"CPU F32 Relative error vs CPU F64: {error:.6e}")
        timings_dict["mean"]["CPU_F32"].append(cpu_mean_time)
        errors_dict["mean"]["CPU_F32"].append(error)

        # 4. Matrix product
        start = time.time()
        C_np = A_np @ B_np
        cpu_matprod_time = time.time() - start
        print(f"CPU F32 Matrix product time: {cpu_matprod_time:.4f} seconds")
        print(f"Speedup (GPU vs CPU): {cpu_matprod_time / matprod_time:.2f}x")
        C_ref = reference_results[f"matprod_{N}_{'sparse' if sparse else 'dense'}"]
        C_np = C_np.todense() if sparse else C_np
        error = np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref) if np.linalg.norm(C_ref) > 0 else 0
        print(f"CPU F32 Relative error vs CPU F64: {error:.6e}")
        timings_dict["matprod"]["CPU_F32"].append(cpu_matprod_time)
        errors_dict["matprod"]["CPU_F32"].append(error)

        # 5. Inverse (non-sparse only)
        if not sparse:
            start = time.time()
            invA_np = np.linalg.inv(A_np)
            cpu_inv_time = time.time() - start
            print(f"CPU F32 Matrix inverse time: {cpu_inv_time:.4f} seconds")
            print(f"Speedup (GPU vs CPU): {cpu_inv_time / inv_time:.2f}x")
            invA_ref = reference_results[f"inv_{N}_dense"]
            error = np.linalg.norm(invA_np - invA_ref) / np.linalg.norm(invA_ref) if np.linalg.norm(invA_ref) > 0 else 0
            print(f"CPU F32 Relative error vs CPU F64: {error:.6e}")
            timings["inv"]["CPU_F32"].append(cpu_inv_time)
            errors["inv"]["CPU_F32"].append(error)

        # 6. SVD (non-sparse only)
        if not sparse:
            start = time.time()
            U_np, S_np, VT_np = np.linalg.svd(A_np, full_matrices=True)
            cpu_svd_time = time.time() - start
            print(f"CPU F32 SVD time: {cpu_svd_time:.4f} seconds")
            print(f"Speedup (GPU vs CPU): {cpu_svd_time / svd_time:.2f}x")
            S_ref = reference_results[f"svd_S_{N}_dense"]
            error = np.linalg.norm(S_np - S_ref) / np.linalg.norm(S_ref) if np.linalg.norm(S_ref) > 0 else 0
            print(f"CPU F32 Relative error vs CPU F64: {error:.6e}")
            timings["svd"]["CPU_F32"].append(cpu_svd_time)
            errors["svd"]["CPU_F32"].append(error)

# Generate CPU F64 reference results, store matrices, and measure times
print("Generating CPU F64 reference results, storing matrices, and measuring times...")
for N in matrix_sizes:
    for sparse in [False, True]:
        if sparse:
            A = sp.random(N, N, density=0.01, format='csr', dtype=np.float64)
            B = sp.random(N, N, density=0.01, format='csr', dtype=np.float64)
            reference_matrices[f"matrices_{N}_sparse"] = (A, B)
            
            # Element-wise product
            start = time.time()
            reference_results[f"elemwise_{N}_sparse"] = (A.multiply(B)).todense()
            reference_times["sparse"]["elemwise"].append(time.time() - start)
            
            # Exponential
            start = time.time()
            expA = A.copy()
            expA.data = np.exp(expA.data)
            reference_times["sparse"]["exp"].append(time.time() - start)
            reference_results[f"exp_{N}_sparse"] = expA.todense()
            
            # Mean
            start = time.time()
            reference_results[f"mean_{N}_sparse"] = np.array(A.mean(axis=1)).flatten()
            reference_times["sparse"]["mean"].append(time.time() - start)
            
            # Matrix product
            start = time.time()
            reference_results[f"matprod_{N}_sparse"] = (A @ B).todense()
            reference_times["sparse"]["matprod"].append(time.time() - start)
        else:
            A = np.random.rand(N, N).astype(np.float64)
            B = np.random.rand(N, N).astype(np.float64)
            reference_matrices[f"matrices_{N}_dense"] = (A, B)
            
            # Element-wise product
            start = time.time()
            reference_results[f"elemwise_{N}_dense"] = (A * B).astype(np.float64)
            reference_times["dense"]["elemwise"].append(time.time() - start)
            
            # Exponential
            start = time.time()
            reference_results[f"exp_{N}_dense"] = np.exp(A).astype(np.float64)
            reference_times["dense"]["exp"].append(time.time() - start)
            
            # Mean
            start = time.time()
            reference_results[f"mean_{N}_dense"] = np.mean(A, axis=1).astype(np.float64)
            reference_times["dense"]["mean"].append(time.time() - start)
            
            # Matrix product
            start = time.time()
            reference_results[f"matprod_{N}_dense"] = np.dot(A, B).astype(np.float64)
            reference_times["dense"]["matprod"].append(time.time() - start)
            
            # Inverse
            start = time.time()
            reference_results[f"inv_{N}_dense"] = np.linalg.inv(A).astype(np.float64)
            reference_times["dense"]["inv"].append(time.time() - start)
            
            # SVD
            start = time.time()
            U, S, VT = np.linalg.svd(A, full_matrices=True)
            reference_results[f"svd_S_{N}_dense"] = S.astype(np.float64)
            reference_times["dense"]["svd"].append(time.time() - start)

# Initialize timing and error lists
for op in timings.values():
    for key in op:
        op[key] = []
for op in sparse_timings.values():
    for key in op:
        op[key] = []
for op in errors.values():
    for key in op:
        op[key] = []
for op in sparse_errors.values():
    for key in op:
        op[key] = []

# Run experiments (exclude CPU F64 for dense)
for size_idx, N in enumerate(matrix_sizes):
    # Dense experiments
    run_experiments(np.float32, "F32", sparse=False, size_idx=size_idx)
    if xp == cp:  # Run GPU F64 if CuPy is available
        run_experiments(np.float64, "F64", sparse=False, size_idx=size_idx)
    # Sparse experiments
    run_experiments(np.float32, "F32", sparse=True, size_idx=size_idx)
    run_experiments(np.float64, "F64", sparse=True, size_idx=size_idx)

# Plotting
operations = [
    ("elemwise", "Element-wise product"),
    ("exp", "Exponential of each element"),
    ("mean", "Means of the rows"),
    ("matprod", "Matrix product"),
    ("inv", "Inverse of a matrix"),
    ("svd", "Singular Value Decomposition")
]

# Create output directory for plots
Path("plots").mkdir(exist_ok=True)

# Plot dense matrix timing results (relative to CPU F64)
plt.figure(figsize=(15, 10))
for i, (op_key, op_name) in enumerate(operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (Relative Time)")
    plt.xlabel("Matrix size n×n")
    plt.ylabel("Time relative to CPU F64")
    plt.xticks(matrix_sizes)
    
    ref_times = reference_times["dense"][op_key]
    if xp == cp:
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(timings[op_key]["GPU_F32"], ref_times)], 
                 marker='o', label="GPU F32", color="blue")
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(timings[op_key]["GPU_F64"], ref_times)], 
                 marker='^', label="GPU F64", color="red")
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(timings[op_key]["CPU_F32"], ref_times)], 
                 marker='s', label="CPU F32", color="purple")
    else:
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(timings[op_key]["CPU_F32"], ref_times)], 
                 marker='s', label="CPU F32", color="purple")
    
    plt.legend()
    plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("plots/dense_matrix_timing_results.png")
plt.close()

# Plot dense matrix accuracy results
plt.figure(figsize=(15, 10))
for i, (op_key, op_name) in enumerate(operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (Relative Error)")
    plt.xlabel("Matrix size n×n")
    plt.ylabel("Relative Error (log-scale)")
    plt.yscale("log")
    plt.xticks(matrix_sizes)
    
    if xp == cp:
        plt.plot(matrix_sizes, errors[op_key]["GPU_F32"], marker='o', label="GPU F32", color="blue")
        plt.plot(matrix_sizes, errors[op_key]["GPU_F64"], marker='^', label="GPU F64", color="red")
        plt.plot(matrix_sizes, errors[op_key]["CPU_F32"], marker='s', label="CPU F32", color="purple")
    else:
        plt.plot(matrix_sizes, errors[op_key]["CPU_F32"], marker='s', label="CPU F32", color="purple")
    
    plt.legend()
    plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("plots/dense_matrix_accuracy_results.png")
plt.close()

# Plot sparse matrix timing results (relative to CPU F64)
plt.figure(figsize=(15, 7))
sparse_operations = operations[:4]  # Exclude inverse and SVD for sparse
for i, (op_key, op_name) in enumerate(sparse_operations, 1):
    plt.subplot(2, 2, i)
    plt.title(f"{op_name} (Sparse, Relative Time)")
    plt.xlabel("Matrix size n×n")
    plt.ylabel("Time relative to CPU F64")
    plt.xticks(matrix_sizes)
    
    ref_times = reference_times["sparse"][op_key]
    if xp == cp:
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(sparse_timings[op_key]["GPU_F32"], ref_times)], 
                 marker='o', label="GPU F32", color="blue")
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(sparse_timings[op_key]["GPU_F64"], ref_times)], 
                 marker='^', label="GPU F64", color="red")
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(sparse_timings[op_key]["CPU_F32"], ref_times)], 
                 marker='s', label="CPU F32", color="purple")
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(sparse_timings[op_key]["CPU_F64"], ref_times)], 
                 marker='d', label="CPU F64", color="green")
    else:
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(sparse_timings[op_key]["CPU_F32"], ref_times)], 
                 marker='s', label="CPU F32", color="purple")
        plt.plot(matrix_sizes, [t / r if r > 0 else 1 for t, r in zip(sparse_timings[op_key]["CPU_F64"], ref_times)], 
                 marker='d', label="CPU F64", color="green")
    
    plt.legend()
    plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("plots/sparse_matrix_timing_results.png")
plt.close()

# Plot sparse matrix accuracy results
plt.figure(figsize=(15, 7))
for i, (op_key, op_name) in enumerate(sparse_operations, 1):
    plt.subplot(2, 2, i)
    plt.title(f"{op_name} (Sparse, Relative Error)")
    plt.xlabel("Matrix size n×n")
    plt.ylabel("Relative Error (log-scale)")
    plt.yscale("log")
    plt.xticks(matrix_sizes)
    
    if xp == cp:
        plt.plot(matrix_sizes, sparse_errors[op_key]["GPU_F32"], marker='o', label="GPU F32", color="blue")
        plt.plot(matrix_sizes, sparse_errors[op_key]["GPU_F64"], marker='^', label="GPU F64", color="red")
        plt.plot(matrix_sizes, sparse_errors[op_key]["CPU_F32"], marker='s', label="CPU F32", color="purple")
        plt.plot(matrix_sizes, sparse_errors[op_key]["CPU_F64"], marker='d', label="CPU F64", color="green")
    else:
        plt.plot(matrix_sizes, sparse_errors[op_key]["CPU_F32"], marker='s', label="CPU F32", color="purple")
        plt.plot(matrix_sizes, sparse_errors[op_key]["CPU_F64"], marker='d', label="CPU F64", color="green")
    
    plt.legend()
    plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("plots/sparse_matrix_accuracy_results.png")
plt.close()

print("\nPlots saved in 'plots' directory: dense_matrix_timing_results.png, dense_matrix_accuracy_results.png, sparse_matrix_timing_results.png, sparse_matrix_accuracy_results.png")
