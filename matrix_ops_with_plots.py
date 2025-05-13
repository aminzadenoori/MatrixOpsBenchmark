import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import platform
from pathlib import Path
import subprocess
from sklearn.utils.extmath import randomized_svd

# Get operating system name
os_name = platform.system()
if os_name == "Darwin":
    os_name = "macOS"

try:
    if os_name == "Windows":
        result = subprocess.check_output("wmic cpu get name", shell=True).decode().strip()
        cpu_model = result.split("\n")[1].strip()
    elif os_name == "Linux":
        result = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
        cpu_model = result.split(":")[1].strip()
    elif os_name == "macOS":
        result = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
        cpu_model = result
    else:
        cpu_model = platform.processor() or "Unknown CPU"
except Exception:
    cpu_model = "Unknown CPU"
print(cpu_model)

# Get GPU model (if CuPy is available)
gpu_model = "No GPU (CPU only)"
cp = None
try:
    import cupy as cp
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_model = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        pynvml.nvmlShutdown()
    except:
        if cp.cuda.runtime.getDeviceCount() > 0:
            gpu_model = f"CUDA Device {cp.cuda.runtime.getDevice()}"
except ImportError:
    pass

# Try to import CuPy; fall back to NumPy if not available
try:
    import cupy as cp
    xp = cp
    print("Using CuPy (GPU)")
except ImportError:
    xp = np
    print("Using NumPy (CPU)")

# Matrix sizes to test
matrix_sizes = [1000, 2000, 3000, 4000, 5000]
num_runs = 10  # Number of runs for each experiment

# Reference results, matrices, and times
reference_results = {}
reference_matrices = {}
reference_times = {
    "dense": {
        "elemwise": [], "exp": [], "mean": [], "matprod": [], "inv": [], "svd": []
    },
    "sparse": {
        "elemwise": [], "exp": [], "mean": [], "matprod": [], "svd": []
    }
}

# Timing results (list of lists for multiple runs)
timings = {
    "elemwise": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "exp": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "mean": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "matprod": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "inv": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "svd": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
}

sparse_timings = {
    "elemwise": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
    "exp": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
    "mean": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
    "matprod": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
    "svd": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
}

# Error results
errors = {
    "elemwise": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "exp": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "mean": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "matprod": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "inv": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
    "svd": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes]},
}

sparse_errors = {
    "elemwise": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
    "exp": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
    "mean": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
    "matprod": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
    "svd": {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]},
}

# Function to convert between CuPy and NumPy arrays
def to_numpy(array):
    if xp == cp:
        return array.get()
    return array

# Function to run experiments
def run_experiments(dtype, label, sparse=False, size_idx=0, run_idx=0):
    print(f"\nRun {run_idx+1}/{num_runs} with {label} {'(sparse)' if sparse else '(non-sparse)'}...")
    timings_dict = sparse_timings if sparse else timings
    errors_dict = sparse_errors if sparse else errors
    N = matrix_sizes[size_idx]
    print(f"Matrix size: {N}x{N}")
    
    A_ref, B_ref = reference_matrices[f"matrices_{N}_{'sparse' if sparse else 'dense'}"]
    
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
    C_ref = reference_results[f"elemwise_{N}_{'sparse' if sparse else 'dense'}"]
    C_np = to_numpy(C.todense() if sparse else C)
    error = np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref) if np.linalg.norm(C_ref) > 0 else 0
    timings_dict["elemwise"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(elemwise_time)
    errors_dict["elemwise"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(error)

    # 2. Exponential
    start = time.time()
    if sparse:
        expA = A.copy()
        expA.data = xp.exp(expA.data)
    else:
        expA = xp.exp(A)
    if xp == cp:
        cp.cuda.Stream.null.synchronize()
    exp_time = time.time() - start
    expA_ref = reference_results[f"exp_{N}_{'sparse' if sparse else 'dense'}"]
    expA_np = to_numpy(expA.todense() if sparse else expA)
    error = np.linalg.norm(expA_np - expA_ref) / np.linalg.norm(expA_ref) if np.linalg.norm(expA_ref) > 0 else 0
    timings_dict["exp"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(exp_time)
    errors_dict["exp"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(error)

    # 3. Means of the rows
    start = time.time()
    if sparse:
        meanRows = xp.array(A.mean(axis=1)).flatten()
    else:
        meanRows = xp.mean(A, axis=1)
    if xp == cp:
        cp.cuda.Stream.null.synchronize()
    mean_time = time.time() - start
    meanRows_ref = reference_results[f"mean_{N}_{'sparse' if sparse else 'dense'}"]
    meanRows_np = to_numpy(meanRows)
    error = np.linalg.norm(meanRows_np - meanRows_ref) / np.linalg.norm(meanRows_ref) if np.linalg.norm(meanRows_ref) > 0 else 0
    timings_dict["mean"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(mean_time)
    errors_dict["mean"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(error)

    # 4. Matrix product
    start = time.time()
    C = A @ B
    if xp == cp:
        cp.cuda.Stream.null.synchronize()
    matprod_time = time.time() - start
    C_ref = reference_results[f"matprod_{N}_{'sparse' if sparse else 'dense'}"]
    C_np = to_numpy(C.todense() if sparse else C)
    error = np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref) if np.linalg.norm(C_ref) > 0 else 0
    timings_dict["matprod"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(matprod_time)
    errors_dict["matprod"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(error)

    # 5. Inverse (non-sparse)
    if not sparse:
        start = time.time()
        invA = xp.linalg.inv(A)
        if xp == cp:
            cp.cuda.Stream.null.synchronize()
        inv_time = time.time() - start
        invA_ref = reference_results[f"inv_{N}_dense"]
        invA_np = to_numpy(invA)
        error = np.linalg.norm(invA_np - invA_ref) / np.linalg.norm(invA_ref) if np.linalg.norm(invA_ref) > 0 else 0
        timings["inv"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(inv_time)
        errors["inv"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(error)

    # 6. Randomized SVD (both dense and sparse)
    start = time.time()
    A_np = to_numpy(A.todense() if sparse else A)
    n_components = min(N, 100)  # Adjust based on matrix size
    U, S, VT = randomized_svd(A_np, n_components=n_components, random_state=run_idx)
    svd_time = time.time() - start
    S_ref = reference_results[f"svd_S_{N}_{'sparse' if sparse else 'dense'}"][:n_components]
    error = np.linalg.norm(S - S_ref) / np.linalg.norm(S_ref) if np.linalg.norm(S_ref) > 0 else 0
    timings_dict["svd"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(svd_time)
    errors_dict["svd"][f"{'GPU' if xp == cp else 'CPU'}_{label}"][size_idx].append(error)

    # CPU F32 comparison
    if xp == cp:
        A_np = A_ref.astype(np.float32)
        B_np = B_ref.astype(np.float32)
        if sparse:
            A_np = sp.csr_matrix(A_np)
            B_np = sp.csr_matrix(B_np)

        start = time.time()
        if sparse:
            C_np = A_np.multiply(B_np)
        else:
            C_np = A_np * B_np
        cpu_elemwise_time = time.time() - start
        C_ref = reference_results[f"elemwise_{N}_{'sparse' if sparse else 'dense'}"]
        C_np = C_np.todense() if sparse else C_np
        error = np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref) if np.linalg.norm(C_ref) > 0 else 0
        timings_dict["elemwise"]["CPU_F32"][size_idx].append(cpu_elemwise_time)
        errors_dict["elemwise"]["CPU_F32"][size_idx].append(error)

        start = time.time()
        if sparse:
            expA_np = A_np.copy()
            expA_np.data = np.exp(expA_np.data)
        else:
            expA_np = np.exp(A_np)
        cpu_exp_time = time.time() - start
        expA_ref = reference_results[f"exp_{N}_{'sparse' if sparse else 'dense'}"]
        expA_np = expA_np.todense() if sparse else expA_np
        error = np.linalg.norm(expA_np - expA_ref) / np.linalg.norm(expA_ref) if np.linalg.norm(expA_ref) > 0 else 0
        timings_dict["exp"]["CPU_F32"][size_idx].append(cpu_exp_time)
        errors_dict["exp"]["CPU_F32"][size_idx].append(error)

        start = time.time()
        if sparse:
            meanRows_np = np.array(A_np.mean(axis=1)).flatten()
        else:
            meanRows_np = np.mean(A_np, axis=1)
        cpu_mean_time = time.time() - start
        meanRows_ref = reference_results[f"mean_{N}_{'sparse' if sparse else 'dense'}"]
        error = np.linalg.norm(meanRows_np - meanRows_ref) / np.linalg.norm(meanRows_ref) if np.linalg.norm(meanRows_ref) > 0 else 0
        timings_dict["mean"]["CPU_F32"][size_idx].append(cpu_mean_time)
        errors_dict["mean"]["CPU_F32"][size_idx].append(error)

        start = time.time()
        C_np = A_np @ B_np
        cpu_matprod_time = time.time() - start
        C_ref = reference_results[f"matprod_{N}_{'sparse' if sparse else 'dense'}"]
        C_np = C_np.todense() if sparse else C_np
        error = np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref) if np.linalg.norm(C_ref) > 0 else 0
        timings_dict["matprod"]["CPU_F32"][size_idx].append(cpu_matprod_time)
        errors_dict["matprod"]["CPU_F32"][size_idx].append(error)

        if not sparse:
            start = time.time()
            invA_np = np.linalg.inv(A_np)
            cpu_inv_time = time.time() - start
            invA_ref = reference_results[f"inv_{N}_dense"]
            error = np.linalg.norm(invA_np - invA_ref) / np.linalg.norm(invA_ref) if np.linalg.norm(invA_ref) > 0 else 0
            timings["inv"]["CPU_F32"][size_idx].append(cpu_inv_time)
            errors["inv"]["CPU_F32"][size_idx].append(error)

        start = time.time()
        A_np_dense = A_np.todense() if sparse else A_np
        U_np, S_np, VT_np = randomized_svd(A_np_dense, n_components=n_components, random_state=run_idx)
        cpu_svd_time = time.time() - start
        S_ref = reference_results[f"svd_S_{N}_{'sparse' if sparse else 'dense'}"][:n_components]
        error = np.linalg.norm(S_np - S_ref) / np.linalg.norm(S_ref) if np.linalg.norm(S_ref) > 0 else 0
        timings_dict["svd"]["CPU_F32"][size_idx].append(cpu_svd_time)
        errors_dict["svd"]["CPU_F32"][size_idx].append(error)

# Generate CPU F64 reference results
print("Generating CPU F64 reference results...")
for N in matrix_sizes:
    for sparse in [False, True]:
        if sparse:
            A = sp.random(N, N, density=0.01, format='csr', dtype=np.float64)
            B = sp.random(N, N, density=0.01, format='csr', dtype=np.float64)
            reference_matrices[f"matrices_{N}_sparse"] = (A, B)
            
            start = time.time()
            reference_results[f"elemwise_{N}_sparse"] = (A.multiply(B)).todense()
            reference_times["sparse"]["elemwise"].append(time.time() - start)
            
            start = time.time()
            expA = A.copy()
            expA.data = np.exp(expA.data)
            reference_times["sparse"]["exp"].append(time.time() - start)
            reference_results[f"exp_{N}_sparse"] = expA.todense()
            
            start = time.time()
            reference_results[f"mean_{N}_sparse"] = np.array(A.mean(axis=1)).flatten()
            reference_times["sparse"]["mean"].append(time.time() - start)
            
            start = time.time()
            reference_results[f"matprod_{N}_sparse"] = (A @ B).todense()
            reference_times["sparse"]["matprod"].append(time.time() - start)
            
            start = time.time()
            n_components = min(N, 100)
            U, S, VT = randomized_svd(A, n_components=n_components, random_state=0)
            reference_results[f"svd_S_{N}_sparse"] = S
            reference_times["sparse"]["svd"].append(time.time() - start)
        else:
            A = np.random.rand(N, N).astype(np.float64)
            B = np.random.rand(N, N).astype(np.float64)
            reference_matrices[f"matrices_{N}_dense"] = (A, B)
            
            start = time.time()
            reference_results[f"elemwise_{N}_dense"] = (A * B).astype(np.float64)
            reference_times["dense"]["elemwise"].append(time.time() - start)
            
            start = time.time()
            reference_results[f"exp_{N}_dense"] = np.exp(A).astype(np.float64)
            reference_times["dense"]["exp"].append(time.time() - start)
            
            start = time.time()
            reference_results[f"mean_{N}_dense"] = np.mean(A, axis=1).astype(np.float64)
            reference_times["dense"]["mean"].append(time.time() - start)
            
            start = time.time()
            reference_results[f"matprod_{N}_dense"] = np.dot(A, B).astype(np.float64)
            reference_times["dense"]["matprod"].append(time.time() - start)
            
            start = time.time()
            reference_results[f"inv_{N}_dense"] = np.linalg.inv(A).astype(np.float64)
            reference_times["dense"]["inv"].append(time.time() - start)
            
            start = time.time()
            n_components = min(N, 100)
            U, S, VT = randomized_svd(A, n_components=n_components, random_state=0)
            reference_results[f"svd_S_{N}_dense"] = S
            reference_times["dense"]["svd"].append(time.time() - start)

# Run experiments
for size_idx, N in enumerate(matrix_sizes):
    for run_idx in range(num_runs):
        run_experiments(np.float32, "F32", sparse=False, size_idx=size_idx, run_idx=run_idx)
        if xp == cp:
            run_experiments(np.float64, "F64", sparse=False, size_idx=size_idx, run_idx=run_idx)
        run_experiments(np.float32, "F32", sparse=True, size_idx=size_idx, run_idx=run_idx)
        run_experiments(np.float64, "F64", sparse=True, size_idx=size_idx, run_idx=run_idx)

# Plotting
operations = [
    ("elemwise", "Element-wise product"),
    ("exp", "Exponential of each element"),
    ("mean", "Means of the rows"),
    ("matprod", "Matrix product"),
    ("inv", "Inverse of a matrix"),
    ("svd", "Randomized SVD")
]

sparse_operations = operations[:4] + [operations[5]]  # Exclude inverse

Path("plots").mkdir(exist_ok=True)

# Define matrix sizes explicitly
matrix_sizes = [1000, 2000, 3000, 4000, 5000]

# Define color mapping
color_map = {
    'CPU': 'purple',
    'GPU': 'blue'
}

# Check for GPU availability
gpu_available = False
try:
    import cupy
    if cupy.cuda.runtime.getDeviceCount() > 0:
        gpu_available = True
except ImportError:
    pass

# CPU Dense Timing (F64 Dense Baseline, CPU_F32)
plt.figure(figsize=(20, 12))
for i, (op_key, op_name) in enumerate(operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (CPU Dense, Time)", fontsize=12)
    plt.xlabel("Size of Matrix (n×n)", fontsize=10)
    plt.ylabel("Time (seconds, log-scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(range(len(matrix_sizes)), matrix_sizes, fontsize=9)
    
    # Collect min and max times for y-limits
    all_times = []
    if "CPU_F32" in timings[op_key]:
        for t in timings[op_key]["CPU_F32"]:
            if len(t) > 0:
                all_times.extend(t)
    ref_times = reference_times["dense"][op_key]
    all_times.extend(ref_times)
    if all_times:
        min_time = min(all_times) / 2
        max_time = max(all_times) * 2
        plt.ylim(min_time, max_time)
    
    # F64 Dense Baseline
    plt.plot(range(len(matrix_sizes)), ref_times, 'x', label="F64 Dense Baseline", color="black", markersize=8)
    
    # CPU_F32 Box Plot
    data = [t if len(t) > 0 else [0] * num_runs for t in timings[op_key]["CPU_F32"]]
    plt.boxplot(data, positions=range(len(matrix_sizes)), widths=0.2, patch_artist=True, 
                boxprops=dict(facecolor=color_map["CPU"], edgecolor='black'),
                whiskerprops=dict(color='black'), capprops=dict(color='black'),
                medianprops=dict(color='red'), showfliers=True)
    
    plt.legend(["F64 Dense Baseline", "CPU_F32"], fontsize=9, loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.7)

plt.tight_layout(pad=2.0)
plt.savefig("plots/cpu_dense_timing_results.png", dpi=300)
plt.close()

# CPU Dense Accuracy (CPU_F32)
plt.figure(figsize=(20, 12))
for i, (op_key, op_name) in enumerate(operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (CPU Dense, Error)", fontsize=12)
    plt.xlabel("Size of Matrix (n×n)", fontsize=10)
    plt.ylabel("Relative Error (log-scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(range(len(matrix_sizes)), matrix_sizes, fontsize=9)
    
    # Collect min and max errors for y-limits
    all_errors = []
    if "CPU_F32" in errors[op_key]:
        for e in errors[op_key]["CPU_F32"]:
            if len(e) > 0:
                all_errors.extend(e)
    if all_errors:
        min_error = max(min(all_errors) / 2, 1e-20)
        max_error = max(all_errors) * 2
        plt.ylim(min_error, max_error)
    
    # CPU_F32 Box Plot
    data = [e if len(e) > 0 else [1e-20] * num_runs for e in errors[op_key]["CPU_F32"]]
    plt.boxplot(data, positions=range(len(matrix_sizes)), widths=0.2, patch_artist=True, 
                boxprops=dict(facecolor=color_map["CPU"], edgecolor='black'),
                whiskerprops=dict(color='black'), capprops=dict(color='black'),
                medianprops=dict(color='red'), showfliers=True)
    
    plt.legend(["CPU_F32"], fontsize=9, loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.7)

plt.tight_layout(pad=2.0)
plt.savefig("plots/cpu_dense_accuracy_results.png", dpi=300)
plt.close()

# CPU Sparse Timing (F64 Sparse Baseline, CPU_F32)
plt.figure(figsize=(20, 12))
for i, (op_key, op_name) in enumerate(sparse_operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (CPU Sparse, Time)", fontsize=12)
    plt.xlabel("Size of Matrix (n×n)", fontsize=10)
    plt.ylabel("Time (seconds, log-scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(range(len(matrix_sizes)), matrix_sizes, fontsize=9)
    
    # Collect min and max times for y-limits
    all_times = []
    if "CPU_F32" in sparse_timings[op_key]:
        for t in sparse_timings[op_key]["CPU_F32"]:
            if len(t) > 0:
                all_times.extend(t)
    ref_times = reference_times["sparse"][op_key]
    all_times.extend(ref_times)
    if all_times:
        min_time = min(all_times) / 2
        max_time = max(all_times) * 2
        plt.ylim(min_time, max_time)
    
    # F64 Sparse Baseline
    plt.plot(range(len(matrix_sizes)), ref_times, 'x', label="F64 Sparse Baseline", color="black", markersize=8)
    
    # CPU_F32 Box Plot
    data = [t if len(t) > 0 else [0] * num_runs for t in sparse_timings[op_key]["CPU_F32"]]
    plt.boxplot(data, positions=range(len(matrix_sizes)), widths=0.2, patch_artist=True, 
                boxprops=dict(facecolor=color_map["CPU"], edgecolor='black'),
                whiskerprops=dict(color='black'), capprops=dict(color='black'),
                medianprops=dict(color='red'), showfliers=True)
    
    plt.legend(["F64 Sparse Baseline", "CPU_F32"], fontsize=9, loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.7)

plt.tight_layout(pad=2.0)
plt.savefig("plots/cpu_sparse_timing_results.png", dpi=300)
plt.close()

# CPU Sparse Accuracy (CPU_F32)
plt.figure(figsize=(20, 12))
for i, (op_key, op_name) in enumerate(sparse_operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (CPU Sparse, Error)", fontsize=12)
    plt.xlabel("Size of Matrix (n×n)", fontsize=10)
    plt.ylabel("Relative Error (log-scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(range(len(matrix_sizes)), matrix_sizes, fontsize=9)
    
    # Collect min and max errors for y-limits
    all_errors = []
    if "CPU_F32" in sparse_errors[op_key]:
        for e in sparse_errors[op_key]["CPU_F32"]:
            if len(e) > 0:
                all_errors.extend(e)
    if all_errors:
        min_error = max(min(all_errors) / 2, 1e-20)
        max_error = max(all_errors) * 2
        plt.ylim(min_error, max_error)
    
    # CPU_F32 Box Plot
    data = [e if len(e) > 0 else [1e-20] * num_runs for e in sparse_errors[op_key]["CPU_F32"]]
    plt.boxplot(data, positions=range(len(matrix_sizes)), widths=0.2, patch_artist=True, 
                boxprops=dict(facecolor=color_map["CPU"], edgecolor='black'),
                whiskerprops=dict(color='black'), capprops=dict(color='black'),
                medianprops=dict(color='red'), showfliers=True)
    
    plt.legend(["CPU_F32"], fontsize=9, loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.7)

plt.tight_layout(pad=2.0)
plt.savefig("plots/cpu_sparse_accuracy_results.png", dpi=300)
plt.close()

# GPU Plots (only if GPU is available)
if gpu_available:
    # GPU Dense Timing (F64 Dense Baseline, GPU_F32)
    plt.figure(figsize=(20, 12))
    for i, (op_key, op_name) in enumerate(operations, 1):
        plt.subplot(2, 3, i)
        plt.title(f"{op_name} (GPU Dense, Time)", fontsize=12)
        plt.xlabel("Size of Matrix (n×n)", fontsize=10)
        plt.ylabel("Time (seconds, log-scale)", fontsize=10)
        plt.yscale("log")
        plt.xticks(range(len(matrix_sizes)), matrix_sizes, fontsize=9)
        
        # Collect min and max times for y-limits
        all_times = []
        if "GPU_F32" in timings[op_key]:
            for t in timings[op_key]["GPU_F32"]:
                if len(t) > 0:
                    all_times.extend(t)
        ref_times = reference_times["dense"][op_key]
        all_times.extend(ref_times)
        if all_times:
            min_time = min(all_times) / 2
            max_time = max(all_times) * 2
            plt.ylim(min_time, max_time)
        
        # F64 Dense Baseline
        plt.plot(range(len(matrix_sizes)), ref_times, 'x', label="F64 Dense Baseline", color="black", markersize=8)
        
        # GPU_F32 Box Plot
        data = [t if len(t) > 0 else [0] * num_runs for t in timings[op_key]["GPU_F32"]]
        plt.boxplot(data, positions=range(len(matrix_sizes)), widths=0.2, patch_artist=True, 
                    boxprops=dict(facecolor=color_map["GPU"], edgecolor='black'),
                    whiskerprops=dict(color='black'), capprops=dict(color='black'),
                    medianprops=dict(color='red'), showfliers=True)
        
        plt.legend(["F64 Dense Baseline", "GPU_F32"], fontsize=9, loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.tight_layout(pad=2.0)
    plt.savefig("plots/gpu_dense_timing_results.png", dpi=300)
    plt.close()

    # GPU Dense Accuracy (GPU_F32)
    plt.figure(figsize=(20, 12))
    for i, (op_key, op_name) in enumerate(operations, 1):
        plt.subplot(2, 3, i)
        plt.title(f"{op_name} (GPU Dense, Error)", fontsize=12)
        plt.xlabel("Size of Matrix (n×n)", fontsize=10)
        plt.ylabel("Relative Error (log-scale)", fontsize=10)
        plt.yscale("log")
        plt.xticks(range(len(matrix_sizes)), matrix_sizes, fontsize=9)
        
        # Collect min and max errors for y-limits
        all_errors = []
        if "GPU_F32" in errors[op_key]:
            for e in errors[op_key]["GPU_F32"]:
                if len(e) > 0:
                    all_errors.extend(e)
        if all_errors:
            min_error = max(min(all_errors) / 2, 1e-20)
            max_error = max(all_errors) * 2
            plt.ylim(min_error, max_error)
        
        # GPU_F32 Box Plot
        data = [e if len(e) > 0 else [1e-20] * num_runs for e in errors[op_key]["GPU_F32"]]
        plt.boxplot(data, positions=range(len(matrix_sizes)), widths=0.2, patch_artist=True, 
                    boxprops=dict(facecolor=color_map["GPU"], edgecolor='black'),
                    whiskerprops=dict(color='black'), capprops=dict(color='black'),
                    medianprops=dict(color='red'), showfliers=True)
        
        plt.legend(["GPU_F32"], fontsize=9, loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.tight_layout(pad=2.0)
    plt.savefig("plots/gpu_dense_accuracy_results.png", dpi=300)
    plt.close()

    # GPU Sparse Timing (F64 Sparse Baseline, GPU_F32)
    plt.figure(figsize=(20, 12))
    for i, (op_key, op_name) in enumerate(sparse_operations, 1):
        plt.subplot(2, 3, i)
        plt.title(f"{op_name} (GPU Sparse, Time)", fontsize=12)
        plt.xlabel("Size of Matrix (n×n)", fontsize=10)
        plt.ylabel("Time (seconds, log-scale)", fontsize=10)
        plt.yscale("log")
        plt.xticks(range(len(matrix_sizes)), matrix_sizes, fontsize=9)
        
        # Collect min and max times for y-limits
        all_times = []
        if "GPU_F32" in sparse_timings[op_key]:
            for t in sparse_timings[op_key]["GPU_F32"]:
                if len(t) > 0:
                    all_times.extend(t)
        ref_times = reference_times["sparse"][op_key]
        all_times.extend(ref_times)
        if all_times:
            min_time = min(all_times) / 2
            max_time = max(all_times) * 2
            plt.ylim(min_time, max_time)
        
        # F64 Sparse Baseline
        plt.plot(range(len(matrix_sizes)), ref_times, 'x', label="F64 Sparse Baseline", color="black", markersize=8)
        
        # GPU_F32 Box Plot
        data = [t if len(t) > 0 else [0] * num_runs for t in sparse_timings[op_key]["GPU_F32"]]
        plt.boxplot(data, positions=range(len(matrix_sizes)), widths=0.2, patch_artist=True, 
                    boxprops=dict(facecolor=color_map["GPU"], edgecolor='black'),
                    whiskerprops=dict(color='black'), capprops=dict(color='black'),
                    medianprops=dict(color='red'), showfliers=True)
        
        plt.legend(["F64 Sparse Baseline", "GPU_F32"], fontsize=9, loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.tight_layout(pad=2.0)
    plt.savefig("plots/gpu_sparse_timing_results.png", dpi=300)
    plt.close()

    # GPU Sparse Accuracy (GPU_F32)
    plt.figure(figsize=(20, 12))
    for i, (op_key, op_name) in enumerate(sparse_operations, 1):
        plt.subplot(2, 3, i)
        plt.title(f"{op_name} (GPU Sparse, Error)", fontsize=12)
        plt.xlabel("Size of Matrix (n×n)", fontsize=10)
        plt.ylabel("Relative Error (log-scale)", fontsize=10)
        plt.yscale("log")
        plt.xticks(range(len(matrix_sizes)), matrix_sizes, fontsize=9)
        
        # Collect min and max errors for y-limits
        all_errors = []
        if "GPU_F32" in sparse_errors[op_key]:
            for e in sparse_errors[op_key]["GPU_F32"]:
                if len(e) > 0:
                    all_errors.extend(e)
        if all_errors:
            min_error = max(min(all_errors) / 2, 1e-20)
            max_error = max(all_errors) * 2
            plt.ylim(min_error, max_error)
        
        # GPU_F32 Box Plot
        data = [e if len(e) > 0 else [1e-20] * num_runs for e in sparse_errors[op_key]["GPU_F32"]]
        plt.boxplot(data, positions=range(len(matrix_sizes)), widths=0.2, patch_artist=True, 
                    boxprops=dict(facecolor=color_map["GPU"], edgecolor='black'),
                    whiskerprops=dict(color='black'), capprops=dict(color='black'),
                    medianprops=dict(color='red'), showfliers=True)
        
        plt.legend(["GPU_F32"], fontsize=9, loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.tight_layout(pad=2.0)
    plt.savefig("plots/gpu_sparse_accuracy_results.png", dpi=300)
    plt.close()

print("\nPlots saved in 'plots' directory.")
