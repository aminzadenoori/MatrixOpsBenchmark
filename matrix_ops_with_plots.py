import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import platform
from pathlib import Path
from sklearn.utils.extmath import randomized_svd

# CPU model detection using py-cpuinfo
def get_cpu_model():
    try:
        from cpuinfo import get_cpu_info
        cpu_info = get_cpu_info()
        cpu_model = cpu_info.get('brand_raw', 'Unknown CPU')
        return cpu_model
    except ImportError:
        import platform
        return platform.processor() or "Unknown CPU"
    except Exception:
        return "Unknown CPU"

cpu_model = get_cpu_model()
print(f"CPU Model: {cpu_model}")

# GPU model detection (OS-independent)
def get_gpu_model():
    gpu_model = "No GPU (CPU only)"
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_model = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_model, bytes):
                gpu_model = gpu_model.decode('utf-8')
        pynvml.nvmlShutdown()
    except ImportError:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                gpu_model = f"CUDA Device {cp.cuda.runtime.getDevice()}"
        except ImportError:
            pass
    except Exception:
        pass
    return gpu_model

gpu_model = get_gpu_model()
print(f"GPU Model: {gpu_model}")

# Get Python version
python_version = platform.python_version()

# Import NumPy and CuPy
np_xp = np  # Always available
try:
    import cupy as cp
    cp_xp = cp
    gpu_available = cp.cuda.runtime.getDeviceCount() > 0
    print("CuPy (GPU) available" if gpu_available else "CuPy available but no GPU detected")
except ImportError:
    cp_xp = None
    gpu_available = False
    print("CuPy (GPU) not available")

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

# Timing and error results
timings = {op: {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], 
                "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]} 
           for op in ["elemwise", "exp", "mean", "matprod", "inv", "svd"]}
sparse_timings = {op: {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], 
                       "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]} 
                  for op in ["elemwise", "exp", "mean", "matprod", "svd"]}
errors = {op: {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], 
               "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]} 
          for op in ["elemwise", "exp", "mean", "matprod", "inv", "svd"]}
sparse_errors = {op: {"GPU_F32": [[] for _ in matrix_sizes], "GPU_F64": [[] for _ in matrix_sizes], 
                      "CPU_F32": [[] for _ in matrix_sizes], "CPU_F64": [[] for _ in matrix_sizes]} 
                 for op in ["elemwise", "exp", "mean", "matprod", "svd"]}

# Convert between CuPy and NumPy arrays, ensuring dense output
def to_numpy(array, xp):
    if xp == cp_xp:
        if hasattr(array, 'get'):
            array = array.get()
        if sp.issparse(array):
            array = array.todense()
    elif sp.issparse(array):
        array = array.todense()
    return np.asarray(array, dtype=np.float64)

# Compute relative error safely
def compute_relative_error(result, reference):
    result = np.asarray(result, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    ref_norm = np.linalg.norm(reference)
    if ref_norm > 1e-20:
        error = np.linalg.norm(result - reference) / ref_norm
    else:
        error = 0.0
    return error

# Function to run experiments
def run_experiments(dtype, label, sparse=False, size_idx=0, run_idx=0, device="CPU"):
    xp = cp_xp if device == "GPU" and gpu_available else np_xp
    print(f"\nRun {run_idx+1}/{num_runs} with {device}_{label} {'(sparse)' if sparse else '(non-sparse)'}...")
    timings_dict = sparse_timings if sparse else timings
    errors_dict = sparse_errors if sparse else errors
    N = matrix_sizes[size_idx]
    print(f"Matrix size: {N}x{N}")
    
    A_ref, B_ref = reference_matrices[f"matrices_{N}_{'sparse' if sparse else 'dense'}"]
    
    if sparse:
        A = A_ref.astype(dtype)
        B = B_ref.astype(dtype)
        if device == "GPU" and gpu_available:
            A = cp_xp.sparse.csr_matrix(A)
            B = cp_xp.sparse.csr_matrix(B)
    else:
        A = xp.array(A_ref.astype(dtype))
        B = xp.array(B_ref.astype(dtype))

    # Reference results for error computation
    ref_key = f"{'sparse' if sparse else 'dense'}"
    C_ref = reference_results[f"elemwise_{N}_{ref_key}"]
    expA_ref = reference_results[f"exp_{N}_{ref_key}"]
    meanRows_ref = reference_results[f"mean_{N}_{ref_key}"]
    matprod_ref = reference_results[f"matprod_{N}_{ref_key}"]
    svd_S_ref = reference_results[f"svd_S_{N}_{ref_key}"]
    inv_ref = reference_results[f"inv_{N}_dense"] if not sparse else None

    # 1. Element-wise product
    start = time.time()
    if sparse:
        C = A.multiply(B)
    else:
        C = A * B
    if device == "GPU" and gpu_available:
        cp_xp.cuda.Stream.null.synchronize()
    elemwise_time = time.time() - start
    C_np = to_numpy(C, xp)
    error = compute_relative_error(C_np, C_ref)
    timings_dict["elemwise"][f"{device}_{label}"][size_idx].append(max(elemwise_time, 1e-10))  # Avoid zero timings
    errors_dict["elemwise"][f"{device}_{label}"][size_idx].append(error)

    # 2. Exponential
    start = time.time()
    if sparse:
        expA = A.copy()
        expA.data = xp.exp(expA.data)
    else:
        expA = xp.exp(A)
    if device == "GPU" and gpu_available:
        cp_xp.cuda.Stream.null.synchronize()
    exp_time = time.time() - start
    expA_np = to_numpy(expA, xp)
    error = compute_relative_error(expA_np, expA_ref)
    timings_dict["exp"][f"{device}_{label}"][size_idx].append(max(exp_time, 1e-10))
    errors_dict["exp"][f"{device}_{label}"][size_idx].append(error)

    # 3. Means of the rows
    start = time.time()
    if sparse:
        meanRows = xp.array(A.mean(axis=1)).flatten()
    else:
        meanRows = xp.mean(A, axis=1)
    if device == "GPU" and gpu_available:
        cp_xp.cuda.Stream.null.synchronize()
    mean_time = time.time() - start
    meanRows_np = to_numpy(meanRows, xp)
    error = compute_relative_error(meanRows_np, meanRows_ref)
    timings_dict["mean"][f"{device}_{label}"][size_idx].append(max(mean_time, 1e-10))
    errors_dict["mean"][f"{device}_{label}"][size_idx].append(error)

    # 4. Matrix product
    start = time.time()
    C = A @ B
    if device == "GPU" and gpu_available:
        cp_xp.cuda.Stream.null.synchronize()
    matprod_time = time.time() - start
    C_np = to_numpy(C, xp)
    error = compute_relative_error(C_np, matprod_ref)
    timings_dict["matprod"][f"{device}_{label}"][size_idx].append(max(matprod_time, 1e-10))
    errors_dict["matprod"][f"{device}_{label}"][size_idx].append(error)

    # 5. Inverse (non-sparse)
    if not sparse:
        start = time.time()
        invA = xp.linalg.inv(A)
        if device == "GPU" and gpu_available:
            cp_xp.cuda.Stream.null.synchronize()
        inv_time = time.time() - start
        invA_np = to_numpy(invA, xp)
        error = compute_relative_error(invA_np, inv_ref)
        timings_dict["inv"][f"{device}_{label}"][size_idx].append(max(inv_time, 1e-10))
        errors_dict["inv"][f"{device}_{label}"][size_idx].append(error)

    # 6. Randomized SVD
    start = time.time()
    n_components = min(N, 100)
    A_np = to_numpy(A, xp)
    U, S, VT = randomized_svd(A_np, n_components=n_components, random_state=run_idx)
    svd_time = time.time() - start
    error = compute_relative_error(S, svd_S_ref[:n_components])
    timings_dict["svd"][f"{device}_{label}"][size_idx].append(max(svd_time, 1e-10))
    errors_dict["svd"][f"{device}_{label}"][size_idx].append(error)

# Generate reference results (CPU_F64 Dense for dense, CPU_F64 Sparse for sparse)
print("Generating reference results...")
for N in matrix_sizes:
    # Dense: CPU_F64
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

    # Sparse: CPU_F64
    A = sp.random(N, N, density=0.01, format='csr', dtype=np.float64)
    B = sp.random(N, N, density=0.01, format='csr', dtype=np.float64)
    reference_matrices[f"matrices_{N}_sparse"] = (A, B)
    
    start = time.time()
    result = A.multiply(B)
    reference_results[f"elemwise_{N}_sparse"] = to_numpy(result, np_xp)
    reference_times["sparse"]["elemwise"].append(time.time() - start)
    
    start = time.time()
    expA = A.copy()
    expA.data = np.exp(expA.data)
    reference_results[f"exp_{N}_sparse"] = to_numpy(expA, np_xp)
    reference_times["sparse"]["exp"].append(time.time() - start)
    
    start = time.time()
    reference_results[f"mean_{N}_sparse"] = np.array(A.mean(axis=1)).flatten()
    reference_times["sparse"]["mean"].append(time.time() - start)
    
    start = time.time()
    reference_results[f"matprod_{N}_sparse"] = to_numpy(A @ B, np_xp)
    reference_times["sparse"]["matprod"].append(time.time() - start)
    
    start = time.time()
    A_np = A.todense()
    U, S, VT = randomized_svd(A_np, n_components=n_components, random_state=0)
    reference_results[f"svd_S_{N}_sparse"] = S
    reference_times["sparse"]["svd"].append(time.time() - start)

# Run experiments for CPU and GPU separately
for size_idx, N in enumerate(matrix_sizes):
    for run_idx in range(num_runs):
        # CPU experiments
        run_experiments(np.float32, "F32", sparse=False, size_idx=size_idx, run_idx=run_idx, device="CPU")
        run_experiments(np.float64, "F64", sparse=False, size_idx=size_idx, run_idx=run_idx, device="CPU")
        run_experiments(np.float32, "F32", sparse=True, size_idx=size_idx, run_idx=run_idx, device="CPU")
        run_experiments(np.float64, "F64", sparse=True, size_idx=size_idx, run_idx=run_idx, device="CPU")
        
        # GPU experiments (if available)
        if gpu_available:
            run_experiments(np.float32, "F32", sparse=False, size_idx=size_idx, run_idx=run_idx, device="GPU")
            run_experiments(np.float64, "F64", sparse=False, size_idx=size_idx, run_idx=run_idx, device="GPU")
            run_experiments(np.float32, "F32", sparse=True, size_idx=size_idx, run_idx=run_idx, device="GPU")
            run_experiments(np.float64, "F64", sparse=True, size_idx=size_idx, run_idx=run_idx, device="GPU")

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

# Define color and marker mapping
color_map = {
    'CPU_F32': 'purple',
    'CPU_F64': 'green',
    'GPU_F32': 'blue',
    'GPU_F64': 'cyan'
}
marker_map = {
    'CPU_F32': 'o',
    'CPU_F64': 'x',
    'GPU_F32': '^',
    'GPU_F64': 's'
}

# Create plots directory
Path("plots").mkdir(exist_ok=True)

# Combined CPU and GPU Dense Timing Plot
plt.figure(figsize=(20, 12))
for i, (op_key, op_name) in enumerate(operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (Dense, Time)", fontsize=12)
    plt.xlabel("Size of Matrix (n×n)", fontsize=10)
    plt.ylabel("Time (seconds, log-scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(matrix_sizes, matrix_sizes, fontsize=9)

    # Collect all valid timings (non-zero, non-empty)
    all_times = []
    for key in ["CPU_F32", "CPU_F64", "GPU_F32", "GPU_F64"]:
        if key in timings[op_key]:
            for t_list in timings[op_key][key]:
                if len(t_list) > 0:
                    all_times.extend([t for t in t_list if t > 0])
    
    # Set y-axis limits safely
    if all_times:
        min_time = max(min(all_times) / 2, 1e-10)  # Ensure positive lower bound
        max_time = max(all_times) * 2
        plt.ylim(min_time, max_time)
    else:
        plt.ylim(1e-10, 1e-1)  # Default range if no data
        print(f"Warning: No valid timing data for {op_name} (Dense)")

    # Plot lines without reference
    lines = []
    labels = []
    for key in ["CPU_F32", "CPU_F64"]:
        mean_times = [np.mean(t) if len(t) > 0 else 1e-10 for t in timings[op_key][key]]
        if all(t <= 1e-10 for t in mean_times):
            print(f"Warning: No valid timing data for {key} in {op_name} (Dense)")
            continue
        line, = plt.plot(matrix_sizes, mean_times, color=color_map[key], marker=marker_map[key], 
                         linestyle='-', markersize=8)
        lines.append(line)
        labels.append(f"{key} Dense ({cpu_model})")

    if gpu_available:
        for key in ["GPU_F32", "GPU_F64"]:
            mean_times = [np.mean(t) if len(t) > 0 else 1e-10 for t in timings[op_key][key]]
            if all(t <= 1e-10 for t in mean_times):
                print(f"Warning: No valid timing data for {key} in {op_name} (Dense)")
                continue
            line, = plt.plot(matrix_sizes, mean_times, color=color_map[key], marker=marker_map[key], 
                             linestyle='-', markersize=8)
            lines.append(line)
            labels.append(f"{key} Dense ({gpu_model})")

    plt.grid(True, which="both", ls="--", alpha=0.7)

# Add single legend outside subplots
plt.legend(lines[:len(labels)], labels, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
plt.figtext(0.5, 0.01, f"Timing for CPU ({cpu_model}) and GPU ({gpu_model}) Dense Matrices, Python {python_version}", 
            ha="center", fontsize=12, wrap=True)
plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
plt.savefig("plots/cpu_gpu_dense_timing_results.png", dpi=300, bbox_inches='tight')
plt.close()

# Combined CPU and GPU Sparse Timing Plot
plt.figure(figsize=(20, 12))
for i, (op_key, op_name) in enumerate(sparse_operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (Sparse, Time)", fontsize=12)
    plt.xlabel("Size of Matrix (n×n)", fontsize=10)
    plt.ylabel("Time (seconds, log-scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(matrix_sizes, matrix_sizes, fontsize=9)

    # Collect all valid timings (non-zero, non-empty)
    all_times = []
    for key in ["CPU_F32", "CPU_F64", "GPU_F32", "GPU_F64"]:
        if key in sparse_timings[op_key]:
            for t_list in sparse_timings[op_key][key]:
                if len(t_list) > 0:
                    all_times.extend([t for t in t_list if t > 0])
    
    # Set y-axis limits safely
    if all_times:
        min_time = max(min(all_times) / 2, 1e-10)  # Ensure positive lower bound
        max_time = max(all_times) * 2
        plt.ylim(min_time, max_time)
    else:
        plt.ylim(1e-10, 1e-1)  # Default range if no data
        print(f"Warning: No valid timing data for {op_name} (Sparse)")

    # Plot lines without reference
    lines = []
    labels = []
    for key in ["CPU_F32", "CPU_F64"]:
        mean_times = [np.mean(t) if len(t) > 0 else 1e-10 for t in sparse_timings[op_key][key]]
        if all(t <= 1e-10 for t in mean_times):
            print(f"Warning: No valid timing data for {key} in {op_name} (Sparse)")
            continue
        line, = plt.plot(matrix_sizes, mean_times, color=color_map[key], marker=marker_map[key], 
                         linestyle='-', markersize=8)
        lines.append(line)
        labels.append(f"{key} Sparse ({cpu_model})")

    if gpu_available:
        for key in ["GPU_F32", "GPU_F64"]:
            mean_times = [np.mean(t) if len(t) > 0 else 1e-10 for t in sparse_timings[op_key][key]]
            if all(t <= 1e-10 for t in mean_times):
                print(f"Warning: No valid timing data for {key} in {op_name} (Sparse)")
                continue
            line, = plt.plot(matrix_sizes, mean_times, color=color_map[key], marker=marker_map[key], 
                             linestyle='-', markersize=8)
            lines.append(line)
            labels.append(f"{key} Sparse ({gpu_model})")

    plt.grid(True, which="both", ls="--", alpha=0.7)

# Add single legend outside subplots
plt.legend(lines[:len(labels)], labels, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
plt.figtext(0.5, 0.01, f"Timing for CPU ({cpu_model}) and GPU ({gpu_model}) Sparse Matrices, Python {python_version}", 
            ha="center", fontsize=12, wrap=True)
plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
plt.savefig("plots/cpu_gpu_sparse_timing_results.png", dpi=300, bbox_inches='tight')
plt.close()

# Combined CPU and GPU Dense Accuracy Plot (Mean Errors)
plt.figure(figsize=(20, 12))
for i, (op_key, op_name) in enumerate(operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (Dense, Mean Error)", fontsize=12)
    plt.xlabel("Size of Matrix (n×n)", fontsize=10)
    plt.ylabel("Mean Relative Error (log-scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(matrix_sizes, matrix_sizes, fontsize=9)

    all_errors = []
    for key in ["CPU_F32", "CPU_F64", "GPU_F32", "GPU_F64"]:
        if key in errors[op_key]:
            for e in errors[op_key][key]:
                if len(e) > 0:
                    all_errors.extend(e)
    if all_errors:
        min_error = max(min(all_errors) / 2, 1e-20)
        max_error = max(all_errors) * 2
        plt.ylim(min_error, max_error)

    # Plot lines without legend
    lines = []
    labels = []
    for key in ["CPU_F32", "CPU_F64"]:
        mean_errors = [np.mean(e) if len(e) > 0 else 1e-20 for e in errors[op_key][key]]
        if all(e == 1e-20 for e in mean_errors):
            print(f"Warning: No error data for {key} in {op_name} (Dense)")
            continue
        line, = plt.plot(matrix_sizes, mean_errors, color=color_map[key], marker=marker_map[key], 
                         linestyle='-', markersize=8)
        lines.append(line)
        labels.append(f"{key} Dense ({cpu_model})")

    if gpu_available:
        for key in ["GPU_F32", "GPU_F64"]:
            mean_errors = [np.mean(e) if len(e) > 0 else 1e-20 for e in errors[op_key][key]]
            if all(e == 1e-20 for e in mean_errors):
                print(f"Warning: No error data for {key} in {op_name} (Dense)")
                continue
            line, = plt.plot(matrix_sizes, mean_errors, color=color_map[key], marker=marker_map[key], 
                             linestyle='-', markersize=8)
            lines.append(line)
            labels.append(f"{key} Dense ({gpu_model})")

    plt.grid(True, which="both", ls="--", alpha=0.7)

# Add single legend outside subplots
plt.legend(lines[:len(labels)], labels, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
plt.figtext(0.5, 0.01, f"Mean Accuracy for CPU ({cpu_model}) and GPU ({gpu_model}) Dense Matrices vs CPU F64 Reference, Python {python_version}", 
            ha="center", fontsize=12, wrap=True)
plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
plt.savefig("plots/cpu_gpu_dense_mean_accuracy_results.png", dpi=300, bbox_inches='tight')
plt.close()

# Combined CPU and GPU Sparse Accuracy Plot (Mean Errors)
plt.figure(figsize=(20, 12))
for i, (op_key, op_name) in enumerate(sparse_operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (Sparse, Mean Error)", fontsize=12)
    plt.xlabel("Size of Matrix (n×n)", fontsize=10)
    plt.ylabel("Mean Relative Error (log-scale)", fontsize=10)
    plt.yscale("log")
    plt.xticks(matrix_sizes, matrix_sizes, fontsize=9)

    all_errors = []
    for key in ["CPU_F32", "CPU_F64", "GPU_F32", "GPU_F64"]:
        if key in sparse_errors[op_key]:
            for e in sparse_errors[op_key][key]:
                if len(e) > 0:
                    all_errors.extend(e)
    if all_errors:
        min_error = max(min(all_errors) / 2, 1e-20)
        max_error = max(all_errors) * 2
        plt.ylim(min_error, max_error)

    # Plot lines without legend
    lines = []
    labels = []
    for key in ["CPU_F32", "CPU_F64"]:
        mean_errors = [np.mean(e) if len(e) > 0 else 1e-20 for e in sparse_errors[op_key][key]]
        if all(e == 1e-20 for e in mean_errors):
            print(f"Warning: No error data for {key} in {op_name} (Sparse)")
            continue
        line, = plt.plot(matrix_sizes, mean_errors, color=color_map[key], marker=marker_map[key], 
                         linestyle='-', markersize=8)
        lines.append(line)
        labels.append(f"{key} Sparse ({cpu_model})")

    if gpu_available:
        for key in ["GPU_F32", "GPU_F64"]:
            mean_errors = [np.mean(e) if len(e) > 0 else 1e-20 for e in sparse_errors[op_key][key]]
            if all(e == 1e-20 for e in mean_errors):
                print(f"Warning: No error data for {key} in {op_name} (Sparse)")
                continue
            line, = plt.plot(matrix_sizes, mean_errors, color=color_map[key], marker=marker_map[key], 
                             linestyle='-', markersize=8)
            lines.append(line)
            labels.append(f"{key} Sparse ({gpu_model})")

    plt.grid(True, which="both", ls="--", alpha=0.7)

# Add single legend outside subplots
plt.legend(lines[:len(labels)], labels, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
plt.figtext(0.5, 0.01, f"Mean Accuracy for CPU ({cpu_model}) and GPU ({gpu_model}) Sparse Matrices vs CPU F64 Reference, Python {python_version}", 
            ha="center", fontsize=12, wrap=True)
plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
plt.savefig("plots/cpu_gpu_sparse_mean_accuracy_results.png", dpi=300, bbox_inches='tight')
plt.close()

# Individual Accuracy Plots (8 separate plots: CPU/GPU, Dense/Sparse, F32/F64)
def plot_accuracy(errors_dict, op_list, title_prefix, device, dtype, matrix_type, filename):
    plt.figure(figsize=(20, 12))
    for i, (op_key, op_name) in enumerate(op_list, 1):
        plt.subplot(2, 3, i)
        plt.title(f"{op_name} ({matrix_type}, {device} {dtype}, Error)", fontsize=12)
        plt.xlabel("Size of Matrix (n×n)", fontsize=10)
        plt.ylabel("Relative Error (log-scale)", fontsize=10)
        plt.yscale("log")
        plt.xticks(matrix_sizes, matrix_sizes, fontsize=9)
        
        all_errors = []
        key = f"{device}_{dtype}"
        if key in errors_dict[op_key]:
            for e in errors_dict[op_key][key]:
                if len(e) > 0:
                    all_errors.extend(e)
        if all_errors:
            min_error = max(min(all_errors) / 2, 1e-20)
            max_error = max(all_errors) * 2
            plt.ylim(min_error, max_error)
        
        data = [e if len(e) > 0 else [1e-20] * num_runs for e in errors_dict[op_key][key]]
        box = plt.boxplot(data, positions=matrix_sizes, widths=200, patch_artist=True,
                          boxprops=dict(facecolor=color_map[key], edgecolor='black'),
                          whiskerprops=dict(color='black'), capprops=dict(color='black'),
                          medianprops=dict(color='red'), showfliers=True)
        
        plt.grid(True, which="both", ls="--", alpha=0.7)

    # Add single legend outside subplots
    plt.legend([box['boxes'][0]], [f"{device} {dtype} {matrix_type} ({cpu_model if device == 'CPU' else gpu_model})"], 
               fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.figtext(0.5, 0.01, 
                f"Accuracy for {device} ({cpu_model if device == 'CPU' else gpu_model}) {dtype} {matrix_type} Matrices vs CPU F64 Reference, Python {python_version}", 
                ha="center", fontsize=12, wrap=True)
    plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
    plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

# Generate individual accuracy plots
plot_accuracy(errors, operations, "Dense", "CPU", "F32", "Dense", "cpu_f32_dense_accuracy_results.png")
plot_accuracy(errors, operations, "Dense", "CPU", "F64", "Dense", "cpu_f64_dense_accuracy_results.png")
plot_accuracy(sparse_errors, sparse_operations, "Sparse", "CPU", "F32", "Sparse", "cpu_f32_sparse_accuracy_results.png")
plot_accuracy(sparse_errors, sparse_operations, "Sparse", "CPU", "F64", "Sparse", "cpu_f64_sparse_accuracy_results.png")

if gpu_available:
    plot_accuracy(errors, operations, "Dense", "GPU", "F32", "Dense", "gpu_f32_dense_accuracy_results.png")
    plot_accuracy(errors, operations, "Dense", "GPU", "F64", "Dense", "gpu_f64_dense_accuracy_results.png")
    plot_accuracy(sparse_errors, sparse_operations, "Sparse", "GPU", "F32", "Sparse", "gpu_f32_sparse_accuracy_results.png")
    plot_accuracy(sparse_errors, sparse_operations, "Sparse", "GPU", "F64", "Sparse", "gpu_f64_sparse_accuracy_results.png")

print("\nPlots saved in 'plots' directory.")
