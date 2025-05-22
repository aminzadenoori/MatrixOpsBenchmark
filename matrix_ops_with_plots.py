import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import platform
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
import gc  # For garbage collection as fallback

# CPU model detection
def get_cpu_model():
    try:
        from cpuinfo import get_cpu_info
        cpu_info = get_cpu_info()
        return cpu_info.get('brand_raw', 'Unknown CPU')
    except ImportError:
        return platform.processor() or "Unknown CPU"
    except Exception:
        return "Unknown CPU"

cpu_model = get_cpu_model()
print(f"CPU Model: {cpu_model}")

# GPU model detection
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
np_xp = np
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    cp_xp = cp
    gpu_available = cp.cuda.runtime.getDeviceCount() > 0
    print("CuPy (GPU) available" if gpu_available else "CuPy available but no GPU detected")
except ImportError:
    cp_xp = None
    cp_sp = None
    gpu_available = False
    print("CuPy (GPU) not available")

# Matrix sizes to test (columns, rows fixed at 10000)
fixed_rows = 10000
matrix_sizes = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
num_runs = 1

# Timing results (for CPU_F32, CPU_F64, GPU_F32, GPU_F64)
timings = {op: {
    "CPU_F32": [[] for _ in matrix_sizes],
    "CPU_F64": [[] for _ in matrix_sizes],
    "GPU_F32": [[] for _ in matrix_sizes],
    "GPU_F64": [[] for _ in matrix_sizes]
} for op in ["exp", "mean", "matprod", "inv", "svd"]}
sparse_timings = {op: {
    "CPU_F32": [[] for _ in matrix_sizes],
    "CPU_F64": [[] for _ in matrix_sizes],
    "GPU_F32": [[] for _ in matrix_sizes],
    "GPU_F64": [[] for _ in matrix_sizes]
} for op in ["exp", "mean", "matprod", "svd"]}

# Error results (for CPU_F32, GPU_F32, GPU_F64; CPU_F64 is reference)
errors = {op: {
    "CPU_F32": [[] for _ in matrix_sizes],
    "GPU_F32": [[] for _ in matrix_sizes],
    "GPU_F64": [[] for _ in matrix_sizes]
} for op in ["exp", "mean", "matprod", "inv", "svd"]}
sparse_errors = {op: {
    "CPU_F32": [[] for _ in matrix_sizes],
    "GPU_F32": [[] for _ in matrix_sizes],
    "GPU_F64": [[] for _ in matrix_sizes]
} for op in ["exp", "mean", "matprod", "svd"]}

# Convert between CuPy/NumPy arrays and sparse matrices, ensuring dense NumPy output
def to_numpy(array, xp):
    try:
        print(f"to_numpy: Input type = {type(array)}, xp = {xp.__name__}")
        if xp == cp_xp and isinstance(array, cp_sp.csr_matrix):  # CuPy sparse matrix
            print("Converting CuPy sparse matrix to dense NumPy")
            array = array.todense().get()  # Convert to dense CuPy, then to NumPy
        elif xp == cp_xp and hasattr(array, 'get'):  # CuPy dense array
            print("Converting CuPy dense array to NumPy")
            array = array.get()  # Convert CuPy dense to NumPy
        elif sp.issparse(array):  # SciPy sparse matrix
            print("Converting SciPy sparse matrix to dense NumPy")
            array = array.todense()  # Convert to dense NumPy
        else:
            print(f"Converting array of type {type(array)} directly to NumPy")
        result = np.asarray(array, dtype=np.float64)
        print(f"to_numpy: Output type = {type(result)}")
        return result
    except Exception as e:
        raise ValueError(f"Failed to convert array to NumPy: {str(e)}, array type: {type(array)}")

# Compute relative error safely
def compute_relative_error(result, reference):
    result = np.asarray(result, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    ref_norm = np.linalg.norm(reference)
    if ref_norm > 1e-20:
        error = np.linalg.norm(result - reference) / ref_norm
    else:
        print(f"Small reference norm detected: {ref_norm}")
        error = 0.0
    return error

# Function to compute reference results (CPU_F64) for a given size
def compute_references(N, matrix, size_idx):
    reference_results = {}
    reference_times = {
        "dense": {"exp": [], "mean": [], "matprod": [], "inv": [], "svd": []},
        "sparse": {"exp": [], "mean": [], "matprod": [], "svd": []}
    }

    print(f"Generating reference results for size {fixed_rows}x{N}...")
    # Dense reference: CPU_F64
    A_ref = matrix  # View of the float64 matrix

    # Exponential
    start = time.time()
    result = np.exp(A_ref)
    reference_results[f"exp_{N}_dense"] = result.astype(np.float64)
    reference_times["dense"]["exp"].append(time.time() - start)
    del result
    result = None
    gc.collect()

    # Mean of rows
    start = time.time()
    result = np.mean(A_ref, axis=1)
    reference_results[f"mean_{N}_dense"] = result.astype(np.float64)
    reference_times["dense"]["mean"].append(time.time() - start)
    del result
    result = None
    gc.collect()

    # Matrix product (A @ A.T for rectangular)
    start = time.time()
    result = np.dot(A_ref, A_ref.T)  # Output is 10000x10000
    reference_results[f"matprod_{N}_dense"] = result.astype(np.float64)
    reference_times["dense"]["matprod"].append(time.time() - start)
    del result
    result = None
    gc.collect()

    # PInverse 
    start = time.time()
    result = np.linalg.pinv(A_ref)
    reference_results[f"inv_{N}_dense"] = result.astype(np.float64)
    reference_times["dense"]["inv"].append(time.time() - start)
    del result
    result = None
    gc.collect()

    # Randomized SVD
    start = time.time()
    n_components = min(fixed_rows, N, 100)
    U, S, VT = randomized_svd(A_ref, n_components=n_components, random_state=0)
    reference_results[f"svd_S_{N}_dense"] = S
    reference_times["dense"]["svd"].append(time.time() - start)
    del U, S, VT
    U = S = VT = None
    gc.collect()

    # Sparse reference: CPU_F64 (1% non-zero elements)
    mask = np.random.rand(fixed_rows, N) < 0.01
    A_sparse_ref = sp.csr_matrix(A_ref * mask, dtype=np.float64)
    
    # Exponential
    start = time.time()
    expA = A_sparse_ref.copy()  # Necessary for CSR data modification
    expA.data = np.exp(expA.data)
    reference_results[f"exp_{N}_sparse"] = to_numpy(expA, np_xp)
    reference_times["sparse"]["exp"].append(time.time() - start)
    del expA
    expA = None
    gc.collect()

    # Mean of rows
    start = time.time()
    result = np.array(A_sparse_ref.mean(axis=1)).flatten()
    reference_results[f"mean_{N}_sparse"] = result
    reference_times["sparse"]["mean"].append(time.time() - start)
    del result
    result = None
    gc.collect()

    # Matrix product (A @ A.T for rectangular)
    start = time.time()
    result = A_sparse_ref @ A_sparse_ref.T
    reference_results[f"matprod_{N}_sparse"] = to_numpy(result, np_xp)
    reference_times["sparse"]["matprod"].append(time.time() - start)
    del result
    result = None
    gc.collect()

    # Randomized SVD
    start = time.time()
    A_np = A_sparse_ref.todense()  # Necessary for SVD
    U, S, VT = randomized_svd(A_np, n_components=n_components, random_state=0)
    reference_results[f"svd_S_{N}_sparse"] = S
    reference_times["sparse"]["svd"].append(time.time() - start)
    del A_np, U, S, VT
    A_np = U = S = VT = None
    gc.collect()

    # Store CPU_F64 timings
    for op in ["exp", "mean", "matprod", "inv", "svd"]:
        if op in reference_times["dense"]:
            timings[op]["CPU_F64"][size_idx].append(reference_times["dense"][op][0] if reference_times["dense"][op] else 1e-10)
    for op in ["exp", "mean", "matprod", "svd"]:
        sparse_timings[op]["CPU_F64"][size_idx].append(reference_times["sparse"][op][0])

    # Clear temporary matrices
    del A_ref, A_sparse_ref, mask
    A_ref = A_sparse_ref = mask = None
    gc.collect()

    return reference_results

# Modified run_experiments function
def run_experiments(dtype, label, sparse=False, size_idx=0, run_idx=0, device="CPU", matrix=None, reference_results=None):
    xp = cp_xp if device == "GPU" and gpu_available else np_xp
    sp_xp = cp_sp if device == "GPU" and gpu_available else sp
    print(f"\nRun {run_idx+1}/{num_runs} with {device}_{label} {'(sparse)' if sparse else '(dense)'}...")
    N = matrix_sizes[size_idx]
    print(f"Matrix size: {fixed_rows}x{N}")

    timings_dict = sparse_timings if sparse else timings
    errors_dict = sparse_errors if sparse else errors
    ref_key = f"{'sparse' if sparse else 'dense'}"
    expA_ref = reference_results[f"exp_{N}_{ref_key}"]
    meanRows_ref = reference_results[f"mean_{N}_{ref_key}"]
    matprod_ref = reference_results[f"matprod_{N}_{ref_key}"]
    svd_S_ref = reference_results[f"svd_S_{N}_{ref_key}"]
    inv_ref = reference_results.get(f"inv_{N}_dense", None) if not sparse else None

    if sparse:
        mask = np.random.rand(fixed_rows, N) < 0.01
        A = sp.csr_matrix(matrix * mask, dtype=dtype)
        if device == "GPU" and gpu_available:
            A = cp_sp.csr_matrix(A)
    else:
        A = xp.array(matrix, dtype=dtype)

    # Exponential
    if device == "GPU" and gpu_available:
        cp_xp.get_default_memory_pool().free_all_blocks()
    start = time.time()
    if sparse:
        expA = A.copy()
        expA.data = xp.exp(expA.data)
    else:
        expA = xp.exp(A)
    if device == "GPU" and gpu_available:
        cp_xp.cuda.Stream.null.synchronize()
        if sparse:
            assert not cp_xp.any(cp_xp.isnan(expA.data)), f"NaN detected in exp {device}_{label}"
            assert expA.data.size > 0, f"Empty array in exp {device}_{label}"
        else:
            assert not cp_xp.any(cp_xp.isnan(expA)), f"NaN detected in exp {device}_{label}"
            assert expA.size > 0, f"Empty array in exp {device}_{label}"
    exp_time = time.time() - start
    expA_np = to_numpy(expA, xp)
    error = compute_relative_error(expA_np, expA_ref)
    print(f"{device}_{label} exp error: {error}")
    timings_dict["exp"][f"{device}_{label}"][size_idx].append(max(exp_time, 1e-10))
    errors_dict["exp"][f"{device}_{label}"][size_idx].append(error)
    del expA, expA_np
    expA = expA_np = None
    if device == "GPU" and gpu_available:
        cp_xp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    # Mean of rows
    if device == "GPU" and gpu_available:
        cp_xp.get_default_memory_pool().free_all_blocks()
    start = time.time()
    if sparse:
        meanRows = xp.array(A.mean(axis=1)).flatten()
    else:
        meanRows = xp.mean(A, axis=1)
    if device == "GPU" and gpu_available:
        cp_xp.cuda.Stream.null.synchronize()
        assert not cp_xp.any(cp_xp.isnan(meanRows)), f"NaN detected in mean {device}_{label}"
        assert meanRows.size > 0, f"Empty array in mean {device}_{label}"
    mean_time = time.time() - start
    meanRows_np = to_numpy(meanRows, xp)
    error = compute_relative_error(meanRows_np, meanRows_ref)
    print(f"{device}_{label} mean error: {error}")
    timings_dict["mean"][f"{device}_{label}"][size_idx].append(max(mean_time, 1e-10))
    errors_dict["mean"][f"{device}_{label}"][size_idx].append(error)
    del meanRows, meanRows_np
    meanRows = meanRows_np = None
    if device == "GPU" and gpu_available:
        cp_xp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    # Matrix product (A @ A.T for rectangular)
    if device == "GPU" and gpu_available:
        cp_xp.get_default_memory_pool().free_all_blocks()
    start = time.time()
    C = A @ A.T  # Output is 10000x10000
    if device == "GPU" and gpu_available:
        cp_xp.cuda.Stream.null.synchronize()
        if sparse:
            assert not cp_xp.any(cp_xp.isnan(C.data)), f"NaN detected in matprod {device}_{label}"
            assert C.data.size > 0, f"Empty array in matprod {device}_{label}"
        else:
            assert not cp_xp.any(cp_xp.isnan(C)), f"NaN detected in matprod {device}_{label}"
            assert C.size > 0, f"Empty array in matprod {device}_{label}"
    matprod_time = time.time() - start
    C_np = to_numpy(C, xp)
    error = compute_relative_error(C_np, matprod_ref)
    print(f"{device}_{label} matprod error: {error}")
    timings_dict["matprod"][f"{device}_{label}"][size_idx].append(max(matprod_time, 1e-10))
    errors_dict["matprod"][f"{device}_{label}"][size_idx].append(error)
    del C, C_np
    C = C_np = None
    if device == "GPU" and gpu_available:
        cp_xp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    # Inverse (non-sparse, only for square matrix)
    if not sparse:
        if device == "GPU" and gpu_available:
            cp_xp.get_default_memory_pool().free_all_blocks()
        start = time.time()
        invA = xp.linalg.pinv(A)
        if device == "GPU" and gpu_available:
            cp_xp.cuda.Stream.null.synchronize()
            assert not cp_xp.any(cp_xp.isnan(invA)), f"NaN detected in inv {device}_{label}"
            assert invA.size > 0, f"Empty array in inv {device}_{label}"
        inv_time = time.time() - start
        invA_np = to_numpy(invA, xp)
        error = compute_relative_error(invA_np, inv_ref)
        print(f"{device}_{label} inv error: {error}")
        timings_dict["inv"][f"{device}_{label}"][size_idx].append(max(inv_time, 1e-10))
        errors_dict["inv"][f"{device}_{label}"][size_idx].append(error)
        del invA, invA_np
        invA = invA_np = None
        if device == "GPU" and gpu_available:
            cp_xp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    elif not sparse:
        timings_dict["inv"][f"{device}_{label}"][size_idx].append(1e-10)
        errors_dict["inv"][f"{device}_{label}"][size_idx].append(0.0)

    # Randomized SVD
    if device == "GPU" and gpu_available:
        cp_xp.get_default_memory_pool().free_all_blocks()
    start = time.time()
    n_components = min(fixed_rows, N, 100)
    A_np = to_numpy(A, xp)
    U, S, VT = randomized_svd(A_np, n_components=n_components, random_state=run_idx)
    svd_time = time.time() - start
    error = compute_relative_error(S, svd_S_ref[:n_components])
    print(f"{device}_{label} svd error: {error}")
    timings_dict["svd"][f"{device}_{label}"][size_idx].append(max(svd_time, 1e-10))
    errors_dict["svd"][f"{device}_{label}"][size_idx].append(error)
    del A_np, U, S, VT
    A_np = U = S = VT = None
    if device == "GPU" and gpu_available:
        cp_xp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    # Clear input matrix and mask
    del A
    if sparse:
        del mask
        mask = None
    A = None
    gc.collect()

# Process each size sequentially
for size_idx, N in enumerate(matrix_sizes):
    print(f"\nProcessing matrix size {fixed_rows}x{N}...")
    # Generate single float64 matrix for current size
    matrix = np.random.rand(fixed_rows, N).astype(np.float64)

    # Compute reference results (CPU_F64)
    reference_results = compute_references(N, matrix, size_idx)

    # Run experiments (CPU_F32, GPU_F32, GPU_F64)
    for run_idx in range(num_runs):
        run_experiments(np.float32, "F32", sparse=False, size_idx=size_idx, run_idx=run_idx, device="CPU", matrix=matrix, reference_results=reference_results)
        if gpu_available:
            run_experiments(np.float32, "F32", sparse=False, size_idx=size_idx, run_idx=run_idx, device="GPU", matrix=matrix, reference_results=reference_results)
            run_experiments(np.float64, "F64", sparse=False, size_idx=size_idx, run_idx=run_idx, device="GPU", matrix=matrix, reference_results=reference_results)
        run_experiments(np.float32, "F32", sparse=True, size_idx=size_idx, run_idx=run_idx, device="CPU", matrix=matrix, reference_results=reference_results)
        if gpu_available:
            run_experiments(np.float32, "F32", sparse=True, size_idx=size_idx, run_idx=run_idx, device="GPU", matrix=matrix, reference_results=reference_results)
            run_experiments(np.float64, "F64", sparse=True, size_idx=size_idx, run_idx=run_idx, device="GPU", matrix=matrix, reference_results=reference_results)

    # Flush memory completely, keeping only results
    print(f"Flushing memory for size {fixed_rows}x{N}...")
    del matrix, reference_results
    matrix = reference_results = None
    gc.collect()

# Plotting
operations = [
    ("exp", "Exponential of each element"),
    ("mean", "Means of the rows"),
    ("matprod", "Matrix product A @ A.T"),
    ("inv", "Inverse of a matrix"),
    ("svd", "Randomized SVD")
]
sparse_operations = operations[:3] + [operations[4]]  # Exclude inverse

color_map = {'CPU_F32': 'purple', 'GPU_F32': 'blue', 'GPU_F64': 'cyan'}
marker_map = {'CPU_F32': 'o', 'GPU_F32': '^', 'GPU_F64': 's'}

Path("plots").mkdir(exist_ok=True)

# Dense Timing Plot
plt.figure(figsize=(15, 10))
for i, (op_key, op_name) in enumerate(operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (Dense, Time)", fontsize=10)
    plt.xlabel(f"Matrix Size (10000×N)", fontsize=8)
    plt.ylabel("Time (s, log)", fontsize=8)
    plt.yscale("log")
    plt.xticks(matrix_sizes, [f"{n//1000}k" for n in matrix_sizes], fontsize=7)
    plt.yticks(fontsize=7)

    all_times = []
    for key in ["CPU_F32", "CPU_F64", "GPU_F32", "GPU_F64"]:
        for t_list in timings[op_key][key]:
            if len(t_list) > 0:
                all_times.extend([t for t in t_list if t > 0])

    if all_times:
        min_time = max(min(all_times) / 2, 1e-10)
        max_time = max(all_times) * 2
        plt.ylim(min_time, max_time)
    else:
        plt.ylim(1e-10, 1e-1)
        print(f"Warning: No valid timing data for {op_name} (Dense)")

    lines = []
    labels = []
    for key in ["CPU_F32", "CPU_F64"]:
        mean_times = [np.mean(t) if len(t) > 0 else 1e-10 for t in timings[op_key][key]]
        line, = plt.plot(matrix_sizes, mean_times, color=color_map.get(key, 'green'), marker=marker_map.get(key, 'x'), linestyle='-', markersize=6)
        lines.append(line)
        labels.append(f"{key} Dense ({cpu_model})")

    if gpu_available:
        for key in ["GPU_F32", "GPU_F64"]:
            mean_times = [np.mean(t) if len(t) > 0 else 1e-10 for t in timings[op_key][key]]
            line, = plt.plot(matrix_sizes, mean_times, color=color_map[key], marker=marker_map[key], linestyle='-', markersize=6)
            lines.append(line)
            labels.append(f"{key} Dense ({gpu_model})")

    plt.grid(True, which="both", ls="--", alpha=0.7)

plt.legend(lines[:len(labels)], labels, fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
plt.figtext(0.5, 0.01, f"Timing for CPU ({cpu_model}) and GPU ({gpu_model}) Dense Matrices (10000×N), Python {python_version}", 
            ha="center", fontsize=9, wrap=True)
plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
plt.savefig("plots/cpu_gpu_dense_timing_results.png", dpi=300, bbox_inches='tight')
plt.close()

# Sparse Timing Plot
plt.figure(figsize=(15, 10))
for i, (op_key, op_name) in enumerate(sparse_operations, 1):
    plt.subplot(2, 2, i)
    plt.title(f"{op_name} (Sparse, 1% non-zero, Time)", fontsize=10)
    plt.xlabel(f"Matrix Size (10000×N)", fontsize=8)
    plt.ylabel("Time (s, log)", fontsize=8)
    plt.yscale("log")
    plt.xticks(matrix_sizes, [f"{n//1000}k" for n in matrix_sizes], fontsize=7)
    plt.yticks(fontsize=7)

    all_times = []
    for key in ["CPU_F32", "CPU_F64", "GPU_F32", "GPU_F64"]:
        for t_list in sparse_timings[op_key][key]:
            if len(t_list) > 0:
                all_times.extend([t for t in t_list if t > 0])

    if all_times:
        min_time = max(min(all_times) / 2, 1e-10)
        max_time = max(all_times) * 2
        plt.ylim(min_time, max_time)
    else:
        plt.ylim(1e-10, 1e-1)
        print(f"Warning: No valid timing data for {op_name} (Sparse)")

    lines = []
    labels = []
    for key in ["CPU_F32", "CPU_F64"]:
        mean_times = [np.mean(t) if len(t) > 0 else 1e-10 for t in sparse_timings[op_key][key]]
        line, = plt.plot(matrix_sizes, mean_times, color=color_map.get(key, 'green'), marker=marker_map.get(key, 'x'), linestyle='-', markersize=6)
        lines.append(line)
        labels.append(f"{key} Sparse ({cpu_model})")

    if gpu_available:
        for key in ["GPU_F32", "GPU_F64"]:
            mean_times = [np.mean(t) if len(t) > 0 else 1e-10 for t in sparse_timings[op_key][key]]
            line, = plt.plot(matrix_sizes, mean_times, color=color_map[key], marker=marker_map[key], linestyle='-', markersize=6)
            lines.append(line)
            labels.append(f"{key} Sparse ({gpu_model})")

    plt.grid(True, which="both", ls="--", alpha=0.7)

plt.legend(lines[:len(labels)], labels, fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
plt.figtext(0.5, 0.01, f"Timing for CPU ({cpu_model}) and GPU ({gpu_model}) Sparse Matrices (10000×N, 1% non-zero), Python {python_version}", 
            ha="center", fontsize=9, wrap=True)
plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
plt.savefig("plots/cpu_gpu_sparse_timing_results.png", dpi=300, bbox_inches='tight')
plt.close()

# Dense Accuracy Plot
plt.figure(figsize=(15, 10))
for i, (op_key, op_name) in enumerate(operations, 1):
    plt.subplot(2, 3, i)
    plt.title(f"{op_name} (Dense, Relative Error)", fontsize=10)
    plt.xlabel(f"Matrix Size (10000×N)", fontsize=8)
    plt.ylabel("Relative Error (log)", fontsize=8)
    plt.yscale("log")
    plt.xticks(matrix_sizes, [f"{n//1000}k" for n in matrix_sizes], fontsize=7)
    plt.yticks(fontsize=7)

    all_errors = []
    for key in ["CPU_F32", "GPU_F32", "GPU_F64"]:
        for e_list in errors[op_key][key]:
            if len(e_list) > 0:
                all_errors.extend([e for e in e_list])

    if not all_errors:
        print(f"Warning: No error data for {op_name} (Dense)")
        plt.ylim(1e-20, 1e-1)
        continue

    non_zero_errors = [e for e in all_errors if e > 0]
    min_error = min(non_zero_errors) / 2 if non_zero_errors else 1e-20
    max_error = max(all_errors) * 2 if max(all_errors) > 0 else 1e-1
    plt.ylim(max(min_error, 1e-20), max_error)

    lines = []
    labels = []
    for key in ["CPU_F32"]:
        mean_errors = [np.mean(e) if len(e) > 0 else 0.0 for e in errors[op_key][key]]
        line, = plt.plot(matrix_sizes, mean_errors, color=color_map[key], marker=marker_map[key], linestyle='-', markersize=6)
        lines.append(line)
        labels.append(f"{key} Dense ({cpu_model})")

    if gpu_available:
        for key in ["GPU_F32", "GPU_F64"]:
            mean_errors = [np.mean(e) if len(e) > 0 else 0.0 for e in errors[op_key][key]]
            line, = plt.plot(matrix_sizes, mean_errors, color=color_map[key], marker=marker_map[key], linestyle='-', markersize=6)
            lines.append(line)
            labels.append(f"{key} Dense ({gpu_model})")

    plt.grid(True, which="both", ls="--", alpha=0.7)

plt.legend(lines[:len(labels)], labels, fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
plt.figtext(0.5, 0.01, f"Relative Error for CPU ({cpu_model}) and GPU ({gpu_model}) Dense Matrices (10000×N), Python {python_version}", 
            ha="center", fontsize=9, wrap=True)
plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
plt.savefig("plots/cpu_gpu_dense_error_results.png", dpi=300, bbox_inches='tight')
plt.close()

# Sparse Accuracy Plot
plt.figure(figsize=(15, 10))
for i, (op_key, op_name) in enumerate(sparse_operations, 1):
    plt.subplot(2, 2, i)
    plt.title(f"{op_name} (Sparse, 1% non-zero, Relative Error)", fontsize=10)
    plt.xlabel(f"Matrix Size (10000×N)", fontsize=8)
    plt.ylabel("Relative Error (log)", fontsize=8)
    plt.yscale("log")
    plt.xticks(matrix_sizes, [f"{n//1000}k" for n in matrix_sizes], fontsize=7)
    plt.yticks(fontsize=7)

    all_errors = []
    for key in ["CPU_F32", "GPU_F32", "GPU_F64"]:
        for e_list in sparse_errors[op_key][key]:
            if len(e_list) > 0:
                all_errors.extend([e for e in e_list])

    if not all_errors:
        print(f"Warning: No error data for {op_name} (Sparse)")
        plt.ylim(1e-20, 1e-1)
        continue

    non_zero_errors = [e for e in all_errors if e > 0]
    min_error = min(non_zero_errors) / 2 if non_zero_errors else 1e-20
    max_error = max(all_errors) * 2 if max(all_errors) > 0 else 1e-1
    plt.ylim(max(min_error, 1e-20), max_error)

    lines = []
    labels = []
    for key in ["CPU_F32"]:
        mean_errors = [np.mean(e) if len(e) > 0 else 0.0 for e in sparse_errors[op_key][key]]
        line, = plt.plot(matrix_sizes, mean_errors, color=color_map[key], marker=marker_map[key], linestyle='-', markersize=6)
        lines.append(line)
        labels.append(f"{key} Sparse ({cpu_model})")

    if gpu_available:
        for key in ["GPU_F32", "GPU_F64"]:
            mean_errors = [np.mean(e) if len(e) > 0 else 0.0 for e in sparse_errors[op_key][key]]
            line, = plt.plot(matrix_sizes, mean_errors, color=color_map[key], marker=marker_map[key], linestyle='-', markersize=6)
            lines.append(line)
            labels.append(f"{key} Sparse ({gpu_model})")

    plt.grid(True, which="both", ls="--", alpha=0.7)

plt.legend(lines[:len(labels)], labels, fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
plt.figtext(0.5, 0.01, f"Relative Error for CPU ({cpu_model}) and GPU ({gpu_model}) Sparse Matrices (10000×N, 1% non-zero), Python {python_version}", 
            ha="center", fontsize=9, wrap=True)
plt.tight_layout(pad=2.0, rect=[0, 0.05, 0.95, 1])
plt.savefig("plots/cpu_gpu_sparse_error_results.png", dpi=300, bbox_inches='tight')
plt.close()

# Clear plotting data to minimize memory
del operations, sparse_operations, color_map, marker_map
operations = sparse_operations = color_map = marker_map = None
gc.collect()

print("\nPlots saved in 'plots' directory.")
