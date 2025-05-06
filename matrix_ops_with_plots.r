# Load required R packages
library(reticulate)
library(ggplot2)
library(gridExtra)
library(tictoc)

# Import Python modules via reticulate
np <- import("numpy")
sp <- import("scipy.sparse")
plt <- import("matplotlib.pyplot")
time <- import("time")
platform <- import("platform")
pathlib <- import("pathlib")
subprocess <- import("subprocess")
tryCatch({
  cp <- import("cupy")
  pynvml <- import("pynvml")
}, error = function(e) {
  cat("CuPy or pynvml not available, falling back to NumPy (CPU)\n")
})

# Get operating system name
os_name <- platform$system()
if (os_name == "Darwin") os_name <- "macOS"

# Get CPU model
cpu_model <- tryCatch({
  if (os_name == "Windows") {
    result <- subprocess$check_output("wmic cpu get name", shell = TRUE)$decode()$strip()
    cpu_model <- strsplit(result, "\n")[[1]][2]$strip()
  } else if (os_name == "Linux") {
    result <- subprocess$check_output("lscpu | grep 'Model name'", shell = TRUE)$decode()$strip()
    cpu_model <- strsplit(result, ":")[[1]][2]$strip()
  } else if (os_name == "macOS") {
    result <- subprocess$check_output("sysctl -n machdep.cpu.brand_string", shell = TRUE)$decode()$strip()
    cpu_model <- result
  } else {
    cpu_model <- platform$processor() %||% "Unknown CPU"
  }
  cpu_model
}, error = function(e) {
  "Unknown CPU"
})
cat("CPU Model:", cpu_model, "\n")

# Get GPU model (if CuPy and pynvml are available)
gpu_model <- "No GPU (CPU only)"
if (exists("cp") && exists("pynvml")) {
  tryCatch({
    pynvml$nvmlInit()
    device_count <- pynvml$nvmlDeviceGetCount()
    if (device_count > 0) {
      handle <- pynvml$nvmlDeviceGetHandleByIndex(0)
      gpu_model <- pynvml$nvmlDeviceGetName(handle)$decode("utf-8")
    }
    pynvml$nvmlShutdown()
  }, error = function(e) {
    if (cp$cuda$runtime$getDeviceCount() > 0) {
      gpu_model <- sprintf("CUDA Device %d", cp$cuda$runtime$getDevice())
    }
  })
} else if (exists("cp")) {
  if (cp$cuda$runtime$getDeviceCount() > 0) {
    gpu_model <- sprintf("CUDA Device %d", cp$cuda$runtime$getDevice())
  }
}
cat("GPU Model:", gpu_model, "\n")

# Select array module (CuPy for GPU if available, else NumPy)
xp <- if (exists("cp")) cp else np
cat(sprintf("Using %s\n", if (exists("cp")) "CuPy (GPU)" else "NumPy (CPU)"))

# Matrix sizes to test
matrix_sizes <- c(1000, 2000, 3000, 4000, 5000)

# Initialize storage for reference results, matrices, and times
reference_results <- list()
reference_matrices <- list()
reference_times <- list(
  dense = list(
    elemwise = numeric(),
    exp = numeric(),
    mean = numeric(),
    matprod = numeric(),
    inv = numeric(),
    svd = numeric()
  ),
  sparse = list(
    elemwise = numeric(),
    exp = numeric(),
    mean = numeric(),
    matprod = numeric()
  )
)

# Timing results for plotting
timings <- list(
  elemwise = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  exp = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  mean = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  matprod = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  inv = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  svd = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric())
)

# Sparse timing results
sparse_timings <- list(
  elemwise = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric(), CPU_F64 = numeric()),
  exp = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric(), CPU_F64 = numeric()),
  mean = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric(), CPU_F64 = numeric()),
  matprod = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric(), CPU_F64 = numeric sefer()),
)

# Error results for plotting
errors <- list(
  elemwise = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  exp = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  mean = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  matprod = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  inv = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric()),
  svd = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric())
)

# Sparse error results
sparse_errors <- list(
  elemwise = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric(), CPU_F64 = numeric()),
  exp = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric(), CPU_F64 = numeric()),
  mean = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric(), CPU_F64 = numeric()),
  matprod = list(GPU_F32 = numeric(), GPU_F64 = numeric(), CPU_F32 = numeric(), CPU_F64 = numeric())
)

# Function to convert CuPy to NumPy arrays
to_numpy <- function(array) {
  if (identical(xp, cp)) {
    return(array$get())
  }
  return(array)
}

# Function to run experiments
run_experiments <- function(dtype, label, sparse = FALSE, size_idx) {
  cat(sprintf("\nRunning experiments with %s precision %s...\n", 
              label, ifelse(sparse, "(sparse)", "(non-sparse)")))
  timings_dict <- if (sparse) sparse_timings else timings
  errors_dict <- if (sparse) sparse_errors else errors
  N <- matrix_sizes[size_idx + 1]
  cat(sprintf("\nMatrix size: %dx%d\n", N, N))
  
  # Retrieve reference matrices
  matrices <- reference_matrices[[sprintf("matrices_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
  A_ref <- matrices[[1]]
  B_ref <- matrices[[2]]
  
  # Convert matrices to appropriate dtype and device
  if (sparse) {
    A <- A_ref$astype(dtype)
    B <- B_ref$astype(dtype)
    if (identical(xp, cp)) {
      A <- cp$sparse$csr_matrix(A)
      B <- cp$sparse$csr_matrix(B)
    }
  } else {
    A <- xp$array(A_ref$astype(dtype))
    B <- xp$array(B_ref$astype(dtype))
  }
  
  # 1. Element-wise product
  tic("Element-wise product")
  if (sparse) {
    C <- A$multiply(B)
  } else {
    C <- A * B
  }
  if (identical(xp, cp)) cp$cuda$Stream$null$synchronize()
  elemwise_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
  cat(sprintf("Element-wise product time: %.4f seconds\n", elemwise_time))
  C_ref <- reference_results[[sprintf("elemwise_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
  C_np <- to_numpy(if (sparse) C$todense() else C)
  error <- np$linalg$norm(C_np - C_ref) / np$linalg$norm(C_ref)
  if (is.nan(error) || np$linalg$norm(C_ref) == 0) error <- 0
  cat(sprintf("Relative error vs CPU F64: %.6e\n", error))
  timings_dict$elemwise[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- elemwise_time
  errors_dict$elemwise[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- error
  
  # 2. Exponential
  tic("Exponential")
  if (sparse) {
    expA <- A$copy()
    expA$data <- xp$exp(expA$data)
  } else {
    expA <- xp$exp(A)
  }
  if (identical(xp, cp)) cp$cuda$Stream$null$synchronize()
  exp_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
  cat(sprintf("Exponential time: %.4f seconds\n", exp_time))
  expA_ref <- reference_results[[sprintf("exp_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
  expA_np <- to_numpy(if (sparse) expA$todense() else expA)
  error <- np$linalg$norm(expA_np - expA_ref) / np$linalg$norm(expA_ref)
  if (is.nan(error) || np$linalg$norm(expA_ref) == 0) error <- 0
  cat(sprintf("Relative error vs CPU F64: %.6e\n", error))
  timings_dict$exp[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- exp_time
  errors_dict$exp[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- error
  
  # 3. Means of the rows
  tic("Means of rows")
  if (sparse) {
    meanRows <- xp$array(A$mean(axis = 1))$flatten()
  } else {
    meanRows <- xp$mean(A, axis = 1)
  }
  if (identical(xp, cp)) cp$cuda$Stream$null$synchronize()
  mean_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
  cat(sprintf("Means of rows time: %.4f seconds\n", mean_time))
  meanRows_ref <- reference_results[[sprintf("mean_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
  meanRows_np <- to_numpy(meanRows)
  error <- np$linalg$norm(meanRows_np - meanRows_ref) / np$linalg$norm(meanRows_ref)
  if (is.nan(error) || np$linalg$norm(meanRows_ref) == 0) error <- 0
  cat(sprintf("Relative error vs CPU F64: %.6e\n", error))
  timings_dict$mean[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- mean_time
  errors_dict$mean[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- error
  
  # 4. Matrix product
  tic("Matrix product")
  C <- A$`@`(B)
  if (identical(xp, cp)) cp$cuda$Stream$null$synchronize()
  matprod_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
  cat(sprintf("Matrix product time: %.4f seconds\n", matprod_time))
  C_ref <- reference_results[[sprintf("matprod_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
  C_np <- to_numpy(if (sparse) C$todense() else C)
  error <- np$linalg$norm(C_np - C_ref) / np$linalg$norm(C_ref)
  if (is.nan(error) || np$linalg$norm(C_ref) == 0) error <- 0
  cat(sprintf("Relative error vs CPU F64: %.6e\n", error))
  timings_dict$matprod[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- matprod_time
  errors_dict$matprod[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- error
  
  # 5. Inverse (non-sparse only)
  if (!sparse) {
    tic("Matrix inverse")
    invA <- xp$linalg$inv(A)
    if (identical(xp, cp)) cp$cuda$Stream$null$synchronize()
    inv_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
    cat(sprintf("Matrix inverse time: %.4f seconds\n", inv_time))
    invA_ref <- reference_results[[sprintf("inv_%d_dense", N)]]
    invA_np <- to_numpy(invA)
    error <- np$linalg$norm(invA_np - invA_ref) / np$linalg$norm(invA_ref)
    if (is.nan(error) || np$linalg$norm(invA_ref) == 0) error <- 0
    cat(sprintf("Relative error vs CPU F64: %.6e\n", error))
    timings$inv[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- inv_time
    errors$inv[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- error
  } else {
    cat("Matrix inverse skipped for sparse matrices.\n")
  }
  
  # 6. SVD (non-sparse only)
  if (!sparse) {
    tic("SVD")
    svd_result <- xp$linalg$svd(A, full_matrices = TRUE)
    if (identical(xp, cp)) cp$cuda$Stream$null$synchronize()
    svd_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
    cat(sprintf("SVD time: %.4f seconds\n", svd_time))
    S_ref <- reference_results[[sprintf("svd_S_%d_dense", N)]]
    S_np <- to_numpy(svd_result[[2]])
    error <- np$linalg$norm(S_np - S_ref) / np$linalg$norm(S_ref)
    if (is.nan(error) || np$linalg$norm(S_ref) == 0) error <- 0
    cat(sprintf("SVD (S) relative error vs CPU F64: %.6e\n", error))
    timings$svd[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- svd_time
    errors$svd[[sprintf("%s_%s", if (identical(xp, cp)) "GPU" else "CPU", label)]][size_idx + 1] <- error
  } else {
    cat("SVD skipped for sparse matrices.\n")
  }
  
  # CPU F32 execution for comparison (if running on GPU)
  if (identical(xp, cp)) {
    cat("\nRunning CPU F32 for comparison...\n")
    A_np <- A_ref$astype(np$float32)
    B_np <- B_ref$astype(np$float32)
    if (sparse) {
      A_np <- sp$csr_matrix(A_np)
      B_np <- sp$csr_matrix(B_np)
    }
    
    # 1. Element-wise product
    tic("CPU F32 Element-wise product")
    if (sparse) {
      C_np <- A_np$multiply(B_np)
    } else {
      C_np <- A_np * B_np
    }
    cpu_elemwise_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
    cat(sprintf("CPU F32 Element-wise product time: %.4f seconds\n", cpu_elemwise_time))
    cat(sprintf("Speedup (GPU vs CPU): %.2fx\n", cpu_elemwise_time / elemwise_time))
    C_ref <- reference_results[[sprintf("elemwise_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
    C_np <- if (sparse) C_np$todense() else C_np
    error <- np$linalg$norm(C_np - C_ref) / np$linalg$norm(C_ref)
    if (is.nan(error) || np$linalg$norm(C_ref) == 0) error <- 0
    cat(sprintf("CPU F32 Relative error vs CPU F64: %.6e\n", error))
    timings_dict$elemwise$CPU_F32[size_idx + 1] <- cpu_elemwise_time
    errors_dict$elemwise$CPU_F32[size_idx + 1] <- error
    
    # 2. Exponential
    tic("CPU F32 Exponential")
    if (sparse) {
      expA_np <- A_np$copy()
      expA_np$data <- np$exp(expA_np$data)
    } else {
      expA_np <- np$exp(A_np)
    }
    cpu_exp_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
    cat(sprintf("CPU F32 Exponential time: %.4f seconds\n", cpu_exp_time))
    cat(sprintf("Speedup (GPU vs CPU): %.2fx\n", cpu_exp_time / exp_time))
    expA_ref <- reference_results[[sprintf("exp_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
    expA_np <- if (sparse) expA_np$todense() else expA_np
    error <- np$linalg$norm(expA_np - expA_ref) / np$linalg$norm(expA_ref)
    if (is.nan(error) || np$linalg$norm(expA_ref) == 0) error <- 0
    cat(sprintf("CPU F32 Relative error vs CPU F64: %.6e\n", error))
    timings_dict$exp$CPU_F32[size_idx + 1] <- cpu_exp_time
    errors_dict$exp$CPU_F32[size_idx + 1] <- error
    
    # 3. Means of rows
    tic("CPU F32 Means of rows")
    if (sparse) {
      meanRows_np <- np$array(A_np$mean(axis = 1))$flatten()
    } else {
      meanRows_np <- np$mean(A_np, axis = 1)
    }
    cpu_mean_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
    cat(sprintf("CPU F32 Means of rows time: %.4f seconds\n", cpu_mean_time))
    cat(sprintf("Speedup (GPU vs CPU): %.2fx\n", cpu_mean_time / mean_time))
    meanRows_ref <- reference_results[[sprintf("mean_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
    error <- np$linalg$norm(meanRows_np - meanRows_ref) / np$linalg$norm(meanRows_ref)
    if (is.nan(error) || np$linalg$norm(meanRows_ref) == 0) error <- 0
    cat(sprintf("CPU F32 Relative error vs CPU F64: %.6e\n", error))
    timings_dict$mean$CPU_F32[size_idx + 1] <- cpu_mean_time
    errors_dict$mean$CPU_F32[size_idx + 1] <- error
    
    # 4. Matrix product
    tic("CPU F32 Matrix product")
    C_np <- A_np$`@`(B_np)
    cpu_matprod_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
    cat(sprintf("CPU F32 Matrix product time: %.4f seconds\n", cpu_matprod_time))
    cat(sprintf("Speedup (GPU vs CPU): %.2fx\n", cpu_matprod_time / matprod_time))
    C_ref <- reference_results[[sprintf("matprod_%d_%s", N, ifelse(sparse, "sparse", "dense"))]]
    C_np <- if (sparse) C_np$todense() else C_np
    error <- np$linalg$norm(C_np - C_ref) / np$linalg$norm(C_ref)
    if (is.nan(error) || np$linalg$norm(C_ref) == 0) error <- 0
    cat(sprintf("CPU F32 Relative error vs CPU F64: %.6e\n", error))
    timings_dict$matprod$CPU_F32[size_idx + 1] <- cpu_matprod_time
    errors_dict$matprod$CPU_F32[size_idx + 1] <- error
    
    # 5. Inverse (non-sparse only)
    if (!sparse) {
      tic("CPU F32 Matrix inverse")
      invA_np <- np$linalg$inv(A_np)
      cpu_inv_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
      cat(sprintf("CPU F32 Matrix inverse time: %.4f seconds\n", cpu_inv_time))
      cat(sprintf("Speedup (GPU vs CPU): %.2fx\n", cpu_inv_time / inv_time))
      invA_ref <- reference_results[[sprintf("inv_%d_dense", N)]]
      error <- np$linalg$norm(invA_np - invA_ref) / np$linalg$norm(invA_ref)
      if (is.nan(error) || np$linalg$norm(invA_ref) == 0) error <- 0
      cat(sprintf("CPU F32 Relative error vs CPU F64: %.6e\n", error))
      timings$inv$CPU_F32[size_idx + 1] <- cpu_inv_time
      errors$inv$CPU_F32[size_idx + 1] <- error
    }
    
    # 6. SVD (non-sparse only)
    if (!sparse) {
      tic("CPU F32 SVD")
      svd_result_np <- np$linalg$svd(A_np, full_matrices = TRUE)
      cpu_svd_time <- toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic
      cat(sprintf("CPU F32 SVD time: %.4f seconds\n", cpu_svd_time))
      cat(sprintf("Speedup (GPU vs CPU): %.2fx\n", cpu_svd_time / svd_time))
      S_ref <- reference_results[[sprintf("svd_S_%d_dense", N)]]
      error <- np$linalg$norm(svd_result_np[[2]] - S_ref) / np$linalg$norm(S_ref)
      if (is.nan(error) || np$linalg$norm(S_ref) == 0) error <- 0
      cat(sprintf("CPU F32 Relative error vs CPU F64: %.6e\n", error))
      timings$svd$CPU_F32[size_idx + 1] <- cpu_svd_time
      errors$svd$CPU_F32[size_idx + 1] <- error
    }
  }
}

# Generate CPU F64 reference results
cat("Generating CPU F64 reference results, storing matrices, and measuring times...\n")
for (N in matrix_sizes) {
  for (sparse in c(FALSE, TRUE)) {
    if (sparse) {
      A <- sp$random(N, N, density = 0.01, format = "csr", dtype = np$float64)
      B <- sp$random(N, N, density = 0.01, format = "csr", dtype = np$float64)
      reference_matrices[[sprintf("matrices_%d_sparse", N)]] <- list(A, B)
      
      tic("Sparse elemwise")
      reference_results[[sprintf("elemwise_%d_sparse", N)]] <- A$multiply(B)$todense()
      reference_times$sparse$elemwise <- c(reference_times$sparse$elemwise, 
                                           toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
      
      tic("Sparse exp")
      expA <- A$copy()
      expA$data <- np$exp(expA$data)
      reference_results[[sprintf("exp_%d_sparse", N)]] <- expA$todense()
      reference_times$sparse$exp <- c(reference_times$sparse$exp, 
                                      toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
      
      tic("Sparse mean")
      reference_results[[sprintf("mean_%d_sparse", N)]] <- np$array(A$mean(axis = 1))$flatten()
      reference_times$sparse$mean <- c(reference_times$sparse$mean, 
                                       toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
      
      tic("Sparse matprod")
      reference_results[[sprintf("matprod_%d_sparse", N)]] <- A$`@`(B)$todense()
      reference_times$sparse$matprod <- c(reference_times$sparse$matprod, 
                                          toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
    } else {
      A <- np$random$rand(N, N)$astype(np$float64)
      B <- np$random$rand(N, N)$astype(np$float64)
      reference_matrices[[sprintf("matrices_%d_dense", N)]] <- list(A, B)
      
      tic("Dense elemwise")
      reference_results[[sprintf("elemwise_%d_dense", N)]] <- (A * B)$astype(np$float64)
      reference_times$dense$elemwise <- c(reference_times$dense$elemwise, 
                                          toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
      
      tic("Dense exp")
      reference_results[[sprintf("exp_%d_dense", N)]] <- np$exp(A)$astype(np$float64)
      reference_times$dense$exp <- c(reference_times$dense$exp, 
                                    toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
      
      tic("Dense mean")
      reference_results[[sprintf("mean_%d_dense", N)]] <- np$mean(A, axis = 1)$astype(np$float64)
      reference_times$dense$mean <- c(reference_times$dense$mean, 
                                      toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
      
      tic("Dense matprod")
      reference_results[[sprintf("matprod_%d_dense", N)]] <- np$dot(A, B)$astype(np$float64)
      reference_times$dense$matprod <- c(reference_times$dense$matprod, 
                                         toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
      
      tic("Dense inv")
      reference_results[[sprintf("inv_%d_dense", N)]] <- np$linalg$inv(A)$astype(np$float64)
      reference_times$dense$inv <- c(reference_times$dense$inv, 
                                     toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
      
      tic("Dense svd")
      svd_result <- np$linalg$svd(A, full_matrices = TRUE)
      reference_results[[sprintf("svd_S_%d_dense", N)]] <- svd_result[[2]]$astype(np$float64)
      reference_times$dense$svd <- c(reference_times$dense$svd, 
                                    toc(quiet = TRUE)$toc - toc(quiet = TRUE)$tic)
    }
  }
}

# Initialize timing and error lists
for (op in names(timings)) {
  for (key in names(timings[[op]])) {
    timings[[op]][[key]] <- numeric(length(matrix_sizes))
  }
}
for (op in names(sparse_timings)) {
  for (key in names(sparse_timings[[op]])) {
    sparse_timings[[op]][[key]] <- numeric(length(matrix_sizes))
  }
}
for (op in names(errors)) {
  for (key in names(errors[[op]])) {
    errors[[op]][[key]] <- numeric(length(matrix_sizes))
  }
}
for (op in names(sparse_errors)) {
  for (key in names(sparse_errors[[op]])) {
    sparse_errors[[op]][[key]] <- numeric(length(matrix_sizes))
  }
}

# Run experiments
for (size_idx in seq_along(matrix_sizes) - 1) {
  # Dense experiments
  run_experiments(np$float32, "F32", sparse = FALSE, size_idx = size_idx)
  if (exists("cp")) run_experiments(np$float64, "F64", sparse = FALSE, size_idx = size_idx)
  # Sparse experiments
  run_experiments(np$float32, "F32", sparse = TRUE, size_idx = size_idx)
  run_experiments(np$float64, "F64", sparse = TRUE, size_idx = size_idx)
}

# Plotting
operations <- list(
  c("elemwise", "Element-wise product"),
  c("exp", "Exponential of each element"),
  c("mean", "Means of the rows"),
  c("matprod", "Matrix product"),
  c("inv", "Inverse of a matrix"),
  c("svd", "Singular Value Decomposition")
)

# Create output directory for plots
pathlib$Path("plots")$mkdir(exist_ok = TRUE)

# Plot dense matrix timing results
plots <- list()
for (i in seq_along(operations)) {
  op_key <- operations[[i]][1]
  op_name <- operations[[i]][2]
  data <- data.frame(
    Size = matrix_sizes,
    CPU_F64 = reference_times$dense[[op_key]]
  )
  if (exists("cp")) {
    data$GPU_F32 <- timings[[op_key]]$GPU_F32
    data$GPU_F64 <- timings[[op_key]]$GPU_F64
    data$CPU_F32 <- timings[[op_key]]$CPU_F32
  } else {
    data$CPU_F32 <- timings[[op_key]]$CPU_F32
  }
  p <- ggplot(data, aes(x = Size)) +
    geom_line(aes(y = CPU_F64, color = "CPU F64 Reference"), linetype = "dashed") +
    geom_point(aes(y = CPU_F64, color = "CPU F64 Reference"), shape = 4) +
    { if (exists("cp")) geom_line(aes(y = GPU_F32, color = "GPU F32")) } +
    { if (exists("cp")) geom_point(aes(y = GPU_F32, color = "GPU F32"), shape = 1) } +
    { if (exists("cp")) geom_line(aes(y = GPU_F64, color = "GPU F64")) } +
    { if (exists("cp")) geom_point(aes(y = GPU_F64, color = "GPU F64"), shape = 2) } +
    geom_line(aes(y = CPU_F32, color = "CPU F32")) +
    geom_point(aes(y = CPU_F32, color = "CPU F32"), shape = 15) +
    scale_y_log10() +
    labs(
      title = sprintf("%s on %s\nCPU: %s, GPU: %s (Time)", op_name, os_name, cpu_model, gpu_model),
      x = "Matrix size n×n",
      y = "Time (seconds, log-scale)"
    ) +
    theme_minimal() +
    theme(legend.title = element_blank())
  plots[[i]] <- p
}

# Save dense timing plots
png("plots/dense_matrix_timing_results.png", width = 1200, height = 800)
do.call(grid.arrange, c(plots, ncol = 3))
dev.off()

# Plot dense matrix accuracy results
plots <- list()
for (i in seq_along(operations)) {
  op_key <- operations[[i]][1]
  op_name <- operations[[i]][2]
  data <- data.frame(
    Size = matrix_sizes
  )
  if (exists("cp")) {
    data$GPU_F32 <- errors[[op_key]]$GPU_F32
    data$GPU_F64 <- errors[[op_key]]$GPU_F64
    data$CPU_F32 <- errors[[op_key]]$CPU_F32
  } else {
    data$CPU_F32 <- errors[[op_key]]$CPU_F32
  }
  p <- ggplot(data, aes(x = Size)) +
    { if (exists("cp")) geom_line(aes(y = GPU_F32, color = "GPU F32")) } +
    { if (exists("cp")) geom_point(aes(y = GPU_F32, color = "GPU F32"), shape = 1) } +
    { if (exists("cp")) geom_line(aes(y = GPU_F64, color = "GPU F64")) } +
    { if (exists("cp")) geom_point(aes(y = GPU_F64, color = "GPU F64"), shape = 2) } +
    geom_line(aes(y = CPU_F32, color = "CPU F32")) +
    geom_point(aes(y = CPU_F32, color = "CPU F32"), shape = 15) +
    scale_y_log10() +
    labs(
      title = sprintf("%s on %s\nCPU: %s, GPU: %s (Relative Error)", op_name, os_name, cpu_model, gpu_model),
      x = "Matrix size n×n",
      y = "Relative Error (log-scale)"
    ) +
    theme_minimal() +
    theme(legend.title = element_blank())
  plots[[i]] <- p
}

# Save dense accuracy plots
png("plots/dense_matrix_accuracy_results.png", width = 1200, height = 800)
do.call(grid.arrange, c(plots, ncol = 3))
dev.off()

# Plot sparse matrix timing results
sparse_operations <- operations[1:4]
plots <- list()
for (i in seq_along(sparse_operations)) {
  op_key <- sparse_operations[[i]][1]
  op_name <- sparse_operations[[i]][2]
  data <- data.frame(
    Size = matrix_sizes,
    CPU_F64_Ref = reference_times$sparse[[op_key]],
    CPU_F64 = sparse_timings[[op_key]]$CPU_F64
  )
  if (exists("cp")) {
    data$GPU_F32 <- sparse_timings[[op_key]]$GPU_F32
    data$GPU_F64 <- sparse_timings[[op_key]]$GPU_F64
    data$CPU_F32 <- sparse_timings[[op_key]]$CPU_F32
  } else {
    data$CPU_F32 <- sparse_timings[[op_key]]$CPU_F32
  }
  p <- ggplot(data, aes(x = Size)) +
    geom_line(aes(y = CPU_F64_Ref, color = "CPU F64 Reference"), linetype = "dashed") +
    geom_point(aes(y = CPU_F64_Ref, color = "CPU F64 Reference"), shape = 4) +
    { if (exists("cp")) geom_line(aes(y = GPU_F32, color = "GPU F32")) } +
    { if (exists("cp")) geom_point(aes(y = GPU_F32, color = "GPU F32"), shape = 1) } +
    { if (exists("cp")) geom_line(aes(y = GPU_F64, color = "GPU F64")) } +
    { if (exists("cp")) geom_point(aes(y = GPU_F64, color = "GPU F64"), shape = 2) } +
    geom_line(aes(y = CPU_F32, color = "CPU F32")) +
    geom_point(aes(y = CPU_F32, color = "CPU F32"), shape = 15) +
    geom_line(aes(y = CPU_F64, color = "CPU F64")) +
    geom_point(aes(y = CPU_F64, color = "CPU F64"), shape = 5) +
    scale_y_log10() +
    labs(
      title = sprintf("%s on %s\nCPU: %s, GPU: %s (Sparse, Time)", op_name, os_name, cpu_model, gpu_model),
      x = "Matrix size n×n",
      y = "Time (seconds, log-scale)"
    ) +
    theme_minimal() +
    theme(legend.title = element_blank())
  plots[[i]] <- p
}

# Save sparse timing plots
png("plots/sparse_matrix_timing_results.png", width = 1200, height = 600)
do.call(grid.arrange, c(plots, ncol = 2))
dev.off()

# Plot sparse matrix accuracy results
plots <- list()
for (i in seq_along(sparse_operations)) {
  op_key <- sparse_operations[[i]][1]
  op_name <- sparse_operations[[i]][2]
  data <- data.frame(
    Size = matrix_sizes,
    CPU_F64 = sparse_errors[[op_key]]$CPU_F64
  )
  if (exists("cp")) {
    data$GPU_F32 <- sparse_errors[[op_key]]$GPU_F32
    data$GPU_F64 <- sparse_errors[[op_key]]$GPU_F64
    data$CPU_F32 <- sparse_errors[[op_key]]$CPU_F32
  } else {
    data$CPU_F32 <- sparse_errors[[op_key]]$CPU_F32
  }
  p <- ggplot(data, aes(x = Size)) +
    { if (exists("cp")) geom_line(aes(y = GPU_F32, color = "GPU F32")) } +
    { if (exists("cp")) geom_point(aes(y = GPU_F32, color = "GPU F32"), shape = 1) } +
    { if (exists("cp")) geom_line(aes(y = GPU_F64, color = "GPU F64")) } +
    { if (exists("cp")) geom_point(aes(y = GPU_F64, color = "GPU F64"), shape = 2) } +
    geom_line(aes(y = CPU_F32, color = "CPU F32")) +
    geom_point(aes(y = CPU_F32, color = "CPU F32"), shape = 15) +
    geom_line(aes(y = CPU_F64, color = "CPU F64")) +
    geom_point(aes(y = CPU_F64, color = "CPU F64"), shape = 5) +
    scale_y_log10() +
    labs(
      title = sprintf("%s on %s\nCPU: %s, GPU: %s (Sparse, Relative Error)", op_name, os_name, cpu_model, gpu_model),
      x = "Matrix size n×n",
      y = "Relative Error (log-scale)"
    ) +
    theme_minimal() +
    theme(legend.title = element_blank())
  plots[[i]] <- p
}

# Save sparse accuracy plots
png("plots/sparse_matrix_accuracy_results.png", width = 1200, height = 600)
do.call(grid.arrange, c(plots, ncol = 2))
dev.off()

cat("\nPlots saved in 'plots' directory: dense_matrix_timing_results.png, dense_matrix_accuracy_results.png, sparse_matrix_timing_results.png, sparse_matrix_accuracy_results.png\n")
