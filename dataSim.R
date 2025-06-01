# Load Splatter (if not already loaded)
library(splatter)

# Create parameters with batchCells (from your previous code)
params <- newSplatParams(
  nGenes = 50000,           # Number of genes
  batchCells = 10000,        # Number of cells in the batch (single batch here)
  group.prob = c(0.6, 0.4), # Probabilities for two groups
  de.prob = 0.2            # Probability of differential expression
)

# Simulate data with two groups
set.seed(123)
sim <- splatSimulateGroups(params, verbose = TRUE)

# Verify the number of cells
ncol(sim)  # Should return 200 (total cells)

# Extract count matrix
count_matrix <- as.matrix(counts(sim))

# Save to CSV at C:/
write.csv(count_matrix, "count_matrix.csv", row.names = TRUE)

# Show the current working directory
getwd()
