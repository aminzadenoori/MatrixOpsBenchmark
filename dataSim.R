# Load libraries
library(splatter)
library(scater)
library(scran)
library(ggplot2)

# Create parameters
params <- newSplatParams(
  nGenes = 1000,
  batchCells = 200,
  group.prob = c(0.6, 0.2 ,0.2),
  de.prob = 0.2
)

# Simulate data
set.seed(123)
sim <- splatSimulateGroups(params, verbose = TRUE)

# Normalize data
sim <- logNormCounts(sim)

# Run dimensionality reduction
sim <- runPCA(sim, ncomponents = 2)
sim <- runTSNE(sim, dimred = "PCA")
sim <- runUMAP(sim, dimred = "PCA")

# Plot visualizations
# PCA
pca_plot <- plotPCA(sim, colour_by = "Group", shape_by = "Group")
ggsave("pca_plot.png", pca_plot, width = 6, height = 4)

# UMAP
umap_plot <- plotUMAP(sim, colour_by = "Group", shape_by = "Group")
ggsave("umap_plot.png", umap_plot, width = 6, height = 4)

# Heatmap for top DE genes
markers <- findMarkers(sim, groups = colData(sim)$Group, test = "wilcox")
top_markers <- rownames(markers$Group1)[1:10]
plotHeatmap(sim, features = top_markers, columns = order(colData(sim)$Group), 
            colour_columns_by = "Group", center = TRUE, scale = TRUE)

# Gene expression for first gene
plotExpression(sim, features = rownames(sim)[1], x = "Group", colour_by = "Group")

# QC metrics
sim <- addPerCellQC(sim)
plotColData(sim, x = "Group", y = "sum", colour_by = "Group")
