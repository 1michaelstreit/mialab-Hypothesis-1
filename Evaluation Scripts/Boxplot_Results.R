# Load necessary libraries
# install.packages(c("tidyverse", "readr", "ggplot2")) # Uncomment and run if you need to install packages
#library(tidyverse)
#library(readr)
#library(ggplot2)

# Load the Data ---
rm(list = ls())    # Remove all objects
graphics.off()     # Close all plots
gc()               # Free memory

# Determine script directory
if (interactive()) {
  # Inside RStudio
  script_dir <- dirname(getActiveDocumentContext()$path)
} else {
  # When running with Rscript
  script_dir <- dirname(normalizePath(commandArgs(trailingOnly = FALSE)[[1]]))
}

# select file
################################################################

file_name = "results.csv"
trial = "../experiment_results/forest/min_max/2025-12-05-14-46-37"

csv_path <- file.path(script_dir, trial, file_name)
################################################################

# Load CSV
df <- read.csv(csv_path, sep = ";")

# Boxplot for DICE Score by Label

# Ensure LABEL is treated as a factor for correct ordering on the x-axis
df$LABEL <- factor(df$LABEL)

plot_dice <- ggplot(df, aes(x = LABEL, y = DICE)) +
  geom_boxplot(aes(fill = LABEL), alpha = 0.7) +
  # Add points on top of the boxplot to show individual data points
  geom_jitter(color = "black", size = 0.8, alpha = 0.6, width = 0.2) +
  
  # Customize labels and title
  labs(
    title = "Distribution of DICE Score by Anatomical Label",
    x = "Anatomical Label",
    y = "DICE Score (Similarity)",
  ) +
  scale_fill_viridis_d(option = "plasma") +
  theme_minimal() +
  
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "none", # Remove the redundant legend
    plot.background = element_rect(fill = "white", colour = "white")
  ) +
  
  ylim(0, 1)

print(plot_dice)
# Save the plot to a file
ggsave(file.path(script_dir,trial,"boxplot_dice_by_label.png"), plot = plot_dice, width = 10, height = 6, units = "in", dpi = 300)

cat("Saved 'boxplot_dice_by_label.png'\n")




# Boxplot for HDRFDST Score by Label

plot_hdrfdst <- ggplot(df, aes(x = LABEL, y = HDRFDST)) +
  geom_boxplot(aes(fill = LABEL), alpha = 0.7, outlier.colour = "red", outlier.shape = 8) +
  # Add points on top (optional, but good for visualizing raw data density)
  geom_jitter(color = "darkgrey", size = 0.8, alpha = 0.6, width = 0.2) +
  
  # Customize labels and title
  labs(
    title = "Distribution of HDRFDST Score by Anatomical Label",
    x = "Anatomical Label",
    y = "HDRFDST (Boundary Error)",
  ) +
  scale_fill_viridis_d(option = "plasma") +
  theme_bw() +
  
  theme(
  plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
  axis.title = element_text(size = 12),
  axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
  legend.position = "none",
  plot.background = element_rect(fill = "white", colour = "white")
  )

print(plot_hdrfdst)
# Save the plot to a file
ggsave(file.path(script_dir,trial,"boxplot_hdrfdst_by_label.png"), plot = plot_hdrfdst, width = 10, height = 6, units = "in", dpi = 300)

cat("Saved 'boxplot_hdrfdst_by_label.png'\n")