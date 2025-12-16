# Load necessary libraries
#install.packages(c("tidyverse", "readr", "ggplot2")) # Uncomment and run if you need to install packages
library(tidyverse)
library(readr)
library(ggplot2)
library(rstudioapi)


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

# choose trials to compare
trials <- data.frame(
  PATH = c(
    "../experiment_results/forest/z_score/2025-12-16-00-17-06",
    "../experiment_results/forest/percentile/2025-12-16-00-30-28",
    "../experiment_results/forest/none/2025-12-16-00-44-22",
    "../experiment_results/forest/histogram_matching/2025-12-16-00-37-29",
    "../experiment_results/forest/min_max/2025-12-16-00-23-44"
  ),
  
  METHOD = c(
    "z_score",
    "percentile",
    "None",
    "Histogram_Matching",
    "min_max"
    ),
  stringsAsFactors = FALSE
)


################################################################



# -------------------------------
# Combine all trials into one data frame
# -------------------------------
all_data <- data.frame()

for (i in 1:nrow(trials)) {
  trial <- trials$METHOD[i]
  csv_path <- file.path(script_dir, trials$PATH[i], file_name)
  
  # Load CSV
  df <- read.csv(csv_path, sep = ";")
  
  # Keep only labels of interest
  df <- df %>% filter(LABEL %in% c("Amygdala", "GreyMatter", "Hippocampus", "Thalamus", "WhiteMatter"))
  
  # Add method as a column
  df$METHOD <- trial
  
  # Append to combined data frame
  all_data <- rbind(all_data, df)
}

# Ensure LABEL and METHOD are factors for proper ordering
all_data$LABEL <- factor(all_data$LABEL, levels = c("Amygdala", "GreyMatter", "Hippocampus", "Thalamus", "WhiteMatter"))
all_data$METHOD <- factor(all_data$METHOD, levels = trials$METHOD)

# -------------------------------
# Boxplot: DICE (all methods)
# -------------------------------
plot_dice <- ggplot(all_data, aes(x = LABEL, y = DICE, fill = METHOD)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7) +
  scale_fill_viridis_d(option = "plasma") +
  scale_color_viridis_d(option = "plasma") +
  labs(
    title = "DICE Score Comparison Across Normalization Methods",
    x = "Anatomical Label",
    y = "DICE Score"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "right",
    plot.background = element_rect(fill = "white", colour = "white")
  ) +
  coord_cartesian(ylim = c(0.3,1))

print(plot_dice)

# Save DICE plot
ggsave(file.path(script_dir, "boxplot_dice_all_methods.png"),
       plot = plot_dice, width = 12, height = 6, units = "in", dpi = 300)
cat("Saved 'boxplot_dice_all_methods.png'\n")

# -------------------------------
# Boxplot: HDRFDST (all methods)
# -------------------------------
plot_hdrfdst <- ggplot(all_data, aes(x = LABEL, y = HDRFDST, fill = METHOD)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7, outlier.colour = "red", outlier.shape = 8) +
  scale_fill_viridis_d(option = "plasma") +
  scale_color_viridis_d(option = "plasma") +
  labs(
    title = "HDRFDST Score Comparison Across Normalization Methods",
    x = "Anatomical Label",
    y = "HDRFDST"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "right",
    plot.background = element_rect(fill = "white", colour = "white")
  )

print(plot_hdrfdst)

# Save HDRFDST plot
ggsave(file.path(script_dir, "boxplot_hdrfdst_all_methods.png"),
       plot = plot_hdrfdst, width = 12, height = 6, units = "in", dpi = 300)
cat("Saved 'boxplot_hdrfdst_all_methods.png'\n")

