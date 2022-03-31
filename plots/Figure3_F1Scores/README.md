**Brief Summary of Data Visualization Pipeline**
1. **subgroup_cell_classes:** _Takes the summary of predicted classes for all trained models (all_y_pred) and returns grouped classes for the lymphocyte lineages (grouped_y_pred)._

2. **f1_by_class.py:** _Calculates f-1 scores for each class and model in both all_y_pred and grouped_y_pred, returning them as f1.csv and f1.group.csv, respectively._

3. **plot_f1_heatmaps:** _Takes f1.csv and f1.group.csv and plots them as a heatmap using the ggplot2 package._
