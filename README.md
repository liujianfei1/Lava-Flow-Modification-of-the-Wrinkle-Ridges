# Lava-Flow-Modification-of-the-Wrinkle-Ridges
A collection of Python scripts for plotting wrinkle ridge topography and calculating geophysical parameters such as thickness, two-way travel time, loss tangent, and yield stress.
The scripts are used for generating cross-sections and various plots related to subsurface and surface structures.

## Scripts Description

- **Thcikness.py**  
  Calculates the relationship between fill thickness and two-way travel time, and generates **Fig. 4** in the article.

- **Loss_tangent.py**  
  Plots the loss tangent of four radar tracks, corresponding to **Fig. 5**.

- **WH_distance.py**  
  Plots 18 cross-sections of wrinkle ridges and illustrates how parameters (width, height, and crater distance) vary with distance, producing **Figs. 6b–d**.

- **Radar_elevation.py**  
  Generates elevation profiles for four radar tracks, corresponding to **Fig. 7b**.

- **Cross_Elevation.py**  
  Produces eight cross-sectional profiles of wrinkle ridges after Kriging interpolation, corresponding to **Fig. 7c**.

- **Relocate.py**  
  Relocate pick based on energy, the codes "caltime.py, readroi.py, re_dupli.py, findlatlon.py" are related functions. Relocate.py relocates the surface pick at the maximum value within ±10 pixels, and the subsurface interface pick at the maximum value within ±7 pixels.
