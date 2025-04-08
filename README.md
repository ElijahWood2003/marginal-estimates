# Efficient Marginal Distribution Estimation in Markov Random Fields

**Researcher**: [Elijah Wood]  
**Institution**: University of North Carolina at Chapel Hill  
**Date**: [April 2025]  
**Advisor**: [P.S. Thiagarajan]  

![MRF Visualization Example](images/mrf_visualization.png)  
*Example output from the MRF simulator showing node states evolving over time*

## Project Overview

This repository contains Python implementations for:
- **Markov Random Fields (MRF)**: Gibbs sampling, marginal distribution estimation and example MRF
- **LAS (Live-And-Safe Marked Graph)**: 
- **Factored MDPs (FMDP)**: 

Developed as part of undergraduate research at UNC Chapel Hill, this work focuses on developing computationally efficient methods for estimating marginal distributions in MRFs, with applications in computer vision, spatial statistics, and probabilistic graphical models.

## Key Features

### MRF Implementation
- Gibbs sampling with configurable burn-in and thinning
- Neighbor-aware conditional probability tables (CPTs)
- Interactive Tkinter visualization
- Console-based ASCII visualization
- Automatic CPT propagation

### LAS Algorithm
- Location-aware sampling for variance reduction
- Adaptive neighborhood weighting
- Convergence diagnostics

### FMDP Extension
- Factored state representation
- Approximate dynamic programming
- Integration with MRF inference