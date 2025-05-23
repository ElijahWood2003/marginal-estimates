# Marginal Distribution Estimates of Markov Random Fields

**Researcher**: Elijah Wood    
**Institution**: University of North Carolina at Chapel Hill  
**Date**: April 2025  
**Advisor**: P.S. Thiagarajan  

![MRF Visualization Example](images/mrf_visualization.png)  
**
<!-- TODO: add visualization comparing old methods to new method for sampling-->

## Project Overview

This repository contains Python implementations for:
- **Markov Random Fields (MRF)**: Gibbs sampling, marginal distribution estimation, CPT auto-propagation
- **Live-And-Safe Marked Graph (LAS)**: MRF to LAS conversion, auto-tokenizer
- **Factored MDPs (FMDP)**: MRF + LAS to FMDP conversion, greedy strategy for marginal distribution estimates

Developed as part of undergraduate research at UNC Chapel Hill, this work focuses on developing computationally efficient methods for estimating marginal distributions in MRFs, with applications in computer vision, spatial statistics, and probabilistic graphical models.

## Key Features

### MRF Simulation
- Includes hashing for efficient conditional probability tables (CPTs)
- Auto propagate CPT over all domains with random values
- Gibbs sampling with configurable burn-in and thinning

### LAS Marked Graph
- Live-and-Safe Marked Graphs for conversion to FMDP
- Easy conversion from MRF to LAS
- Automatically sets tokens based on acyclic orientation

### Factored MDP Extension
- Factored MDP simulation for stationary distributions
- Easy conversion from MRF -> LAS -> FMDP
- Uses properties inherent in creation to estimate MRF distribution