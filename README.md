# Survival Deep Learning Models
This project explores the application of deep learning methods to survival analysis, with a specific focus on modeling HIV incidence using high-dimensional longitudinal cytokine biomarker data. 
The study employs three deep learning models—DeepSurv, DeepHit, and Dynamic DeepHit—to predict time-to-event outcomes and investigate the importance of incorporating time-varying covariates.
We further explore handling missing values with missForest.
## Project Objectives
1. Evaluate the performance of deep learning survival models on complete-case and imputed datasets.
2. Investigate different strategies for incorporating longitudinal cytokine profiles into predictive models (Such as using the summary statistics of the cytokine data).
3. Demonstrate the utility of using Dynamic DeepHit for handling time-varying covariates in high-dimensional survival data.
## Dataset
The data (CAP004) used for this analysis was obtained from CAPRISA (https://www.caprisa.org/Pages/CAPRISAStudies).
## Paper
Title: Deep learning models for the analysis of high-dimensional survival data with time-varying covariates while handling missing data (https://doi.org/10.1007/s44163-025-00429-z).
## Authors
Sarah Ogutu, Mohanad Mohammed, Henry Mwambi
## Supplementary materials
These Github repositories DeepSurv (https://github.com/jaredleekatzman/DeepSurv), DeepHit (https://nbviewer.org/github/havakv/pycox/tree/master/examples/), and Dynamic DeepHit (https://github.com/chl8856/Dynamic-DeepHit/tree/master) were instrumental in creating the Python script codes provided here.
## References
Ogutu, S., Mohammed, M. & Mwambi, H. Deep learning models for the analysis of high-dimensional survival data with time-varying covariates while handling missing data. Discov Artif Intell 5, 176 (2025)
## Code Description
The Python scripts are provided, and they give a point-by-point step of how the analysis was achieved. 
