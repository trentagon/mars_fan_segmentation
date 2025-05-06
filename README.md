# Mars Fan Segmentation

This repository contains instructions and code for analyzing images of alluvial fans and deltas on Mars's surface, with a primary focus on the Eberswalde crater. The project involves training and applying a self-supervised deep learning algorithm to segment images of these fans.

## Directory Structure

- `data/`: contains all of the data for this project, classified into `raw`, `clean`, and `ai_ready`. The `label` folder includes the geologic map used as validation. Note that the data folders are empty to start because the data is not hosted on github (but the label is).
- `notebooks_DataAnalysis/`: contains all of the jupyter notebooks needed to download, process, and analyze the data for this project.

  - `Download_Data.ipynb`: downloads the raw, clean, or AI ready data.
  - `Data_Cleaning.ipynb`: data pipeline for going from raw to clean data.
  - `Prepare_AI_Ready_Data.ipynb`: data pipeline for going from clean to AI ready data.
  - `EDA.ipynb`: exploratory data analysis.
  - `Dimensionality_Reduction.ipynb`: a sample analysis to demonstrate dimensionality reduction and other simple ML analysis.

- `notebooks_ClassicML/`: contains all of the jupyter notebooks for classic machine learning analysis of the data.

  - `AutoML_Hyperparameter_Tuning.ipynb`: using PyCaret to automatically optimize our machine learning approach.
  - `Clustering_Analysis.ipynb`: an analysis with several simple clustering algorithms and deep dive into K Means Clustering.
  - `Computational_Time_Analysis.ipynb`: an analysis of clustering and training time for a variety of different approaches.
  - `Model_Training_Assessment.ipynb`: a demonstration of reinforcement learning with best practices (although our data is not suited for this approach).

- `notebooks_DeepLearning/`: contains jupyter notebooks for deep learning analysis of the data.

  - `AnalyzeResults_CTX_masked_only.ipynb`: analysis of the deep learning reesults from training on masked CTX images.
  - `AnalyzeResults_CTX_only.ipynb`: analysis of the deep learning reesults from training on unmasked CTX images.
  - `Download_Results.ipynb`: downloads the results of the deep learning training and application.
  - `KMeansCluster_Baseline.ipynb`: K means clustering analysis for comparison to the deep learning analysis (may be empty, see Veronica's fork of this repository if so).
  - `Train_Saha_CTX.ipynb`: This is a sample of how to train the Saha et al. (2022) algorithm on our data.

- `notebooks_Research/`: contains jupyter notebooks related to ongoing development of the project.

## Getting Started

The code in this repository is written in the Python programming language. To get started, follow these steps:

1. **Download and Prepare Data**:

   - Go to the `Download_Data.ipynb` notebook for instructions on downloading the data and a description of data sources.
   - There are 3 levels of data available for download, or you can start from the raw data and reproduce our entire analysis.

2. **Explore the AI Ready Data**:

   - Ensure you have the AI ready data, either by downloading directly or following our data pipeline.
   - Run the `EDA.ipynb` notebook to view the training images and see a simple statistical analysis.

## Where does the data come from?

This is a custom dataset that was curated from a number of different sources, listed below.

### 1. HRSC + MOLA blended DEM (200 meters per pixel)

Files: `blendDEM.tif`, `blendSLOPE.tif`

Link: https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m

This is a global data set that blends observations from the Mars Orbiter Laser Altimeter (MOLA), an instrument aboard NASA's Mars Global Surveyor spacecraft (MGS), and the High-Resolution Stereo Camera (HRSC), an instrument aboard the European Space Agency's Mars Express (MEX) spacecraft. MGS launched in 1996 and MEX launched in 2003. The raw dataset is a digital elevation model, and we have calculated a slope map.

### 2. CTX grayscale image (5 meters per pixel) and DEM (6 meters per pixel)

Files: `ctxIMG.tif`, `ctxDEM.tif`, `ctxSLOPE.tif`

Link: https://murray-lab.caltech.edu/CTX/

Link: https://github.com/GALE-Lab/Mars_DEMs

The Bruce Murray Laboratory for Planetary Visualization created a global mosaic of Mars using grayscale images from the Context Camera (CTX) onboard the Mars Reconnaissance Orbiter (MRO) acquired between 2006 and 2020. A DEM was created from these grayscale images (see link) and then we calculated a slope map.

### 3. THEMIS infrared images (100 meters per pixel)

Files: `dayIR.tif`, `nightIR.tif`

Link: https://www.mars.asu.edu/data/thm_dir/

Link: https://www.mars.asu.edu/data/thm_nir_100m/

These mosaics represent the Thermal Emission Imaging System (THEMIS) daytime and nighttime infrared (IR) 100 meter/pixel images released in the summer of 2014 by Arizona State University.

### 4. HRSC panchromatic images (10-20 meters per pixel)

Files: `hrscND.tif`, `hrscP1.tif`, `hrscP2.tif`, `hrscS1.tif`, `hrscS2.tif`

Link: https://ode.rsl.wustl.edu/mars/index.aspx

Mars Express' High Resolution Stero Camera (HRSC) has five panchromatic detectors that each look down from the spacecraft at a different angle: there is one nadir (straight-down) channel (ND), two "stereo" channels S1 and S2 that are each angled forward and backward along the spacecraft's path at 18.9°, and two "photometric" channels P1 and P2 that are at slightly smaller angles. These images were accessed through the Mars orbital data explorer under Mars Express HRSC calibrated data PDS3 Version 4 Reduced Data Record.

## How to install and use this repository

Clone this repository from github:

`git clone https://github.com/UW-MLGEO/MLGEO2024_MarsFans.git`

In order to run the Jupyter notebooks, we need a python installation with the following libraries:

- Python 3.8+
- Jupyter Notebook
- TensorFlow
- rioxarray
- scikit-image
- matplotlib
- geopandas
- pyproj
- dask

If you already have a python environment with those libraries, you're good to go. If you don't, use conda or pip to install them in your Python environment.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
