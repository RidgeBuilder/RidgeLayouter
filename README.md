# RidgeLayouter
The code repository documents the layout algorithms and dataset used in the paper "RidgeBuilder: Interactive Authoring of Expressive Ridgeline Plots."

## File Description
### Dataset
The dataset is located in the /src/dataset folder, consisting of a total of 20 data files, each corresponding to a ridge plot that is to be laid out. The filenames are in the format N_M_C.csv, where N indicates that there are N ridges recorded in the data file, and M signifies that each ridge has 10N data points. Each pair (N,M) corresponds to two ridge plots, with C serving as a distinguishing marker for the same data scale. The CSV files contain N rows and 10M columns. 

### Code
The /src folder contains the layout algorithms used in RidgeBuilder, including optimization algorithms, statistical sorting (mean, maximum), and peak-based sorting.

### Test & Visualize
The /results_vis folder contains the visualization effect images that should be obtained if the algorithm is correctly executed. The code includes a visualize function.

## Getting Started
1. Configure the environment and install necessary packages. The python version must fulfill python >= 3.10.
```
conda create -n my_env python==3.10
conda activate my_env
pip install -r requirements.txt
```
2. Run the code
```
cd src
python main.py
```
3. If the code execute correctly, then an image will be generated in the /src/output folder. The image will be the same as the one in /result_vis .

