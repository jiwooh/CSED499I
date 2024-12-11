# CSED499I
* 24' Fall Research Project I @ POSETCH MLLab
* Jiwoo Hong, POSTECH CSE 22
* Research Advisor: Prof. [Dongwoo Kim](https://dongwookim-ml.github.io/)

## Overview
(To be written)

## Instructions
### [Optional] graphcast
* **You can skip this entire step if you want to run this repository right away.**
* Use graphcast_prediction_saver.ipynb (recommended to run on [Google Colab](https://colab.research.google.com/github/jiwooh/CSED499I/blob/master/graphcast_prediction_saver.ipynb))
  * On cell `Choose the model`, select `Mesh size: 4, 5, 6` for each run, keeping other values as default
  * On cell `Get and filter the list of available example datasets`, select `source: era5, date: 2022-01-01, res: 0.25, levels: 13, steps: 01` (default choice)
  * On cell `Choose training and eval data to extract`, continue with the default choice
  * Keep running the cells until the end of the notebook, it might take some time
* Download exported .nc datasets on Colab to ./ directory

### GINR
* [Optional] Open `./npy_processor.ipynb` to make dataset files (.npy/.npz) and place them as below
  ```plaintext
  ./GINR/dataset/
    ├── gcm4/
    │   ├── fourier_m4.npy, points_m4.npy
    ├── gcm4to5/
    │   ├── fourier_m4.npy, points_m4.npy
    │   └── npz_files
    │       └── data_m5.npz
    ├── gcm4to6/
    │   ├── fourier_m4.npy, points_m4.npy
    │   └── npz_files
    │       └── data_m6.npz
    ├── gcm5/
    │   └── fourier_m5.npy, points_m5.npy
    ├── gcm5to6/
    │   ├── fourier_m5.npy, points_m5.npy
    │   └── npz_files
    │       └── data_m6.npz
    ├── gcm6/
    │   └── fourier_m6.npy, points_m6.npy
    └── targets/
        └── data_m5.npz, data_m6.npz
    ```
  * rename all `_m4`, `_m5`, `_m6` to make files `fourier.npy`, `points.npy`, `data.npz`, except for `targets/` directory
  * **These are already provided in the repository.**
* Install dependencies (**Linux is required**, virtual environment is recommended)
  ```bash
  cd ./GINR
  # Create your virtual environment here
  pip install -r requirements.txt
  ```
* Build and install PyMesh (Again, Linux is required, virtual environment is recommended)
  ```bash
  git clone https://github.com/PyMesh/PyMesh.git
  cd PyMesh
  git submodule update --init
  sudo apt-get install libmpfr-dev libgmp-dev libboost-all-dev
  echo "SET(CMAKE_C_FLAGS " -fcommon ${CMAKE_C_FLAGS}")" >> ./GINR/PyMesh/third_party/mmg/CMakeLists.txt # Handle CMake error
  ./setup.py build
  ./setup.py install
  ```
* Train models and evaluate them with commands inside `./GINR/run.sh`
  * Check `./GINR/run.sh` for more information
  * Do not run it on the shell directly
* Evaluate and generate MSE comparison plots by executing `./GINR/eval_mse.py`

## Based Researches
[![1](https://img.shields.io/static/v1?label=google-deepmind&message=graphcast&color=181717)](https://github.com/google-deepmind/graphcast)
[![2](https://img.shields.io/static/v1?label=danielegrattarola&message=GINR&color=181717)](https://github.com/danielegrattarola/GINR)
[![3](https://img.shields.io/static/v1?label=PyMesh&message=PyMesh&color=181717)](https://github.com/PyMesh/PyMesh)

* Graphcast: [Learning skillful medium-range global weather forecasting](https://www.science.org/doi/10.1126/science.adi2336)
* GINR: [Generalised Implicit Neural Representations](https://arxiv.org/abs/2205.15674)
