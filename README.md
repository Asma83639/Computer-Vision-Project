# COE-49413 Computer Vision Semester Project

This repository contains the code and results for the COE-49413 Computer Vision semester project at the American University of Sharjah. The project explores various computer vision techniques across four distinct tasks, applying and extending state-of-the-art methods, solving a local problem, and implementing ensemble learning.


## Project Overview

This project covers the following four tasks as outlined in the course requirements:

1.  **Reproduce and Adapt:** Implementing the PaDiM anomaly detection framework \cite{padim} and adapting it to the AITEX fabric defect dataset \cite{aitex}.
2.  **Extend and Innovate:** Extending the PaDiM framework by replacing the backbone model with DenseNet-121 \cite{densenet} and incorporating Otsu's thresholding \cite{otsu}.
3.  **Solve a Local Problem:** Developing a solution for real-time traffic congestion monitoring in the UAE using YOLOv8 \cite{yolo} and ByteTrack \cite{bytetrack}.
4.  **Ensemble Learning:** Combining HOG+SVM, HOG+MLP, and a CNN model using majority voting for face mask classification \cite{facemask_dataset}.

## Project Structure

The repository is organized into folders corresponding to each task:

.├── Task 1 Reproduce and Adapt/│   └── PaDiM-Anomaly-Detection-Localization/  # Codebase for Task 1│       ├── datasets/AITEX/                    # Organized AITEX data│       ├── AITEX_result/                      # Results for AITEX dataset│       └── main.py                            # Main script for Task 1│       └── datasets/mvtec.py                  # Dataset loader (adapted for AITEX)│       └── ... (other necessary scripts/utils)├── Task 2 Extend and Innovate/│   └── PaDiM-Anomaly-Detection-Localization/  # Codebase for Task 2 (Modified PaDiM)│       ├── datasets/MvTec/                    # MVTec AD dataset used for evaluation│       ├── MvTec_result_Purposed_Model/       # Results for Task 2│       └── main.py                            # Main script with DenseNet/Otsu mods│       └── datasets/mvtec.py                  # Dataset loader│       └── ... (other necessary scripts/utils)├── Task 3/│   └── Traffic_Congestion_Monitoring.ipynb    # Jupyter Notebook for Task 3│   └── input.mp4                              # Example input video (if included)│   └── ByteTrack/                             # Cloned ByteTrack repository│   └── ... (potentially requirements.txt for Task 3)├── Task 4/│   ├── Face_Mask_Classification.ipynb         # Jupyter Notebook for Task 4│   ├── CW_Dataset/                            # Face Mask dataset source│   ├── Face_Mask_Models/                      # Saved models (SVM, MLP, CNN) and plots│   └── ... (potentially requirements.txt for Task 4)├── Project.pdf                                # Project description PDF└── README.md                                  # This file*(Note: Actual file names and exact structure might vary slightly based on implementation details.)*

## Installation / Dependencies

It is highly recommended to use separate virtual environments (e.g., Conda, venv) for each task or group of tasks due to potentially conflicting library versions.

### Task 1 & 2 (PaDiM - Anomaly Detection)

* **Python:** 3.7+ (as per original PaDiM repo, though newer versions might work)
* **PyTorch & Torchvision:** Version compatible with original PaDiM (e.g., `torch~=1.7`, `torchvision~=0.8`) or newer versions if adapted (e.g., `torch>=1.9`). Install specific versions if needed:
    ```bash
    # Example for specific older versions (adjust based on your environment/CUDA)
    # pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
    pip install torch torchvision torchaudio
    ```
* **Other Libraries:**
    ```bash
    pip install numpy pandas tqdm scikit-learn matplotlib Pillow scipy scikit-image
    ```

### Task 3 (Traffic Monitoring)

* **Python:** 3.8+ recommended
* **PyTorch:** Required by YOLOv8.
* **Ultralytics YOLOv8:**
    ```bash
    pip install ultralytics
    ```
* **ByteTrack:** Clone the repository and install requirements (see notebook cells for specific commands):
    ```bash
    git clone [https://github.com/ifzhang/ByteTrack.git](https://github.com/ifzhang/ByteTrack.git)
    cd ByteTrack
    # Modify requirements if needed (e.g., onnx version)
    pip install -r requirements.txt
    python setup.py develop
    pip install cython_bbox onemetric loguru lap thop
    cd ..
    ```
* **Supervision:**
    ```bash
    pip install supervision==0.1.0 # Or a newer compatible version
    ```
* **OpenCV:**
    ```bash
    pip install opencv-python
    ```

### Task 4 (Ensemble - Face Mask Classification)

* **Python:** 3.8+ recommended
* **TensorFlow/Keras:** Used for the CNN model.
    ```bash
    pip install tensorflow # Or tensorflow-gpu if CUDA is set up
    ```
* **Scikit-learn:** Used for SVM, MLP, HOG, metrics, etc.
    ```bash
    pip install scikit-learn scikit-image
    ```
* **Other Libraries:**
    ```bash
    pip install numpy pandas matplotlib seaborn joblib opencv-python
    ```

## Running the Code

### Task 1: Reproduce and Adapt (PaDiM on AITEX)

1.  **Navigate:** `cd "Task 1 Reproduce and Adapt/PaDiM-Anomaly-Detection-Localization"`
2.  **Dataset:** Ensure the AITEX dataset is downloaded and organized under `datasets/AITEX/` (structure: `train/good`, `test/good`, `test/bad`, `ground_truth/bad`). Verify the dataset loader (`datasets/mvtec.py`) is correctly adapted to read this structure and apply appropriate preprocessing (e.g., tiling/resizing).
3.  **Run:** Execute the main script, specifying the AITEX data path and desired backbone (original PaDiM used `wide_resnet50_2` or `resnet18`).
    ```bash
    python main.py --data_path ./datasets/AITEX --save_path ./AITEX_result --arch wide_resnet50_2
    ```
4.  **Output:** Results (ROCAUC scores, visualizations) are saved in `./AITEX_result`.

### Task 2: Extend and Innovate (PaDiM Extension)

1.  **Navigate:** `cd "Task 2 Extend and Innovate/PaDiM-Anomaly-Detection-Localization"`
2.  **Dataset:** Ensure the MVTec AD dataset is available at the path specified by `--data_path` (default `./datasets/MvTec`).
3.  **Run:** Execute the modified `main.py` script, specifying the DenseNet backbone.
    ```bash
    python main.py --data_path ./datasets/MvTec --save_path ./MvTec_result_Purposed_Model --arch densenet121
    ```
4.  **Output:** Results (ROCAUC scores, visualizations using Otsu threshold) are saved in `./MvTec_result_Purposed_Model`.

### Task 3: Solve a Local Problem (Traffic Monitoring)

1.  **Navigate:** `cd "Task 3"`
2.  **Setup:** Open `Traffic_Congestion_Monitoring.ipynb` in Google Colab (GPU recommended) or a local environment with required dependencies installed.
3.  **Dependencies:** Run installation cells within the notebook. Restart runtime if prompted.
4.  **Input Video:** Place the input video file (e.g., `input.mp4`) in the expected location or update `SOURCE_VIDEO_PATH`.
5.  **Run:** Execute notebook cells sequentially.
6.  **Output:** Annotated frames or video showing detected and tracked vehicles.

### Task 4: Ensemble Learning (Face Mask Classification)

1.  **Navigate:** `cd "Task 4"`
2.  **Setup:** Open `Face_Mask_Classification.ipynb` (or the relevant notebook) in Google Colab (GPU recommended for CNN training) or a local environment.
3.  **Dataset:** Ensure the Face Mask dataset (`CW_Dataset`) is accessible via the paths defined in the notebook (`TRAIN_IMG_PATH`, `TEST_IMG_PATH`, etc.).
4.  **Run Training & Individual Evaluation:** Execute the notebook cells up to and including the `main()` function call (e.g., Cell [65]). This trains HOG+SVM, HOG+MLP, CNN, evaluates them individually, and saves the models to `Face_Mask_Models`.
5.  **Run Ensemble Evaluation:** Execute the subsequent cell(s) containing the ensemble logic (loading models, combining predictions via majority vote, evaluating ensemble accuracy, comparing results).
6.  **Output:** Accuracy metrics, classification reports, confusion matrices, and comparison plots saved in `Face_Mask_Models`. Visualizations of predictions on sample images are also generated.

## Results

* **Task 1 & 2:** Numerical results (Image/Pixel ROCAUC) are printed to the console during execution and stored in the respective `*_result*/` directories. Anomaly map visualizations are saved in `pictures_*` subdirectories.
* **Task 3:** Output is primarily visual (annotated video frames/output video). Quantitative analysis (e.g., vehicle counts) depends on specific implementation details within the notebook.
* **Task 4:** Performance metrics (Accuracy, Precision, Recall, F1-Score) for individual models and the ensemble are printed. Confusion matrices and comparison plots are saved to `Face_Mask_Models`.

## References

[Provide numbered references corresponding to citations in the text, e.g.:]
[1] T. Defard, A. Setkov, A. Loesch, and R. Audigier, "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization," [*arXiv preprint arXiv:2011.08785*, 2020](https://arxiv.org/pdf/2011.08785).
[2] J. Silvestre-Blanes, et al., "AFID: a public fabric image database for defect detection," [*AUTEX Research Journal*](https://www.researchgate.net/publication/334068330_A_Public_Fabric_Database_for_Defect_Detection_Methods_and_Results), 2019.
[3] G. Huang, et al., "Densely connected convolutional networks," *CVPR*, 2017.
[4] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, 1979.
[5] Ultralytics YOLOv8. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
[6] Y. Zhang, et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," *ECCV*, 2022.

