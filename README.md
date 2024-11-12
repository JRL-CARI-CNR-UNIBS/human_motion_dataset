# Human Motion Dataset

This repository contains scripts and data for building and processing a human motion dataset. The dataset includes keypoint positions, joint angles, and body parameters data. The pipeline processes the data through several stages, including conversion, inverse kinematics, merging, labelling, and preprocessing.


## Repository Structure

- `scripts/`: Contains all the Python scripts used in the data processing pipeline.
  - `pkl_to_csv.py`: Converts pickle files containing raw keypoint data to CSV format.
  - `perform_inverse_kinematics.py`: Performs inverse kinematics on the raw keypoint data.
  - `merge_kpt_ik.py`: Merges the raw keypoint data with the inverse kinematics data.
  - `create_labelled_dataset.py`: Creates a labelled dataset from the merged data using trigger data to split the data into labelled segments.
  - `preprocess_dataset.py`: Resamples, filters, and differentiates the labelled dataset.
- `data/`: Contains the raw and processed data files.
- `run_pipeline.bash`: Bash script to run the entire data processing pipeline.


## Prerequisites

Ensure you have the following installed:

- Python 3
- Required Python packages (listed in `requirements.txt`)
- Human Kinematic Model

You can install the **required Python packages** using pip:

```bash
pip install -r requirements.txt
```

You can install the **human kinematic model** from source:
1. Clone the repository:
```
git clone https://github.com/JRL-CARI-CNR-UNIBS/human_kinematic_model.git -b f_michele_devel
```
2. Follow the instructions in that repository to build the package.


## Downloading the Data

Before running the pipeline, you need to download the `data` folder from the following link:

[Download Data Folder](https://drive.google.com/drive/folders/1ytaC6sb4ZdSuqsTpfRiYRbUrc54kMPvQ?usp=drive_link)

Place the downloaded `data` folder in the root of the repository.


## Running the Pipeline

To run and build the dataset, execute the `run_pipeline.bash` script:

```bash
./run_pipeline.bash
```

This script will sequentially execute the following steps:

1. Convert pickle files to CSV format.
2. Perform inverse kinematics on the raw keypoint data.
3. Merge the raw keypoint data with the inverse kinematics data.
4. Create a labelled dataset from the merged data.
5. Resample, filter, and differentiate the labelled dataset.