#!/bin/bash

# Convert pickle files containing raw keypoint data to CSV format
echo "Converting pickle files to CSV..."
python3 pkl_to_csv.py
echo "Conversion complete."

# Perform inverse kinematics on the raw keypoint data
echo -e "\nPerforming inverse kinematics on raw keypoint data..."
python3 perform_inverse_kinematics.py
echo "Inverse kinematics complete."

# Merge the raw keypoint data with the inverse kinematics data
echo -e "\nMerging raw keypoint data with inverse kinematics data..."
python3 merge_kpt_ik.py
echo "Merge complete."

# Create a dataset from the merged data using the trigger data to split the data into labelled segments
echo -e "\nCreating labelled dataset from merged data..."
python3 create_labelled_dataset.py
echo "Labelled dataset creation complete."

# Resample, filter, and differentiate the labelled dataset
echo -e "\nResampling, filtering, and differentiating the labelled dataset..."
python3 preprocess_dataset.py
echo "Preprocessing complete."