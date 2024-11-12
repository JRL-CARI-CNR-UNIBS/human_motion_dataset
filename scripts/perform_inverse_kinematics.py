import os
import glob
import pandas as pd
import numpy as np
from human_model_binding import Human28DOF, JointLimits, Keypoints
from load_params import load_parameters

PARAM_DIR = {'FOLDER': 'config', 'FILE': 'params.yaml'}
INPUT_DIR = {'FOLDER': 'data', 'SUBFOLDER': '2_csv_raw'}
OUTPUT_DIR = {'FOLDER': 'data', 'SUBFOLDER': '3_ik_results'}


def process_csv_file(csv_file_path, output_file_path, column_names_input,
                     qbounds, kpt_names, joint_names, param_names, n_dof, n_param):
    # Read the CSV file
    df = pd.read_csv(csv_file_path,
                     usecols=column_names_input)

    # Create human kinematic model and initialize configuration and parameters
    human_model = Human28DOF()
    q = np.zeros(n_dof)
    param = np.zeros(n_param)
    keypoints = Keypoints()    

    # Process the keypoints using the human_kinematic_model library
    processed_df = pd.DataFrame(columns=joint_names+param_names)
    for row in df.values:
        # Extract the keypoint number from the row
        row_kpt = {}
        for i in range(0, len(row), 3):
            kpt_num = int(df.columns[i].split('kp')[1].split('_')[0])
            row_kpt.update({kpt_names[kpt_num]: row[i:i+3]})

        # Extract a dict of keypoints from the row
        keypoints.set_keypoints(row_kpt)

        # Perform inverse kinematics
        try:
            q, param = human_model.inverse_kinematics(keypoints, qbounds, q, param)
        except Exception as e:
            q = np.array([np.nan]*n_dof)
            param = np.array([np.nan]*n_param)

            for value in row_kpt.values():
                if (np.isnan(value).any()):
                    continue

        ik_row = pd.DataFrame([np.concatenate((q, param))], columns=joint_names+param_names)
        processed_df = pd.concat([processed_df, ik_row], ignore_index=True)
        
    # Save the processed data to a new CSV file
    processed_df.to_csv(output_file_path, index=False)


def main():
    # Determine the directory of the script being executed
    pkg_dir = os.path.dirname(os.path.abspath(__file__)).split('scripts')[0]

    # Define path to folders
    input_folder = os.path.join(pkg_dir, INPUT_DIR['FOLDER'], INPUT_DIR['SUBFOLDER'])
    output_folder = os.path.join(pkg_dir, OUTPUT_DIR['FOLDER'], OUTPUT_DIR['SUBFOLDER'])
    os.makedirs(output_folder, exist_ok=True)

    # Load parameters
    param_file = os.path.join(pkg_dir, PARAM_DIR['FOLDER'], PARAM_DIR['FILE'])
    params = load_parameters(param_file)

    # Define column names for input data
    column_names_input = [f'human_kp{i}_{axis}'
                          for i in params['SELECTED_KEYPOINTS'] for axis in ['x', 'y', 'z']]
    
    # Initialize joint limits
    qbounds = [JointLimits(-np.pi, np.pi)]*params['N_DOF']

    # Set the shoulder rot y joint limits
    qbounds[12] = JointLimits(params['SHOULDER_ROT_Y_JOINT_LIMITS'][0],
                              params['SHOULDER_ROT_Y_JOINT_LIMITS'][1]) # right shoulder
    qbounds[16] = JointLimits(params['SHOULDER_ROT_Y_JOINT_LIMITS'][0],
                              params['SHOULDER_ROT_Y_JOINT_LIMITS'][1]) # left shoulder
    qbounds[20] = JointLimits(params['SHOULDER_ROT_Y_JOINT_LIMITS'][0],
                              params['SHOULDER_ROT_Y_JOINT_LIMITS'][1]) # right hip
    qbounds[24] = JointLimits(params['SHOULDER_ROT_Y_JOINT_LIMITS'][0],
                              params['SHOULDER_ROT_Y_JOINT_LIMITS'][1]) # left hip

    # Iterate through all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    for csv_file_path in csv_files:
        # Generate the output file path
        output_file_path = os.path.join(output_folder, os.path.basename(csv_file_path))

        # Process the CSV file
        process_csv_file(csv_file_path, output_file_path, column_names_input,
                         qbounds, params['SELECTED_KPT_NAMES'], params['JOINT_NAMES'], params['PARAM_NAMES'],
                         params['N_DOF'], params['N_PARAM'])
        print(f"Processed data saved to {output_file_path}")


if __name__ == '__main__':
    main()