import os
import pandas as pd

TRIGGER_DIR = {'FOLDER': 'data', 'SUBFOLDER': '1_trigger'}
KPT_DIR = {'FOLDER': 'data', 'SUBFOLDER': '2_csv_raw'}
IK_DIR = {'FOLDER': 'data', 'SUBFOLDER': '3_ik_results'}
MERGED_DIR = {'FOLDER': 'data', 'SUBFOLDER': '4_merged'}


def main():
    # Determine the directory of the script being executed
    pkg_dir = os.path.dirname(os.path.abspath(__file__)).split('scripts')[0]

    # Construct paths relative to the package's directory
    keypoint_folder = os.path.join(pkg_dir, KPT_DIR['FOLDER'], KPT_DIR['SUBFOLDER'])
    ik_folder = os.path.join(pkg_dir, IK_DIR['FOLDER'], IK_DIR['SUBFOLDER'])
    merged_folder = os.path.join(pkg_dir, MERGED_DIR['FOLDER'], MERGED_DIR['SUBFOLDER'])
    os.makedirs(merged_folder, exist_ok=True)

    # Merge the keypoint and IK results CSV files
    keypoint_files = [f for f in os.listdir(keypoint_folder)]
    ik_files = [f for f in os.listdir(ik_folder)]

    for kf in keypoint_files:
        if kf not in ik_files:
            print(f"File {kf} not found in the IK results folder. Skipping.")
            continue

        subject_id = kf.split('_')[1].split('.csv')[0]
        
        keypoint_df = pd.read_csv(os.path.join(keypoint_folder, kf))
        ik_df = pd.read_csv(os.path.join(ik_folder, ('sub_' + subject_id + '.csv')))

        # Merge the DataFrames
        merged_df = pd.concat([keypoint_df, ik_df], axis=1)

        # Change the timestamp column name to "Timestamp"
        merged_df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)

        # Save the merged DataFrame to a CSV file
        merged_df.to_csv(os.path.join(merged_folder, kf), index=False)

        print(f"Successfully merged {kf} and saved the result to {merged_folder}")


if __name__ == "__main__":
    main()