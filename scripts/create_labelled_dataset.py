import os
import pandas as pd
from load_params import load_parameters

PARAM_DIR = {'FOLDER': 'config', 'FILE': 'params.yaml'}
TRIGGER_DIR = {'FOLDER': 'data', 'SUBFOLDER': '1_trigger'}
MERGED_DIR = {'FOLDER': 'data', 'SUBFOLDER': '4_merged'}
OUTPUT_DIR = {'FOLDER': 'data', 'SUBFOLDER': '5_labelled'}


def main():
    # Determine the directory of the script being executed
    pkg_dir = os.path.dirname(os.path.abspath(__file__)).split('scripts')[0]

    # Construct paths relative to the package's directory
    trigger_folder = os.path.join(pkg_dir, TRIGGER_DIR['FOLDER'], TRIGGER_DIR['SUBFOLDER'])
    merged_folder = os.path.join(pkg_dir, MERGED_DIR['FOLDER'], MERGED_DIR['SUBFOLDER'])
    output_folder = os.path.join(pkg_dir, OUTPUT_DIR['FOLDER'], OUTPUT_DIR['SUBFOLDER'])
    os.makedirs(output_folder, exist_ok=True)

    # Load the parameters from the YAML file
    params = load_parameters(os.path.join(pkg_dir, PARAM_DIR['FOLDER'], PARAM_DIR['FILE']))
    tasks = params['TASKS']

    # Load the merged files and the trigger files
    merged_files = [os.path.splitext(f)[0] for f in os.listdir(merged_folder)]
    trigger_files = [f for f in os.listdir(trigger_folder)]

    # Create an empty dictionary to store the resulting dataframes, one for each task
    output_dfs = {}
    for task_name in tasks.keys():
        output_dfs[task_name] = pd.DataFrame()

    # Loop through all subjects
    final_df = pd.DataFrame()
    for trigger_file in trigger_files:
        sub_name = os.path.splitext(trigger_file)[0]

        # Load the trigger file with the timestamps
        trigger_df = pd.read_csv(os.path.join(trigger_folder, trigger_file))

        if sub_name in merged_files:
            # Load the merged file with the keypoints and the IK results
            merged_df = pd.read_csv(os.path.join(merged_folder, sub_name + '.csv'))
            
            # Add new columns to the merged_df
            merged_df['Subject'] = sub_name
            merged_df['Instruction_id'] = None
            merged_df['Task_name'] = None
            merged_df['Velocity'] = None

            # Iterate through the tasks and their instruction IDs
            for task_name, instruction_ids in tasks.items():
                # Select only the rows corresponding to the current task from the trigger_df
                current_df = trigger_df[trigger_df['Task_name'] == task_name]
                
                for i in range(len(instruction_ids)):
                    print(f"Processing instruction {instruction_ids[i]} for task {task_name} for subject {sub_name}...")

                    # Get the start timestamp for the current instruction
                    start_timestamps = current_df[current_df['Instruction_id'] == instruction_ids[i]]['Timestamp']
                    
                    # Get the end timestamp for the current instruction
                    if i == len(instruction_ids) - 1:
                        # Detect changes in the 'Task_name' column
                        task_name_changes = trigger_df['Task_name'] != trigger_df['Task_name'].shift(1)
                        task_name_changes.iloc[0] = False   # The first row is never a change

                        idxs = task_name_changes.values & (trigger_df['Task_name'].shift(1) == task_name).values

                        # Get the timestamps where the 'Task_name' changes
                        end_timestamps = trigger_df.loc[idxs, 'Timestamp'].tolist()  # type: ignore

                        # If it is the last row of the trigger_df, add the last timestamp
                        if trigger_df['Task_name'].iloc[-1] == task_name and \
                            trigger_df['Instruction_id'].iloc[-1] == instruction_ids[i]:
                            
                            end_timestamps.append(merged_df['Timestamp'].iloc[-1])

                    else:
                        end_timestamps = current_df[current_df['Instruction_id'] == instruction_ids[i + 1]]['Timestamp']
                    
                    # Iterate through the start and end timestamps and assign
                    # the instruction ID, the task name, and the velocity to the merged_df
                    for start, end in zip(start_timestamps, end_timestamps):
                        mask = (merged_df['Timestamp'] >= start) & (merged_df['Timestamp'] <= end)
                        merged_df.loc[mask, 'Instruction_id'] = instruction_ids[i]
                        merged_df.loc[mask, 'Task_name'] = task_name
                        merged_df.loc[mask, 'Velocity'] = current_df[current_df['Timestamp'] == start]['Velocity'].values[0]

            # Append the merged_df to the final_df
            final_df = pd.concat([final_df, merged_df], ignore_index=True)

    # Define column names
    kpt_names = [f'human_kp{i}_{dim}' for i in range(params['N_KPTS']) for dim in ['x', 'y', 'z']]
    col_names = ['Subject', 'Instruction_id', 'Task_name', 'Velocity', 'Timestamp'] \
                 + kpt_names + params['JOINT_NAMES'] + params['PARAM_NAMES']
    
    # Store dataframes for each task in the output_dfs dictionary (and change the order of the columns)
    for task_name in output_dfs.keys():
        df = final_df[final_df['Task_name'] == task_name]
        output_dfs[task_name] = df[col_names]

    # Save the resulting dataframes to CSV files
    for task_name, df in output_dfs.items():
        df.to_csv(os.path.join(output_folder, f'dataset_{task_name}.csv'), index=False)
        print(f"Saved the labelled dataset for the {task_name} task to {output_folder}")


if __name__ == '__main__':
    main()