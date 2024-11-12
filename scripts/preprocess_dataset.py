import os, json
import pandas as pd
from load_params import load_parameters
from utils import build_configuration_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

PARAM_DIR = {'FOLDER': 'config', 'FILE': 'params.yaml'}
MERGED_DIR = {'FOLDER': 'data', 'SUBFOLDER': '4_merged'}
INPUT_DIR = {'FOLDER': 'data', 'SUBFOLDER': '5_labelled'}
OUTPUT_DIR = {'FOLDER': 'data', 'SUBFOLDER': '6_preprocessed'}


def process_task(sub, instr, vel, task, params, input_folder,
                 conf_names_filt, conf_names_vel, conf_names_acc, conf_names_jerk,
                 kpt_names, kpt_names_filt, kpt_names_vel, kpt_names_acc,
                 var_to_plot, plot_counter, max_plots):
    
    print(f"Building configuration dataset for {sub} - 'PICK-&-PLACE task' - Instruction {instr} - {vel} velocity...")

    if plot_counter >= max_plots:
        var_to_plot = None

    # Load the correct DataFrame
    input_df = pd.read_csv(os.path.join(input_folder, f'dataset_{task}.csv'))

    temp_df = build_configuration_dataset(input_df, sub, instr, task, vel, 
                                          params['JOINT_NAMES'], conf_names_filt, conf_names_vel, conf_names_acc, conf_names_jerk,
                                          params['PARAM_NAMES'], kpt_names,
                                          filter_window=params['SAVGOL_FILTER_WINDOW'], sav_gol_order=params['SAVGOL_ORDER'],
                                          new_dt=params['RESAMPLING_TIME'],
                                          var_to_plot=var_to_plot, differentiate_cartesian=params['DIFFERENTIATE_CARTESIAN'],
                                          kpt_names_filt=kpt_names_filt, kpt_names_vel=kpt_names_vel, kpt_names_acc=kpt_names_acc)
    return temp_df


def main():
    # Determine the directory of the script being executed
    pkg_dir = os.path.dirname(os.path.abspath(__file__)).split('scripts')[0]

    # Construct paths relative to the package's directory
    merged_folder = os.path.join(pkg_dir, MERGED_DIR['FOLDER'], MERGED_DIR['SUBFOLDER'])
    input_folder = os.path.join(pkg_dir, INPUT_DIR['FOLDER'], INPUT_DIR['SUBFOLDER'])
    output_folder = os.path.join(pkg_dir, OUTPUT_DIR['FOLDER'], OUTPUT_DIR['SUBFOLDER'])
    os.makedirs(output_folder, exist_ok=True)

    # Load the parameters from the YAML file
    params = load_parameters(os.path.join(pkg_dir, PARAM_DIR['FOLDER'], PARAM_DIR['FILE']))

    # Define the names of the dataframe columns
    kpt_names = [f'human_kp{i}_{dim}' for i in range(params['N_KPTS']) for dim in ['x', 'y', 'z']]

    conf_names_filt = [f'filt_{conf_name}' for conf_name in params['JOINT_NAMES']]
    conf_names_vel = [f'd{conf_name}' for conf_name in params['JOINT_NAMES']]
    conf_names_acc = [f'dd{conf_name}' for conf_name in params['JOINT_NAMES']]
    conf_names_jerk = [f'ddd{conf_name}' for conf_name in params['JOINT_NAMES']]
    kpt_names_filt = [f'filt_{kpt_name}' for kpt_name in kpt_names]
    kpt_names_vel = [f'd_{kpt_name}' for kpt_name in kpt_names]
    kpt_names_acc = [f'dd_{kpt_name}' for kpt_name in kpt_names]

    # Save the lists containing the names of the columns
    data = {
        "conf_names": params['JOINT_NAMES'],
        "conf_names_filt": conf_names_filt,
        "conf_names_vel": conf_names_vel,
        "conf_names_acc": conf_names_acc,
        "conf_names_jerk": conf_names_jerk,
        "param_names": params['PARAM_NAMES'],
        "kpt_names": kpt_names,
        "kpt_names_filt": kpt_names_filt,
        "kpt_names_vel": kpt_names_vel,
        "kpt_names_acc": kpt_names_acc
    }

    # Save the dictionary to a JSON file
    print("Saving column names to JSON file...")
    json_file_path = os.path.join(output_folder, 'column_names.json')
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    # Build the configuration dataset for each subject (resampling, filtering, and differentiating the configuration data)
    subjects = [f.split('.')[0] for f in os.listdir(merged_folder)]

    var_to_plot = 'q_right_elbow_rot_z'
    max_plots = 0 # Maximum number of plots to show (performance reasons)
    plot_counter = 0

    # Create an empty dictionary to store the resulting dataframes, one for each task
    output_dfs = {}
    for task_name in params['TASKS'].keys():
        output_dfs[task_name] = pd.DataFrame()

    for task, instr_ids in params['TASKS'].items():
        final_df = pd.DataFrame()
        futures = []
        plot_counter = 0

        with ThreadPoolExecutor(max_workers=params['MAX_THREADS']) as executor:
            for sub in subjects:
                for instr in instr_ids:
                    for vel in params['VELOCITIES']:
                        futures.append(executor.submit(process_task,
                                                       sub, instr, vel, task, params, input_folder,
                                                       conf_names_filt, conf_names_vel, conf_names_acc, conf_names_jerk,
                                                       kpt_names, kpt_names_filt, kpt_names_vel, kpt_names_acc,
                                                       var_to_plot, plot_counter, max_plots))
                        plot_counter += 1

            for future in as_completed(futures):
                temp_df = future.result()
                final_df = pd.concat([final_df, temp_df], ignore_index=True, axis=0)

            output_dfs[task] = final_df

    # Save the resulting dataframes to CSV files
    for task_name, df in output_dfs.items():
        print(f"Saving the preprocessed dataset for the {task_name} task to {output_folder}...")
        df.to_csv(os.path.join(output_folder, f'dataset_{task_name}.csv'), index=False)


if __name__ == '__main__':
    main()