import os
import pandas as pd

PKL_DIR = {'FOLDER': 'data', 'SUBFOLDER': '1_pkl_raw', 'FILE': 'measurement_data.pkl'}
CSV_DIR = {'FOLDER': 'data', 'SUBFOLDER': '2_csv_raw'}


def main():
    # Determine the directory of the script being executed
    pkg_dir = os.path.dirname(os.path.abspath(__file__)).split('scripts')[0]

    # Select pickle file
    pickle_file = os.path.join(pkg_dir,
                            PKL_DIR['FOLDER'],
                            PKL_DIR['SUBFOLDER'],
                            PKL_DIR['FILE'])

    # Select the directory to save the CSV files
    csv_file_path = os.path.join(pkg_dir,
                                CSV_DIR['FOLDER'],
                                CSV_DIR['SUBFOLDER'])
    os.makedirs(csv_file_path, exist_ok=True)

    print(f"Converting data from {pickle_file}.")

    try:
        # Open and load the pickle file
        with open(pickle_file, 'rb') as pickle_file:
            data = pd.read_pickle(pickle_file)

        for sub, df in data.items():
            # resets the index of the DataFrame
            df.reset_index(drop=True, inplace=True)

            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(csv_file_path,f'{sub}.csv'), index=False)
            print(f"Data from {sub} saved to CSV file.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()