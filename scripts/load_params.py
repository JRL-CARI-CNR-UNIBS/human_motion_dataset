import yaml, os

# Determine the directory of the script being executed
PKG_DIR = os.path.dirname(os.path.abspath(__file__)).split('scripts')[0]

# param folder
PARAM_FILE = os.path.join(PKG_DIR, 'config', 'params.yaml')

def load_parameters(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters

# Example usage
if __name__ == "__main__":
    parameters = load_parameters(PARAM_FILE)
    print(parameters)