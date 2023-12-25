# Python script to create the proposed folder structure for the "omega_predictor" project.

import os

def create_folder_structure(base_path):
    # Define the folder structure
    folders = [
        "config",
        "models",
        "services",
        "services/prediction_service",
        "services/data_service",
        "data",
        "data/processed",
        "data/raw",
        "utils",
        "tests",
        "notebooks"
    ]

    # Create each folder
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        # Create an __init__.py file in each folder to make it a package
        init_file = os.path.join(path, '__init__.py')
        with open(init_file, 'w') as f:
            pass  # Just create an empty __init__.py file

    # Create additional files at the root
    additional_files = ["requirements.txt", "main.py", "README.md"]
    for file in additional_files:
        file_path = os.path.join(base_path, file)
        with open(file_path, 'w') as f:
            pass  # Just create an empty file

    print(f"Folder structure created in {base_path}")

# Set the base path to the current directory (or change to your desired root directory)
base_path = '..'  # Current directory
create_folder_structure(base_path)
