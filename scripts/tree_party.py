import os
import shutil

# Define the root directory of your project (adjust if necessary)
project_root = "C:/Users/franc/PycharmProjects/omega_pre"

# Define the directories to move into the 'src' folder
directories_to_move = ["config", "data", "David", "notebooks", "scripts", "svc", "utils"]

# Create the 'src' directory if it doesn't already exist
src_directory = os.path.join(project_root, "src")
os.makedirs(src_directory, exist_ok=True)

# Move the specified directories into the 'src' directory
for dir_name in directories_to_move:
    full_dir_path = os.path.join(project_root, dir_name)
    new_location = os.path.join(src_directory, dir_name)

    if os.path.exists(full_dir_path):
        shutil.move(full_dir_path, new_location)
        print(f"Moved: {dir_name}")
    else:
        print(f"Directory not found, skipping: {dir_name}")

print("Reorganization complete.")
