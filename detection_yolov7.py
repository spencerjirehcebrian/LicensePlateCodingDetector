import os
import subprocess
import shutil
import time


def run(file_path, file_name, close_loading, start, stop, update_text):
    # Activate the Conda environment
    conda_env = "myenv"  # Replace with your Conda environment name
    activate_cmd = f"conda activate {conda_env}"

    # Set the path to the yolov7-object-tracking directory
    yolov7_dir = "yolov7-object-tracking"

    # Change the current working directory to yolov7-object-tracking
    os.chdir(yolov7_dir)

    # Define the commands to execute within the yolov7-object-tracking directory
    commands = [
        f'python detect.py --weights ../weights/best_exp4.pt --source "{file_path}"',
    ]

    start()
    update_text("DETECTION STATUS: Detection Currently Running...")
    # Execute the commands in the Conda environment
    for cmd in commands:
        full_cmd = f"{activate_cmd} && {cmd}"
        subprocess.run(full_cmd, shell=True, check=True)
    stop()
        
    # Change back to the original working directory (optional)
    os.chdir("..")  # Move up one directory
    
    current_file_name = f"yolov7-object-tracking/runs/detect/object_tracking/{file_name}"
    new_file_name = f"yolov7-object-tracking/runs/detect/object_tracking/yolov7_{file_name}"
    os.rename(current_file_name, new_file_name)

    # Source file path
    source_file = f"yolov7-object-tracking/runs/detect/object_tracking/yolov7_{file_name}"

    # Destination directory
    destination_directory = "output_videos/"

    # Copy the file to the destination directory
    shutil.copy(source_file, destination_directory)

    # Delete the source file
    os.remove(source_file)

    # Remove the source directory (if it's empty)
    source_directory = os.path.dirname(source_file)
    if not os.listdir(source_directory):
        os.rmdir(source_directory)

    close_loading(f"output_videos/yolov7_{file_name}")
