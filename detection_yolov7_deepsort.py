import os
import subprocess

def run(file_path, file_name, close_loading, start, stop, update_text):
        # Activate the Conda environment
    conda_env = 'myenv'  # Replace with your Conda environment name
    activate_cmd = f'conda activate {conda_env}'

    # Set the path to the yolov7-object-tracking directory
    yolov7_dir = 'yolov7-deepsort-tracking'

    # Change the current working directory to yolov7-object-tracking
    os.chdir(yolov7_dir)

    # Define the commands to execute within the yolov7-object-tracking directory
    commands = [
        f'python run.py --weights ../weights/best_license_plate.pt --source "{file_path}" --video_output "{file_name}"',
    ]

    start()
    update_text("DETECTION STATUS: Detection Currently Running...")
    # Execute the commands in the Conda environment
    for cmd in commands:
        full_cmd = f'{activate_cmd} && {cmd}'
        subprocess.run(full_cmd, shell=True, check=True)
    stop()
    # Change back to the original working directory (optional)
    os.chdir('..')  # Move up one directory
    
    close_loading(f"output_videos/deepsort_{file_name}")
