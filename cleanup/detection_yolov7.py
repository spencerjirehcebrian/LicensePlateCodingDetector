import subprocess
import urllib.request
import os
import tkinter as tk
from tkinter import ttk
import cv2


def setup_yolov7():
    # Clone the repository
    if not os.path.exists("yolov7"):
        clone_command = "git clone https://github.com/WongKinYiu/yolov7.git"
        subprocess.run(clone_command, shell=True, check=True)

    # Download the requirements.txt file using urllib
    requirements_url = (
        "https://raw.githubusercontent.com/WongKinYiu/yolov7/u5/requirements.txt"
    )
    requirements_file = "requirements.txt"

    try:
        urllib.request.urlretrieve(requirements_url, requirements_file)
    except Exception as e:
        print(f"Error downloading requirements.txt: {e}")

    # try:
    #     # Open the Anaconda shell and run 'pip install' command
    #     command = f'conda activate your_environment_name && pip install -r requirements.txt'
    #     subprocess.run(command, shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error: {e}")

def get_weights():
    print('Checking Weights...')
    # List of URLs to download
    urls = [
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
    ]

    # Directory where you want to save the downloaded files
    download_directory = "yolov7/yolov7_weights/"

    # Check if the directory exists, and if not, create it
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    # Download files
    for url in urls:
        filename = url.split("/")[-1]  # Extract the filename from the URL
        file_path = f"{download_directory}/{filename}"

        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            
def convert_to_frames(file_path):
    # Output folder for frames
    output_folder = '/yolov7/conversion_frames'
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(file_path)

    # Initialize a frame counter
    frame_count = 0
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    
def convert_to_video(file_path_output):
    # Output folder for frames
    output_folder = 'yolov7/conversion_frames'
    os.makedirs(output_folder, exist_ok=True)
    
    # Output video file
    output_video = file_path_output

    # Get the list of frame files
    frame_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')]
    frame_files.sort()  # Sort frames in ascending order

    # Get the height and width of the frames
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    # Initialize a VideoWriter object to create the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

    # Write each frame to the output video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    print('conversion done')

def run_yolov7_detection(file_path):
    # # Activate the Conda environment
    # conda_env = "myenv"
    # activate_command = f"conda activate {conda_env}"
    # subprocess.run(activate_command, shell=True, check=True)

    # # Define the relative path to the 'yolov7' directory
    # relative_path = 'yolov7'

    # # Get the current working directory
    # current_directory = os.getcwd()

    # # Construct the absolute path to the 'yolov7' directory
    # yolov7_dir = os.path.join(current_directory, relative_path)

    # # Change the working directory to the 'yolov7' directory
    # os.chdir(yolov7_dir)

    # # Run the 'detect.py' script
    # detect_command = "python detect.py --weights /yolov7_weights/yolov7.pt --conf 0.1 --source /conversion_frames"
    # subprocess.run(detect_command, shell=True, check=True)
    # Define the command to run
    command = "python yolov7/detect.py --weights /yolov7_weights/yolov7.pt --conf 0.1 --source /conversion_frames"

    # Activate your Conda environment (replace 'your_env_name' with your actual environment name)
    conda_activate_command = "conda activate myenv"

    # Combine the activation and script execution commands
    full_command = f"{conda_activate_command} && {command}"
    subprocess.run(full_command, shell=True)
    
    
def run(file_path):
    setup_yolov7()
    if not os.path.exists("yolov7/yolov7_weights"):
        get_weights()
    print('Converting to frames...')
    #convert_to_frames(file_path)
    print('Running Detection on Images...')
    run_yolov7_detection(file_path)
    print('Converting to Video...')
    #convert_to_video('output_video.mp4')
    print('Detection Complete')
 
    
run('random')    

    
