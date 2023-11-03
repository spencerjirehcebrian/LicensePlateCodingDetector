from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import time
import detection_faster_rcnn, detection_yolov7, detection_yolov7_deepsort
import os
import threading
from tkinter import messagebox
import re


global_file_path_undetected = ""
global_file_path_detected = ""
global_file_name = ""
global_frame_numbers = 0
global_average_frame = 0
global_current_frame = 0



file_open = False
video_paused = False  # Track video pause state

# Define a variable for the video capture
cap = None

# Set the desired frame rate (e.g., 30 frames per second)
desired_frame_rate = 30

# Set the playback speed factor
playback_speed_factor = 1.5  # 1.5 normal speedsme

root = Tk()
root.attributes("-fullscreen", True)
root.title("License Plate Coding Detection (CPS - Proof of Concept)")

selected_option = StringVar()

frm = ttk.Frame(root, padding=20)
frm.pack()

ttk.Label(frm, text="License Plate Coding Detection (CPS - Proof of Concept)").pack(
    padx=10, pady=2
)

def start_timer():
    global start_time
    start_time = time.time()
    update_timer()

def stop_timer():
    time_label.after_cancel(timer_id)

def update_timer():
    elapsed_time = round(time.time() - start_time, 2)
    time_label.config(text=f"DETECTION TIME ELAPSED: {elapsed_time} seconds")
    global timer_id
    timer_id = time_label.after(100, update_timer)  # Update every 100 milliseconds (0.1 seconds)
    
# def update_text(status, time, output, average, frame):
#     global global_average_frame, global_current_frame
#     if status != 0:
#         status_label.config(text=status)
#     if time != 0:
#         time_label.config(text=time)
#     if output != 0:
#         output_label.config(text=output)
#     if average != 0:
#         global_current_frame += 1
#         global_average_frame += average
#         meow = global_average_frame/global_current_frame
#         average_label.config(text=meow)
#     if frame != 0:
#         frame_label.config(text=frame)

def update_text(status):
    global global_average_frame, global_current_frame
    if status != 0:
        status_label.config(text=status)

def close_window():
    root.destroy()

def confirm_run():
    result = messagebox.askyesno("Confirmation", "Begin the detection?")
    if result:
        # Add your code to proceed with the action here
        run_model()
        
# Add Pause and Play buttons
def pause_video():
    global video_paused
    video_paused = True

def play_video():
    global video_paused
    if cap is not None:
        video_paused = False
        while not video_paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (480, 270))
            photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            video_label.config(image=photo)
            video_label.image = photo
            root.update()
            time.sleep(
                1 / (desired_frame_rate * playback_speed_factor)
            )  # Adjust playback speed

def video_button():
    if (video_paused == False):
        play_button.config(text="Play")
        pause_video()
        
    else:
        play_button.config(text="Pause")
        play_video()
        

def center_window(window, width, height):
    # Get the screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calculate the X and Y coordinates for the center of the screen
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # Set the window's geometry to center it on the screen
    window.geometry(f"{width}x{height}+{x}+{y}")


def run_model():
    selected_value = selected_option.get()

    status_label.config(text=f"DETECTION STATUS: Loading Model...")
    pause_video()

    def close_loading(final_video):
        close_button["state"] = NORMAL
        run_button["state"] = NORMAL
        for button in radio_buttons:
            button["state"] = NORMAL
        status_label.config(
            text=f"DETECTION STATUS: Detection Complete"
        )
        stop_timer()
        
        label_text = time_label.cget("text")
        label_float = 0
        match = re.search(r"(\d+\.\d+)", label_text)
        if match:
            float_value = float(match.group(1))
            label_float = float_value
        else:
            label_float.config(text="MODEL AVG. DETECTION TIME PER FRAME: ERROR")

        global file_path, global_frame_numbers    
        file_path = final_video
        cap = cv2.VideoCapture(file_path)
        cap.set(cv2.CAP_PROP_FPS, desired_frame_rate)
        
        averages = label_float/global_frame_numbers
        
        average_label.config(text=f"MODEL AVG. DETECTION TIME PER FRAME: {averages:.2f} FPS")
        messagebox.showinfo("Complete", f"Inference Complete at {averages:.2f} FPS")
        play_video()

    print(f"Selected radio option: {selected_value}")
    run_button["state"] = DISABLED
    close_button["state"] = DISABLED
    for button in radio_buttons:
        button["state"] = DISABLED
    if selected_value == "faster_rcnn":
        thread = threading.Thread(
            target=detection_faster_rcnn.run,
            args=(global_file_path_undetected, global_file_name, close_loading, start_timer, stop_timer, update_text)
        )
        thread.start()
    elif selected_value == "yolov7":
        thread = threading.Thread(
            target=detection_yolov7.run,
            args=(global_file_path_undetected, global_file_name, close_loading, start_timer, stop_timer, update_text)
        )
        thread.start()
    elif selected_value == "yolov7_deep_sort":
        thread = threading.Thread(
            target=detection_yolov7_deepsort.run,
            args=(global_file_path_undetected, global_file_name, close_loading, start_timer, stop_timer, update_text)
        )
        thread.start()


def check_radio_selection():
    if selected_option.get() == 0:
        run_button["state"] = DISABLED
    elif file_open == True:
        run_button["state"] = NORMAL


def open_file():
    global cap, file_open, global_frame_numbers
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("MP4 Files", ("*.mp4")),
            # ("Video Files", ("*.mp4", "*.avi", "*.mkv")),
            # ("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
        ]
    )
    file_open = True
    global global_file_path_undetected, global_file_name
    global_file_path_undetected = file_path
    global_file_name = os.path.basename(file_path)
    if file_path:
        if cap is not None:
            cap.release()  # Release the previous video capture, if any
        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            # If the selected file is an image
            image = Image.open(file_path)
            image = image.resize((640, 480))
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo
            video_label.config(
                image=None
            )  # Clear video label if it was previously displayed
        elif file_path.lower().endswith((".mp4", ".avi", ".mkv")):
            # If the selected file is a video
            cap = cv2.VideoCapture(file_path)
            global_frame_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_label.config(text=f"VIDEO FRAMES: {global_frame_numbers} frames")
            cap.set(cv2.CAP_PROP_FPS, desired_frame_rate)
            play_video()
        play_button["state"] = DISABLED
        video_name_label.config(text=f"Video File Path: {file_path}")


image_label = ttk.Label(frm)
image_label.pack()

video_label = ttk.Label(frm)
video_label.pack()

video_name_label = ttk.Label(frm)
video_name_label.pack()

btn_open = ttk.Button(frm, text="Import Video File (mp4 format)", command=open_file)
btn_open.pack(padx=10, pady=2)

play_button = ttk.Button(frm, text="Pause", command=video_button, state=DISABLED)
play_button.pack(padx=10, pady=2)

# pause_button = ttk.Button(frm, text="Pause", command=pause_video)
# pause_button.pack(padx=10, pady=2)


# Create and place the radio buttons in a loop
options = [
    ("Faster RCNN", "faster_rcnn"),
    ("YOLOv7", "yolov7"),
    ("YOLOv7 + DeepSort", "yolov7_deep_sort"),
]

radio_buttons = []  # Store the Radiobuttons in a list

for text, value in options:
    radiobutton = ttk.Radiobutton(
        frm,
        text=text,
        variable=selected_option,
        value=value,
        command=check_radio_selection,
    )
    radiobutton.pack(anchor="w", padx=10, pady=2)
    radio_buttons.append(radiobutton)

status_label = ttk.Label(frm, text="DETECTION STATUS: Inactive")
status_label.pack(anchor="e", padx=10, pady=2)

time_label = ttk.Label(frm, text="DETECTION TIME ELAPSED: --s")
time_label.pack(anchor="e", padx=10, pady=2)

frame_label = ttk.Label(frm, text="VIDEO FRAMES: --")
frame_label.pack(anchor="e", padx=10, pady=2)

average_label = ttk.Label(frm, text="MODEL AVG. DETECTION TIME PER FRAME: -- FPS")
average_label.pack(anchor="e", padx=10, pady=2)

output_label = ttk.Label(frm, text="")
output_label.pack(padx=10, pady=2)

run_button = ttk.Button(
    frm, text="Run License Plate Coding Detection", command=confirm_run, state=DISABLED
)
run_button.pack(padx=10, pady=10)

close_button = ttk.Button(frm, text="Exit Application", command=close_window)
close_button.pack(padx=10, pady=0)


root.mainloop()
