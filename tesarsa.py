import tkinter as tk
import time

def start_timer():
    global start_time
    start_time = time.time()
    update_timer()

def stop_timer():
    timer_label.after_cancel(timer_id)

def update_timer():
    elapsed_time = round(time.time() - start_time, 2)
    timer_label.config(text=f"Time Elapsed: {elapsed_time} seconds")
    global timer_id
    timer_id = timer_label.after(100, update_timer)  # Update every 100 milliseconds (0.1 seconds)

root = tk.Tk()
root.title("Timer")

timer_label = tk.Label(root, text="Time Elapsed: 0.00 seconds", font=("Helvetica", 16))
timer_label.pack(pady=20)

start_button = tk.Button(root, text="Start Timer", command=start_timer)
start_button.pack()

stop_button = tk.Button(root, text="Stop Timer", command=stop_timer)
stop_button.pack()

root.mainloop()
