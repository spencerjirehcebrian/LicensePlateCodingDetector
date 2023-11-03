import tkinter as tk

def update_count(count):
    count_label.config(text=str(count))
    count_label.after(1000, update_count, count + 1)

root = tk.Tk()
root.title("Count Up Label")

count_label = tk.Label(root, text="0", font=("Helvetica", 24))
count_label.pack(pady=50)

update_count(0)  # Start the count-up function with an initial count of 0

root.mainloop()
