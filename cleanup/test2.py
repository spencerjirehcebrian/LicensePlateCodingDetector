from jupyter_client import MultiKernelManager
import jupyter_client
import json

# Create a kernel manager
km = MultiKernelManager()

# Start a kernel
kernel_name = 'python3'  # You can specify the kernel you want to use
kernel_id = km.start_kernel(kernel_name=kernel_name)

# Connect to the kernel
kc = km.get_kernel(kernel_id)

# Run Jupyter commands
code = """
print("Hello, Jupyter!")
"""

msg_id = kc.execute(code)
reply = kc.get_shell_msg(msg_id)
content = reply['content']

# Print the output
print(content['text'])
