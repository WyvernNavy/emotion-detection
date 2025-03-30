import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from keras.saving import register_keras_serializable

# Define and register the missing function
@register_keras_serializable()
def gray_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

# Load the trained model
model = load_model('emotion_model.h5', custom_objects={'gray_to_rgb': gray_to_rgb})

# Define emotion labels (order must match your training labels)
EMOTIONS = ['happy', 'sad', 'angry', 'surprised', 'neutral']

# Create Tkinter UI
root = tk.Tk()
root.title("Real-time Emotion Recognition")
root.geometry("1000x800")

# Label for the video feed
video_label = Label(root)
video_label.pack(side=tk.TOP)

# Label for showing the detected emotion and probability
emotion_label = Label(root, text="", font=("Arial", 20), fg="blue")
emotion_label.pack(side=tk.TOP)

# Create a Matplotlib figure for the pentagonal graph
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.axis('off')

# Compute pentagon vertices
angles = [2 * math.pi * i / 5 - math.pi / 2 for i in range(5)]
vertices = [(math.cos(angle), math.sin(angle)) for angle in angles]

# Plot pentagon outline
pentagon_x = [v[0] for v in vertices] + [vertices[0][0]]
pentagon_y = [v[1] for v in vertices] + [vertices[0][1]]
ax.plot(pentagon_x, pentagon_y, 'k-', lw=2)

# Label each vertex with its corresponding emotion
for i, (x, y) in enumerate(vertices):
    ax.text(x * 1.15, y * 1.15, EMOTIONS[i], ha='center', va='center', fontsize=12)

# Create circle patches at each vertex
vertex_patches = []
for i, (x, y) in enumerate(vertices):
    circle = patches.Circle((x, y), 0.1, facecolor='lightgray', edgecolor='black', zorder=2)
    ax.add_patch(circle)
    vertex_patches.append(circle)

# Create a marker for the moving point
point_marker, = ax.plot([0], [0], 'ro', markersize=12)

# Embed the Matplotlib figure in Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Global variables
last_emotion_index = None
hold_counter = 0
max_hold = 100

def update_frame():
    global last_emotion_index, hold_counter

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (48, 48))
    resized_frame = resized_frame / 255.0
    resized_frame = np.expand_dims(resized_frame, axis=-1)
    resized_frame = np.expand_dims(resized_frame, axis=0)

    # Get emotion predictions
    preds = model.predict(resized_frame)[0]
    most_prob_index = np.argmax(preds)
    most_prob_emotion = EMOTIONS[most_prob_index]
    prob_value = preds[most_prob_index]

    # Update hold counter
    if most_prob_index == last_emotion_index:
        hold_counter += 1
    else:
        hold_counter = 0
        last_emotion_index = most_prob_index

    # Interpolate color from green to red
    f = min(hold_counter / max_hold, 1.0)
    interp_color = (f, 1 - f, 0)

    # Update vertex patches
    for i, patch in enumerate(vertex_patches):
        patch.set_facecolor(interp_color if i == most_prob_index else 'lightgray')

    # Update the emotion label
    emotion_label.config(text=f"Most Likely Emotion: {most_prob_emotion} ({prob_value * 100:.1f}%)")

    # Calculate new position for red dot
    angle = angles[most_prob_index]
    new_x = math.cos(angle) * prob_value
    new_y = math.sin(angle) * prob_value
    point_marker.set_data([new_x], [new_y])
    canvas.draw()

    # Convert frame for Tkinter display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    # Schedule next update
    root.after(10, update_frame)

# Start the frame update loop
update_frame()

# Run Tkinter event loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
