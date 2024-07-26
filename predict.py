import tkinter as tk
from tkinter import filedialog, Canvas, Label
from PIL import Image, ImageTk
from ultralytics import YOLO

model = YOLO('C:/Users/user/Desktop/Fauna Find/models/6animals.pt')

# results = model('C:/Users/user/Desktop/Fauna Find/dog.jpeg')

def process_image(image_path):
    
    results = model(image_path)

    names_dict = results[0].names
    top1_index = results[0].probs.top1
    top1_conf = results[0].probs.top1conf.item()

    print("Top 1 Prediction:", names_dict[top1_index], "with confidence:", top1_conf)

    img = Image.open(image_path)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)
    canvas.image = img
    canvas.create_image(0, 0, anchor=tk.NW, image=img)

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)
        result_label.config(text="Image loaded successfully.")



root = tk.Tk()
root.title("Fauna Find")

canvas = Canvas(root, width=400, height=400, bg="white")
canvas.pack()

result_label = Label(root, text="", bg="white", justify=tk.LEFT, anchor=tk.NW)
result_label.pack(fill=tk.BOTH, expand=True)

load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

root.mainloop()