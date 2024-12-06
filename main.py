import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

from chromadb.config import Settings
import chromadb
import tensorflow as tf
import numpy as np
import os
from diffusers import StableDiffusionImg2ImgPipeline

# -----------------------------------
# TensorFlow: Image Feature Extraction
# -----------------------------------
global temp_img
# Load pre-trained model for feature extraction
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='avg')
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu")  # Use CPU (remove this line if running on GPU)

def preprocess_image(image):
    if image.shape[-1] == 4:  # Check if the image has 4 channels
        image = image[:, :, :3]  # Remove the alpha channel
    img = tf.image.resize(image, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def extract_features(image):
    preprocessed_image = preprocess_image(image)
    features = model.predict(tf.expand_dims(preprocessed_image, axis = 0))
    return features


# -----------------------------------
# ChromaDB: Storing Image Embeddings
# -----------------------------------

client = chromadb.Client(Settings())
collection = client.create_collection("image_embeddings")

def add_image_embedding(image_id, embedding, location):
    collection.add(
        ids=[image_id.split(".")[0]],
        embeddings=[embedding],
        metadatas=[{'image_id': location}]
    )
# # -----------------------------------
# # Workflow to Save and Search Images
# # -----------------------------------

def save_image(image_id, image):
    # Extract image features (embeddings)
    embedding = extract_features(image)
    # Save the embedding into ChromaDB
    add_image_embedding(image_id, embedding)

# -----------------------------------
# customtkinter
# -----------------------------------

directory_path = "./database"

# List all files in the directory
file_names = os.listdir(directory_path)


file_name = "database.txt"

index = 0

file_name_list = list()
embeding_list = list()
location_list = list()

with open(file_name, "r") as file:
    for line in file:
        if index == 0:
            file_name_list.append(line.strip())
            index = index + 1
        elif index == 1:
            array = np.array([float(num) for num in line.split()])
            embeding_list.append(array)
            index = index + 1
        elif index == 2:
            location_list.append(line.strip())
            index = 0


for i in range(len(file_name_list)):
    add_image_embedding(file_name_list[i], embeding_list[i], location_list[i])


# Initialize the CustomTkinter theme
ctk.set_appearance_mode("System")  # Use "Dark" or "Light" for fixed modes
ctk.set_default_color_theme("blue")  # Customize with other color themes if needed

# Create the main app window
app = ctk.CTk()
app.attributes('-fullscreen', True)  # Make the window full-screen without borders
app.title("Full-Screen App with Sidebar and Image Display")

# Get screen dimensions
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

# Calculate widths for each section
menu_width = int(screen_width * 0.1)
middle_width = int(screen_width * 0.45)
right_width = screen_width - menu_width - middle_width


# Function to load and display an image in the top half of the middle section
def load_image():
    global temp_img

    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )

    if file_path:
        # Open, resize, and display the image
        print(file_path)

        img = Image.open(file_path)
        img = img.resize(((224, 224)), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        # opencv_img = cv2.imread(file_path)
        opencv_img = Image.open(file_path)
        opencv_img = np.array(opencv_img)
        # Update the label with the image
        image_label.configure(image=img_tk, text="")  # Clear the text
        image_label.image = img_tk  # Keep a reference to avoid garbage collection


        embed = extract_features(opencv_img)

        results = collection.query(
            query_embeddings=embed[0],
            n_results=3  # Change this number for the number of resu6666666alts you want
        )

        print(results)
        index = 0
        for i in results['metadatas'][0]:
            if index == 0:
                temp_img = i['image_id']
            image = Image.open(i['image_id'])
            image_resized = image.resize((224, 224))  # Resize image to fit label
            photo = ImageTk.PhotoImage(image_resized)
            x = round(index % 2)
            y = round(index / 2)
            label = ctk.CTkLabel(master=right_frame, text="", image=photo)  # No text, only image
            label.place(x=125 + x * 300, y=125 + y * 300)
            index = index + 1


        # input_image = Image.open("./database/dog_002.jpg").convert("RGB")  # Input image
        # input_image = input_image.resize((128, 128))  # Resize to model-compatible size (e.g., 512x512)
        # prompt = "fancy dog"
        # output = pipe(
        #     prompt=prompt,
        #     image=input_image,
        #     strength=0.7,  # How much of the original image to keep (0.0 = no change, 1.0 = full generation)
        #     guidance_scale=7.5  # Controls adherence to the text prompt
        # )
        # result_image = output.images[0]
        # result_image.save("output.jpg")
        # result_image.show()
        # right_image_label.configure(image=img_tk, text="")
        # right_image_label.image = img_tkq


# Function to close the app when "Q" is pressed
def quit_app(event=None):
    app.quit()

def diffusion_fun():
    global temp_img
    print(temp_img)
    file_name = temp_img.split("/")[-1]
    animal = file_name.split("_")[0]
    print(animal)
    input_image = Image.open(temp_img).convert("RGB")  # Input image
    input_image = input_image.resize((512, 512))  # Resize to model-compatible size (e.g., 512x512)
    prompt = "fancy " + animal
    output = pipe(
        prompt=prompt,
        image=input_image,
        strength=0.7,  # How much of the original image to keep (0.0 = no change, 1.0 = full generation)
        guidance_scale=7.5  # Controls adherence to the text prompt
    )
    result_image = output.images[0]
    result_image.save("output.jpg")
    result_image.show()
    # right_image_label.configure(image=img_tk, text="")
    # right_image_label.image = img_tkq

# Bind the "Q" key to the quit function
app.bind("q", quit_app)
app.bind("Q", quit_app)  # Bind both lowercase and uppercase Q

# Create the sidebar frame (10% width)
sidebar = ctk.CTkFrame(app, width=menu_width, height=screen_height, corner_radius=0)
sidebar.grid(row=0, column=0, sticky="ns")  # Sticky "ns" makes it fill vertically


# Add an "Upload Image" button
upload_button = ctk.CTkButton(sidebar, text="Upload Image", command=load_image)
upload_button.pack(pady=20, padx=20, fill="x")

diffusers_button = ctk.CTkButton(sidebar, text="diffusion", command=diffusion_fun)
diffusers_button.pack(pady=40, padx=20, fill="x")

# Create the middle frame (45% width)
middle_frame = ctk.CTkFrame(app, width=middle_width, height=screen_height)
middle_frame.grid(row=0, column=1, sticky="nsew")

# Split the middle frame horizontally into two equal frames
top_middle_frame = ctk.CTkFrame(middle_frame, width=middle_width, height=screen_height // 2, corner_radius=0)
top_middle_frame.pack(side="top", fill="both", expand=True)

bottom_middle_frame = ctk.CTkFrame(middle_frame, width=middle_width, height=screen_height // 2, corner_radius=0)
bottom_middle_frame.pack(side="top", fill="both", expand=True)

# Add a label to display the image in the top half of the middle frame
image_label = ctk.CTkLabel(top_middle_frame, text="Image will appear here", font=("Arial", 24))
image_label.pack(pady=20)

# Add a placeholder in the bottom half of the middle frame
bottom_middle_label = ctk.CTkLabel(bottom_middle_frame, text="Bottom Half of Middle", font=("Arial", 24))
bottom_middle_label.pack(pady=20)

# Create the right frame (45% width) for additional content if needed
right_frame = ctk.CTkFrame(app, width=right_width, height=screen_height)
right_frame.grid(row=0, column=2, sticky="nsew")

right_image_label = ctk.CTkLabel(right_frame)
right_image_label.pack(pady=20)


# Configure grid layout for resizing behavior
app.grid_columnconfigure(0, weight=0)  # Sidebar remains fixed size
app.grid_columnconfigure(1, weight=1)  # Middle section takes up remaining space
app.grid_columnconfigure(2, weight=1)  # Right section takes up remaining space
app.grid_rowconfigure(0, weight=1)  # Allows vertical fill of screen

app.mainloop()