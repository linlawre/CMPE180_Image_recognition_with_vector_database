import os
import tensorflow as tf
import chromadb
from chromadb.config import Settings
import cv2

model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(image):
    img = tf.image.resize(image, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def extract_features(image):
    preprocessed_image = preprocess_image(image)
    features = model.predict(tf.expand_dims(preprocessed_image, axis=0))
    return features

# -----------------------------------
# ChromaDB: Storing Image Embeddings
# -----------------------------------

client = chromadb.Client(Settings())
collection = client.create_collection("image_embeddings")

def add_image_embedding(image_id, embedding):
    collection.add(
        ids=[image_id],
        embeddings=[embedding.flatten().tolist()],
        metadatas=[{'image_id': image_id}]
    )

# # -----------------------------------
# # Workflow to Save and Search Images
# # -----------------------------------

def save_image(image_id, image):
    # Extract image features (embeddings)
    embedding = extract_features(image)
    print(embedding)
    add_image_embedding(image_id, embedding)



# Specify the directory path
directory_path = "./database"

# List all files in the directory
file_names = os.listdir(directory_path)


file_name = "database.txt"

with open(file_name, "w") as file:

# Print the file names
    for file_name in file_names:
        name_without_extension = os.path.splitext(file_name)[0]
        location = directory_path + "/" + file_name
        image1 = cv2.imread("./database/" + file_name)
        embed = extract_features(image1)
        file.write(file_name)
        file.write("\n")

        for i in embed[0]:
            file.write(str(i))
            file.write(" ")

        file.write("\n")
        file.write("./database/" + file_name)
        file.write("\n")