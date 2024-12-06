# Image recognition with vector databas

Envireoment: python 3.10

```bash
pip install customtkinter tkinter PIL langchain langchain-ollama chromadb numpy opencv-python ast diffusers
```

## instruction:
install all the library before use, and then run the main.py file.
first time running will take a while because it has to download 2 models.

## Description:
This app use tenserflow model to convert image to 1280 dims array for Chromadb embedding. It can be use to search simular type of image from the database.
and then use diffusion model to transfor the images from chromadb to new image.
Chromadb uses KNN to search the image, you can use the test1 test2 test3 images to see the result



