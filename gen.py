import os
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline

def fuse_images(image_dir):
    """
    Load all .jpeg images from a folder and merge them into an averaged image.
    """
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.jpeg')]
    if not image_files:
        raise ValueError("No .jpeg image files found. Please check the folder path and file formats.")

    # Initialize the accumulator
    accumulated_image = None
    count = 0

    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        image = image.resize((512, 512))
        image_np = np.array(image).astype(np.float32)

        if accumulated_image is None:
            accumulated_image = np.zeros_like(image_np)
        accumulated_image += image_np
        count += 1

    # Compute the averaged image
    fused_image_np = accumulated_image / count
    fused_image_np = np.clip(fused_image_np, 0, 255).astype(np.uint8)
    fused_image = Image.fromarray(fused_image_np)

    return fused_image

def generate_image(fused_image, prompt, output_path, num_inference_steps=50, guidance_scale=7.5):
    """
    Use Stable Diffusion to generate a new image based on the merged image and a text prompt.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained Stable Diffusion Img2Img model
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline = pipeline.to(device)

    # Set the seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(42)

    # Resize the merged image
    fused_image = fused_image.resize((512, 512))

    # Generate the image
    if device == "cuda":
        with torch.autocast(device_type=device):
            images = pipeline(
                prompt=prompt,
                image=fused_image,
                strength=0.75,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images
    else:
        images = pipeline(
            prompt=prompt,
            image=fused_image,
            strength=1,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images

    # Save the generated image
    images[0].save(output_path)
    print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    # Path to the folder containing images
    image_dir = "./image"  # Replace with the path to the folder containing .jpeg files
    # Path to save the generated image
    output_path = "generated_image.png"
    # Text prompt
    prompt = "A beautiful painting blending the essence of multiple photographs."

    # Merge multiple images
    try:
        fused_image = fuse_images(image_dir)
        fused_image.save("fused_image.jpeg")  # Save the merged image
        print("Merged image saved as fused_image.jpeg")

        # Use Stable Diffusion to generate a new image
        generate_image(fused_image, prompt, output_path)
    except ValueError as e:
        print(f"Error: {e}")