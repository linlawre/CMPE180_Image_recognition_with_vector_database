import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

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
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    if file_path:
        # Open, resize, and display the image
        img = Image.open(file_path)
        img = img.resize((int(middle_width * 0.8), int(screen_height * 0.4)), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        # Update the label with the image
        image_label.configure(image=img_tk, text="")  # Clear the text
        image_label.image = img_tk  # Keep a reference to avoid garbage collection


# Function to close the app when "Q" is pressed
def quit_app(event=None):
    app.quit()


# Bind the "Q" key to the quit function
app.bind("q", quit_app)
app.bind("Q", quit_app)  # Bind both lowercase and uppercase Q

# Create the sidebar frame (10% width)
sidebar = ctk.CTkFrame(app, width=menu_width, height=screen_height, corner_radius=0)
sidebar.grid(row=0, column=0, sticky="ns")  # Sticky "ns" makes it fill vertically

# Add menu buttons to the sidebar
menu_buttons = ["Home", "Settings", "Profile", "About", "Logout"]
for i, text in enumerate(menu_buttons):
    button = ctk.CTkButton(sidebar, text=text)
    button.pack(pady=10, padx=20, fill="x")  # Padding and fill for button layout

# Add an "Upload Image" button
upload_button = ctk.CTkButton(sidebar, text="Upload Image", command=load_image)
upload_button.pack(pady=20, padx=20, fill="x")

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

# Configure grid layout for resizing behavior
app.grid_columnconfigure(0, weight=0)  # Sidebar remains fixed size
app.grid_columnconfigure(1, weight=1)  # Middle section takes up remaining space
app.grid_columnconfigure(2, weight=1)  # Right section takes up remaining space
app.grid_rowconfigure(0, weight=1)  # Allows vertical fill of screen

app.mainloop()