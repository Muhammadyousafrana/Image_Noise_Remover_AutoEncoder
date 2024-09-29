import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('autotuned_model.keras')

# Function to add noise to the image
def add_noise(image, noise_factor=0.5):
    image = np.array(image).astype('float32') / 255.0
    if len(image.shape) == 2:  # If grayscale, add channel dimension
        image = np.expand_dims(image, axis=-1)
    noisy_image = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    return np.clip(noisy_image, 0., 1.)

# Sidebar for user input
st.sidebar.title("Autoencoder Settings")
noise_factor = st.sidebar.slider("Noise Factor", 0.1, 1.0, 0.5)
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

st.title("Autoencoder Image Denoising")
st.write("This app uses a convolutional autoencoder to denoise an image.")

# If an image is uploaded, show the result
if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file).convert('L').resize((28, 28))
    img = np.array(img).astype('float32') / 255.0
    st.image(img, caption="Uploaded Image", use_column_width=True, clamp=True)

    # Add noise to the image
    noisy_img = add_noise(img, noise_factor)
    st.image(noisy_img.squeeze(), caption="Noisy Image", use_column_width=True, clamp=True)

    # Make prediction
    noisy_img_exp = np.expand_dims(noisy_img, axis=0)
    denoised_img = model.predict(noisy_img_exp).squeeze()

    # Show the denoised image
    st.image(denoised_img, caption="Denoised Image", use_column_width=True, clamp=True)

    # Display side-by-side comparison
    st.subheader("Comparison")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(noisy_img.squeeze(), cmap='gray')
    axs[1].set_title("Noisy")
    axs[2].imshow(denoised_img, cmap='gray')
    axs[2].set_title("Denoised")
    for ax in axs:
        ax.axis('off')
    st.pyplot(fig)
else:
    st.write("Please upload an image to get started!")

# Footer styling
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: black;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
<div class="footer">
    <p>Developed by Muhammad Yousaf Rana | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)
