import streamlit as st
import openai
from openai import OpenAI
import io
import logging
import time
from PIL import Image
import requests

# ------------------------------------------------------------------------------
# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Set OpenAI API Key (ensure it's in your .streamlit/secrets.toml or environment variables)
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found in secrets. Please add 'OPENAI_API_KEY' to your secrets.")
    st.stop()

# ------------------------------------------------------------------------------
# Instantiate the OpenAI client.
try:
    client = OpenAI()
except Exception as e:
    logger.error("Failed to instantiate OpenAI client: %s", e)
    st.error("Failed to instantiate OpenAI client. Please check your configuration.")
    st.stop()

# ------------------------------------------------------------------------------
# Generate a marketing ad image using OpenAI's Text-to-Image API (DALL‑E‑3).
def generate_marketing_ad_openai(prompt: str, size: str = "1024x1024") -> str:
    """
    Calls OpenAI's image generation endpoint (using DALL‑E‑3) to produce an image based on the prompt.
    Returns the URL of the generated image.
    """
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        logger.info("Successfully generated marketing image: %s", image_url)
        return image_url
    except Exception as e:
        logger.error("Error generating image: %s", e)
        st.error("Failed to generate image. Please try again.")
        return None

# ------------------------------------------------------------------------------
# Create image variations using OpenAI's DALL-E 2 variation endpoint.
def create_image_variations_openai(input_image: Image.Image, n: int, size: str = "1024x1024") -> list:
    """
    Converts the input image into PNG bytes and sends it to OpenAI's image variation endpoint.
    Returns a list of URLs for the generated variations.
    """
    try:
        # Convert the input image into a PNG bytes buffer.
        img_buffer = io.BytesIO()
        input_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        response = client.images.create_variation(
            model="dall-e-2",
            image=img_buffer,
            n=n,
            size=size,
        )
        urls = [item.url for item in response.data]
        logger.info("Successfully created %s image variation(s): %s", n, urls)
        return urls
    except Exception as e:
        logger.error("Error creating image variations: %s", e)
        st.error("Failed to create image variations. Please try again.")
        return []

# ------------------------------------------------------------------------------
# Main application function.
def main():
    st.title("Image Generation & Variation with OpenAI API")
    
    mode = st.selectbox("Select Mode", [
        "Generate Marketing Ad & Variations"
    ])

    if mode == "Generate Marketing Ad & Variations":
        st.header("Marketing Ad Generator")
        prompt = st.text_input(
            "Marketing Prompt",
            value="Introducing our new summer collection, vibrant, modern, eye-catching"
        )
        size = st.selectbox(
            "Image Size",
            ["256x256", "512x512", "1024x1024"],
            index=2  # default to 1024x1024
        )
        
        # Generate the marketing ad image.
        if st.button("Generate Marketing Ad"):
            with st.spinner("Generating marketing ad via OpenAI..."):
                image_url = generate_marketing_ad_openai(prompt, size)
                if image_url:
                    st.image(image_url, caption="Generated Marketing Ad", use_column_width=True)
                    st.session_state.generated_image_url = image_url
        
        # If an image was generated, offer to generate variations.
        if "generated_image_url" in st.session_state:
            st.subheader("Generate Variations of the Generated Ad")
            n_variations = st.number_input("Number of Variations", min_value=1, max_value=8, value=1, step=1)
            if st.button("Generate Variations"):
                with st.spinner("Generating image variations via OpenAI..."):
                    # Download the generated image from its URL.
                    try:
                        response = requests.get(st.session_state.generated_image_url)
                        base_image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    except Exception as e:
                        logger.error("Error downloading generated image: %s", e)
                        st.error("Failed to download the generated image for variations.")
                        return
                    # Create variations.
                    variation_urls = create_image_variations_openai(base_image, n_variations, size)
                    if variation_urls:
                        st.markdown("### Variations")
                        for idx, url in enumerate(variation_urls):
                            st.image(url, caption=f"Variation {idx+1}", use_column_width=True)

if __name__ == "__main__":
    main()
