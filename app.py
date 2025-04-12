import streamlit as st
import io
import logging
import time
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import json
from dotenv import load_dotenv
import base64
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas

# Load environment variables from .env (if present) or from Streamlit secrets.
load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_KEY") or (st.secrets.get("STABILITY_KEY") if st.secrets else None)

if not STABILITY_KEY:
    st.error("Stability AI key not found. Please set it in your environment or in .streamlit/secrets.toml.")
    st.stop()

# ------------------------------------------------------------------------------
# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Helper: Get file data regardless of type.
def get_file_data(file_obj):
    try:
        return file_obj.getvalue()
    except AttributeError:
        return file_obj.read()

# ------------------------------------------------------------------------------
# REST Request Helpers
def send_generation_request(host, params, files=None):
    """
    Synchronously sends a REST request to a Stability AI endpoint.
    """
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }
    if files is None:
        files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        if isinstance(image, str):
            files["image"] = open(image, 'rb')
        else:
            files["image"] = image
    if mask is not None and mask != '':
        if isinstance(mask, str):
            files["mask"] = open(mask, 'rb')
        else:
            files["mask"] = mask
    if len(files) == 0:
        files["none"] = ''
    logger.info(f"Sending request to Stability AI: {host}")
    response = requests.post(host, headers=headers, files=files, data=params)
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    return response

def send_async_generation_request(host, params, files=None):
    """
    Sends an asynchronous REST request to a Stability AI endpoint and polls for the final result.
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }
    if files is None:
        files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        if hasattr(image, "getvalue"):
            files["image"] = image
        else:
            files["image"] = image
    if mask is not None and mask != '':
        files["mask"] = mask
    if len(files) == 0:
        files["none"] = ''
    st.write(f"Sending async request to {host}...")
    response = requests.post(host, headers=headers, files=files, data=params)
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    response_dict = response.json()
    generation_id = response_dict.get("id", None)
    if generation_id is None:
        raise Exception("Expected id in async response")
    timeout = int(os.getenv("WORKER_TIMEOUT", 500))
    start = time.time()
    while True:
        poll_response = requests.get(
            f"https://api.stability.ai/v2beta/results/{generation_id}",
            headers={**headers, "Accept": "*/*"}
        )
        if not poll_response.ok:
            raise Exception(f"HTTP {poll_response.status_code}: {poll_response.text}")
        if poll_response.status_code != 202:
            break
        time.sleep(10)
        if time.time() - start > timeout:
            raise Exception(f"Timeout after {timeout} seconds")
    return poll_response

# ------------------------------------------------------------------------------
# Feature Functions (Stability AI endpoints)

def generate_marketing_ad_stability(prompt: str, negative_prompt: str, aspect_ratio: str, seed: int, output_format: str, size: str="1024x1024") -> str:
    """Generate a marketing ad using the 'ultra' endpoint."""
    host = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "output_format": output_format,
        "size": size
    }
    try:
        response = send_generation_request(host, params)
        image_bytes = response.content
        filename = f"stability_generated_{seed}.{output_format}"
        with open(filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Marketing ad generated and saved as {filename}")
        return filename
    except Exception as e:
        logger.error("Marketing ad generation error: %s", e)
        st.error("Failed to generate marketing ad with Stability AI.")
        return None

def generate_control_sketch_stability(prompt: str, negative_prompt: str, control_strength: float, seed: int, output_format: str, sketch_file) -> str:
    """Generate image using the control/sketch endpoint."""
    host = "https://api.stability.ai/v2beta/stable-image/control/sketch"
    params = {
        "control_strength": control_strength,
        "seed": seed,
        "output_format": output_format,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": "temp_sketch.png"
    }
    try:
        sketch_bytes = get_file_data(sketch_file)
        temp_filename = f"temp_sketch_{seed}.png"
        with open(temp_filename, "wb") as f:
            f.write(sketch_bytes)
        params["image"] = temp_filename
        response = send_generation_request(host, params)
        image_bytes = response.content
        result_filename = f"control_sketch_{seed}.{output_format}"
        with open(result_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Control Sketch result saved as {result_filename}")
        return result_filename
    except Exception as e:
        logger.error("Control Sketch error: %s", e)
        st.error("Failed to generate Control Sketch image.")
        return None

def generate_control_structure_stability(prompt: str, negative_prompt: str, control_strength: float, seed: int, output_format: str, structure_file) -> str:
    """Generate image using the control/structure endpoint."""
    host = "https://api.stability.ai/v2beta/stable-image/control/structure"
    params = {
        "control_strength": control_strength,
        "seed": seed,
        "output_format": output_format,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": "temp_structure.png"
    }
    try:
        structure_bytes = get_file_data(structure_file)
        temp_filename = f"temp_structure_{seed}.png"
        with open(temp_filename, "wb") as f:
            f.write(structure_bytes)
        params["image"] = temp_filename
        response = send_generation_request(host, params)
        image_bytes = response.content
        result_filename = f"control_structure_{seed}.{output_format}"
        with open(result_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Control Structure result saved as {result_filename}")
        return result_filename
    except Exception as e:
        logger.error("Control Structure error: %s", e)
        st.error("Failed to generate Control Structure image.")
        return None

def generate_search_and_recolor(image_file, prompt: str, select_prompt: str, negative_prompt: str, grow_mask: int, seed: int, output_format: str) -> str:
    """Generate image using the search-and-recolor endpoint."""
    host = "https://api.stability.ai/v2beta/stable-image/edit/search-and-recolor"
    temp_filename = f"temp_upload_{seed}.png"
    image_bytes = get_file_data(image_file)
    with open(temp_filename, "wb") as f:
        f.write(image_bytes)
    params = {
        "grow_mask": grow_mask,
        "seed": seed,
        "mode": "search",
        "output_format": output_format,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "select_prompt": select_prompt,
        "image": temp_filename
    }
    try:
        response = send_generation_request(host, params)
        image_bytes = response.content
        new_seed = response.headers.get("seed", seed)
        edited_filename = f"edited_searchrecolor_{os.path.splitext(os.path.basename(temp_filename))[0]}_{new_seed}.{output_format}"
        with open(edited_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Search and Recolor result saved as {edited_filename}")
        return edited_filename
    except Exception as e:
        logger.error("Search and Recolor error: %s", e)
        st.error("Failed to generate search and recolor image.")
        return None

def generate_search_and_replace(image_file, prompt: str, search_prompt: str, negative_prompt: str, seed: int, output_format: str) -> str:
    """Generate image using the search-and-replace endpoint."""
    host = "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace"
    temp_filename = f"temp_upload_{seed}.png"
    image_bytes = get_file_data(image_file)
    with open(temp_filename, "wb") as f:
        f.write(image_bytes)
    params = {
        "seed": seed,
        "mode": "search",
        "output_format": output_format,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "search_prompt": search_prompt,
        "image": temp_filename
    }
    try:
        response = send_generation_request(host, params)
        image_bytes = response.content
        new_seed = response.headers.get("seed", seed)
        edited_filename = f"edited_searchreplace_{os.path.splitext(os.path.basename(temp_filename))[0]}_{new_seed}.{output_format}"
        with open(edited_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Search and Replace result saved as {edited_filename}")
        return edited_filename
    except Exception as e:
        logger.error("Search and Replace error: %s", e)
        st.error("Failed to generate search and replace image.")
        return None

def generate_replace_background_and_relight(subject_image_file, background_prompt: str, background_reference_file, foreground_prompt: str, negative_prompt: str, preserve_original_subject: float, original_background_depth: float, keep_original_background: bool, light_source_strength: float, light_reference_file, light_source_direction: str, seed: int, output_format: str) -> str:
    """Generate image using the replace-background-and-relight endpoint."""
    host = "https://api.stability.ai/v2beta/stable-image/edit/replace-background-and-relight"
    subject_bytes = get_file_data(subject_image_file)
    subject_filename = f"temp_subject_{seed}.png"
    with open(subject_filename, "wb") as f:
        f.write(subject_bytes)
    files = {"subject_image": open(subject_filename, "rb")}
    if background_reference_file:
        bg_bytes = get_file_data(background_reference_file)
        bg_filename = f"temp_bg_{seed}.png"
        with open(bg_filename, "wb") as f:
            f.write(bg_bytes)
        files["background_reference"] = open(bg_filename, "rb")
    if light_reference_file:
        light_bytes = get_file_data(light_reference_file)
        light_filename = f"temp_light_{seed}.png"
        with open(light_filename, "wb") as f:
            f.write(light_bytes)
        files["light_reference"] = open(light_filename, "rb")
    
    params = {
        "output_format": output_format,
        "background_prompt": background_prompt,
        "foreground_prompt": foreground_prompt,
        "negative_prompt": negative_prompt,
        "preserve_original_subject": preserve_original_subject,
        "original_background_depth": original_background_depth,
        "keep_original_background": keep_original_background,
        "seed": seed
    }
    if light_source_direction != "none":
        params["light_source_direction"] = light_source_direction
    if light_source_direction != "none" or (light_reference_file is not None):
        params["light_source_strength"] = light_source_strength
    try:
        response = send_async_generation_request(host, params, files=files)
        image_bytes = response.content
        new_seed = response.headers.get("seed", seed)
        base_name = os.path.splitext(os.path.basename(subject_filename))[0]
        edited_filename = f"edited_replacebg_{base_name}_{new_seed}.{output_format}"
        with open(edited_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Replace Background and Relight result saved as {edited_filename}")
        return edited_filename
    except Exception as e:
        logger.error("Replace Background and Relight error: %s", e)
        st.error("Failed to generate replace background and relight image.")
        return None

def generate_upscale_creative(prompt: str, negative_prompt: str, creativity: float, seed: int, output_format: str, up_image_file) -> str:
    """Generate an upscaled creative image asynchronously."""
    host = "https://api.stability.ai/v2beta/stable-image/upscale/creative"
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "creativity": creativity,
        "output_format": output_format
    }
    try:
        response = send_async_generation_request(host, {**params, "image": up_image_file})
        image_bytes = response.content
        filename = f"upscaled_creative_{seed}.{output_format}"
        with open(filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Upscaled creative image saved as {filename}")
        return filename
    except Exception as e:
        logger.error("Upscale Creative error: %s", e)
        st.error("Failed to upscale image creatively.")
        return None

# ------------------------------------------------------------------------------
# New: Function to overlay text onto an image using an editable text box via JS.
def editable_text_overlay(image_path: str, initial_text: str, font_size: int, font_color: str, output_format: str) -> None:
    """
    Renders an HTML component with the image in the background and a contentEditable
    div overlaid. The user can edit the text directly.
    """
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        # Create an HTML snippet that overlays an editable text box
        html_code = f"""
        <html>
        <head>
          <style>
            .container {{
              position: relative;
              text-align: center;
              color: {font_color};
            }}
            .overlay {{
              position: absolute;
              bottom: 20px;
              left: 50%;
              transform: translateX(-50%);
              font-size: {font_size}px;
              font-family: Arial, sans-serif;
              font-weight: bold;
              text-shadow: 2px 2px 4px #000000;
              background: transparent;
            }}
          </style>
        </head>
        <body>
          <div class="container">
            <img src="data:image/{output_format};base64,{img_b64}" style="width:100%;">
            <div class="overlay" contenteditable="true">{initial_text}</div>
          </div>
        </body>
        </html>
        """
        # Render the HTML code as a component.
        st.components.v1.html(html_code, height=600)
    except Exception as e:
        st.error("Editable overlay failed, please check.")
        logger.error("Editable overlay error: %s", e)

# ------------------------------------------------------------------------------
# Main Application UI with Sidebar Controls
def main():
    st.title("Marketing Content Generator")
    st.markdown("## Enjoy a spacious view of your creative outputs!")
    
    # Sidebar: Task type selection and inputs.
    with st.sidebar:
        st.header("Settings")
        task_type = st.selectbox("Select Task Type:", options=[
            "Marketing Ad",
            "Control Sketch",
            "Control Structure",
            "Search and Recolor",
            "Search and Replace",
            "Replace Background and Relight",
            "Upscale Creative"
        ])
        st.markdown("#### Common Inputs")
        prompt = st.text_input("Prompt", "Introducing our new summer collection, vibrant, modern, eye-catching")
        seed = st.number_input("Seed", value=0, step=1)
        output_format = st.selectbox("Output Format", ["jpeg", "png", "webp"], index=0)
        use_previous = st.checkbox("Use Previously Generated Image", value=False)
        
        # Overlay text settings (optional)
        st.markdown("#### Overlay Text Settings (Optional)")
        overlay_text_input = st.text_input("Overlay Text", "")
        if overlay_text_input:
            font_size = st.number_input("Font Size", min_value=10, max_value=200, value=40, step=1)
            font_color = st.color_picker("Font Color", value="#FFFFFF")
        else:
            font_size, font_color = None, None
        
        # Option to enable editable text overlay (JS)
        editable_overlay_js = st.checkbox("Editable Overlay (JS)", value=False)
        
        # Drawing overlay in sidebar
        st.markdown("#### Drawing Overlay (Optional)")
        drawing_mode = st.selectbox("Drawing Tool:", options=["freedraw", "line", "rect", "circle"], index=0)
        stroke_width = st.slider("Stroke Width", 1, 25, 3)
        stroke_color = st.color_picker("Stroke Color", "#ff0000")
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            height=150,
            drawing_mode=drawing_mode,
            key="canvas_overlay",
        )
        
        # Task-specific inputs:
        if task_type == "Marketing Ad":
            negative_prompt = st.text_input("Negative Prompt", "")
            aspect_ratio = st.selectbox("Aspect Ratio", ["21:9", "16:9", "3:2", "5:4", "1:1"], index=2)
            size = st.selectbox("Image Size", ["256x256", "512x512", "1024x1024"], index=2)
            if st.button("Generate Marketing Ad"):
                with st.spinner("Generating marketing ad..."):
                    filename = generate_marketing_ad_stability(prompt, negative_prompt, aspect_ratio, seed, output_format, size)
                    if filename:
                        st.session_state.generated_image = filename

        elif task_type == "Control Sketch":
            control_strength = st.slider("Control Strength", 0.0, 1.0, 0.7, 0.05)
            negative_prompt = st.text_input("Negative Prompt", "")
            if use_previous and "generated_image" in st.session_state:
                sketch_file = open(st.session_state.generated_image, "rb")
            else:
                sketch_file = st.file_uploader("Upload Sketch Image", type=["jpg", "jpeg", "png"])
            if st.button("Generate Control Sketch"):
                if not sketch_file:
                    st.error("Please provide a sketch image or use a previously generated image.")
                else:
                    with st.spinner("Generating Control Sketch..."):
                        filename = generate_control_sketch_stability(prompt, negative_prompt, control_strength, seed, output_format, sketch_file)
                        if filename:
                            st.session_state.generated_image = filename

        elif task_type == "Control Structure":
            control_strength = st.slider("Control Strength", 0.0, 1.0, 0.7, 0.05)
            negative_prompt = st.text_input("Negative Prompt", "")
            if use_previous and "generated_image" in st.session_state:
                structure_file = open(st.session_state.generated_image, "rb")
            else:
                structure_file = st.file_uploader("Upload Structure Image", type=["jpg", "jpeg", "png"])
            if st.button("Generate Control Structure"):
                if not structure_file:
                    st.error("Please provide a structure image or use a previously generated image.")
                else:
                    with st.spinner("Generating Control Structure..."):
                        filename = generate_control_structure_stability(prompt, negative_prompt, control_strength, seed, output_format, structure_file)
                        if filename:
                            st.session_state.generated_image = filename

        elif task_type == "Search and Recolor":
            select_prompt = st.text_input("Select Prompt", "chicken")
            negative_prompt = st.text_input("Negative Prompt", "")
            if use_previous and "generated_image" in st.session_state:
                image_file = open(st.session_state.generated_image, "rb")
            else:
                image_file = st.file_uploader("Upload Image for Recolor", type=["jpg", "jpeg", "png"])
            grow_mask = st.number_input("Grow Mask", min_value=1, value=3, step=1)
            if st.button("Generate Search and Recolor"):
                if not image_file:
                    st.error("Please provide an image for search and recolor.")
                else:
                    with st.spinner("Generating Search and Recolor..."):
                        filename = generate_search_and_recolor(image_file, prompt, select_prompt, negative_prompt, grow_mask, seed, output_format)
                        if filename:
                            st.session_state.generated_image = filename

        elif task_type == "Search and Replace":
            search_prompt = st.text_input("Search Prompt", "chicken")
            negative_prompt = st.text_input("Negative Prompt", "")
            if use_previous and "generated_image" in st.session_state:
                image_file = open(st.session_state.generated_image, "rb")
            else:
                image_file = st.file_uploader("Upload Image for Search and Replace", type=["jpg", "jpeg", "png"])
            if st.button("Generate Search and Replace"):
                if not image_file:
                    st.error("Please provide an image for search and replace.")
                else:
                    with st.spinner("Generating Search and Replace..."):
                        filename = generate_search_and_replace(image_file, prompt, search_prompt, negative_prompt, seed, output_format)
                        if filename:
                            st.session_state.generated_image = filename

        elif task_type == "Replace Background and Relight":
            background_prompt = st.text_input("Background Prompt", "pastel landscape")
            foreground_prompt = st.text_input("Foreground Prompt", "")
            negative_prompt = st.text_input("Negative Prompt", "")
            preserve_original_subject = st.slider("Preserve Original Subject", 0.0, 1.0, 0.6, 0.05)
            original_background_depth = st.slider("Original Background Depth", 0.0, 1.0, 0.5, 0.05)
            keep_original_background = st.checkbox("Keep Original Background")
            light_source_direction = st.selectbox("Light Source Direction", ["none", "left", "right", "above", "below"], index=0)
            light_source_strength = st.slider("Light Source Strength", 0.0, 1.0, 0.3, 0.05) if light_source_direction != "none" else None
            if use_previous and "generated_image" in st.session_state:
                subject_image_file = open(st.session_state.generated_image, "rb")
            else:
                subject_image_file = st.file_uploader("Upload Subject Image", type=["jpg", "jpeg", "png"])
            background_reference_file = st.file_uploader("Upload Background Reference (Optional)", type=["jpg", "jpeg", "png"])
            light_reference_file = st.file_uploader("Upload Light Reference (Optional)", type=["jpg", "jpeg", "png"])
            if st.button("Generate Replace Background and Relight"):
                if not subject_image_file:
                    st.error("Please provide a subject image for replace background and relight.")
                else:
                    with st.spinner("Generating Replace Background and Relight..."):
                        filename = generate_replace_background_and_relight(
                            subject_image_file, background_prompt, background_reference_file,
                            foreground_prompt, negative_prompt, preserve_original_subject,
                            original_background_depth, keep_original_background,
                            light_source_strength if light_source_strength is not None else 0,
                            light_reference_file, light_source_direction, seed, output_format
                        )
                        if filename:
                            st.session_state.generated_image = filename

        elif task_type == "Upscale Creative":
            creativity = st.number_input("Creativity", 0.0, 1.0, 0.30, 0.01)
            negative_prompt = st.text_input("Negative Prompt", "")
            if use_previous and "generated_image" in st.session_state:
                up_image_file = open(st.session_state.generated_image, "rb")
            else:
                up_image_file = st.file_uploader("Upload Image to Upscale", type=["jpg", "jpeg", "png"])
            if st.button("Generate Upscale Creative"):
                if not up_image_file:
                    st.error("Please provide an image to upscale.")
                else:
                    with st.spinner("Generating Upscale Creative..."):
                        filename = generate_upscale_creative(prompt, negative_prompt, creativity, seed, output_format, up_image_file)
                        if filename:
                            st.session_state.generated_image = filename

    # Main Output Display Area.
    st.markdown("## Output Image")
    if "generated_image" in st.session_state:
        output_img_path = st.session_state.generated_image

        # Apply overlay text (if provided) using our standard overlay function.
        if overlay_text and not editable_overlay_js:
            try:
                output_img_path = overlay_text_on_image(output_img_path, overlay_text, int(font_size), font_color, output_format)
            except Exception as e:
                st.error("Failed to apply text overlay.")
        # If editable overlay (JS) is selected, display an HTML component with editable text.
        if editable_overlay_js:
            try:
                # Convert the image to base64.
                with open(output_img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                img_b64 = base64.b64encode(img_bytes).decode()
                html_code = f"""
                <html>
                <head>
                  <style>
                    .container {{
                      position: relative;
                      width: 100%;
                    }}
                    .overlay-image {{
                      width: 100%;
                    }}
                    .editable-text {{
                      position: absolute;
                      bottom: 20px;
                      left: 50%;
                      transform: translateX(-50%);
                      font-size: {font_size if font_size else 40}px;
                      color: {font_color if font_color else "#FFFFFF"};
                      text-shadow: 2px 2px 4px #000000;
                      background-color: transparent;
                    }}
                  </style>
                </head>
                <body>
                  <div class="container">
                    <img class="overlay-image" src="data:image/{output_format};base64,{img_b64}">
                    <div class="editable-text" contenteditable="true">{overlay_text}</div>
                  </div>
                </body>
                </html>
                """
                st.components.v1.html(html_code, height=600)
            except Exception as e:
                st.error("Editable overlay failed, please check.")
                logger.error("Editable overlay error: %s", e)
        else:
            st.image(output_img_path, caption="Result", use_column_width=True)
        
        try:
            with open(output_img_path, "rb") as f:
                image_bytes = f.read()
            st.download_button(label="Download Result Image", data=image_bytes, file_name=os.path.basename(output_img_path), mime="image/png")
        except Exception as e:
            st.error("Unable to prepare download for the result image.")

if __name__ == "__main__":
    main()
