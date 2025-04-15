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

# Folder to store generated images and videos
IMAGE_DIR = "generated_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Load environment variables from .env or Streamlit secrets.
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
# Helper to get file data.
def get_file_data(file_obj):
    try:
        return file_obj.getvalue()
    except AttributeError:
        return file_obj.read()

# ------------------------------------------------------------------------------
# REST Request Helpers
def send_generation_request(host, params, files=None):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }
    if files is None:
        files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != "":
        if isinstance(image, str):
            files["image"] = open(image, "rb")
        else:
            files["image"] = image
    if mask is not None and mask != "":
        if isinstance(mask, str):
            files["mask"] = open(mask, "rb")
        else:
            files["mask"] = mask
    if len(files) == 0:
        files["none"] = ""
    logger.info(f"Sending request to Stability AI: {host}")
    response = requests.post(host, headers=headers, files=files, data=params)
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    return response

def send_async_generation_request(host, params, files=None):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }
    if files is None:
        files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != "":
        if hasattr(image, "getvalue"):
            files["image"] = image
        else:
            files["image"] = image
    if mask is not None and mask != "":
        files["mask"] = mask
    if len(files) == 0:
        files["none"] = ""
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
# Generation Functions (for image tasks) with Unique Filenames
def generate_marketing_ad_stability(prompt: str, negative_prompt: str, aspect_ratio: str, seed: int,
                                      output_format: str, size: str="1024x1024") -> str:
    host = "https://api.stability.ai/v2beta/stable-image/generate/core"
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
        timestamp = int(time.time() * 1000)
        filename = os.path.join(IMAGE_DIR, f"stability_generated_{seed}_{timestamp}.{output_format}")
        with open(filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Marketing ad generated and saved as {filename}")
        return filename
    except Exception as e:
        logger.error("Marketing ad generation error: %s", e)
        st.error("Failed to generate marketing ad with Stability AI.")
        return None

def generate_control_sketch_stability(prompt: str, negative_prompt: str, control_strength: float,
                                        seed: int, output_format: str, sketch_file) -> str:
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
        timestamp = int(time.time() * 1000)
        result_filename = os.path.join(IMAGE_DIR, f"control_sketch_{seed}_{timestamp}.{output_format}")
        with open(result_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Control Sketch result saved as {result_filename}")
        return result_filename
    except Exception as e:
        logger.error("Control Sketch error: %s", e)
        st.error("Failed to generate Control Sketch image.")
        return None

def generate_control_structure_stability(prompt: str, negative_prompt: str, control_strength: float,
                                           seed: int, output_format: str, structure_file) -> str:
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
        timestamp = int(time.time() * 1000)
        result_filename = os.path.join(IMAGE_DIR, f"control_structure_{seed}_{timestamp}.{output_format}")
        with open(result_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Control Structure result saved as {result_filename}")
        return result_filename
    except Exception as e:
        logger.error("Control Structure error: %s", e)
        st.error("Failed to generate Control Structure image.")
        return None

def generate_search_and_recolor(image_file, prompt: str, select_prompt: str, negative_prompt: str,
                                  grow_mask: int, seed: int, output_format: str) -> str:
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
        timestamp = int(time.time() * 1000)
        edited_filename = os.path.join(IMAGE_DIR,
            f"edited_searchrecolor_{os.path.splitext(os.path.basename(temp_filename))[0]}_{new_seed}_{timestamp}.{output_format}")
        with open(edited_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Search and Recolor result saved as {edited_filename}")
        return edited_filename
    except Exception as e:
        logger.error("Search and Recolor error: %s", e)
        st.error("Failed to generate search and recolor image.")
        return None

def generate_search_and_replace(image_file, prompt: str, search_prompt: str, negative_prompt: str,
                                seed: int, output_format: str) -> str:
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
        timestamp = int(time.time() * 1000)
        edited_filename = os.path.join(IMAGE_DIR,
            f"edited_searchreplace_{os.path.splitext(os.path.basename(temp_filename))[0]}_{new_seed}_{timestamp}.{output_format}")
        with open(edited_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Search and Replace result saved as {edited_filename}")
        return edited_filename
    except Exception as e:
        logger.error("Search and Replace error: %s", e)
        st.error("Failed to generate search and replace image.")
        return None

def generate_replace_background_and_religh(subject_image_file, background_prompt: str, background_reference_file,
                                             foreground_prompt: str, negative_prompt: str, preserve_original_subject: float,
                                             original_background_depth: float, keep_original_background: bool, light_source_strength: float,
                                             light_reference_file, light_source_direction: str, seed: int,
                                             output_format: str) -> str:
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
        timestamp = int(time.time() * 1000)
        base_name = os.path.splitext(os.path.basename(subject_filename))[0]
        edited_filename = os.path.join(IMAGE_DIR, f"edited_replacebg_{base_name}_{new_seed}_{timestamp}.{output_format}")
        with open(edited_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Replace Background result saved as {edited_filename}")
        return edited_filename
    except Exception as e:
        logger.error("Replace Background and Relight error: %s", e)
        st.error("Failed to generate replace background and relight image.")
        return None

def generate_upscale_creative(prompt: str, negative_prompt: str, creativity: float, seed: int,
                              output_format: str, up_image_file) -> str:
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
        timestamp = int(time.time() * 1000)
        filename = os.path.join(IMAGE_DIR, f"upscaled_creative_{seed}_{timestamp}.{output_format}")
        with open(filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Upscaled creative image saved as {filename}")
        return filename
    except Exception as e:
        logger.error("Upscale Creative error: %s", e)
        st.error("Failed to upscale image creatively.")
        return None

# ------------------------------------------------------------------------------
# New: Image-to-Video Generation Function with Resizing to 768x768
def generate_image_to_video(image_file, seed: int, cfg_scale: float, motion_bucket_id: int) -> str:
    """
    Resizes the uploaded image to 768x768, sends a POST request to the Stability AI
    image-to-video endpoint, polls for the video result, saves it locally, and returns the filename.
    """
    host = "https://api.stability.ai/v2beta/image-to-video"
    try:
        # Resize the image to 768x768 using PIL.
        pil_image = Image.open(image_file)
        resized_image = pil_image.resize((768, 768))
        img_buffer = io.BytesIO()
        resized_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        # Initiate video generation with the resized image.
        response = requests.post(
            host,
            headers={"authorization": f"Bearer {STABILITY_KEY}"},
            files={"image": img_buffer},
            data={
                "seed": seed,
                "cfg_scale": cfg_scale,
                "motion_bucket_id": motion_bucket_id
            }
        )
        generation_id = response.json().get("id")
        if not generation_id:
            st.error("Failed to initiate video generation; no generation id returned.")
            return None
        
        # API expects an ID of exactly 64 characters. If not, alert the user.
        if len(generation_id) != 64:
            st.error(f"Generation id '{generation_id}' is not 64 characters long (length: {len(generation_id)}).")
            return None
        
        st.info("Generation ID: " + generation_id)
        poll_url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
        while True:
            poll_response = requests.get(
                poll_url,
                headers={
                    "accept": "video/*",
                    "authorization": f"Bearer {STABILITY_KEY}"
                }
            )
            if poll_response.status_code == 202:
                st.info("Generation in-progress, waiting 10 seconds...")
                time.sleep(10)
            elif poll_response.status_code == 200:
                st.info("Video generation complete!")
                break
            else:
                raise Exception(str(poll_response.json()))
        video_content = poll_response.content
        timestamp = int(time.time() * 1000)
        video_filename = os.path.join(IMAGE_DIR, f"image_to_video_{seed}_{timestamp}.mp4")
        with open(video_filename, "wb") as f:
            f.write(video_content)
        return video_filename
    except Exception as e:
        st.error("Error during video generation: " + str(e))
        return None

# ------------------------------------------------------------------------------
# Editable Overlay HTML Generation Function (for images)
def get_editable_overlay_html(image_path: str, overlay_text: str, font_size: int, font_color: str,
                              font_type: str, font_weight: str, font_style: str, border_weight: int,
                              border_color: str, output_format: str) -> str:
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        if border_weight == 0:
            border_css = "none"
        else:
            border_css = f"{border_weight}px solid {border_color}"
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
              font-size: {font_size}px;
              font-family: {font_type};
              font-weight: {font_weight};
              font-style: {font_style};
              color: {font_color};
              text-shadow: 2px 2px 4px #000000;
              background: transparent;
              resize: both;
              overflow: auto;
              padding: 5px;
              cursor: move;
              border: {border_css};
            }}
          </style>
        </head>
        <body>
          <div class="container" id="container">
            <img class="overlay-image" src="data:image/{output_format};base64,{img_b64}">
            <div class="editable-text" id="editable-text" contenteditable="false">{overlay_text}</div>
          </div>
          <button id="download-btn" style="margin-top:10px;padding:10px 20px;background-color:#0073b1;color:#fff;border:none;border-radius:5px;cursor:pointer;">
            Download Overlayed Image
          </button>
          <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
          <script>
            const editableText = document.getElementById('editable-text');
            let isDragging = false;
            editableText.addEventListener('mousedown', function(e) {{
                isDragging = true;
                let shiftX = e.clientX - editableText.getBoundingClientRect().left;
                let shiftY = e.clientY - editableText.getBoundingClientRect().top;
                function onMouseMove(e) {{
                    if (isDragging) {{
                        editableText.style.left = (e.pageX - shiftX) + 'px';
                        editableText.style.top = (e.pageY - shiftY) + 'px';
                    }}
                }}
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', function() {{
                    isDragging = false;
                    document.removeEventListener('mousemove', onMouseMove);
                }}, {{ once: true }});
            }});
            editableText.ondragstart = function() {{
                return false;
            }};
            editableText.addEventListener('dblclick', function(e) {{
                if (editableText.contentEditable === "false") {{
                    editableText.contentEditable = "true";
                    editableText.style.border = "1px dashed #fff";
                }} else {{
                    editableText.contentEditable = "false";
                    editableText.style.border = "{border_css}";
                }}
            }});
            document.getElementById('download-btn').addEventListener('click', function() {{
                html2canvas(document.getElementById('container')).then(canvas => {{
                    var link = document.createElement('a');
                    link.download = 'overlay_result.png';
                    link.href = canvas.toDataURL();
                    link.click();
                }});
            }});
          </script>
        </body>
        </html>
        """
        return html_code
    except Exception as e:
        st.error("Failed to generate editable overlay HTML.")
        logger.error("Editable overlay HTML error: %s", e)
        return ""

# ------------------------------------------------------------------------------
# Main Application UI with Left Sidebar & Right Pane (Recent Media in an Expander)
def main():
    # Inject CSS for a scrollable container within the Recent Media expander.
    st.markdown("""
    <style>
      [data-testid="stExpander"] > div > div {
         max-height: 300px;
         overflow-y: auto;
      }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state containers if they do not exist.
    if "recent_images" not in st.session_state:
        st.session_state.recent_images = sorted(
            [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))],
            key=lambda x: os.path.getmtime(x),
            reverse=True
        )
    if "recent_videos" not in st.session_state:
        st.session_state.recent_videos = sorted(
            [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
             if f.lower().endswith(".mp4")],
            key=lambda x: os.path.getmtime(x),
            reverse=True
        )
    if "selected_video" not in st.session_state:
        st.session_state.selected_video = None

    # Left Sidebar: Settings and Controls
    with st.sidebar:
        st.title("Marketing Content Generator")
        st.header("Settings")
        task_type = st.selectbox("Select Task Type:", options=[
            "Marketing Ad",
            "Control Sketch",
            "Control Structure",
            "Search and Recolor",
            "Search and Replace",
            "Replace Background and Relight",
            "Upscale Creative",
            "Image to Video"
        ])
        st.markdown("#### Common Inputs")
        common_prompt = st.text_input("Prompt", "Introducing our new summer collection, vibrant, modern, eye-catching")
        seed = st.number_input("Seed", value=0, step=1)
        output_format = st.selectbox("Output Format", ["jpeg", "png", "webp"], index=0)
        
        st.markdown("#### Editable Overlay Settings")
        overlay_text_input = st.text_input("Overlay Text", "Your Ad Slogan Here")
        font_type = st.selectbox("Font Type", options=["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia"], index=0)
        font_weight = st.selectbox("Font Weight", options=["normal", "bold"], index=0)
        font_style = st.selectbox("Font Style", options=["normal", "italic", "bold italic"], index=0)
        font_size = st.number_input("Font Size", min_value=10, max_value=200, value=40, step=1)
        font_color = st.color_picker("Font Color", value="#FFFFFF")
        border_weight = st.number_input("Border Weight", min_value=0, max_value=10, value=0, step=1)
        border_color = st.color_picker("Border Color", value="#000000")
        st.session_state.overlay_text = overlay_text_input
        st.session_state.font_type = font_type
        st.session_state.font_weight = font_weight
        st.session_state.font_style = font_style
        st.session_state.font_size = font_size
        st.session_state.font_color = font_color
        st.session_state.border_weight = border_weight
        st.session_state.border_color = border_color
        st.session_state.editable_overlay_js = True

        # Task-Specific Inputs
        if task_type == "Marketing Ad":
            negative_prompt = st.text_input("Negative Prompt", "")
            aspect_ratio = st.selectbox("Aspect Ratio", ["21:9", "16:9", "3:2", "5:4", "1:1"], index=2)
            size = st.selectbox("Image Size", ["256x256", "512x512", "1024x1024"], index=2)
            marketing_new_prompt = st.text_input("Marketing Ad Prompt", common_prompt, key="marketing_new_prompt")
            if st.button("Generate Marketing Ad"):
                with st.spinner("Generating marketing ad..."):
                    filename = generate_marketing_ad_stability(marketing_new_prompt, negative_prompt, aspect_ratio, seed, output_format, size)
                    if filename:
                        st.session_state.generated_image = filename
                        st.session_state.recent_images.append(filename)

        elif task_type == "Control Sketch":
            control_strength = st.slider("Control Strength", 0.0, 1.0, 0.7, 0.05)
            negative_prompt = st.text_input("Negative Prompt", "")
            sketch_new_prompt = st.text_input("Sketch Prompt", common_prompt, key="sketch_new_prompt")
            if "generated_image" in st.session_state and st.session_state.generated_image:
                st.info("Using selected recent image: " + os.path.basename(st.session_state.generated_image))
                sketch_file = open(st.session_state.generated_image, "rb")
            else:
                sketch_file = st.file_uploader("Upload Sketch Image", type=["jpg", "jpeg", "png"])
            if st.button("Generate Control Sketch"):
                if not sketch_file:
                    st.error("Please upload a sketch image.")
                else:
                    with st.spinner("Generating Control Sketch..."):
                        filename = generate_control_sketch_stability(sketch_new_prompt, negative_prompt, control_strength, seed, output_format, sketch_file)
                        if filename:
                            st.session_state.generated_image = filename
                            st.session_state.recent_images.append(filename)

        elif task_type == "Control Structure":
            control_strength = st.slider("Control Strength", 0.0, 1.0, 0.7, 0.05)
            negative_prompt = st.text_input("Negative Prompt", "")
            structure_new_prompt = st.text_input("Structure Prompt", common_prompt, key="structure_new_prompt")
            if "generated_image" in st.session_state and st.session_state.generated_image:
                st.info("Using selected recent image: " + os.path.basename(st.session_state.generated_image))
                structure_file = open(st.session_state.generated_image, "rb")
            else:
                structure_file = st.file_uploader("Upload Structure Image", type=["jpg", "jpeg", "png"])
            if st.button("Generate Control Structure"):
                if not structure_file:
                    st.error("Please upload a structure image.")
                else:
                    with st.spinner("Generating Control Structure..."):
                        filename = generate_control_structure_stability(structure_new_prompt, negative_prompt, control_strength, seed, output_format, structure_file)
                        if filename:
                            st.session_state.generated_image = filename
                            st.session_state.recent_images.append(filename)

        elif task_type == "Search and Recolor":
            select_prompt = st.text_input("Select Prompt", "chicken")
            negative_prompt = st.text_input("Negative Prompt", "")
            recolor_new_prompt = st.text_input("Recolor Prompt", common_prompt, key="recolor_new_prompt")
            if "generated_image" in st.session_state and st.session_state.generated_image:
                st.info("Using selected recent image: " + os.path.basename(st.session_state.generated_image))
                image_file = open(st.session_state.generated_image, "rb")
            else:
                image_file = st.file_uploader("Upload Image for Recolor", type=["jpg", "jpeg", "png"])
            grow_mask = st.number_input("Grow Mask", min_value=1, value=3, step=1)
            if st.button("Generate Search and Recolor"):
                if not image_file:
                    st.error("Please upload an image for search and recolor.")
                else:
                    with st.spinner("Generating Search and Recolor..."):
                        filename = generate_search_and_recolor(image_file, recolor_new_prompt, select_prompt, negative_prompt, grow_mask, seed, output_format)
                        if filename:
                            st.session_state.generated_image = filename
                            st.session_state.recent_images.append(filename)

        elif task_type == "Search and Replace":
            search_prompt = st.text_input("Search Prompt", "chicken")
            negative_prompt = st.text_input("Negative Prompt", "")
            replacement_prompt = st.text_input("Replacement Prompt", common_prompt, key="replacement_new_prompt")
            if "generated_image" in st.session_state and st.session_state.generated_image:
                st.info("Using selected recent image: " + os.path.basename(st.session_state.generated_image))
                image_file = open(st.session_state.generated_image, "rb")
            else:
                image_file = st.file_uploader("Upload Image for Search and Replace", type=["jpg", "jpeg", "png"])
            if st.button("Generate Search and Replace"):
                if not image_file:
                    st.error("Please upload an image for search and replace.")
                else:
                    with st.spinner("Generating Search and Replace..."):
                        filename = generate_search_and_replace(image_file, replacement_prompt, search_prompt, negative_prompt, seed, output_format)
                        if filename:
                            st.session_state.generated_image = filename
                            st.session_state.recent_images.append(filename)

        elif task_type == "Replace Background and Relight":
            background_prompt = st.text_input("Background Prompt", "pastel landscape")
            foreground_prompt = st.text_input("Foreground Prompt", "")
            negative_prompt = st.text_input("Negative Prompt", "")
            preserve_original_subject = st.slider("Preserve Original Subject", 0.0, 1.0, 0.6, 0.05)
            original_background_depth = st.slider("Original Background Depth", 0.0, 1.0, 0.5, 0.05)
            keep_original_background = st.checkbox("Keep Original Background")
            light_source_direction = st.selectbox("Light Source Direction", ["none", "left", "right", "above", "below"], index=0)
            light_source_strength = st.slider("Light Source Strength", 0.0, 1.0, 0.3, 0.05) if light_source_direction != "none" else None
            relight_new_prompt = st.text_input("Relight Prompt", common_prompt, key="relight_new_prompt")
            if "generated_image" in st.session_state and st.session_state.generated_image:
                st.info("Using selected recent image: " + os.path.basename(st.session_state.generated_image))
                subject_image_file = open(st.session_state.generated_image, "rb")
            else:
                subject_image_file = st.file_uploader("Upload Subject Image", type=["jpg", "jpeg", "png"])
            background_reference_file = st.file_uploader("Upload Background Reference (Optional)", type=["jpg", "jpeg", "png"])
            light_reference_file = st.file_uploader("Upload Light Reference (Optional)", type=["jpg", "jpeg", "png"])
            if st.button("Generate Replace Background and Relight"):
                if not subject_image_file:
                    st.error("Please upload a subject image.")
                else:
                    with st.spinner("Generating Replace Background and Relight..."):
                        filename = generate_replace_background_and_religh(
                            subject_image_file, relight_new_prompt, background_reference_file,
                            foreground_prompt, negative_prompt, preserve_original_subject,
                            original_background_depth, keep_original_background,
                            light_source_strength if light_source_strength is not None else 0,
                            light_reference_file, light_source_direction, seed, output_format
                        )
                        if filename:
                            st.session_state.generated_image = filename
                            st.session_state.recent_images.append(filename)

        elif task_type == "Upscale Creative":
            creativity = st.number_input("Creativity", 0.0, 1.0, 0.30, 0.01)
            negative_prompt = st.text_input("Negative Prompt", "")
            upscaling_new_prompt = st.text_input("Upscale Prompt", common_prompt, key="upscale_new_prompt")
            if "generated_image" in st.session_state and st.session_state.generated_image:
                st.info("Using selected recent image: " + os.path.basename(st.session_state.generated_image))
                up_image_file = open(st.session_state.generated_image, "rb")
            else:
                up_image_file = st.file_uploader("Upload Image to Upscale", type=["jpg", "jpeg", "png"])
            if st.button("Generate Upscale Creative"):
                if not up_image_file:
                    st.error("Please upload an image to upscale.")
                else:
                    with st.spinner("Generating Upscale Creative..."):
                        filename = generate_upscale_creative(upscaling_new_prompt, negative_prompt, creativity, seed, output_format, up_image_file)
                        if filename:
                            st.session_state.generated_image = filename
                            st.session_state.recent_images.append(filename)

        elif task_type == "Image to Video":
            cfg_scale = st.number_input("CFG Scale", value=1.8, step=0.1, format="%.1f")
            motion_bucket_id = st.number_input("Motion Bucket ID", value=127, step=1)
            if "generated_image" in st.session_state and st.session_state.generated_image:
                st.info("Using selected recent image: " + os.path.basename(st.session_state.generated_image))
                video_image_file = open(st.session_state.generated_image, "rb")
            else:
                video_image_file = st.file_uploader("Upload Image for Video", type=["jpg", "jpeg", "png"])
            if st.button("Generate Image to Video"):
                if not video_image_file:
                    st.error("Please upload an image for video generation.")
                else:
                    with st.spinner("Generating video... This may take some time."):
                        video_filename = generate_image_to_video(video_image_file, seed, cfg_scale, motion_bucket_id)
                        if video_filename:
                            st.session_state.generated_video = video_filename
                            st.session_state.recent_videos.append(video_filename)
                            st.success("Video generation complete!")

    # Main Layout: Two Columns (Left: Output; Right: Recent Media Expander)
    col_left, col_right = st.columns([3, 1])
    
    
    
    with col_right:
        with st.expander("Recent Images", expanded=True):
            
            selected_images = []
            if st.session_state.recent_images:
                for idx, img_path in enumerate(reversed(st.session_state.recent_images)):
                    st.image(img_path, use_column_width=True)
                    if st.checkbox("Select this image", key=f"recent_checkbox_{idx}"):
                        selected_images.append(img_path)
            if st.button("Apply Selected Image"):
                if len(selected_images) == 0:
                    st.error("No image selected. Please select one image.")
                elif len(selected_images) > 1:
                    st.error("Multiple images selected. Please select only one.")
                else:
                    st.session_state.generated_image = selected_images[0]
                    try:
                        with open(st.session_state.generated_image, "rb") as f:
                            image_bytes = f.read()
                        st.download_button(label="Download Selected Image", data=image_bytes,
                                        file_name=os.path.basename(st.session_state.generated_image), mime="image/png")
                    except Exception as e:
                        st.error("Unable to prepare download for the generated image.")
                    st.success("Selected image applied.")
            
        with st.expander("Recent videos", expanded=True):
            if st.session_state.recent_videos:
                for idx, video_path in enumerate(reversed(st.session_state.recent_videos)):
                    st.image(st.session_state.generated_image, use_column_width=True)
                    st.write(os.path.basename(video_path))
                    if st.button("Show Video", key=f"show_video_{idx}"):
                        st.session_state.selected_video = video_path
                        
    with col_left:
        if "generated_image" in st.session_state:
            output_img_path = st.session_state.generated_image
            try:
                html_code = get_editable_overlay_html(
                    output_img_path,
                    st.session_state.overlay_text,
                    st.session_state.font_size,
                    st.session_state.font_color,
                    st.session_state.font_type,
                    st.session_state.font_weight,
                    st.session_state.font_style,
                    st.session_state.border_weight,
                    st.session_state.border_color,
                    output_format
                )
                components.html(html_code, height=400)
            except Exception as e:
                st.error("Editable overlay failed, please check.")
                logger.error("Editable overlay error: %s", e)
            
    # Modal window for selected video
        if st.session_state.selected_video:
            
            st.video(st.session_state.selected_video)
            try:
                with open(st.session_state.selected_video, "rb") as vf:
                    video_bytes = vf.read()
                st.download_button("Download Video", video_bytes,
                                    file_name=os.path.basename(st.session_state.selected_video), mime="video/mp4")
            except Exception as e:
                st.error("Download failed: " + str(e))
            st.session_state.selected_video = None

if __name__ == "__main__":
    main()
