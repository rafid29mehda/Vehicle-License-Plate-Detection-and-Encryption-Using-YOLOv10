import streamlit as st
import os
import requests
import numpy as np
from PIL import Image, ImageDraw
import io
import random

##############################################################################
# 1. Attempt to import YOLOv10 from the ultralytics package (THU-MIG/yolov10)
##############################################################################
try:
    from ultralytics import YOLOv10
except ImportError:
    st.error("Could not import YOLOv10. Please confirm THU-MIG/yolov10 installation in requirements.txt.")
    st.stop()

##############################################################################
# 2. Chaotic Logistic Map Encryption Functions
##############################################################################
def logistic_map(r, x):
    return r * x * (1 - x)

def generate_key(seed, n):
    key = []
    x = seed
    for _ in range(n):
        x = logistic_map(3.9, x)
        key.append(int(x * 255) % 256)
    return np.array(key, dtype=np.uint8)

def shuffle_pixels(img_array, seed):
    h, w, c = img_array.shape
    num_pixels = h * w
    flattened = img_array.reshape(-1, c)
    indices = np.arange(num_pixels)

    random.seed(seed)
    random.shuffle(indices)

    shuffled = flattened[indices]
    return shuffled.reshape(h, w, c), indices

def encrypt_image(img_array, seed):
    """
    Encrypt the given image array using a two-layer XOR + pixel shuffling approach.
    """
    h, w, c = img_array.shape
    flat_image = img_array.flatten()

    # First chaotic key
    chaotic_key_1 = generate_key(seed, len(flat_image))
    # XOR-based encryption (first layer)
    encrypted_flat_1 = [p ^ chaotic_key_1[i] for i, p in enumerate(flat_image)]
    encrypted_array_1 = np.array(encrypted_flat_1, dtype=np.uint8).reshape(h, w, c)

    # Shuffle
    shuffled_array, _ = shuffle_pixels(encrypted_array_1, seed)

    # Second chaotic key
    chaotic_key_2 = generate_key(seed * 1.1, len(flat_image))
    shuffled_flat = shuffled_array.flatten()
    encrypted_flat_2 = [p ^ chaotic_key_2[i] for i, p in enumerate(shuffled_flat)]
    doubly_encrypted_array = np.array(encrypted_flat_2, dtype=np.uint8).reshape(h, w, c)

    return doubly_encrypted_array

##############################################################################
# 3. YOLOv10 License Plate Detection
##############################################################################
@st.cache_data(show_spinner=False)
def load_model(weights_path: str):
    """
    Loads the YOLOv10 model from local .pt weights (Ultralytics style).
    """
    model = YOLOv10(weights_path)  # e.g., 'best.pt'
    return model

def detect_license_plates(model, pil_image):
    """
    Runs YOLOv10 detection on the PIL image using ultralytics-style output:
      results -> list of ultralytics.engine.results.Results
        each Results has .boxes, .masks, .names, etc.

    We'll handle only the first Results object (single image).
    """
    np_image = np.array(pil_image)
    results = model.predict(np_image)

    # 1) Check how many Results objects we have
    if not results:
        print("No results returned by model.")
        return pil_image, []

    # 2) Take the first Results object
    r = results[0]

    # Debug: print the entire Results object
    print("Raw model output (first Results object):", r)

    # 3) If r.boxes is None or empty, we have no detections
    if not hasattr(r, 'boxes') or r.boxes is None or len(r.boxes) == 0:
        print("No boxes found in results[0].")
        return pil_image, []

    # 4) Parse bounding boxes
    bboxes = []
    draw = ImageDraw.Draw(pil_image)

    # r.boxes is an ultralytics.engine.results.Boxes object
    # We can iterate over each box in r.boxes:
    for box in r.boxes:
        # box has .xyxy, .conf, .cls as 1D tensors
        # e.g. box.xyxy[0] is [x1, y1, x2, y2]
        # box.conf[0] is confidence
        # box.cls[0] is class ID
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        # If your license plate class is 0:
        if cls_id == 0:
            x1, y1, x2, y2 = map(int, coords)
            bboxes.append((x1, y1, x2, y2))

            # Optional: draw bounding box for visualization
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    return pil_image, bboxes

##############################################################################
# 4. Streamlit App
##############################################################################
def main():
    st.title("YOLOv10 + Chaotic Encryption Demo")
    st.write(
        """
        **Instructions**:
        1. Provide an image (URL or file upload).
        2. If a license plate is detected, only that region will be **encrypted** using Chaotic Logistic Map.
        3. Download the final result.
        """
    )

    # A. Model weights path
    default_model_path = "best.pt"
    model_path = st.sidebar.text_input("YOLOv10 Weights (.pt)", value=default_model_path)

    if not os.path.isfile(model_path):
        st.warning(f"Model file '{model_path}' not found. Upload or provide a correct path.")
        st.stop()

    with st.spinner("Loading YOLOv10 model..."):
        model = load_model(model_path)
    st.success("Model loaded successfully!")

    # B. Image input
    st.subheader("Image Input")
    image_url = st.text_input("Image URL (optional)")
    uploaded_file = st.file_uploader("Or upload an image file", type=["jpg", "jpeg", "png"])

    # C. Encryption seed slider
    key_seed = st.slider("Encryption Key Seed (0 < seed < 1)", 0.001, 0.999, 0.5, step=0.001)

    if st.button("Detect & Encrypt"):
        # 1) Load the image from URL or file
        if image_url and not uploaded_file:
            try:
                response = requests.get(image_url, timeout=10)
                image_bytes = io.BytesIO(response.content)
                pil_image = Image.open(image_bytes).convert("RGB")
            except Exception as e:
                st.error(f"Failed to load image from URL. Error: {str(e)}")
                return
        elif uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
        else:
            st.warning("Please either paste a valid URL or upload an image.")
            return

        st.image(pil_image, caption="Original Image", use_container_width=True)

        # 2) Detect plates
        with st.spinner("Detecting license plates..."):
            image_with_boxes, bboxes = detect_license_plates(model, pil_image.copy())

        st.image(image_with_boxes, caption="Detected Plate(s)", use_container_width=True)
        if not bboxes:
            st.warning("No license plates detected.")
            return

        # 3) Encrypt bounding box regions
        with st.spinner("Encrypting license plates..."):
            np_img = np.array(pil_image)
            encrypted_np = np_img.copy()
            for (x1, y1, x2, y2) in bboxes:
                plate_region = encrypted_np[y1:y2, x1:x2]
                encrypted_region = encrypt_image(plate_region, key_seed)
                encrypted_np[y1:y2, x1:x2] = encrypted_region

            encrypted_image = Image.fromarray(encrypted_np)

        st.image(encrypted_image, caption="Encrypted Image", use_container_width=True)

        # 4) Download link
        buf = io.BytesIO()
        encrypted_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="Download Encrypted Image",
            data=buf,
            file_name="encrypted_plate.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
