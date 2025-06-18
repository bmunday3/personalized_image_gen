from typing import Optional
import streamlit as st
from PIL import Image
from utils import make_grid, generate_image
from dotenv import load_dotenv

import os
import numpy as np
import pandas as pd

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

st.set_page_config(
    page_title="Personalized Brand-Aligned Image Generation",
    page_icon="imgs/logos/databricks-symbol-color.svg",
    initial_sidebar_state="expanded",
    layout="wide"
)

def set_image(key: str, img: Image.Image):
    st.session_state[key] = img

def convert_to_img(data):
    pixel_data = np.array(data["predictions"], dtype=np.uint8)
    img = Image.fromarray(pixel_data, 'RGB')    
    return img

# ~~~~~~~~~~~~~~~
# side bar config
# ~~~~~~~~~~~~~~~
st.sidebar.markdown("\n")
st.sidebar.image("imgs/logos/small-scale-lockup-full-color-rgb.png")

st.sidebar.markdown("This Streamlit app is powered by Stable Diffusion models that were fine tuned on Databricks. These models are served as real time REST endpoints using Databricks model serving.")
st.sidebar.markdown("For more information, visit this Databricks solution accelerator to learn more.")
st.sidebar.link_button(label="Solution Accelerator", type="primary", use_container_width=True, url="https://www.databricks.com/solutions/accelerators/creating-brand-aligned-images-using-gen-ai")

st.sidebar.divider()
st.sidebar.markdown("# Brand Selection")
brand = st.sidebar.radio(label="Select brand for image generation", options=["McDonalds Happy Meal", "Stanley Tumbler"]) # , "Starbucks Cold Brew"
st.sidebar.markdown("\n")
st.sidebar.markdown("\n")
st.sidebar.divider()

example_image_paths = ["imgs/examples/arms.png", "imgs/examples/river.png", "imgs/examples/wrigley.png"]
example_image_captions = ["A photo of a happy meal with arms and legs", "A photo of a happy meal floating on the chicago river with the skyline in the background", "A photo of a happy meal at a table in center field of wrigley field"]


# header
# st.image("<INSERT IMAGE PATH HERE>", width=1000) # unmcomment and provide valid image path to header image for app page
st.markdown("# Brand-Aligned Image Generation")
st.markdown("### Try our fine-tuned stable diffusion model")
prompt = st.text_area(
    label="Prompt",
    value="Enter your text here",
    key="".join(brand.split(" ")).lower()
)

if brand == "Stanley Tumbler":
    prompt = prompt.replace("Stanley", "STAN")

num_inference_steps = st.slider(label="Number of Inference Steps", min_value=25, max_value=50, value=25)

formatted =  pd.DataFrame(
    {"prompt": [prompt], "num_inference_steps": num_inference_steps}
)

left, right = st.columns(2, gap="small")

generate_button = left.button(label=":frame_with_picture: Generate Image", type="secondary", use_container_width=True)
clear_button = right.button(label="Reset", type="secondary", use_container_width=True)

# button to kick off image generation
if generate_button:
    images = []
    with st.spinner(text="Generating image..."):
        for _ in range(3):
            results = generate_image(
                endpoint=os.getenv("SERVING_ENDPOINT"),
                dataset=formatted
                # url=INSTANCE,
                # databricks_token=TOKEN,
            )
            img = convert_to_img(results)
            images.append(img)
    set_image("output_img", img.copy())
    l, m, r = st.columns(3)
    l_img = l.image(images[0], caption=prompt, use_container_width=True)
    m.image(images[1], caption=prompt, use_container_width=True)
    r.image(images[2], caption=prompt, use_container_width=True)

if clear_button:
    set_image("output_img", None)

st.divider()
st.markdown("# Pre-run Examples")
examples_grid = make_grid(1,3,gap="small")
examples_grid[0][0].image(image=example_image_paths[0], caption=example_image_captions[0], use_container_width=True)
examples_grid[0][1].image(image=example_image_paths[1], caption=example_image_captions[1], use_container_width=True)
examples_grid[0][2].image(image=example_image_paths[2], caption=example_image_captions[2], use_container_width=True)


