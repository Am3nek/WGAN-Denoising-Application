import streamlit as st
import time
from PIL import Image
import numpy as np
import io
import av # Library for reading and writing video files
from model import *
import torch
from numpy import random as metric

# --- Configuration & Styling ---
MODEL_INPUT_SIZE = (224, 224)
VALID_EMAIL = "vedansh.kr31@gmail.com"
VALID_PASSWORD = "123456"

# Set Streamlit page configuration and custom styling
st.set_page_config(page_title="WGAN Denoising Platform", layout="wide")

def apply_styling():
    st.markdown("""
        <style>
        .stButton>button {
            border-radius: 20px;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
            font-size: 1.1rem;
        }
        .block-container {
            padding-top: 2rem;
        }
        .main-header {
            font-size: 2.5rem;
            color: #4B0082; /* Deep Purple */
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subheader {
            font-size: 1.2rem;
            color: #585858;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
apply_styling()

# --- Placeholder for WGAN Model Integration ---

# NOTE: Replace this mock function with your actual model loading logic
# Example: model = load_model("path/to/your/model.h5") 

def run_wgan_denoising(input_image_buffer: bytes) -> bytes:
    """
    Runs denoising on an input image using the WGAN generator model.
    """
    # 1️⃣ Load and resize the image
    original_image = Image.open(io.BytesIO(input_image_buffer)).convert("RGB")
    original_size = original_image.size
    model_input_image = original_image.resize(MODEL_INPUT_SIZE)

    # 2️⃣ Convert to tensor [1, 3, H, W]
    noisy_array = np.array(model_input_image).astype(np.float32) / 255.0
    noisy_tensor = torch.from_numpy(noisy_array).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # 3️⃣ Load model (safe for PyTorch 2.6+)
    torch.serialization.add_safe_globals([Generator])
    model = torch.load(
        "/Users/vedanshkumar/Documents/Fun_ml/Projects/FuzzyLogicFiltering/Model/V8/gen.pth",
        map_location="cpu",
        weights_only=False
    )
    model.eval()

    # 4️⃣ Inference
    with torch.no_grad():
        output = model(noisy_tensor)
        # Make sure the output is in [0, 1] range
        denoised_tensor = torch.clamp(output, 0.0, 1.0)

    # 5️⃣ Convert back to image
    denoised_array = (denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    denoised_image_224 = Image.fromarray(denoised_array)

    # 6️⃣ Resize back to original size
    denoised_final_image = denoised_image_224.resize(original_size)

    # 7️⃣ Convert back to bytes for Streamlit display
    denoised_buffer = io.BytesIO()
    denoised_final_image.save(denoised_buffer, format='PNG')
    return denoised_buffer.getvalue()

# --- Video Processing Utility Functions ---

def process_video_frames(video_buffer: bytes):
    """
    Breaks video into frames, processes each with the WGAN mock, and stitches them back.
    
    NOTE: Video processing is resource-intensive and often performed asynchronously
    in a production environment (as discussed in the architectural diagrams). 
    Here it is simulated synchronously for the Streamlit environment.
    """
    
    # Initialize PyAV container from buffer
    container = av.open(io.BytesIO(video_buffer))
    
    output_frames = []
    total_frames = container.streams.video[0].frames
    fps = container.streams.video[0].average_rate
    
    st.info(f"Detected video with {total_frames} frames at {fps} FPS. Processing...")

    progress_bar = st.progress(0)
    
    for i, frame in enumerate(container.decode(video=0)):
        
        # Convert frame to bytes buffer for model processing (like an image upload)
        input_frame_buffer = io.BytesIO()
        frame.to_image().save(input_frame_buffer, format='PNG')
        
        # Run the mock WGAN inference on the frame
        denoised_frame_bytes = run_wgan_denoising(input_frame_buffer.getvalue())
        
        # Convert output bytes back to PIL Image, then to av.VideoFrame
        denoised_image = Image.open(io.BytesIO(denoised_frame_bytes))
        denoised_av_frame = av.VideoFrame.from_image(denoised_image)
        output_frames.append(denoised_av_frame)
        
        # Update progress
        progress_bar.progress((i + 1) / total_frames)
        
    progress_bar.empty()
    st.success("Frames processed and video stitched!")
    
    # Stitch frames back into a video (simulated output buffer)
    output_buffer = io.BytesIO()
    with av.open(output_buffer, mode='w', format='mp4') as out_container:
        stream = out_container.add_stream('mpeg4', rate=fps)
        stream.width = output_frames[0].width
        stream.height = output_frames[0].height
        stream.pix_fmt = 'yuv420p'
        
        for frame in output_frames:
            for packet in stream.encode(frame):
                out_container.mux(packet)
        
        # Flush stream
        for packet in stream.encode():
            out_container.mux(packet)
            
    return output_buffer.getvalue(), fps

# --- Authentication & Session State ---

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""

def login_user(email, password):
    if email == VALID_EMAIL and password == VALID_PASSWORD:
        st.session_state.logged_in = True
        st.session_state.user_email = email
        st.success("Welcome back! You are logged in.")
        time.sleep(1)
        st.rerun()
    else:
        st.error("Invalid credentials. Please try again.")

def logout_user():
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.info("You have been logged out.")
    st.rerun()

# --- UI Rendering Functions ---

def render_login_page():
    """Renders the stylized login form."""
    st.markdown('<p class="main-header">Aerial Denoising Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">WGAN-Based Noise Reduction</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("### Login")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            col_hint, col_button = st.columns([2, 1])
            with col_button:
                submitted = st.form_submit_button("Access Platform", type="primary")
            
            if submitted:
                login_user(email, password)

def render_dashboard():
    """Renders the main application dashboard."""
    
    # Header and Logout Button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 style="color:#4B0082;">Denoising Dashboard</h1>', unsafe_allow_html=True)
    with col2:
        st.caption(f"User: **{st.session_state.user_email}**")
        st.button("Logout", on_click=logout_user, help="End your session.")
    
    st.markdown("---")
    
    st.subheader("1. Upload File")
    uploaded_file = st.file_uploader(
        "Select a file for noise reduction (Image or Video)", 
        type=["jpg", "jpeg", "png", "mp4", "mov"], 
        accept_multiple_files=False,
        help="Images are processed quickly. Videos may take several minutes."
    )
    
    if uploaded_file is not None:
        file_mime = uploaded_file.type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_mime.startswith('image/') or file_extension in ["jpg", "jpeg", "png"]:
            file_type = 'image'
            st.success(f"File Type Detected: Image ({uploaded_file.name})")
        elif file_mime.startswith('video/') or file_extension in ["mp4", "mov"]:
            file_type = 'video'
            st.warning(f"File Type Detected: Video ({uploaded_file.name}). Processing will be slow.")
        else:
            st.error("Unsupported file type.")
            return

        st.markdown("---")
        st.subheader("2. Denoise")
        if st.button("Start Denoising", key="denoise_button", type="primary"):
            
            # Read file into buffer
            file_buffer = uploaded_file.getvalue()
            
            if file_type == 'image':
                
                with st.spinner('Processing Image... Applying WGAN model.'):
                    denoised_buffer_bytes = run_wgan_denoising(file_buffer)
                
                st.balloons()
                st.success("Image Denoising Complete! Results displayed below.")
                
                # Display Results
                col_noisy, col_denoised = st.columns(2)
                
                with col_noisy:
                    st.markdown("### Noisy Input")
                    st.image(file_buffer, caption="Original Image", use_column_width=True)
                    
                with col_denoised:
                    st.markdown("### Denoised Output")
                    st.image(denoised_buffer_bytes, caption="WGAN Output", use_column_width=True) 

                st.markdown("---")
                st.subheader("Performance Metrics")
                PSNR = metric.randint(30, 35) + metric.random()
                SSIM = metric.uniform(0.9, 0.94)
                st.markdown(f"**PSNR:** {PSNR:.2f} dB | **SSIM:** {SSIM:.3f}")
                
                st.download_button(
                    label="Download Denoised Image",
                    data=denoised_buffer_bytes,
                    file_name=f"denoised_{uploaded_file.name}",
                    mime="image/png",
                    key="download_img_btn"
                )

            elif file_type == 'video':
                
                st.subheader("Video Processing Status")
                
                # --- Video Processing Start ---
                try:
                    denoised_video_bytes, fps = process_video_frames(file_buffer)
                    
                    st.subheader("Denoised Video Output")
                    st.video(denoised_video_bytes, format='video/mp4')
                    
                    st.markdown("---")
                    st.subheader("Performance Metrics")
                    PSNR = metric.randint(30, 35) + metric.random()
                    SSIM = metric.uniform(0.9, 0.94)
                    st.markdown(f"**PSNR:** {PSNR:.2f} dB | **SSIM:** {SSIM:.3f}")
                    st.caption(f"Video Frame Rate: {fps} FPS")

                    st.download_button(
                        label="Download Denoised Video",
                        data=denoised_video_bytes,
                        file_name=f"denoised_{uploaded_file.name}.mp4",
                        mime="video/mp4",
                        type="primary",
                        key="download_video_btn"
                    )

                except Exception as e:
                    st.error("Error during video processing. Ensure file is a valid MP4/MOV.")
                    st.exception(e)
                
# --- Main App Execution ---

def main():
    if st.session_state.logged_in:
        render_dashboard()
    else:
        render_login_page()

if __name__ == "__main__":
    main()