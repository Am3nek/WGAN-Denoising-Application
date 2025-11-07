# WGAN Denoising Platform

This project is a web-based platform for denoising images and videos using a Wassertein Generative Adversarial Network (WGAN) model. The application is built with Streamlit and uses a PyTorch-based model for the denoising process.

## Features

- **User Authentication:** Secure login and logout functionality.
- **Image Denoising:** Upload JPG, JPEG, or PNG images for noise reduction.
- **Video Denoising:** Upload MP4 or MOV videos for frame-by-frame denoising.
- **Side-by-Side Comparison:** View the original and denoised files to see the results.
- **Performance Metrics:** Display of Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) for the denoised output.
- **Download Results:** Download the denoised images and videos.

## Model Architecture

The denoising model is a U-Net like generator architecture implemented in PyTorch. It consists of:

- **Downsampling Path:** A series of convolutional blocks (`DownSample`) and max-pooling layers to capture context from the input image.
- **Bridge:** A connecting block at the lowest resolution.
- **Upsampling Path:** A series of up-sampling blocks (`UpSample`) and transposed convolutions to reconstruct the denoised image. Skip connections are used to concatenate features from the downsampling path with the upsampling path to preserve high-frequency details.
- **Final Layer:** A final convolutional layer with a Tanh activation function to produce the denoised image.

## Getting Started

### Prerequisites

- Python 3.7+
- The following Python libraries are required:
  - streamlit
  - torch
  - torchvision
  - numpy
  - Pillow
  - av
  - pytest
  - pytest-mock

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Software-DA
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided, but the dependencies are listed above. You can create one with the specified libraries.)*

### Running the Application

1.  **Start the Streamlit application:**
    ```bash
    streamlit run main.py
    ```

2.  **Access the application:**
    Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Login:**
    Use the following credentials to log in:
    - **Email:** `vedansh.kr31@gmail.com`
    - **Password:** `123456`

## Usage

1.  **Login** to the platform.
2.  On the **Denoising Dashboard**, use the file uploader to **select an image or video** you want to denoise.
3.  Click the **"Start Denoising"** button.
4.  The application will process the file and display the original and denoised versions.
5.  You can then **download** the denoised file.

## Testing

The project includes a comprehensive test suite using `pytest` and `streamlit.testing.v1`.

### Running Tests

1.  **Install testing dependencies:**
    ```bash
    pip install pytest pytest-mock
    ```

2.  **Run all tests:**
    ```bash
    pytest testing.py -v
    ```

3.  **Run specific test classes:**
    ```bash
    pytest testing.py::TestLogin -v
    pytest testing.py::TestImageDenoising -v
    ```

### Manual Testing

The `testing.py` file also contains a checklist for manual testing of file uploads and the complete denoising workflow, which is not fully automatable.
