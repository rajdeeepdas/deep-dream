
**Techniques for adjusting an image iteratively to maximize specific layer activations.**

- **TensorFlow Compatibility:**  
  Adapting TensorFlow 1.x code to TensorFlow 2.x using `tf.compat.v1`, including updating deprecated functions (e.g., replacing `tf.image.resize_bilinear` with `tf.image.resize`).

- **Video Generation with OpenCV:**  
  Capturing intermediate frames of the Deep Dream process and generating a video using OpenCV’s `VideoWriter`.

## Installation Requirements

- Python 3.12 (or a compatible version)
- TensorFlow 2.x
- NumPy
- Pillow
- Matplotlib
- OpenCV-Python
- Imageio (optional, for other uses)

### Install Dependencies

Run the following command in your project directory:

```bash
pip install numpy pillow matplotlib tensorflow opencv-python imageio
```

## Usage

### Prepare an Input Image
Place your input image (e.g., a scenic landscape or any image of your choice) in the project directory. The code is set to load an image named `pilatus800.jpg`. If you want to use a different image, update the corresponding line in the code:

```python
img0 = np.float32(PIL.Image.open('pilatus800.jpg'))
```

For example, change `'pilatus800.jpg'` to `'your_image.jpg'`.

### Run the Script
Open your terminal, navigate to your project directory, and run:

```bash
python deep_dream.py
```

The script will:

- Download the pre-trained Inception model (if it is not already present).
- Load the model and the input image.
- Process the image with Deep Dream across multiple octaves.
- Display the final Deep Dream image using Matplotlib.
- Generate a video (using OpenCV) showing the transformation process and save it as `deep_dream_video.mp4`.

### Adjusting Video Duration
If the generated video is too short (e.g., around 1 second), consider:

- Increasing `iter_n` (iterations per octave) to capture more frames.
- Increasing `octave_n` (number of octaves) to process more scales.
- Lowering the frame rate (fps) in the OpenCV `VideoWriter` (e.g., changing from 30 fps to 10 or 15 fps).
