import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Correctly disable eager execution
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile
import imageio  # (Still available if needed for image display)
import cv2      # For video generation using OpenCV

def main():

    # Step 1 - Download Inception Model

    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = '../data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exists(local_zip_file):
        print("Downloading model...")
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        print("Extracting model...")
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
  
    # Use a noisy image (if needed)

    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
  
    model_fn = 'tensorflow_inception_graph.pb'
    

    # Step 2 - Create TensorFlow Session & Load the Model

    graph = tf.Graph()
    sess = tf.compat.v1.InteractiveSession(graph=graph)
    model_path = os.path.join(data_dir, model_fn)
    with tf.compat.v1.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.compat.v1.placeholder(np.float32, name='input')  # Define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})
    
    # List convolutional layers and their feature counts
    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    
    print('Number of layers:', len(layers))
    print('Total number of feature channels:', sum(feature_nums))
    

    # Helper Functions

    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.compat.v1.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>" % size
        return strip_def
      
    def rename_nodes(graph_def, rename_func):
        res_def = tf.compat.v1.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                if s[0] == '^':
                    n.input[i] = '^' + rename_func(s[1:])
                else:
                    n.input[i] = rename_func(s)
        return res_def
      
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1) * 255)
        plt.imshow(a)
        plt.axis('off')
        plt.show()
        
    def visstd(a, s=0.1):
        """Normalize the image range for visualization."""
        return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5
    
    def T(layer):
        """Helper to get the layer output tensor."""
        return graph.get_tensor_by_name("import/%s:0" % layer)
    
    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        """A simple deep dream render function that shows the result at the end."""
        t_score = tf.reduce_mean(t_obj)  # Define optimization objective
        t_grad = tf.gradients(t_score, t_input)[0]  # Get gradients
        
        img = img0.copy()
        for i in range(iter_n):
            g, score = sess.run([t_grad, t_score], {t_input: img})
            g /= (g.std() + 1e-8)
            img += g * step
            print("Iteration %d, score=%f" % (i, score))
        showarray(visstd(img))
    
    def tffunc(*argtypes, shapes=None):
        """Transform a TF-graph generating function into a regular Python function."""
        if shapes is None:
            shapes = [None] * len(argtypes)
        placeholders = [tf.compat.v1.placeholder(dtype, shape=shape) for dtype, shape in zip(argtypes, shapes)]
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session', sess))
            return wrapper
        return wrap
    
    # Updated resize function: ensure the image has a defined shape.
    def resize(img, size):
        """Resize the image to a given size using bilinear interpolation."""
        img = tf.expand_dims(img, 0)
        # Ensure the image tensor has a shape (1, height, width, 3)
        img = tf.ensure_shape(img, [1, None, None, 3])
        return tf.image.resize(img, size, method='bilinear')[0]
    # When wrapping 'resize', provide a shape for the image placeholder.
    resize = tffunc(np.float32, np.int32, shapes=([None, None, 3], [2]))(resize)
    
    def calc_grad_tiled(img, t_grad, tile_size=512):
        """
        Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries.
        """
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h - sz//2, sz), sz):
            for x in range(0, max(w - sz//2, sz), sz):
                sub = img_shift[y:y+sz, x:x+sz]
                g = sess.run(t_grad, {t_input: sub})
                grad[y:y+sz, x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)
    

    # Deep Dream Rendering Functions

    def render_deepdream(t_obj, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        """
        Render a Deep Dream image by applying gradient ascent to maximize
        the activations of the chosen layer.
        """
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, t_input)[0]
        
        # Split the image into a number of octaves.
        img = img0
        octaves = []
        for i in range(octave_n - 1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        # Generate details octave by octave.
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2]) + hi
            for i in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))
                print("Image: Octave %d, iter %d" % (octave, i))
            showarray(visstd(img))
    
    def render_deepdreamvideo_cv(t_obj, img0, video_filename='deep_dream_video.mp4',
                                 iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        """
        Render a Deep Dream video by generating intermediate frames and saving the
        video using OpenCV's VideoWriter.
        """
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, t_input)[0]
        
        # Split the image into a number of octaves.
        img = img0
        octaves = []
        for i in range(octave_n - 1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        frames = []
        
        # Generate details octave by octave.
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2]) + hi
            for i in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))
                print("CV Video: Octave %d, iter %d" % (octave, i))
                # Capture the current frame.
                frame = visstd(img)
                # Convert to uint8 for video (0–255 range).
                frame_uint8 = np.uint8(np.clip(frame, 0, 1) * 255)
                frames.append(frame_uint8)
        
        # Write frames as video using OpenCV's VideoWriter.
        height, width, channels = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID' or 'H264'
        out = cv2.VideoWriter(video_filename, fourcc, 30, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print("Deep Dream video saved as", video_filename)
    

    # CHOOSE WHICH RENDERING TO RUN
    # For image output:
    # (Make sure you have an image file to work with; you can replace 'pilatus800.jpg'
    # with any image you have.)
    img0 = np.float32(PIL.Image.open('pilatus800.jpg'))
    render_deepdream(tf.square(T('mixed4c')), img0)
    
    # For video output using OpenCV, uncomment the next line:
    render_deepdreamvideo_cv(tf.square(T('mixed4c')), img0, video_filename='deep_dream_video.mp4')

if __name__ == '__main__':
    main()
