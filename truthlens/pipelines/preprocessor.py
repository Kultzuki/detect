import os
import mimetypes
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import ffmpeg

# 1. Initialize the global MTCNN face detector singleton.
# We are routing the device to GPU/MPS if available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device, select_largest=True)

def extract_largest_face(img: Image.Image, margin: int = 20) -> Image.Image:
    """
    Finds the largest face structurally in the image using FaceNet, crops with a margin, 
    and returns the cropped face Image object.
    If no face is detected, it center-crops identically to the legacy fallback.
    """
    detection = mtcnn.detect(img)
    boxes = detection[0]
    if boxes is None or len(boxes) == 0:
        return _fallback_center_crop(img)

    # MTCNN with select_largest=True puts the largest face at index 0.
    box = boxes[0]
    
    # Add Margin around face (jawline/hair)
    width, height = img.size
    left = max(0, int(box[0]) - margin)
    top = max(0, int(box[1]) - margin)
    right = min(width, int(box[2]) + margin)
    bottom = min(height, int(box[3]) + margin)

    return img.crop((left, top, right, bottom))


def _fallback_center_crop(img: Image.Image) -> Image.Image:
    """Legacy center crop logic for files where no human face is identified."""
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))


def detect_file_type(file_path: str) -> str:
    """
    Detects input type from file extension or MIME type.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        str: One of 'image' or 'video'.
        
    Raises:
        ValueError: If the file type cannot be determined or is unsupported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    video_exts = {'.mp4', '.mov', '.avi'}

    if ext in image_exts:
        return 'image'
    if ext in video_exts:
        return 'video'

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'

    raise ValueError(f"Unsupported file type. Extension: {ext}, MIME: {mime_type}")


def process_image(file_path: str) -> np.ndarray:
    """
    Loads an image, center-crops to a square, resizes to 224x224, and returns as an RGB numpy array.
    
    Args:
        file_path (str): Path to the image file.
        
    Returns:
        np.ndarray: A 224x224x3 RGB numpy array.
        
    Raises:
        ValueError: If either dimension of the image is less than 100 pixels.
    """
    with Image.open(file_path) as img:
        img = img.convert("RGB")
        w, h = img.size

        if min(w, h) < 100:
            raise ValueError(f"Image dimensions are too small ({w}x{h}); must be at least 100x100 pixels.")
        
        # New approach: Find the face and crop tightly around it. 
        # If no face is found, it will gracefully fallback to the center crop.
        cropped_img = extract_largest_face(img, margin=30)
        
        # 3. Final standardized resize strictly for the Transformer
        img = cropped_img.resize((224, 224), Image.Resampling.LANCZOS)
        
        return np.array(img)


def process_video(file_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extracts 20 evenly spaced 224x224 RGB frames.
    
    Args:
        file_path (str): Path to the video file.
        
    Returns:
        Tuple[List[np.ndarray], Dict[str, Any]]:
            - List of 20 RGB frames (224x224x3).
            - Metadata dict (fps, duration, resolution, file_size).
    """
    probe = ffmpeg.probe(file_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    if not video_stream:
        raise ValueError("No video stream found in the file.")

    fps_str = video_stream.get('avg_frame_rate', '0/1')
    num, den = fps_str.split('/')
    fps = float(num) / float(den) if float(den) != 0 else 0.0
    
    duration = float(video_stream.get('duration', probe.get('format', {}).get('duration', 0.0)))
    width = int(video_stream.get('width', 0))
    height = int(video_stream.get('height', 0))
    file_size = os.path.getsize(file_path)

    metadata = {
        'fps': fps,
        'duration': duration,
        'resolution': (width, height),
        'file_size': file_size
    }

    # Extract 20 frames evenly spaced
    target_frames = 20
    rate = target_frames / max(duration, 0.1)
    
    try:
        out, _ = (
            ffmpeg
            .input(file_path)
            .filter('fps', fps=rate)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        video_data = np.frombuffer(out, np.uint8)
        frames = video_data.reshape([-1, height, width, 3])
        
        frames_list = []
        for f in frames:
            pil_frame = Image.fromarray(f)
            face_img = extract_largest_face(pil_frame, margin=30)
            resized = face_img.resize((224, 224), Image.Resampling.LANCZOS)
            frames_list.append(np.array(resized))
        
        # Ensure exactly 20 frames
        if len(frames_list) > target_frames:
            indices = np.linspace(0, len(frames_list) - 1, target_frames, dtype=int)
            frames_list = [frames_list[i] for i in indices]
        elif len(frames_list) < target_frames and len(frames_list) > 0:
            missing = target_frames - len(frames_list)
            frames_list.extend([frames_list[-1]] * missing)
        elif len(frames_list) == 0:
            # Fallback for black frames if ffmpeg succeeds but returns no frames
            frames_list = [np.zeros((224, 224, 3), dtype=np.uint8)] * target_frames
            
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode('utf8', errors='ignore') if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg failed to extract frames: {err_msg}")
    
    return frames_list, metadata


def preprocess(file_path: str) -> Dict[str, Any]:
    """
    Main preprocessing entry point.
    
    Args:
        file_path (str): Path to the multimedia file.
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'type': 'image' or 'video'
            - 'frames' (for video/image): The processed visual representation
            - 'metadata': Key information about the file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    file_type = detect_file_type(file_path)
    
    result: Dict[str, Any] = {
        'type': file_type,
        'metadata': {}
    }
    
    if file_type == 'image':
        frame = process_image(file_path)
        result['frames'] = frame
        result['metadata'] = {
            'file_size': os.path.getsize(file_path)
        }
        
    elif file_type == 'video':
        frames, metadata = process_video(file_path)
        result['frames'] = frames
        result['metadata'] = metadata
        
    return result
