import os
import numpy as np
from PIL import Image
import cv2
import decord
from decord import VideoReader, cpu
# decord.bridge.set_bridge('torch')
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
import torch



def load_video(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # decord.bridge.set_bridge('torch')
    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()   # T H W C


    original_size = (img_array.shape[-2], img_array.shape[-3])  # (width, height)
    original_sizes = (original_size,) * total_num_frm


    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[j]) for j in range(total_num_frm)]

    for idx, img in enumerate(clip_imgs):
        os.makedirs('./img/sample_demo_9', exist_ok = True) 
        img.save(f'./img/sample_demo_9/{idx}.png')

    return clip_imgs, original_sizes


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq
