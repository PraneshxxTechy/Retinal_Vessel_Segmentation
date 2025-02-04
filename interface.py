import os
import time
import numpy as np
import cv2
import torch
import gradio as gr
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from model import build_unet
from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy().reshape(-1) > 0.5
    y_pred = y_pred.cpu().numpy().reshape(-1) > 0.5

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

def segment_retinal_image(image):
    """ Hyperparameters """
    H, W = 512, 512
    size = (W, H)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Process image """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1)) / 255.0
    x = np.expand_dims(x, axis=0).astype(np.float32)
    x = torch.from_numpy(x).to(device)

    """ Dummy mask for metric calculation (replace with ground truth if available) """
    mask = np.ones((H, W), dtype=np.uint8)
    y = np.expand_dims(mask, axis=0) / 255.0
    y = np.expand_dims(y, axis=0).astype(np.float32)
    y = torch.from_numpy(y).to(device)

    with torch.no_grad():
        start_time = time.time()
        pred_y = torch.sigmoid(model(x))
        total_time = time.time() - start_time

        metrics_score = calculate_metrics(y, pred_y)
        pred_y = (pred_y[0].cpu().numpy().squeeze(0) > 0.5).astype(np.uint8) * 255

    """ Saving masks """
    ori_mask = mask_parse(mask)
    pred_mask = mask_parse(pred_y)
    line = np.ones((H, 10, 3)) * 128

    cat_images = np.concatenate([image, line, ori_mask, line, pred_mask], axis=1)

    result_path = "results/single_image_result.png"
    cv2.imwrite(result_path, cat_images)

    fps = 1 / total_time
    return cat_images, metrics_score, fps

def process_image(image):
    # Perform segmentation
    result_image, metrics, fps = segment_retinal_image(image)
    jaccard, f1, recall, precision, acc = metrics

    # Convert result_image to uint8 and ensure it's in the range [0, 255]
    result_image = result_image.astype(np.uint8)

    return result_image, jaccard, f1, recall, precision, acc, fps

# Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Retinal Image"),
    outputs=[
        gr.Image(type="numpy", label="Segmented Image"),
        gr.Number(label="FPS")
    ],
    title="Retinal Vessel Segmentation",
    description="Upload a retinal image and the app will segment the vessels."
)

if __name__ == "__main__":
    seeding(42)
    create_dir("results")
    interface.launch()
