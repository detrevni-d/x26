import torch 
import numpy as np
import cv2

EPSILON = 1e-15

def binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    output = (logits > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result


def get_bboxes_from_mask(binary_mask):
    # Ensure the input is binary and of type uint8
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for each contour
    bboxes = []
    masks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # get mask from contour
        mask = np.zeros_like(binary_mask)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        masks.append(mask)
        bboxes.append([x, y, x+w, y+h])  # Format: [x1, y1, x2, y2]
    
    return np.array(bboxes), np.stack(masks)