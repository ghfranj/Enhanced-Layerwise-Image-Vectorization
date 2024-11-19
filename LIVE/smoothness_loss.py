import torch
import numpy as np

def curvature_smoothness_loss(points):
    """
    Curvature-based smoothness loss penalizes high curvature by computing the second derivative.
    Args:
        points: Tensor of shape (N, 2), where N is the number of points.
    Returns:
        curvature_loss: Scalar tensor representing the curvature-based smoothness loss.
    """
    # Calculate second derivative to penalize curvature
    curvature_loss = torch.sum((points[2:] - 2 * points[1:-1] + points[:-2]) ** 2)
    return curvature_loss

def smoothness_loss(x_list, smoothness_weight=1.0, scale=1e-1):  # x[ npoints,2]
    smoothness_loss = 0.
    for x in x_list:
        smoothness_loss += curvature_smoothness_loss(x)

    loss = smoothness_weight * smoothness_loss
    print("smoothness_loss: ", smoothness_loss.item())
    return loss / (len(x_list))

if __name__ == "__main__":
    # Test cases for the curvature-based smoothness loss
    x = torch.tensor([[0,0], [1,1], [2,1], [0.5,0]], dtype=torch.float32)
    scale = 1
    y = smoothness_loss([x], scale)
    print(y)

    x = torch.tensor([[0,0], [1,1], [2,1], [2.,0]], dtype=torch.float32)
    scale = 1
    y = smoothness_loss([x], scale)
    print(y)
