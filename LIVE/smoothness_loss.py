import torch
import numpy as np



def laplacian_smoothness_loss(points):
    """
    Laplacian smoothness loss penalizes deviations of a point from the average of its neighbors.
    Args:
        points: Tensor of shape (N, 2), where N is the number of points.
    Returns:
        laplacian_loss: Scalar tensor representing the Laplacian smoothness loss
    """
    # Compute Laplacian (difference of points from their neighbors)
    laplacian_loss = torch.sum((points[2:] - 2 * points[1:-1] + points[:-2]) ** 2)
    return laplacian_loss

def smoothness_loss(x_list, smoothness_weight=1.1, scale=1e-1):  # x[ npoints,2]
    smoothness_loss = 0.
    for x in x_list:
        smoothness_loss += laplacian_smoothness_loss(x)

    loss = smoothness_weight * smoothness_loss
    print("smoothness_loss: ", smoothness_loss.item())
    return loss / (len(x_list))


if __name__ == "__main__":
    #x = torch.rand([6, 2])
    #x = torch.tensor([[0,0], [1,1], [2,1], [1.5,0]])
    x = torch.tensor([[0,0], [1,1], [2,1], [0.5,0]])
    #x = torch.tensor([[1,0], [2,1], [0,1], [2,0]])
    scale = 1 #0.5
    y = smoothness_loss([x], scale)
    print(y)

    x = torch.tensor([[0,0], [1,1], [2,1], [2.,0]])
    #x = torch.tensor([[1,0], [2,1], [0,1], [2,0]])
    scale = 1 #0.5
    y = smoothness_loss([x], scale)
    print(y)
