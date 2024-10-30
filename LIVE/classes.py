import numpy.random as npr
import torch
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import  LambdaLR
from utils import get_sdf
from skopt.space import Categorical, Real
from skopt import gp_minimize
from sklearn.cluster import KMeans, MeanShift
from scipy.ndimage import binary_closing

class Contour_path_init():
    def __init__(self, pred, gt, format='[bs x c x 2D]', quantile_interval=800, nodiff_thres=0.5861487371810189):
        self.quantile_interval = quantile_interval
        self.nodiff_thres = nodiff_thres
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        self.pred = pred
        self.gt = gt
        # Extract the first image from the batch
        pred_img = pred[0]
        gt_img = gt[0]

        # If the images have multiple channels (e.g., RGB), select one channel
        if pred_img.shape[0] == 3:
            pred_img = pred_img.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
            gt_img = gt_img.transpose(1, 2, 0)
        self.gt_img = gt_img
        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0]) ** 2).sum(0)
            self.reference_gt = copy.deepcopy(np.transpose(gt[0], (1, 2, 0)))
        elif format == '[2D x c]':
            self.map = (np.abs(pred - gt)).sum(-1)
            self.reference_gt = copy.deepcopy(gt[0])
        else:
            raise ValueError
        self.loss = 1e10
        self.format = format
        with torch.no_grad():
            self.optimize_params_bayesian(format=format)
            torch.cuda.empty_cache()

    import matplotlib.pyplot as plt
    def ensure_distinct_and_multiple_of_three(self, points):
        points = points.squeeze(dim=1)
        num_points = points.shape[0]
        all_points = []
        targ_num_points = 180
        if num_points > targ_num_points:
            total_num = 0
            skip = -(num_points // -targ_num_points)
            for i in range(num_points):
                if (i % skip == 0 and len(all_points) < targ_num_points) or points.shape[
                    0] - i <= targ_num_points - len(all_points):
                    all_points.append(points[i])
                    total_num = total_num + 1
            points = torch.stack(all_points)
        if points.shape[0] < 12:
            old_points_num = points.shape[0]
            needed_points_num = 12 - old_points_num
            for i in range(1, needed_points_num + 1):
                first_point = points[0]
                last_point = points[-1]
                points = torch.cat(
                    (points, (last_point + (first_point - last_point) * i / (needed_points_num + 1)).unsqueeze(0)),
                    dim=0)
        # Ensure the number of points is a multiple of 3
        elif points.shape[0] % 3 != 0:
            points = points[:-(points.shape[0] % 3)]
        return torch.FloatTensor(points.flip(dims=[0]))

    def find_border_points(self, component):
        # Find contours (border points)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        border_points = np.vstack(contours)
        border_points = torch.FloatTensor(border_points)
        border_points = self.ensure_distinct_and_multiple_of_three(border_points)
        return border_points

    def calc_mean_color(self, component_binary=None, k=3, masked_pixels=None, mask=None):
        if masked_pixels is None:
            # Extract the mask region from the ground truth image
            masked_pixels = self.gt_img[component_binary == 1]
        if len(masked_pixels.shape) > 2:
            if mask is not None:
                masked_pixels = masked_pixels[mask == 1]
            masked_pixels = masked_pixels.reshape(-1, 3)
        # Check if there are any valid pixels
        if len(masked_pixels) == 0:
            return np.array([0, 0, 0])  # Return black if no pixels found

        # Use k-means clustering to find the most frequent color
        kmeans = KMeans(n_clusters=min(k, len(masked_pixels)), random_state=0)
        kmeans.fit(masked_pixels)
        # ms = MeanShift(bandwidth=30, bin_seeding=True)
        # ms.fit(masked_pixels)
        # Find the largest cluster
        counts = np.bincount(kmeans.labels_)
        most_frequent_cluster_idx = np.argmax(counts)
        most_frequent_color = [kmeans.cluster_centers_[most_frequent_cluster_idx]]
        return most_frequent_color

    def optimize_params_bayesian(self, num_iter=20, format='[bs x c x 2D]'):
        def calculate_loss(quantile_interval, nodiff_thres, opacity, self, format='[bs x c x 2D]'):
            if format == '[bs x c x 2D]':
                map = ((self.pred[0] - self.gt[0]) ** 2).sum(0)
                reference_gt = copy.deepcopy(np.transpose(self.gt[0], (1, 2, 0)))
            elif format == '[2D x c]':
                map = (np.abs(self.pred - self.gt)).sum(-1)
                reference_gt = copy.deepcopy(self.gt[0])
            else:
                raise ValueError
            print("calculating loss...", quantile_interval, nodiff_thres, opacity)
            map[map < nodiff_thres] = 0
            quantile_interval_vals = np.linspace(0., 1., int(quantile_interval))

            quantized_interval = np.quantile(map, quantile_interval_vals)

            quantized_interval = np.unique(quantized_interval)

            quantized_interval = sorted(quantized_interval[1:-1])

            map = np.digitize(map, quantized_interval, right=False)
            map = np.clip(map, 0, 255).astype(np.uint8)
            # Apply Morphological Operations (Erosion followed by Dilation)
            kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel for the operations
            map = cv2.erode(map, kernel, iterations=1)
            map = cv2.dilate(map, kernel, iterations=1)

            idcnt = {}
            for idi in sorted(np.unique(map)):
                idcnt[idi] = (map == idi).sum()
            idcnt.pop(min(idcnt.keys()))

            loss = torch.tensor(100000000000000.)
            i = 3
            while i > 0 and len(idcnt) > 1:
                is_first_path = (np.min(self.pred) == 1)
                is_white_background = False
                i -= 1
                target_id = max(idcnt, key=idcnt.get)
                if is_first_path:
                    num_check = 2

                    # Slice the first and last 5 rows and columns
                    top_rows = self.gt[:, :, :num_check, :].copy()
                    bottom_rows = self.gt[:, :, -num_check:, :].copy()
                    left_columns = self.gt[:, :, :, :num_check].copy()
                    right_columns = self.gt[:, :, :, -num_check:].copy()

                    top_rows[top_rows<0.8] = 0.0
                    bottom_rows[bottom_rows<0.8] = 0.0
                    left_columns[left_columns<0.8] = 0.0
                    right_columns[right_columns<0.8] = 0.0
                    # Define a threshold for "white"
                    threshold = 0.9  # You can adjust this based on your needs
                    # Check if all means are close to 1
                    print(np.mean(top_rows), np.mean(bottom_rows), np.mean(left_columns),np.mean(right_columns))
                    is_white_background = (
                            np.mean(top_rows) >= threshold and
                            np.mean(bottom_rows) >= threshold and
                            np.mean(left_columns) >= threshold and
                            np.mean(right_columns) >= threshold
                    )
                if is_first_path and not is_white_background:
                    binary_pred = ((map == target_id) * 255).astype(np.uint8)
                    # _, binary_pred = cv2.threshold(binary_pred, 1, 255, cv2.THRESH_BINARY)
                    binary_pred = cv2.adaptiveThreshold(
                        (binary_pred).astype(np.uint8),
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        51,
                        2
                    )
                    _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
                        binary_pred.astype(np.uint8), connectivity=4)
                else:
                    _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
                        (map == target_id).astype(np.uint8), connectivity=4)

                csize = [ci[-1] for ci in cstats[1:]]

                target_cid = csize.index(max(csize)) + 1

                # Calculate border points and points closer to center from borders
                component_binary = (component == target_cid).astype(np.uint8)
                loss = self.calc_l2(component_binary, opacity)
                if loss <= self.loss:
                    self.component_binary = component_binary
                    break
                if self.loss < 1e10:
                    break
                idcnt.pop(target_id)
            return loss.item()

        # Define the parameter space
        space = [
            Categorical(list(range(10, 1200, 20)), name='quantile_interval'),
            Categorical(list(np.arange(0.0001, 1.1, 0.01)), name='nodiff_thres'),
            # Categorical(list(np.arange(0.2, 1.04, 0.05)), name='opacity')
        ]

        # Objective function to minimize
        def objective(params):
            quantile_interval, nodiff_thres = params
            opacity = 1.
            return calculate_loss(quantile_interval, nodiff_thres, opacity, self, format)

        # Run Bayesian Optimization
        res = gp_minimize(
            objective,  # Objective function
            space,  # Parameter space
            n_calls=num_iter,  # Number of iterations
            random_state=42,  # Random state for reproducibility
            verbose=False  # Display optimization progress
        )

        # Return the best found parameters
        best_quantile_interval = res.x[0]
        best_nodiff_thres = res.x[1]
        # best_opacity = res.x[2]
        best_opacity = 1.

        print(best_quantile_interval, best_nodiff_thres, self.quantile_interval, self.nodiff_thres)
        print("Best quantile_interval: ", best_quantile_interval)
        print("Best nodiff_thres: ", best_nodiff_thres)
        print("Best opacity: ", best_opacity)
        self.opacity = best_opacity
        self.quantile_interval = best_quantile_interval
        self.nodiff_thres = best_nodiff_thres

    def calc_l2(self, mask, opacity):
        structuring_element = np.ones((2, 2))

        # Apply binary closing to fill the gaps
        filled_mask = binary_closing(mask, structure=structuring_element)
        sdf_weights = get_sdf(filled_mask, method='skfmm', dx=1.0, normalize='lo1')
        sdf_weights_tensor = torch.tensor(sdf_weights, dtype=torch.float32, requires_grad=False)

        mask_tensor = torch.tensor(mask, dtype=torch.float32, requires_grad=False)
        masked_weights = mask_tensor * sdf_weights_tensor
        masked_weights[(masked_weights < masked_weights.mean()) | (
                masked_weights > 0.9 * masked_weights.max() + 0.1 * masked_weights.mean())] = 0
        masked_weights[masked_weights != 0] = 1
        gt_tensor = torch.tensor(self.gt, dtype=torch.float32, requires_grad=False)
        pred_tensor = torch.tensor(self.pred, dtype=torch.float32, requires_grad=False)
        mean_color = torch.tensor(
            self.calc_mean_color(masked_pixels=gt_tensor[0].permute(1,2,0)[masked_weights==1],
                                 mask=mask))[0]
        try:
            mean_color_expanded = mean_color.view(1, 3, 1, 1)  # Shape: (1, 3, 1, 1)
        except:
            return torch.tensor([1000000000000000000000000000000.0])
        mean_color_filled = mask_tensor[None, :, :] * mean_color_expanded  # Shape: (1, 3, H, W)
        mask_expanded = mask_tensor[None, :, :]  # Expand mask to (1, 1, H, W)
        pred_filled = pred_tensor * (1 - mask_expanded*opacity) + mean_color_filled*opacity
        diff_tensor = ((gt_tensor - pred_filled) ** 2).sum((0, 1))

        # overlap loss
        prev_loss = ((gt_tensor - pred_tensor) ** 2).sum((0, 1))
        overlap_loss = torch.sum(diff_tensor[((diff_tensor > prev_loss) & (prev_loss < 0.1))]) * 100
        if diff_tensor.sum() >= prev_loss.sum():
            return torch.tensor([1000000000000000000000000000000.0])
        # torch.sum(mask_tensor[mask_tensor==0] encourages initializing bigger shapes first
        loss = diff_tensor.sum() + overlap_loss.sum() + 0.1*torch.sum(mask_tensor[mask_tensor==0])
        print("Loss: ", loss.item(), diff_tensor.sum(), overlap_loss.sum(), torch.sum(mask_tensor==0))

        if self.loss > loss:
            self.loss = loss
            self.best_color = mean_color
            self.next_pred = pred_filled
            print("new best color obtained: ", self.best_color)
        return loss

    def __call__(self):
        if self.loss >= 1e10:
            with torch.no_grad():
                self.optimize_params_bayesian(format=self.format)
                torch.cuda.empty_cache()
        component_binary = self.component_binary
        border_points = self.find_border_points(component_binary)
        mean_color = [torch.tensor(self.best_color[0]), torch.tensor(self.best_color[1]),
                      torch.tensor(self.best_color[2]), torch.tensor(self.opacity)]
        #======================================================

        self.pred = self.next_pred.numpy()
        self.loss = 1e10
        return {
            'center': [None, None],
            'border_points': border_points,
            'mean_color': mean_color
        }


class linear_decay_lrlambda_f(object):
    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n//self.decay_every
        decay_step = n %self.decay_every
        lr_s = self.decay_ratio**decay_time
        lr_e = self.decay_ratio**(decay_time+1)
        r = decay_step/self.decay_every
        lr = lr_s * (1-r) + lr_e * r
        return lr
