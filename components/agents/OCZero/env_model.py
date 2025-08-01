import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class EnvModel(nn.Module):
    """Class that tries to model the objects dynamics in the
    environment by predicting the objects' next state given
    the current state and action."""

    def __init__(self, n_features, n_actions, hidden_dim=16, feedforward_dim=64):
        super(EnvModel, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        # Use a DETR-like architecture and loss
        # NOTE: A bit too complex IMO, maybe try to simplify by tracking objects

        # Use an embedding so that objects and actions have the same dimension
        self.obj_embedding = nn.Linear(n_features, hidden_dim)
        self.action_embedding = nn.Linear(n_actions, hidden_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=1, dim_feedforward=feedforward_dim, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)

        self.fc_out = nn.Linear(
            hidden_dim, n_features + 1
        )  # + 1 for is-object prediction

    def forward(self, x, actions):
        # x: (batch_size, num_objects, n_features)
        # features: cx,cy,vx,vy,w,h
        # actions: (batch_size, 1)

        # Add a 0 to the padding mask for the action token
        padding_mask = torch.cat(
            [torch.zeros(x.size(0), device=x.device), (x.sum(dim=-1) == 0).all(dim=-1)],
            dim=0,
        )
        actions_one_hot = torch.nn.functional.one_hot(
            actions.flatten().long(), num_classes=self.n_actions, device=x.device
        ).float()

        obj_embeddings = self.obj_embedding(x)
        action_embeddings = self.action_embedding(actions_one_hot).unsqueeze(1)
        embeddings = torch.cat([action_embeddings, obj_embeddings], dim=1)

        transformer_output = self.transformer(
            embeddings.permute(1, 0, 2), src_key_padding_mask=padding_mask
        ).permute(1, 0, 2)

        return self.fc_out(transformer_output)

    def compute_loss(
        self, predictions, targets, iou_weight=1.0, l1_weight=1.0, bce_weight=1.0
    ):
        # targets: (batch_size, num_objects, n_features)
        # predictions: (batch_size, num_objects, n_features + 1)

        batch_size = predictions.shape[0]
        num_pred_objects = predictions.shape[1]

        # Split predictions into object features and objectness scores
        pred_objects = predictions[:, :, :-1]  # (batch_size, num_objects, n_features)
        pred_objectness = predictions[:, :, -1]  # (batch_size, num_objects)

        total_loss = 0.0
        total_iou_loss = 0.0
        total_l1_loss = 0.0
        total_bce_loss = 0.0

        for b in range(batch_size):
            # Get valid targets (non-zero objects)
            target_mask = targets[b].sum(dim=-1) != 0  # (num_objects,)
            valid_targets = targets[b][target_mask]  # (num_valid_targets, n_features)
            num_valid_targets = valid_targets.shape[0]

            if num_valid_targets == 0:
                # No valid targets, all predictions should be background
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred_objectness[b], torch.zeros_like(pred_objectness[b])
                )
                total_bce_loss += bce_loss
                continue

            # Compute cost matrix for Hungarian matching
            cost_matrix = self._compute_cost_matrix(pred_objects[b], valid_targets)

            # Hungarian matching
            pred_indices, target_indices = linear_sum_assignment(
                cost_matrix.detach().cpu().numpy()
            )

            # Convert to tensors
            pred_indices = torch.tensor(pred_indices, device=predictions.device)
            target_indices = torch.tensor(target_indices, device=predictions.device)

            # Create objectness labels
            objectness_labels = torch.zeros(num_pred_objects, device=predictions.device)
            objectness_labels[pred_indices] = 1.0

            # Compute losses for matched predictions
            matched_preds = pred_objects[b][pred_indices]
            matched_targets = valid_targets[target_indices]

            # IoU loss for bounding boxes (cx, cy, w, h are at indices 0, 1, 4, 5)
            iou_loss = self._compute_iou_loss(matched_preds, matched_targets)
            total_iou_loss += iou_loss

            # L1 loss for velocities (vx, vy are at indices 2, 3)
            l1_loss = F.l1_loss(matched_preds[:, 2:4], matched_targets[:, 2:4])
            total_l1_loss += l1_loss

            # Binary cross-entropy for objectness
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_objectness[b], objectness_labels
            )
            total_bce_loss += bce_loss

        # Average losses over batch
        avg_iou_loss = total_iou_loss / batch_size
        avg_l1_loss = total_l1_loss / batch_size
        avg_bce_loss = total_bce_loss / batch_size

        # Combine losses with weights (you can adjust these)
        total_loss = (
            iou_weight * avg_iou_loss
            + l1_weight * avg_l1_loss
            + bce_weight * avg_bce_loss
        )

        return {
            "total_loss": total_loss,
            "iou_loss": avg_iou_loss,
            "l1_loss": avg_l1_loss,
            "bce_loss": avg_bce_loss,
        }

    def _compute_cost_matrix(self, predictions, targets):
        """Compute cost matrix for Hungarian matching."""
        # predictions: (num_pred_objects, n_features)
        # targets: (num_valid_targets, n_features)

        num_preds = predictions.shape[0]
        num_targets = targets.shape[0]

        # Expand for broadcasting
        pred_expanded = predictions.unsqueeze(1).expand(num_preds, num_targets, -1)
        target_expanded = targets.unsqueeze(0).expand(num_preds, num_targets, -1)

        # Compute IoU cost for bounding boxes
        iou_cost = 1.0 - self._compute_iou_batch(pred_expanded, target_expanded)

        return iou_cost

    def _compute_iou_loss(self, predictions, targets):
        """Compute IoU loss for bounding boxes."""
        iou = self._compute_iou_batch(
            predictions.unsqueeze(1), targets.unsqueeze(0)
        ).diagonal()
        return (1.0 - iou).mean()

    def _compute_iou_batch(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes."""
        # boxes: (..., 6) where format is (cx, cy, vx, vy, w, h)
        # Extract bounding box coordinates
        cx1, cy1, w1, h1 = (
            boxes1[..., 0],
            boxes1[..., 1],
            boxes1[..., 4],
            boxes1[..., 5],
        )
        cx2, cy2, w2, h2 = (
            boxes2[..., 0],
            boxes2[..., 1],
            boxes2[..., 4],
            boxes2[..., 5],
        )

        # Convert center-width-height to corner coordinates
        x1_min, y1_min = cx1 - w1 / 2, cy1 - h1 / 2
        x1_max, y1_max = cx1 + w1 / 2, cy1 + h1 / 2
        x2_min, y2_min = cx2 - w2 / 2, cy2 - h2 / 2
        x2_max, y2_max = cx2 + w2 / 2, cy2 + h2 / 2

        # Compute intersection
        inter_xmin = torch.max(x1_min, x2_min)
        inter_ymin = torch.max(y1_min, y2_min)
        inter_xmax = torch.min(x1_max, x2_max)
        inter_ymax = torch.min(y1_max, y2_max)

        inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_width * inter_height

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        # Compute IoU
        iou = inter_area / (union_area + 1e-6)

        return iou

    def get_objects(self, x, actions, threshold=0.5):
        """Get the predicted objects from the model."""
        with torch.no_grad():
            # x: (batch_size, num_objects, n_features)
            # actions: (batch_size, 1)
            predictions = self.forward(x, actions)
            # Get the objects with a prediction above the threshold
            object_scores = predictions[:, :, -1].detach().cpu().numpy()
            objects = (
                predictions[:, :, :-1][object_scores > threshold].detach().cpu().numpy()
            )
            # UGLY DENORMALIZATION
            objects[..., 2:4] *= 8.0
            objects[..., 0] *= 160
            objects[..., 4] *= 160
            objects[..., 1] *= 210
            objects[..., 5] *= 210

        return objects, object_scores[object_scores > threshold]
