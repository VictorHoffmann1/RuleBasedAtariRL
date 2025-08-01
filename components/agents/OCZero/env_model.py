import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class EnvModel(nn.Module):
    """Class that tries to model the objects dynamics in the
    environment by predicting the objects' next state given
    the current state and action."""

    def __init__(
        self, n_features, n_actions, hidden_dim=12, feedforward_dim=12, num_queries=32
    ):
        super(EnvModel, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Use a DETR-like architecture and loss
        # NOTE: A bit too complex IMO, maybe try to simplify by tracking objects

        # Use an embedding so that objects and actions have the same dimension
        self.obj_embedding = nn.Linear(n_features, hidden_dim)
        self.action_embedding = nn.Linear(n_actions, hidden_dim)

        # Learnable object queries for DETR-like approach
        self.object_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

        transformer_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=1, dim_feedforward=feedforward_dim, dropout=0.0
        )
        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=1)

        self.fc_out = nn.Linear(
            hidden_dim, n_features + 1
        )  # + 1 for is-object prediction

    def forward(self, x, actions):
        # x: (batch_size, num_objects, n_features)
        # features: cx,cy,vx,vy,w,h
        # actions: (batch_size, 1)

        # Create padding mask for memory (encoder output)
        x, obj_padding_mask = self.trim(x)  # (batch_size, num_objects, n_features)
        memory_padding_mask = torch.cat(
            [
                torch.zeros((x.size(0), 1), dtype=torch.bool, device=x.device),
                obj_padding_mask,
            ],
            dim=1,
        )  # (batch_size, 1 + num_objects)

        actions_one_hot = (
            torch.nn.functional.one_hot(
                actions.flatten().long(), num_classes=self.n_actions
            )
            .float()
            .to(x.device)
        )

        # Create encoder memory (embeddings of objects and actions)
        obj_embeddings = self.obj_embedding(x)
        action_embeddings = self.action_embedding(actions_one_hot).unsqueeze(1)
        memory = torch.cat([action_embeddings, obj_embeddings], dim=1)
        # memory shape: (batch_size, 1 + num_objects, hidden_dim)

        # Prepare object queries for each batch
        batch_size = x.size(0)
        tgt = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        # tgt shape: (batch_size, num_queries, hidden_dim)

        # Transpose for transformer (seq_len, batch_size, hidden_dim)
        tgt = tgt.permute(1, 0, 2)  # (num_queries, batch_size, hidden_dim)
        memory = memory.permute(1, 0, 2)  # (1 + num_objects, batch_size, hidden_dim)

        # Apply transformer decoder
        transformer_output = self.transformer(
            tgt=tgt, memory=memory, memory_key_padding_mask=memory_padding_mask
        )
        # transformer_output shape: (num_queries, batch_size, hidden_dim)

        # Transpose back and apply final linear layer
        transformer_output = transformer_output.permute(
            1, 0, 2
        )  # (batch_size, num_queries, hidden_dim)

        return self.fc_out(transformer_output)

    def compute_loss(
        self, predictions, targets, iou_weight=1.0, l1_weight=1.0, bce_weight=1.0
    ):
        # targets: (batch_size, num_objects, n_features)
        # predictions: (batch_size, num_queries, n_features + 1)

        batch_size = predictions.shape[0]
        num_queries = predictions.shape[1]

        # Split predictions into object features and objectness scores
        pred_objects = predictions[:, :, :-1]  # (batch_size, num_queries, n_features)
        pred_objectness = predictions[:, :, -1]  # (batch_size, num_queries)

        total_loss = 0.0
        # total_iou_loss = 0.0
        total_l1_pos_loss = 0.0
        total_l1_speed_loss = 0.0
        total_l1_shape_loss = 0.0
        total_bce_loss = 0.0

        for b in range(batch_size):
            # Get valid targets (non-zero objects)
            target_mask = targets[b].abs().sum(dim=-1) != 0  # (num_objects,)
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
            objectness_labels = torch.zeros(num_queries, device=predictions.device)
            objectness_labels[pred_indices] = 1.0

            # Compute losses for matched predictions
            matched_preds = pred_objects[b][pred_indices]
            matched_targets = valid_targets[target_indices]

            # IoU loss for bounding boxes (cx, cy, w, h are at indices 0, 1, 4, 5)
            # iou_loss = self._compute_iou_loss(matched_preds, matched_targets)
            # total_iou_loss += iou_loss

            # L1 loss for positions (cx, cy are at indices 0, 1)
            l1_pos_loss = F.l1_loss(matched_preds[:, :2], matched_targets[:, :2])
            total_l1_pos_loss += l1_pos_loss

            # L1 loss for velocities (vx, vy are at indices 2, 3)
            l1_speed_loss = F.l1_loss(matched_preds[:, 2:4], matched_targets[:, 2:4])
            total_l1_speed_loss += l1_speed_loss

            # L1 loss for shapes (w, h are at indices 4, 5)
            l1_shape_loss = F.l1_loss(matched_preds[:, 4:6], matched_targets[:, 4:6])
            total_l1_shape_loss += l1_shape_loss

            # Binary cross-entropy for objectness
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_objectness[b], objectness_labels
            )
            total_bce_loss += bce_loss

        # Average losses over batch
        # avg_iou_loss = total_iou_loss / batch_size
        avg_l1_speed_loss = total_l1_speed_loss / batch_size
        avg_l1_pos_loss = total_l1_pos_loss / batch_size
        avg_l1_shape_loss = total_l1_shape_loss / batch_size
        avg_bce_loss = total_bce_loss / batch_size

        # Combine losses with weights (you can adjust these)
        total_loss = (
            # iou_weight * avg_iou_loss
            l1_weight * (avg_l1_pos_loss + avg_l1_speed_loss + avg_l1_shape_loss)
            + bce_weight * avg_bce_loss
        )

        return {
            "total_loss": total_loss,
            # "iou_loss": avg_iou_loss,
            "l1_pos_loss": avg_l1_pos_loss,
            "l1_speed_loss": avg_l1_speed_loss,
            "l1_shape_loss": avg_l1_shape_loss,
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
            object_scores = predictions[:, :, -1]  # Keep as tensor
            object_mask = object_scores > threshold

            # Extract objects for each batch item
            objects_list = []
            scores_list = []

            for b in range(predictions.shape[0]):
                batch_mask = object_mask[b]
                if batch_mask.any():
                    batch_objects = (
                        predictions[b, batch_mask, :-1].detach().cpu().numpy()
                    )
                    batch_scores = object_scores[b, batch_mask].detach().cpu().numpy()

                    # UGLY DENORMALIZATION - should be moved to a separate method
                    batch_objects[..., 2:4] *= 8.0
                    batch_objects[..., 0] *= 160
                    batch_objects[..., 4] *= 160
                    batch_objects[..., 1] *= 210
                    batch_objects[..., 5] *= 210

                    objects_list.append(batch_objects)
                    scores_list.append(batch_scores)

        return objects_list, scores_list

    @staticmethod
    def trim(x):
        """
        Remove trailing zero-padded objects from the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) with zero-padded objects.
        Returns:
            torch.Tensor: Tensor with zero-padded objects trimmed, shape (B, max_valid_N, D).
        """

        obj_padding_mask = x.abs().sum(dim=-1) != 0  # (B, N)

        max_valid = obj_padding_mask.sum(dim=1).max()

        return x[:, :max_valid, :], ~obj_padding_mask[
            :, :max_valid
        ]  # Return trimmed tensor and mask
