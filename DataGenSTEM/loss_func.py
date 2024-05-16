import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha  # Weight for penalizing false positives (alpha=1 is neutral, alpha>1 penalizes false positives, alpha<1 penalizes false negatives)

    def forward(self, input, target):
        input = input.view(-1)  # Flatten the input
        target = target.view(-1)  # Flatten the target
        intersection = torch.sum(input * target)
        # Apply a weight to the prediction sum to penalize false positives
        weighted_union = self.alpha * torch.sum(input) + torch.sum(target)
        dice_coeff = (2. * intersection + self.smooth) / (weighted_union + self.smooth)
        dice_coeff = torch.clamp(dice_coeff, 0, 1)
        return 1. - dice_coeff

class GombinatorialLoss(nn.Module):
    def __init__(self, group_size, loss='Dice', epsilon=1e-6, class_weights=None, alpha=2):
        super(GombinatorialLoss, self).__init__()
        self.group_size = group_size
        self.epsilon = epsilon
        self.class_weights = class_weights
        self.loss = loss.lower()
        self.alpha = alpha # for Dice loss

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        total_loss = 0.0

        # Apply sigmoid to outputs if using Dice loss
        if self.loss == 'dice':
            outputs = torch.sigmoid(outputs)

        dice_loss = DiceLoss(alpha=self.alpha) if self.loss == 'dice' else None

        for i in range(batch_size):
            outputs_group1, outputs_group2 = outputs[i, :self.group_size], outputs[i, self.group_size:]
            targets_group1, targets_group2 = targets[i, :self.group_size], targets[i, self.group_size:]

            if self.loss == 'dice':
                loss00 = dice_loss(outputs_group1, targets_group1)
                loss01 = dice_loss(outputs_group1, targets_group2)
                loss10 = dice_loss(outputs_group2, targets_group1)
                loss11 = dice_loss(outputs_group2, targets_group2)
                output_loss = dice_loss(outputs_group1, outputs_group2) # maybe the penatly should be different
                target_loss = dice_loss(targets_group1, targets_group2) # for these two (modify alhpha)
            else:
                loss00 = F.cross_entropy(outputs_group1.unsqueeze(0), targets_group1, weight=self.class_weights, reduction='mean')
                loss01 = F.cross_entropy(outputs_group1.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')
                loss10 = F.cross_entropy(outputs_group2.unsqueeze(0), targets_group1, weight=self.class_weights, reduction='mean')
                loss11 = F.cross_entropy(outputs_group2.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')
                output_loss = F.cross_entropy(outputs_group1.unsqueeze(0), outputs_group2, weight=self.class_weights, reduction='mean')
                target_loss = F.cross_entropy(targets_group1.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')

            # Compute the inverse of loss pairings and sum for current sample
            inverse_loss = 1 / (loss01 + loss10 + self.epsilon) + 1 / (loss00 + loss11 + self.epsilon)
            prediction_loss = 1 / (inverse_loss + self.epsilon)

            # Loss penalizing similar predictions for G1 and G2
            mse_loss = (target_loss - output_loss) ** 2
            mse_loss = torch.sigmoid(mse_loss)

            # Accumulate the loss
            total_loss += prediction_loss * mse_loss

        # Average the accumulated losses over the batch
        return total_loss / batch_size


# Older code where the batch grouping is not considered
# Old but it works reliably on anything with batch_size = 1

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
# 
#     def forward(self, input, target):
#         # Reshape input and target tensors
#         batch_size = input.size(0)
#         num_classes = input.size(1)
#         input = input.view(batch_size, num_classes, -1) # Reshape to [batch_size, num_classes, height * width]
#         target = target.view(batch_size, num_classes, -1) # Reshape to [batch_size, num_classes, height * width]
# 
#         # Compute intersection and union
#         intersection = torch.sum(input * target, dim=2) # Compute intersection along spatial dimensions
#         union = torch.sum(input, dim=2) + torch.sum(target, dim=2) # Compute union along spatial dimensions
# 
#         # Compute Dice coefficient
#         dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
#         # cut off dice coeff to be within expected range
#         dice_coeff = torch.clamp(dice_coeff,0,1)
#         # Compute Dice Loss
#         dice_loss = 1. - dice_coeff
#         dice_loss = dice_loss.mean() #average of dice_loss over batch and classes
#         return dice_loss
# 
# 
# class GombinatorialLoss(nn.Module):
#     def __init__(self, group_size, loss = 'Dice', epsilon=1e-6, class_weights = None):
#         super(GombinatorialLoss, self).__init__()
#         self.group_size = group_size
#         self.epsilon = epsilon  # small value to avoid division by zero
#         self.class_weights = class_weights
#         self.loss = loss.lower()
# 
#     def forward(self, outputs, targets):
#         # Split outputs and targets into groups
#         outputs_group1, outputs_group2 = outputs[:, :self.group_size], outputs[:, self.group_size:]
#         targets_group1, targets_group2 = targets[:, :self.group_size], targets[:, self.group_size:]
# 
#         # Compute loss for each combination
#         if self.loss == 'dice':
#             # Apply sigmoid to outputs
#             outputs = F.sigmoid(outputs)
# 
#             dice_loss = DiceLoss()
#             loss00 = dice_loss(outputs_group1, targets_group1)
#             loss01 = dice_loss(outputs_group1, targets_group2)
#             loss10 = dice_loss(outputs_group2, targets_group1)
#             loss11 = dice_loss(outputs_group2, targets_group2)
#             #output_loss = dice_loss(outputs_group1, outputs_group2)
#             #target_loss = dice_loss(targets_group1, targets_group2)
# 
#         if self.loss == 'crossentropy':
#             loss00 = F.cross_entropy(outputs_group1, targets_group1, weight = self.class_weights, reduction='mean')
#             loss01 = F.cross_entropy(outputs_group1, targets_group2, weight = self.class_weights, reduction='mean')
#             loss10 = F.cross_entropy(outputs_group2, targets_group1, weight = self.class_weights, reduction='mean')
#             loss11 = F.cross_entropy(outputs_group2, targets_group2, weight = self.class_weights, reduction='mean')
#             output_loss = F.cross_entropy(outputs_group1, outputs_group2, weight = self.class_weights, reduction='mean')
#             target_loss = F.cross_entropy(targets_group1, targets_group2, weight = self.class_weights, reduction='mean')
# 
#         # Compute the inverse of loss pairings and sum
#         inverse_loss = 1 / (loss01 + loss10 + self.epsilon) + 1 / (loss00 + loss11 + self.epsilon)
# 
#         # Take the inverse of the sum
#         prediction_loss = 1 / (inverse_loss + self.epsilon)
# 
#         # Loss penalizing similar predictions for G1 and G2
#         #mse_loss = (target_loss - output_loss) ** 2
# 
#         # I think multiplication could work better because we need to change the landscape of the loss (shape)
#         # (predicting the same atomic layer on both groups has to just some local minima?)
#         # not just raise or lower the whole curve 
#         return prediction_loss# * mse_loss