#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import time


def _densenet121_imagenet():
    """Create DenseNet121 with ImageNet weights using the current torchvision API."""
    try:
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        return torchvision.models.densenet.densenet121(weights=weights)
    except AttributeError:
        return torchvision.models.densenet.densenet121(pretrained=True)


def _as_device(tensor, use_cuda):
    return tensor.cuda() if use_cuda else tensor


def _affine_grid(affine_mat, output_size, use_cuda):
    affine_tensor = torch.from_numpy(affine_mat).permute(2, 0, 1).float()
    affine_tensor = _as_device(affine_tensor, use_cuda)
    return F.affine_grid(affine_tensor, output_size, align_corners=False)


def _grid_sample(input_tensor, flow_grid):
    return F.grid_sample(input_tensor, flow_grid, mode='nearest', align_corners=False)


def _upsample_prediction(prediction):
    return F.interpolate(prediction, scale_factor=16, mode='bilinear', align_corners=False)


class reactive_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reactive_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.push_color_trunk = _densenet121_imagenet()
        self.push_depth_trunk = _densenet121_imagenet()
        self.grasp_color_trunk = _densenet121_imagenet()
        self.grasp_depth_trunk = _densenet121_imagenet()

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight)
                elif isinstance(m[1], nn.BatchNorm2d):
                    with torch.no_grad():
                        m[1].weight.fill_(1)
                        m[1].bias.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            output_prob = []
            interm_feat = []

            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(affine_mat_before.cuda(), input_color_data.size(), align_corners=False)
                else:
                    flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=False)

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_color = F.grid_sample(input_color_data.cuda(), flow_grid_before, mode='nearest', align_corners=False)
                    rotate_depth = F.grid_sample(input_depth_data.cuda(), flow_grid_before, mode='nearest', align_corners=False)
                else:
                    rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=False)
                    rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=False)

                # Compute intermediate features
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                interm_feat.append([interm_push_feat, interm_grasp_feat])

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2,3,1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(affine_mat_after.cuda(), interm_push_feat.size(), align_corners=False)
                else:
                    flow_grid_after = F.affine_grid(affine_mat_after, interm_push_feat.size(), align_corners=False)

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest', align_corners=False)),
                                    nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest', align_corners=False))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(affine_mat_before.cuda(), input_color_data.size(), align_corners=False)
            else:
                flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=False)

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(input_color_data.cuda(), flow_grid_before, mode='nearest', align_corners=False)
                rotate_depth = F.grid_sample(input_depth_data.cuda(), flow_grid_before, mode='nearest', align_corners=False)
            else:
                rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=False)
                rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=False)

            # Compute intermediate features
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_push_feat, interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(affine_mat_after.cuda(), interm_push_feat.size(), align_corners=False)
            else:
                flow_grid_after = F.affine_grid(affine_mat_after, interm_push_feat.size(), align_corners=False)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest', align_corners=False)),
                                     nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest', align_corners=False))])

            return self.output_prob, self.interm_feat


class reinforcement_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.push_color_trunk = _densenet121_imagenet()
        self.push_depth_trunk = _densenet121_imagenet()
        self.grasp_color_trunk = _densenet121_imagenet()
        self.grasp_depth_trunk = _densenet121_imagenet()

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight)
                elif isinstance(m[1], nn.BatchNorm2d):
                    with torch.no_grad():
                        m[1].weight.fill_(1)
                        m[1].bias.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(affine_mat_before.cuda(), input_color_data.size(), align_corners=False)
                    else:
                        flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=False)

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(input_color_data.cuda(), flow_grid_before, mode='nearest', align_corners=False)
                        rotate_depth = F.grid_sample(input_depth_data.cuda(), flow_grid_before, mode='nearest', align_corners=False)
                    else:
                        rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=False)
                        rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=False)

                    # Compute intermediate features
                    interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                    interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                    interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                    interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    interm_feat.append([interm_push_feat, interm_grasp_feat])

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(affine_mat_after.cuda(), interm_push_feat.size(), align_corners=False)
                    else:
                        flow_grid_after = F.affine_grid(affine_mat_after, interm_push_feat.size(), align_corners=False)

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest', align_corners=False)),
                                        nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest', align_corners=False))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(affine_mat_before.cuda(), input_color_data.size(), align_corners=False)
            else:
                flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=False)

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(input_color_data.cuda(), flow_grid_before, mode='nearest', align_corners=False)
                rotate_depth = F.grid_sample(input_depth_data.cuda(), flow_grid_before, mode='nearest', align_corners=False)
            else:
                rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=False)
                rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=False)

            # Compute intermediate features
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_push_feat, interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(affine_mat_after.cuda(), interm_push_feat.size(), align_corners=False)
            else:
                flow_grid_after = F.affine_grid(affine_mat_after, interm_push_feat.size(), align_corners=False)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest', align_corners=False)),
                                     nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest', align_corners=False))])

            return self.output_prob, self.interm_feat


    # # OLD VERSION: IMPLICIT ROTATION INSIDE
    # def forward(self, input_color_data, input_depth_data, is_volatile=False):

    #     # Run forward pass through trunk to get intermediate features
    #     if is_volatile:
    #         interm_color_feat = self.color_trunk.features(input_color_data.cuda())
    #         interm_depth_feat = self.depth_trunk.features(input_depth_data.cuda())
    #         interm_feat = torch.cat((interm_color_feat, interm_depth_feat), dim=1)
    #         output_prob = []

    #         # Apply rotations to intermediate features
    #         for rotate_idx in range(self.num_rotations):
    #             rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

    #             # Compute sample grid for rotation BEFORE branches
    #             affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
    #             affine_mat_before.shape = (2,3,1)
    #             affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
    #             flow_grid_before = F.affine_grid(affine_mat_before.cuda(), interm_feat.size(), align_corners=False)

    #             # Rotate intermediate features clockwise
    #             rotate_feat = F.grid_sample(interm_feat, flow_grid_before, mode='nearest', align_corners=False)
    #             # test = rotate_feat.cpu().data.numpy()
    #             # test = np.sum(test[0,:,:,:], axis=0)
    #             # plt.imshow(test)
    #             # plt.show()

    #             # Compute sample grid for rotation AFTER branches
    #             affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
    #             affine_mat_after.shape = (2,3,1)
    #             affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
    #             flow_grid_after = F.affine_grid(affine_mat_after.cuda(), rotate_feat.size(), align_corners=False)

    #             # Forward pass through branches, undo rotation on output predictions, upsample results
    #             output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(rotate_feat), flow_grid_after, mode='nearest', align_corners=False)),
    #                                 nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(rotate_feat), flow_grid_after, mode='nearest', align_corners=False))])

    #         return output_prob, interm_feat

    #     else:
    #         interm_color_feat = self.color_trunk.features(input_color_data.cuda())
    #         interm_depth_feat = self.depth_trunk.features(input_depth_data.cuda())
    #         self.interm_feat = torch.cat((interm_color_feat, interm_depth_feat), dim=1)
    #         self.output_prob = []

    #         # Apply rotations to intermediate features
    #         # for rotate_idx in range(self.num_rotations):
    #         rotate_idx = specific_rotation
    #         rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

    #         # Compute sample grid for rotation BEFORE branches
    #         affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
    #         affine_mat_before.shape = (2,3,1)
    #         affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
    #         flow_grid_before = F.affine_grid(affine_mat_before.cuda(), self.interm_feat.size(), align_corners=False)

    #         # Rotate intermediate features clockwise
    #         rotate_feat = F.grid_sample(self.interm_feat, flow_grid_before, mode='nearest', align_corners=False)

    #         # Compute sample grid for rotation AFTER branches
    #         affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
    #         affine_mat_after.shape = (2,3,1)
    #         affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
    #         flow_grid_after = F.affine_grid(affine_mat_after.cuda(), rotate_feat.size(), align_corners=False)

    #         # Forward pass through branches, undo rotation on output predictions, upsample results
    #         self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(rotate_feat), flow_grid_after, mode='nearest', align_corners=False)),
    #                                  nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(rotate_feat), flow_grid_after, mode='nearest', align_corners=False))])

    #         return self.output_prob, self.interm_feat
