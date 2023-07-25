"""
@author: Xinjing Cheng & Peng Wang

"""

import torch.nn as nn
import torch

class SparseAffinity_Propagate(nn.Module):

    def __init__(self,
                 prop_time,
                 prop_kernel,
                 norm_type='8sum'):
        """
        Sparse Affinity propagation

        x x x x x 1 x x x x x
        x x x x x x x x x x x
        x x x x x x x x x x x
        x x x x x x x x x x x
        x x x x x 2 x x x x x
        7 x x x 8 C 4 x x x 3
        x x x x x 6 x x x x x
        x x x x x x x x x x x
        x x x x x x x x x x x
        x x x x x x x x x x x
        x x x x x 5 x x x x x

        Inputs:
            prop_time: how many steps for CSPN to perform
            prop_kernel: the size of kernel (current only support 3x3)
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super(SparseAffinity_Propagate, self).__init__()
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'

        self.norm_type = norm_type
        assert norm_type in ['8sum', '8sum_abs']

        self.in_feature = 1
        self.out_feature = 1

        '''
        x x x x x 0 x x x x x
        x x x x x x x x x x x
        x x x x x x x x x x x
        x x x x x x x x x x x
        x x x x x 1 x x x x x
        6 x x x 7 C 3 x x x 2
        x x x x x 5 x x x x x
        x x x x x x x x x x x
        x x x x x x x x x x x
        x x x x x x x x x x x
        x x x x x 4 x x x x x
        '''
        grep_kernel = torch.zeros((8, 1, 11, 11))
        grep_kernel[0, 0, 0, 5] = 1
        grep_kernel[1, 0, 4, 5] = 1
        grep_kernel[2, 0, 5, 10] = 1
        grep_kernel[3, 0, 5, 6] = 1
        grep_kernel[4, 0, 10, 5] = 1
        grep_kernel[5, 0, 6, 5] = 1
        grep_kernel[6, 0, 5, 0] = 1
        grep_kernel[7, 0, 5, 4] = 1

        spn_kernel = torch.zeros((8, 8, 11, 11))
        grep_kernel[0, 0, 0, 5] = 1
        grep_kernel[1, 1, 4, 5] = 1
        grep_kernel[2, 2, 5, 10] = 1
        grep_kernel[3, 3, 5, 6] = 1
        grep_kernel[4, 4, 10, 5] = 1
        grep_kernel[5, 5, 6, 5] = 1
        grep_kernel[6, 6, 5, 0] = 1
        grep_kernel[7, 7, 5, 4] = 1
        self.grep_kernel = nn.Parameter(grep_kernel, False)
        self.spn_kernel = nn.Parameter(spn_kernel, False)

    def grep_conv(self, depth):
        return torch.nn.functional.conv2d(depth, self.grep_kernel, bias=None, stride=1, padding=5, dilation=1)

    def spn_conv(self, guidance):
        return torch.nn.functional.conv2d(guidance, self.spn_kernel, bias=None, stride=1, padding=5, dilation=1)

    def forward(self, guidance, blur_depth, sparse_depth=None):
        '''
        guidance: Bx8xHxW
        blur_depth: Bx1xHxW

        '''
        gate_wb, gate_sum = self.affinity_normalization(guidance)

        # pad input and convert to 8 channel 3D features
        raw_depth_input = blur_depth

        #blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
        result_depth = blur_depth

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()

        for i in range(self.prop_time):
            # result_depth = Nx1xHxW
            result_depth = self.grep_conv(result_depth)
            result_depth = (gate_wb * result_depth).sum(1, True)

            if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input

        return result_depth

    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)

        gate_wb = self.spn_conv(guidance)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = (gate_wb_abs).sum(1, True)

        gate_wb = torch.div(gate_wb, abs_weight.clamp(min=1e-6))
        gate_sum = (gate_wb).sum(1, True)

        return gate_wb, gate_sum
