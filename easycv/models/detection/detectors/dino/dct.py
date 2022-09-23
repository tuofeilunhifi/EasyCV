# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import numpy as np
import functools

print = functools.partial(print, flush=True)

class ProcessorDCT(object):
    def __init__(self, n_keep, gt_mask_len):
        self.n_keep = n_keep
        self.gt_mask_len = gt_mask_len

        inputs = np.zeros((self.gt_mask_len, self.gt_mask_len))
        _, zigzag_table = self.zigzag(inputs)
        self.zigzag_table = zigzag_table[:self.n_keep]

    def sigmoid(self, x):
        """Apply the sigmoid operation.
        """
        y = 1. / (1. + 1. / np.exp(x))
        dy = y * (1 - y)
        return y

    def inverse_sigmoid(self, x):
        """Apply the inverse sigmoid operation.
                y = -ln(1-x/x)
        """
        y = -1 * np.log((1 - x) / x)
        return y

    def zigzag(self, input, gt=None):
        """
        Zigzag scan of a matrix
        Argument is a two-dimensional matrix of any size,
        not strictly a square one.
        Function returns a 1-by-(m*n) array,
        where m and n are sizes of an input matrix,
        consisting of its items scanned by a zigzag method.

        Args:
            input (np.array): shape [h,w], value belong to [-127, 128], transformed from gt.
            gt (np.array): shape [h,w], value belong to {0,1}, original instance segmentation gt mask.
        Returns:
            output (np.array): zig-zag encoded values, shape [h*w].
            indicator (np.array): positive sample indicator, shape [h,w].
        """
        # initializing the variables
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = input.shape[0]
        hmax = input.shape[1]
        assert vmax == hmax

        i = 0
        output = np.zeros((vmax * hmax))
        indicator = []

        while ((v < vmax) and (h < hmax)):
            if ((h + v) % 2) == 0:  # going up
                if (v == vmin):
                    output[i] = input[v, h]  # if we got to the first line
                    indicator.append(v * vmax + h)
                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1
                    i = i + 1
                elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                    output[i] = input[v, h]
                    indicator.append(v * vmax + h)
                    v = v + 1
                    i = i + 1
                elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                    output[i] = input[v, h]
                    indicator.append(v * vmax + h)
                    v = v - 1
                    h = h + 1
                    i = i + 1
            else:  # going down
                if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                    output[i] = input[v, h]
                    indicator.append(v * vmax + h)
                    h = h + 1
                    i = i + 1
                elif (h == hmin):  # if we got to the first column
                    output[i] = input[v, h]
                    indicator.append(v * vmax + h)
                    if (v == vmax - 1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                    output[i] = input[v, h]
                    indicator.append(v * vmax + h)
                    v = v + 1
                    h = h - 1
                    i = i + 1
            if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
                output[i] = input[v, h]
                indicator.append(v * vmax + h)
                break
        return output, indicator

    def inverse_zigzag(self, input, vmax, hmax):
        """
        Zigzag scan of a matrix
        Argument is a two-dimensional matrix of any size,
        not strictly a square one.
        Function returns a 1-by-(m*n) array,
        where m and n are sizes of an input matrix,
        consisting of its items scanned by a zigzag method.
        """
        # initializing the variables
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        output = np.zeros((vmax, hmax))
        i = 0
        while ((v < vmax) and (h < hmax)):
            if ((h + v) % 2) == 0:  # going up
                if (v == vmin):
                    output[v, h] = input[i]  # if we got to the first line
                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1
                    i = i + 1
                elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                    output[v, h] = input[i]
                    v = v + 1
                    i = i + 1
                elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                    output[v, h] = input[i]
                    v = v - 1
                    h = h + 1
                    i = i + 1
            else:  # going down
                if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                    output[v, h] = input[i]
                    h = h + 1
                    i = i + 1
                elif (h == hmin):  # if we got to the first column
                    output[v, h] = input[i]
                    if (v == vmax - 1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                    output[v, h] = input[i]
                    v = v + 1
                    h = h - 1
                    i = i + 1
            if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
                output[v, h] = input[i]
                break
        return output