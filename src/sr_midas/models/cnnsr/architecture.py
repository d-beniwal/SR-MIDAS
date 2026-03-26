"""CNN model class for super-resolution (canonical source: SR-MIDAS/super_res_process.py)"""

import torch
import torch.nn as nn


class CNNSR(torch.nn.Module):
    """CNN model class for super-resolution"""

    def __init__(self, l_ch_nrs, l_ker_size, l_act_func, X_channels):
        """Initialization function of the class; runs automatically when class is created
        Args:
            l_ch_nrs (list of int): list with number of channels in each successive CNN layer
            l_ker_size (list of odd int): list with kernel size for each successive CNN layer
            l_act_func (list of str): list with activation function ('r', 's', 'lr', 't') each successive CNN layer
            X_channels (int): no. of channels in input X data
        """

        super(CNNSR, self).__init__()

        self.cnn_ops = []
        cnn_out_chs = tuple(l_ch_nrs)
        cnn_in_chs = (X_channels, ) + cnn_out_chs[:-1]

        for (in_ch, out_ch, ker_sz, act_func) in zip(
            cnn_in_chs, cnn_out_chs, l_ker_size, l_act_func):

            # Add convolution layer
            pad_sz = int(ker_sz / 2)
            self.cnn_ops += [
                torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=ker_sz, padding=pad_sz)
            ]

            # Add activation layer
            if act_func == "r": self.cnn_ops += [torch.nn.ReLU()]
            if act_func == "s": self.cnn_ops += [torch.nn.Sigmoid()]
            if act_func == "lr": self.cnn_ops += [torch.nn.LeakyReLU()]
            if act_func == "t": self.cnn_ops += [torch.nn.Tanh()]

        self.cnn_layers = torch.nn.Sequential(*self.cnn_ops)

    def forward(self, x):

        _out = x
        for layer in self.cnn_layers:
            _out = layer(_out)

        return (_out)
