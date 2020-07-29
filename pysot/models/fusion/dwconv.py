#--------------------------------------------------
#Copyright (c)
#Licensed under the MIT License
#Written by yeyi (18120438@bjtu.edu.cn)
#--------------------------------------------------

import torch
import math

class KernelDWConvFn(torch.autograd.function.Function):
    """2D convolution with kernel.
    Copy from: https://github.com/NVlabs/pacnet/blob/master/pac.py/PacConv2dFn
    """
    @staticmethod
    def forward(ctx, inputs, kernel, weight, bias=None, stride=1, padding=0,
                dilation=1):
        """Forward computation.

        Args:
            inputs: A tensor with shape [batch, channels, height, width]
                representing a batch of images.
            kernel: A tensor with shape [batch, channels, k, k],
                where k = kernel_size
            weight: A tensor with shape [out_channels, in_channels,
                kernel_size, kernel_size].
            bias: None or a tenor with shape [out_channels].

        Returns:
            outputs: A tensor with shape [batch, out_channels, N, N].
            N = number of slide windows.
        """
        (batch_size, channels), input_size = inputs.shape[:2], inputs.shape[2:]
        ctx.in_channels = channels
        ctx.input_size = input_size
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = torch.nn.modules.utils._pair(dilation)
        ctx.padding = torch.nn.modules.utils._pair(padding)
        ctx.stride = torch.nn.modules.utils._pair(stride)

        needs_input_grad = ctx.needs_input_grad
        ctx.save_for_backward(
            inputs if (needs_input_grad[1] or needs_input_grad[2]) else None,
            kernel if (needs_input_grad[0] or needs_input_grad[2]) else None,
            weight if (needs_input_grad[0] or needs_input_grad[1]) else None)
        ctx._backend = torch._thnn.type2backend[inputs.type()]
        out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                        for (i, k, d, p, s) in zip(ctx.input_size, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)])
        # Slide windows, [batch, channels x kernel_size x kernel_size, N x N],
        # where N is the number of slide windows.
        inputs_wins = torch.nn.functional.unfold(inputs, ctx.kernel_size,
                                                 ctx.dilation, ctx.padding,
                                                 ctx.stride)

        kernel_view = kernel.view(*kernel.shape, 1, 1)
        inputs_mul_kernel = inputs_wins.view(
            batch_size, channels, weight.shape[2], weight.shape[3], *out_sz) * kernel_view

        # Matrix multiplication
        outputs = torch.einsum('ijklmn,ojkl->iomn', (inputs_mul_kernel, weight))

        if bias is not None:
            outputs += bias.view(1, -1, 1, 1)
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_outputs):
        grad_inputs = grad_kernel = grad_weight = grad_bias = None
        batch_size, out_channels = grad_outputs.shape[:2]
        output_size = grad_outputs.shape[2:]
        in_channels = ctx.in_channels

        # Compute gradients
        inputs, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_inputs_mul_kernel = torch.einsum('iomn,ojkl->ijklmn',
                                                  (grad_outputs, weight))
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            kernel_view = kernel.view(batch_size, in_channels,
                                      ctx.kernel_size[0],
                                      ctx.kernel_size[1],
                                      1, 1)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            inputs_wins = torch.nn.functional.unfold(inputs, ctx.kernel_size,
                                                     ctx.dilation, ctx.padding,
                                                     ctx.stride)
            inputs_wins = inputs_wins.view(batch_size, in_channels,
                                           ctx.kernel_size[0],
                                           ctx.kernel_size[1],
                                           output_size[0], output_size[1])
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.new()
            grad_inputs_wins = grad_inputs_mul_kernel * kernel_view
            grad_inputs_wins = grad_inputs_wins.view(
                batch_size, -1, output_size[0] * output_size[1])
            grad_inputs = torch.nn.functional.fold(grad_inputs_wins,
                                                   ctx.input_size,
                                                   ctx.kernel_size,
                                                   ctx.dilation,
                                                   ctx.padding,
                                                   ctx.stride)
        if ctx.needs_input_grad[1]:
            grad_kernel = inputs_wins * grad_inputs_mul_kernel
            # grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
            grad_kernel = torch.einsum('ijklmn->ijkl',(grad_kernel,))
        if ctx.needs_input_grad[2]:
            inputs_mul_kernel = inputs_wins * kernel_view
            grad_weight = torch.einsum('iomn,ijklmn->ojkl',
                                       (grad_outputs, inputs_mul_kernel))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_outputs,))
        return (grad_inputs, grad_kernel, grad_weight, grad_bias, None, None,
                None)


class KernelDWConv2d(torch.nn.Module):
    """Implementation of depthwise correlation Convolution with kernel."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        """Constructor."""
        super(KernelDWConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = torch.nn.modules.utils._pair(padding)
        self.dilation = torch.nn.modules.utils._pair(dilation)

        # Parameters: weight, bias
        self.weight = torch.nn.parameter.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size,
                         kernel_size))
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialization
        self.reset_parameters()

    def forward(self, inputs, kernel):
        """Forward computation.

        Args:
            inputs: A tensor with shape [batch, in_channels, height, width]
                representing a batch of images.
            kernel: A tensor with shape [batch, in_channels, kernel_size, kernel_size] representing
                    a batch of template images

        Returns:
            outputs: A tensor with shape [batch, out_channels, N, N].
        """
        outputs = KernelDWConvFn.apply(inputs, kernel, self.weight,
                                     self.bias, self.stride,
                                     self.padding, self.dilation)
        return outputs

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
