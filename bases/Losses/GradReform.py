import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

import math

############################################################
# Common Inner product with manual gradients
class MyLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(MyLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.linearfunction = LinearFunction.apply
    def forward(self, input):
        return self.linearfunction(input, self.weight)

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = input.mm(weight.t())
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight

############################################################

############################################################
# This layer do NOTHING. Yes, it just as f(x) = x
class IdentityMappingReformGrad(nn.Module):
    def __init__(self):
        super(IdentityMappingReformGrad, self).__init__()
        self.IdentityMapping = IdentityMappingFunction.apply
    def forward(self, input):
        return self.IdentityMapping(input)

class IdentityMappingFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.save_for_backward
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input
############################################################

# class _Linear(Function):
#     def forward(self, input, weight):
#         self.save_for_backward(input, weight)
#         output = input.mm(weight.t())
#         return output
#     def backward(self, grad_output):
#         input, weight = self.saved_tensors
#         grad_input = grad_weight = grad_bias = None
#         print("backwarding......")
#         if self.needs_input_grad[0]:
#             grad_input = grad_output.mm(weight)
#         if self.needs_input_grad[1]:
#             grad_weight = grad_output.t().mm(input)
#         return grad_input, grad_weight

# def module_hook(module, grad_input, grad_out):
#     print('module hook')
#     print('grad_out', grad_out)

# def variable_hook(grad):
#     print('variable hook')
#     print('grad', grad)
#     return grad*.1

# class Linear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(Linear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#     def forward(self, input):
#             return _Linear()(input, self.weight)



# linear = Linear(3,1)
# linear.register_backward_hook(module_hook)
# value = Variable(torch.FloatTensor([[1,2,3]]), requires_grad=True)

# res = linear(value)
# res.register_hook(variable_hook)

# res.backward()