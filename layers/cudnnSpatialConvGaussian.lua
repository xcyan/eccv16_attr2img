require 'nn'
require 'cudnn'

local SpatialConvGaussian, parent = torch.class('cudnn.SpatialConvGaussian', 'nn.Module')

function SpatialConvGaussian:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self)
  self.mean_layer = cudnn.SpatialConvWP(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  self.sigma_layer = cudnn.SpatialConvWP(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

end

function SpatialConvGaussian:reset()
  self.mean_layer.reset()
  self.sigma_layer.reset()
  return self
end

function SpatialConvGaussian:initialize(winit, binit, fanin)
  self.mean_layer:initialize(winit, binit, fanin)
  self.sigma_layer:initialize(0, 0, fanin)
end

function SpatialConvGaussian:updateOutput(input)
  tmp_mean = self.mean_layer:updateOutput(input)
  tmp_sigma = self.sigma_layer:updateOutput(input)
  self.output = {tmp_mean, tmp_sigma}
  return self.output
end

function SpatialConvGaussian:updateGradInput(input, gradOutput)
  tmp_mean = self.mean_layer:updateGradInput(input, gradOutput[1])
  tmp_sigma = self.sigma_layer:updateGradInput(input, gradOutput[2])
  self.gradInput = torch.add(tmp_mean, tmp_sigma)
  return self.gradInput
end

function SpatialConvGaussian:accGradParameters(input, gradOutput, scale)
  self.mean_layer:accGradParameters(input, gradOutput[1], scale)
  self.sigma_layer:accGradParameters(input, gradOutput[2], scale)
end

SpatialConvGaussian.sharedAccUpdateGradParameters = SpatialConvGaussian.accUpdateGradParameters

function SpatialConvGaussian:parameters()
  param_mean, grad_mean = self.mean_layer:parameters()
  param_sigma, grad_sigma = self.sigma_layer:parameters()
  param = {param_mean[1], param_mean[2], param_sigma[1], param_sigma[2]}
  grad = {grad_mean[1], grad_mean[2], grad_sigma[1], grad_sigma[2]}
  return param, grad
end

function SpatialConvGaussian:__tostring__()
  return torch.type(self) ..
    string.format(self.mean_layer:__tostring__())
end
