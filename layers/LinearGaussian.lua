require 'nn'
require 'cunn'

local LinearGaussian, parent = torch.class('nn.LinearGaussian', 'nn.Module')

function LinearGaussian:__init(inputSize, outputSize)
  parent.__init(self)

  self.inputSize = inputSize
  self.outputSize = outputSize

  self.mean_layer = nn.LinearWP(inputSize, outputSize)
  self.sigma_layer = nn.LinearWP(inputSize, outputSize)
  --self.mean_layer:reset()
  --self.sigma_layer:reset()
end


function LinearGaussian:reset()
  self.mean_layer:reset()
  self.sigma_layer:reset()
  return self
end

function LinearGaussian:initialize(winit, binit) 
  self.mean_layer:initialize(winit, binit)
  self.sigma_layer:initialize(0, 0)
end

function LinearGaussian:updateOutput(input)
  tmp_mean = self.mean_layer:updateOutput(input)
  tmp_sigma = self.sigma_layer:updateOutput(input)
  self.output = {tmp_mean, tmp_sigma}
  return self.output
end

function LinearGaussian:updateGradInput(input, gradOutput)
  --print(gradOutput[1]:size())
  tmp_mean = self.mean_layer:updateGradInput(input, gradOutput[1])
  tmp_sigma = self.sigma_layer:updateGradInput(input, gradOutput[2])
  self.gradInput = torch.add(tmp_mean, tmp_sigma)
  return self.gradInput
end

function LinearGaussian:accGradParameters(input, gradOutput, scale)
  self.mean_layer:accGradParameters(input, gradOutput[1], scale)
  self.sigma_layer:accGradParameters(input, gradOutput[2], scale)
end

LinearGaussian.sharedAccUpdateGradParameters = LinearGaussian.accUpdateGradParameters

function LinearGaussian:parameters()
  param_mean, grad_mean = self.mean_layer:parameters()
  param_sigma, grad_sigma = self.sigma_layer:parameters()
  param = {param_mean[1], param_mean[2], param_sigma[1], param_sigma[2]}
  grad = {grad_mean[1], grad_mean[2], grad_sigma[1], grad_sigma[2]}
  return param, grad
end

function LinearGaussian:__tostring__()
  return torch.type(self) ..
    string.format('(%d --> %d, %d)', self.inputSize, self.outputSize, self.outputSize)
end

