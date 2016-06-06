require 'nn'
require 'cunn'
local LinearMix2, parent = torch.class('nn.LinearMix2','nn.Module')

function LinearMix2:__init(inputSize_x, inputSize_y, outputSize)
  parent.__init(self)

  self.inputSize_x = inputSize_x
  self.inputSize_y = inputSize_y
  self.outputSize = outputSize

  self.weight = torch.Tensor(outputSize, inputSize_x + inputSize_y):randn(outputSize, inputSize_x + inputSize_y)
  self.bias = torch.Tensor(outputSize):randn(outputSize)
  self.weight_sigma = torch.Tensor(outputSize, inputSize_x):randn(outputSize, inputSize_x)
  self.bias_sigma = torch.Tensor(outputSize):randn(outputSize)

  self.gradWeight = torch.Tensor(outputSize, inputSize_x +inputSize_y):randn(outputSize, inputSize_x + inputSize_y)
  self.gradBias = torch.Tensor(outputSize):randn(outputSize)
  self.gradWeight_sigma = torch.Tensor(outputSize, inputSize_x):randn(outputSize, inputSize_x)
  self.gradBias_sigma = torch.Tensor(outputSize):randn(outputSize)

  self.output = {}
  self.gradInput = {}
  self.output[1] = torch.Tensor(outputSize):zero()
  self.output[2] = torch.Tensor(outputSize):zero()
  self.gradInput[1] = torch.Tensor(inputSize_x):zero()
  self.gradInput[2] = torch.Tensor(inputSize_y):zero()

  self:reset()
end

function LinearMix2:reset(stdv)
  if stdv then
    stdv = stdv *math.sqrt(3)
  else
    stdv = 1./math.sqrt(self.weight:size(2))
    stdv_sigma = 1./math.sqrt(self.weight_sigma:size(2))
  end
  if nn.oldSeed then
    for i = 1,self.weight:size(1) do
      self.weight:select(1, i):apply(function()
        return torch.randn(self.weight:size(2)):mul(stdv)
      end)
      self.bias[i] = torch.randn(self.bias[i]:size()):mul(stdv)
    end
    self.weight_sigma:zero()
    self.bias_sigma:zero()
  else
    self.weight:randn(self.weight:size()):mul(stdv)
    self.bias:randn(self.bias:size()):mul(stdv)
    self.weight_sigma:zero()
    self.bias_sigma:zero()
  end
  -- print(string.format('%d', self.bias:size(1)))
  return self
end

function LinearMix2:initialize(winit, binit)
  
  self.weight[{{}, {1, self.inputSize_x}}]:normal(0,1):div(math.sqrt(self.inputSize_x)):mul(winit)
  self.weight[{{}, {self.inputSize_x + 1, self.inputSize_x + self.inputSize_y}}]:normal(0,1):div(math.sqrt(self.inputSize_y)):mul(winit)
  self.bias:mul(binit)
  
  self.weight_sigma:zero()
  self.bias_sigma:zero()
end

function LinearMix2:updateOutput(input)
  -- input: xstream, ystream

  if input[1]:type() == "torch.CudaTensor" then
    input_cat = torch.cat(input[1]:float(), input[2]:float()):cuda()
  else
    input_cat = torch.cat(input[1], input[2])
  end

  if input[1]:dim() == 1 and input[2]:dim() == 1 then
    -- mean
    self.output[1]:resize(self.bias:size(1))
    self.output[1]:copy(self.bias)
    self.output[1]:addmv(1, self.weight, input_cat)
    -- sigma
    self.output[2]:resize(self.bias_sigma:size(1))
    self.output[2]:copy(self.bias_sigma)
    self.output[2]:addmv(1, self.weight_sigma, input[1])

  elseif input[1]:dim() == 2 and input[2]:dim() == 2 then
    local nframe = input[1]:size(1)
    local nElement = self.output[1]:nElement()
    local nElement2 = self.output[2]:nElement()
    self.output[1]:resize(nframe, self.bias:size(1))
    self.output[2]:resize(nframe, self.bias_sigma:size(1))
    if self.output[1]:nElement() ~= nElement then
      self.output[1]:zero()
    end
    if self.output[2]:nElement() ~= nElement2 then
      self.output[2]:zero()
    end
    if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
      self.addBuffer = input[1].new(nframe):fill(1)
    end
    self.output[1]:addmm(0, self.output[1], 1, input_cat, self.weight:t())
    self.output[1]:addr(1, self.addBuffer, self.bias)

    self.output[2]:addmm(0, self.output[2], 1, input[1], self.weight_sigma:t())
    self.output[2]:addr(1, self.addBuffer, self.bias_sigma)
  else
    error('input must be vector or matrix')
  end

  return self.output
end

function LinearMix2:updateGradInput(input, gradOutput)
  if self.gradInput then
    
    local nElement = self.gradInput[1]:nElement()
    local nElement2 = self.gradInput[2]:nElement()
    self.gradInput[1]:resizeAs(input[1])
    self.gradInput[2]:resizeAs(input[2])
    if self.gradInput[1]:nElement() ~= nElement then
      self.gradInput[1]:zero()
    end
    if self.gradInput[2]:nElement() ~= nElement2 then
      self.gradInput[2]:zero()
    end
    
    if input[1]:dim() == 1 and input[2]:dim() == 1 then
      self.gradInput[1]:addmv(0, 1, self.weight[{{}, {1, self.inputSize_x}}]:t(), gradOutput[1])
      self.gradInput[1]:addmv(1, 1, self.weight_sigma:t(), gradOutput[2])
      self.gradInput[2]:addmv(0, 1, self.weight[{{}, {self.inputSize_x + 1, self.inputSize_x + self.inputSize_y}}]:t(), gradOutput[1])

    elseif input[1]:dim() == 2 and input[2]:dim() == 2 then
      --print(gradOutput)
      --print(self.weight[{{}, {1, self.inputSize_x}}]:size())
      self.gradInput[1]:addmm(0, 1, gradOutput[1], self.weight[{{}, {1, self.inputSize_x}}])
      self.gradInput[1]:addmm(1, 1, gradOutput[2], self.weight_sigma)
      self.gradInput[2]:addmm(0, 1, gradOutput[1], self.weight[{{}, {self.inputSize_x + 1, self.inputSize_x + self.inputSize_y}}])
    end

    return self.gradInput
  end
end

function LinearMix2:accGradParameters(input, gradOutput, scale)
  scale = scale or 1

  if input[1]:type() == "torch.CudaTensor" then
    input_cat = torch.cat(input[1]:float(), input[2]:float()):cuda()
  else
    input_cat = torch.cat(input[1], input[2])
  end

  if input[1]:dim() == 1 and input[2]:dim() == 1 then
    self.gradWeight:addr(scale, gradOutput[1], input_cat)
    self.gradBias:add(scale, gradOutput[1])
    self.gradWeight_sigma:addr(scale, gradOutput[2], input[1])
    self.gradBias_sigma:add(scale, gradOutput[2])

  elseif input[1]:dim() == 2 and input[2]:dim() == 2 then
    self.gradWeight:addmm(scale, gradOutput[1]:t(), input_cat)
    self.gradBias:addmv(scale, gradOutput[1]:t(), self.addBuffer)
    self.gradWeight_sigma:addmm(scale, gradOutput[2]:t(), input[1])
    self.gradBias_sigma:addmv(scale, gradOutput[2]:t(), self.addBuffer)
  end
end

LinearMix2.sharedAccUpdateGradParameters = LinearMix2.accUpdateGradParameters

function LinearMix2:parameters()
  param = {self.weight, self.bias, self.weight_sigma, self.bias_sigma}
  grad = {self.gradWeight, self.gradBias, self.gradWeight_sigma, self.gradBias_sigma}
  return param, grad
end

function LinearMix2:__tostring__()
  return torch.type(self) ..
    string.format('(dim(x) = %d; dim(y) = %d --> dim(m) = %d, dim(s) = %d)', self.inputSize_x, self.inputSize_y, self.outputSize, self.outputSize)
end

