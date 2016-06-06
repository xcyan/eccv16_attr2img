require 'nn'
require 'cunn'
local LinearMix, parent = torch.class('nn.LinearMix', 'nn.Module')

function LinearMix:__init(inputSize_x, inputSize_y, outputSize)
  parent.__init(self)

  self.inputSize_x = inputSize_x
  self.inputSize_y = inputSize_y
  self.outputSize = outputSize

  self.weight = torch.Tensor(outputSize, inputSize_x + inputSize_y)
  self.bias = torch.Tensor(outputSize)

  self.gradWeight = torch.Tensor(outputSize, inputSize_x + inputSize_y)
  self.gradBias = torch.Tensor(outputSize)

  self.gradInput = {}
  self.output = torch.Tensor(outputSize):zero()
  self.gradInput[1] = torch.Tensor(inputSize_x):zero()
  self.gradInput[2] = torch.Tensor(inputSize_y):zero()

  self:reset()
end

function LinearMix:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1./math.sqrt(self.weight:size(2))
  end

  if nn.oldSeed then
    for i = 1,self.weight:size(1) do
      self.weight:select(1, i):apply(function()
        return torch.randn(self.weight:size(2)):mul(stdv)
      end)
      self.bias[i] = torch.randn(self.bias[i]:size()):mul(stdv)
    end

  else
    self.weight:rand(self.weight:size()):mul(stdv)
    self.bias:randn(self.bias:size()):mul(stdv)
  end

  return self
end

function LinearMix:initialize(winit, binit)
  self.weight[{{}, {1, self.inputSize_x}}]:normal(0, 1):div(math.sqrt(self.inputSize_x)):mul(winit)
  self.weight[{{}, {self.inputSize_x + 1, self.inputSize_x + self.inputSize_y}}]:normal(0, 1):div(math.sqrt(self.inputSize_y)):mul(winit)
  self.bias:normal(0, 1):mul(binit)
end

function LinearMix:updateOutput(input)
  -- input: xstream, ystream

  if input[1]:type() == "torch.CudaTensor" then
    input_cat = torch.cat(input[1]:float(), input[2]:float()):cuda()
  else
    input_cat = torch.cat(input[1], input[2])
  end

  if input[1]:dim() == 1 and input[2]:dim() == 1 then
    self.output:resize(self.bias:size(1))
    self.output:copy(self.bias)
    self.output:addmv(1, self.weight, input_cat)

  elseif input[1]:dim() == 2 and input[2]:dim() == 2 then
    local nframe = input[1]:size(1)
    local nElement = self.output:nElement()
    self.output:resize(nframe, self.bias:size(1))
    if self.output:nElement() ~= nElement then
      self.output:zero()
    end
    if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
      self.addBuffer = input[1].new(nframe):fill(1)
    end
    self.output:addmm(0, self.output, 1, input_cat, self.weight:t())
    self.output:addr(1, self.addBuffer, self.bias)

  else
    error('input must be vector or matrix')
  end

  return self.output
end

function LinearMix:updateGradInput(input, gradOutput)
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
      self.gradInput[1]:addmv(0, 1, self.weight[{{}, {1, self.inputSize_x}}]:t(), gradOutput)
      self.gradInput[2]:addmv(0, 1, self.weight[{{}, {self.inputSize_x + 1, self.inputSize_x + self.inputSize_y}}]:t(), gradOutput)

    elseif input[1]:dim() == 2 and input[2]:dim() == 2 then
      self.gradInput[1]:addmm(0, 1, gradOutput, self.weight[{{}, {1, self.inputSize_x}}])
      self.gradInput[2]:addmm(0, 1, gradOutput, self.weight[{{}, {self.inputSize_x + 1, self.inputSize_x + self.inputSize_y}}])
    end

    return self.gradInput
  end
end

function LinearMix:accGradParameters(input, gradOutput, scale)
  scale = scale or 1

  if input[1]:type() == "torch.CudaTensor" then
    input_cat = torch.cat(input[1]:float(), input[2]:float()):cuda()
  else
    input_cat = torch.cat(input[1], input[2])
  end

  if input[1]:dim() == 1 and input[2]:dim() == 1 then
    self.gradWeight:addr(scale, gradOutput, input_cat)
    self.gradBias:add(scale, gradOutput)
  elseif input[1]:dim() == 2 and input[2]:dim() == 2 then
    self.gradWeight:addmm(scale, gradOutput:t(), input_cat)
    self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
  end
end

LinearMix.sharedAccUpdateGradParameters = LinearMix.accUpdateGradParameters

function LinearMix:__tostring__()
  return torch.type(self) ..
    string.format('(dim(x) = %d; dim(y) = %d --> dim(out) = %d', self.inputSize_x, self.inputSize_y, self.outputSize)
end

