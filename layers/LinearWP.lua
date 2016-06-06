local LinearWP, parent = torch.class('nn.LinearWP','nn.Linear')

function LinearWP:__init(inputSize, outputSize)
  self.inputSize = inputSize
  parent.__init(self, inputSize, outputSize)
  --LinearWP:initialize(winit, binit)
  --parent.weight:normal(0,1):mul(winit):div(math.sqrt(inputSize))
  --parent.bias:normal(0,1):mul(binit)
end

function LinearWP:initialize(winit, binit)
  self.weight:normal(0, 1):mul(winit):div(math.sqrt(self.inputSize))
  self.bias:normal(0, 1):mul(binit)
end

