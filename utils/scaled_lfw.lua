require 'paths'
require 'image'
lfw = {}

lfw.path_dataset = 'data/'
lfw.scale = 64

function lfw.setScale(scale)
  lfw.scale = scale
end

function lfw.loadTrainSet(start, stop)
  return lfw.loadDataset(true, start, stop)
end

function lfw.loadTestSet()
  return lfw.loadDataset(false, nil, nil)
end

function lfw.loadDataset(isTrain, start, stop)
  if isTrain then
    datafile = torch.load(lfw.path_dataset .. 'lfw_train.t7')
    data = datafile.trainData
    mask = datafile.trainMask
    attr = datafile.trainAttr
  else
    datafile = torch.load(lfw.path_dataset .. 'lfw_test.t7')
    data = datafile.testData
    mask = datafile.testMask
    attr = datafile.testAttr
  end

  local start = start or 1
  local stop = stop or data:size(1)
  data = data[{ {start, stop} }]
  mask = mask[{ {start, stop} }]
  attr = attr[{ {start, stop} }]
  local N = stop - start + 1

  local dataset = {}
  dataset.data = data:float()
  dataset.mask = mask:float()
  dataset.attr = attr:float()

  function dataset:scaleData()
    local N = dataset.data:size(1)
    dataset.scaled = torch.FloatTensor(N, 3, lfw.scale, lfw.scale)
    dataset.scaled_mask = torch.FloatTensor(N, 1, lfw.scale, lfw.scale)
    for n = 1, N do
      dataset.scaled[n] = image.scale(dataset.data[n], lfw.scale, lfw.scale)
      dataset.scaled_mask[n] = image.scale(dataset.mask[n], lfw.scale, lfw.scale)
    end
  end

  function dataset:size()
    local N = dataset.data:size(1)
    return N
  end

  setmetatable(dataset, {__index = function(self, index)
    local example = {self.scaled[index], self.scaled_mask[index], self.attr[index]}
    return example
  end})

  return dataset
end

