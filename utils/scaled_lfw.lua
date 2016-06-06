require 'paths'
require 'mattorch'
require 'image'
lfw = {}

lfw.path_dataset = '/mnt/brain1/scratch/xcyan/stable/vae-nips20150605/lfw_folds/'
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
  if lfw.scale > 64 then
    filename = string.format('LFW%dx%d_view1.mat', 128, 128)
    maskfname = string.format('LFW%dx%d_view1_fgmask.mat', 128, 128)
  else
    filename = string.format('LFW%dx%d_view1.mat', 64, 64)
    maskfname = string.format('LFW%dx%d_view1_fgmask.mat', 64, 64)
  end
  datafile = mattorch.load(lfw.path_dataset .. filename)
  maskfile = mattorch.load(lfw.path_dataset .. maskfname)

  if isTrain then
    data = datafile.trainImgs / 255
    attr = datafile.trainAttrs
    mask = maskfile.trainMasks
  else
    data = datafile.testImgs / 255
    attr = datafile.testAttrs
    mask = maskfile.testMasks
  end

  local start = start or 1
  local stop = stop or data:size(1)
  data = data[{ {start, stop} }]
  attr = attr[{ {start, stop} }]
  mask = mask[{ {start, stop} }]
  local N = stop - start + 1

  local dataset = {}
  dataset.data = data:float()
  dataset.attr = attr:float()
  dataset.mask = mask:float()
  dataset.testnmpairs = datafile.testNonMatchPairs:float()
  dataset.testmpairs = datafile.testMatchPairs:float()

  function dataset:scaleData()
    local N = dataset.data:size(1)
    dataset.scaled = torch.FloatTensor(N, 3, lfw.scale, lfw.scale)
    dataset.scaledmask = torch.FloatTensor(N, 1, lfw.scale, lfw.scale)
    for n = 1, N do
      dataset.scaled[n] = image.scale(dataset.data[n], lfw.scale, lfw.scale)
      dataset.scaledmask[n] = image.scale(dataset.mask[n], lfw.scale, lfw.scale)
    end
  end

  function dataset:size()
    local N = dataset.data:size(1)
    return N
  end

  setmetatable(dataset, {__index = function(self, index)
    local example = {self.scaled[index], self.attr[index], self.scaledmask[index]}
    return example
  end})

  return dataset
end

