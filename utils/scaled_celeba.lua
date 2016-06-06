require 'paths'
require 'image'
celeba = {}

celeba.path_dataset = 'data/'
celeba.scale = 64

function celeba.setScale(scale)
  celeba.scale = scale
end

function celeba.setPath(path_dataset)
  celeba.path_dataset = path_dataset
end

function celeba.loadDataSet(mode)

  if mode == 'train' then
    loader = torch.load(celeba.path_dataset .. 'celeba_train.t7')
    images = loader.train_images
    attributes = loader.train_attributes
  elseif mode == 'val' then
    loader = torch.load(celeba.path_dataset .. 'celeba_val.t7')
    images = loader.val_images
    attributes = loader.val_attributes
  elseif mode == 'test' then
    loader = torch.load(celeba.path_dataset .. 'celeba_test.t7')
    images = loader.test_images
    attributes = loader.test_attributes
  elseif mode == 'probset' then
    loader = torch.load(celeba.path_dataset .. 'celeba_test.t7')
    local ntest = loader.test_images:size(1)
    local nprob = math.floor(ntest * 0.2)
    images = loader.test_images[{{1, nprob}, {}}]:clone()
    attributes = loader.test_attributes[{{1, nprob}, {}}]:clone()
    collectgarbage()
  elseif mode == 'gallery' then
    loader = torch.load(celeba.path_dataset .. 'celeba_test.t7')
    local ntest = loader.test_images:size(1)
    local nprob = math.floor(ntest * 0.2)
    local ngal = ntest - nprob
    images = loader.test_images[{{nprob+1, ntest}, {}}]:clone()
    attributes = loader.test_attributes[{{nprob+1, ntest}, {}}]:clone()
    collectgarbage()
  end

  local dataset = {}
  dataset.data = images:float()
  dataset.attr = attributes:float()
  
  function dataset:scaleData()
    local N = dataset.data:size(1)
    dataset.scaled = torch.FloatTensor(N, 3, celeba.scale, celeba.scale)
    for n = 1, N do
      dataset.scaled[n] = image.scale(dataset.data[n], celeba.scale, celeba.scale)
    end
  end

  function dataset:size()
    local N = dataset.data:size(1)
    return N
  end

  setmetatable(dataset, {__index = function(self, index)
    local example = {self.scaled[index], self.attr[index]}
    return example
  end})

  return dataset
end

