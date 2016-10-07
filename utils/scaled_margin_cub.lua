require 'image'

cub = {}

cub.path_dataset = 'data/'
cub.scale = 72

function cub.setScale(scale)
  cub.scale = scale
end

function cub.loadTrainSet()
  return cub.loadDataset('train')
end

function cub.loadValSet()
  return cub.loadDataset('val')
end

function cub.loadTestSet()
  return cub.loadDataset('test')
end

function cub.loadDataset(datasplit)
  if datasplit == 'train' then
    loader = torch.load(cub.path_dataset .. 'cub_train80x80.t7')
    data_ori = loader.trainData
    attr_ori = loader.trainAttr
    mask_ori = loader.trainMask
    mask_ori = mask_ori:gt(0.5):float():clone()

    local N = data_ori:size(1) - math.floor(data_ori:size(1) / 10)
    data = torch.FloatTensor(N, 3, data_ori:size(3), data_ori:size(4)):zero()
    attr = torch.FloatTensor(N, attr_ori:size(2)):zero()
    mask = torch.FloatTensor(N, 1, mask_ori:size(3), mask_ori:size(4)):zero()

    local k = 0
    for i = 1, data_ori:size(1) do
      if i % 10 > 0 then
        k = k + 1
        data[{{k}, {}}] = data_ori[{{i}, {}}]:clone()
        attr[{{k}, {}}] = attr_ori[{{i}, {}}]:clone()
        mask[{{k}, {}}] = mask_ori[{{i}, {}}]:clone()
      end
    end
  
  elseif datasplit == 'val' then
    loader = torch.load(cub.path_dataset .. 'cub_train80x80.t7')
    data_ori = loader.trainData
    attr_ori = loader.trainAttr
    mask_ori = loader.trainMask
    mask_ori = mask_ori:gt(0.5):float():clone()
    
    local N = math.floor(data_ori:size(1) / 10)
    data = torch.FloatTensor(N, 3, data_ori:size(3), data_ori:size(4)):zero()
    attr = torch.FloatTensor(N, attr_ori:size(2)):zero()
    mask = torch.FloatTensor(N, 1, mask_ori:size(3), mask_ori:size(4)):zero()

    local k = 0
    for i = 1, data_ori:size(1) do
      if i % 10 == 0 then
        k = k + 1
        data[{{k}, {}}] = data_ori[{{i}, {}}]:clone()
        attr[{{k}, {}}] = attr_ori[{{i}, {}}]:clone()
        mask[{{k}, {}}] = mask_ori[{{i}, {}}]:clone()
      end
    end

  elseif datasplit == 'test' then
    loader = torch.load(cub.path_dataset .. 'official_test80x80.t7')
    data = loader.testData
    attr = loader.testAttr
    mask = loader.testMask
  end

  local N = data:size(1)

  local dataset = {}
  dataset.data = data:float()
  dataset.attr = attr:float()
  dataset.mask = mask:float()
  dataset.mask = mask:gt(0.5):float()

  function dataset:scaleData()
    local N = dataset.data:size(1)
    dataset.scaled = torch.FloatTensor(N, 3, cub.scale, cub.scale)
    dataset.scaled_mask = torch.FloatTensor(N, 1, cub.scale, cub.scale)
    for n = 1, N do
      dataset.scaled[n] = image.scale(dataset.data[n], cub.scale, cub.scale)
      dataset.scaled_mask[n] = image.scale(dataset.mask[n], cub.scale, cub.scale)
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
