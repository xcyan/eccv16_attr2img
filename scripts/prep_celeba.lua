require 'torch'
require 'image'
require 'nn'
dir = require 'pl.dir'

datafolder = 'data/'
imgpath = datafolder .. 'img_align_celeba/'
attrfile = datafolder .. 'list_attr_celeba.txt'

---------
N = 202599
scale = 64
attrDim = 40
all_images = torch.FloatTensor(N, 3, scale, scale):zero()
all_attributes = torch.FloatTensor(N, attrDim):zero()

---------
k = 0
for line in io.lines(attrfile) do
  k = k + 1
  if k > 2 then
    -- load image & attr vec
    local t = string.find(line, ' ')
    local len = string.len(line)
    local fname = string.sub(line, 1, t-1)
    local attrstr = string.sub(line, t+1, len)
    local tFinal = {}
    attrstr:gsub("[-]?%d+", function(i) table.insert(tFinal, i) end)
    local im = image.load(string.format('%s%s', imgpath, fname))
    local cropped_im = image.crop(im, 25, 30, 25+145, 30 + 180)
    local scaled_im = image.scale(cropped_im, scale, scale)
    -- save img & attr vec
    all_images[{{k-2}, {}}] = scaled_im:clone()
    for j = 1, attrDim do
      all_attributes[{{k-2}, {j}}] = tFinal[j]
    end
  end
  if k % 500 == 0 then
    xlua.progress(k, N)
  end
end

--
Ntrain = 162770
Nval = 182637 - Ntrain
Ntest = N - Ntrain - Nval
--
train_images = all_images[{{1,Ntrain}, {}}]:clone()
train_attributes = all_attributes[{{1, Ntrain}, {}}]:clone()

val_images = all_images[{{Ntrain+1,Ntrain+Nval}, {}}]:clone()
val_attributes = all_attributes[{{Ntrain+1, Ntrain+Nval}, {}}]:clone()

test_images = all_images[{{Ntrain+Nval+1, N}, {}}]:clone()
test_attributes = all_attributes[{{Ntrain+Nval+1, N}, {}}]:clone()

--subset_images = all_images[{{1, 1000}, {}}]:clone()
--subset_attributes = all_attributes[{{1, 1000}, {}}]:clone()
collectgarbage()

--
print('saving files to disk...')
torch.save(datafolder .. 'celeba_train.t7', {train_images = train_images,
  train_attributes = train_attributes})
torch.save(datafolder .. 'celeba_val.t7', {val_images = val_images, 
  val_attributes = val_attributes})
torch.save(datafolder .. 'celeba_test.t7', {test_images = test_images, 
  test_attributes = test_attributes})

