require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'mattorch'

require 'layers.LinearWP'
require 'layers.LinearGaussian'
require 'layers.SpatialConvWP'
require 'layers.SpatialConvGaussian'
require 'layers.cudnnSpatialConvWP'
require 'layers.cudnnSpatialConvGaussian'
require 'layers.LinearMix'
require 'layers.LinearMix2'
require 'layers.Reparametrize'
require 'layers.GaussianCriterion'
require 'layers.KLDCriterion'
require 'utils.scaled_lfw'
optim_utils = require 'utils.adam_v2'
image_utils = require 'utils.image_utils'

-- parse command-line options
opts = lapp[[
  --saveFreq    (default 20)        save every saveFreq epochs
  --modelString  (default "")        reload pretrained network
  -p,--plot                         plot while training
  -t,--threads  (default 4)         number of threads
  -g,--gpu      (default -1)        gpu to run on (default cpu)
  --scale       (default 64)        scale of images to train on
  -z,--zdim        (default 256)
  -y,--ydim        (default 73)
  --maxEpoch    (default 0)
  --batchSize    (default 32)
  --nsample  (default 1)       number of samples
  --weightInit   (default 1)
  --biasInit     (default 0.01)
  --weightDecay  (default 0.004)
  --adam        (default 1)         lr mode
  --modelDir    (default '../lfw_models/')
]]

--opts.modelDir = '../lfw_models/'

opts.modelName = string.format('%s_adam%d_bs%d_ns%d_zdim%d_wi%g_bi%g_wd%g', 
  opts.modelString, opts.adam, opts.batchSize, opts.nsample, opts.zdim,
  opts.weightInit, opts.biasInit, opts.weightDecay)

if opts.gpu < 0 or opts.gpu > 3 then opts.gpu = -1 end
print(opts)

torch.manualSeed(1)

-- threads
torch.setnumthreads(opts.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opts.gpu >= 0 then
  cutorch.setDevice(opts.gpu + 1)
  print('<gpu> using device ' .. opts.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

if opts.modelString == '' then
  error('empty modelString')
else
  print('prototxt_cvae_lfw/' .. opts.modelString .. '.lua')

  lfwcvae_module = dofile('prototxt_lfw_cvae/' .. opts.modelString .. '.lua')
  encoder, decoder = lfwcvae_module.create(opts)
end

-- retrieve parameters and gradients
epoch = 0
opts.modelPath = opts.modelDir .. opts.modelName
if not paths.dirp(opts.modelPath) then
  paths.mkdir(opts.modelPath)
end

for i = opts.maxEpoch,1,-opts.saveFreq do
  if paths.filep(opts.modelPath .. string.format('/net-epoch-%d.t7', i)) then
    epoch = i
    loader = torch.load(opts.modelPath .. string.format('/net-epoch-%d.t7', i))
    encoder = loader.encoder
    decoder = loader.decoder
    state = torch.load(opts.modelPath .. '/state.t7')
    LBlist_train = torch.load(opts.modelPath .. '/statistic_train.t7') 
    LBlist_val = torch.load(opts.modelPath .. '/statistic_val.t7')
    print(string.format('resuming from epoch %d', i))
    break
  end
end

cvae = nn.Sequential()
enc_sampling = nn.Sequential()
enc_sampling:add(encoder):add(nn.Reparametrize(opts.zdim))

cvae:add(nn.ParallelTable():add(enc_sampling):add(nn.Copy()))
cvae:add(decoder)
parameters, gradients = cvae:getParameters()

-- print networks
-- TODO: nparams
if opts.gpu >= 0 then
  print('Copying model to gpu')
  encoder:cuda()
  decoder:cuda()
end

-- Optimization criteria
criterionLL = nn.GaussianCriterion()
criterionKLD = nn.KLDCriterion()

-- get/create dataset
ntrain = math.floor(9464 * 0.9)
nval = 9464 - ntrain

print('data preprocessing')
lfw.setScale(opts.scale)
trainData = lfw.loadTrainSet(1, ntrain)
mean, std = image_utils.normalizeGlobal(trainData.data)
--old_min, old_max = image_utils.contrastNormalize(trainData.attr, -1, 1)
trainData:scaleData()

valData = lfw.loadTrainSet(ntrain + 1, ntrain + nval)
image_utils.normalizeGlobal(valData.data, mean, std)
--image_utils.contrastNormalize(valData.attr, -1, 1, old_min, old_max)
valData:scaleData()

ntrain = trainData:size()
nval = valData:size()
print(ntrain)
print(nval)

-- hyperparams
function getAdamParams(opts)
  config = {}
  if opts.adam == 1 then
    config.learningRate = -0.0003
    config.epsilon = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weightDecay = -opts.weightDecay
  elseif opts.adam == 2 then
    config.learningRate = -0.0003
    config.epsilon = 1e-8
    config.beta1 = 0.5
    config.beta2 = 0.999
    config.weightDecay = -opts.weightDecay
  end
  return config
end

config = getAdamParams(opts)
print(config)

local blurK = torch.FloatTensor(5, 5):fill(1/25)

-- training loop
for t = epoch+1, opts.maxEpoch do
  local trainLB = 0
  local trainLL = 0
  local trainKL = 0

  local valLB = 0
  local valLL = 0
  local valKL = 0

  local time = sys.clock()

  -- Make sure batches are always batchSize
  local N_train = ntrain - (ntrain % opts.batchSize)
  local N_val = nval - (nval % opts.batchSize)

  --local samples_gen
  --local samples_gt

  cvae:training()
  for i = 1, N_train, opts.batchSize do
    --xlua.progress(i+opts.batchSize-1, ntrain)
  
    local batch_im = torch.Tensor(opts.batchSize, 3, opts.scale, opts.scale)
    local batch_attr = torch.Tensor(opts.batchSize, opts.ydim)

    local k = 1
    for j = i, i+opts.batchSize-1 do
      local idx = math.random(ntrain)
      local cur_im = trainData[idx][1]:float():clone()
      -- flip
      local flip = math.random(2)-1
      if flip == 1 then
        cur_im = image.vflip(cur_im:float())
      end
      -- color space augmentation
      local randR = torch.rand(1)*0.06+0.97
      local randG = torch.rand(1)*0.06+0.97
      local randB = torch.rand(1)*0.06+0.97
      cur_im[{{1}, {}, {}}]:mul(randR:float()[1])
      cur_im[{{2}, {}, {}}]:mul(randG:float()[1])
      cur_im[{{3}, {}, {}}]:mul(randB:float()[1])
      -- sharpness augmentation
      local cur_im_blurred = image.convolve(cur_im, blurK, 'same')
      local cur_im_residue = torch.add(cur_im, -1, cur_im_blurred)
      local randSh = torch.rand(1)*1.5
      cur_im:add(randSh:float()[1], cur_im_residue)

      cur_im:mul(std)
      cur_im:add(mean)

      local cur_attr = trainData[idx][2]:clone()

      batch_im[{{k},{},{},{}}] = cur_im
      batch_attr[{{k}, {}}] = cur_attr
      k = k + 1
    end
    batch_im:mul(2):add(-1)

    local curLL = 0
    local curKL = 0

    local opfunc = function(x)
      collectgarbage()
      if x ~= parameters then
        parameters:copy(x)
      end

      cvae:zeroGradParameters()
      local f = cvae:forward({{batch_im, batch_attr}, batch_attr})
      local LLerr = criterionLL:forward(f, batch_im)
      local df_dw = criterionLL:backward(f, batch_im)
      cvae:backward({{batch_im, batch_attr}, batch_attr}, df_dw)

      local KLDerr = criterionKLD:forward(enc_sampling:get(1).output, batch_im)
      local de_dw = criterionKLD:backward(enc_sampling:get(1).output, batch_im)
      encoder:backward({batch_im, batch_attr}, de_dw)

      curLL = LLerr
      curKL = KLDerr
      local lowerbound = (LLerr + KLDerr) 
      
      return lowerbound, gradients
    end
    
    x, batchLB = optim_utils.adam_v2(opfunc, parameters, config, state)
    trainLB = trainLB + batchLB[1]
    trainLL = trainLL + curLL
    trainKL = trainKL + curKL
    print(string.format('Epoch %d [%d/%d]:\t%g %g %g', 
      t, i, N_train, batchLB[1]/opts.batchSize, curLL/opts.batchSize, curKL/opts.batchSize))

  end

  if LBlist_train then
    LBlist_train = torch.cat(LBlist_train:float(), torch.Tensor(1,1):fill(trainLB/N_train):float(), 1)
  else
    LBlist_train = torch.Tensor(1,1):fill(trainLB/N_train):float()
  end

  -- val set
  cvae:evaluate()
  for i = 1, N_val, opts.batchSize do
    --xlua.progress(i+opts.batchSize-1, nval)

    local batch_im = torch.Tensor(opts.batchSize, 3, opts.scale, opts.scale)
    local batch_attr = torch.Tensor(opts.batchSize, opts.ydim)

    local k = 1
    for j = i, i+opts.batchSize-1 do
      local idx = math.random(nval)
      local cur_im = valData[idx][1]:float():clone()
      local flip = math.random(2)-1
      if flip == 1 then
        cur_im = image.vflip(cur_im:float())
      end
      local cur_attr = valData[idx][2]:clone()

      cur_im:mul(std)
      cur_im:add(mean)
      batch_im[{{k},{},{},{}}] = cur_im
      batch_attr[{{k}, {}}] = cur_attr
      k = k + 1
    end
    batch_im:mul(2):add(-1)

    cvae:zeroGradParameters()
    local f = cvae:forward({{batch_im, batch_attr}, batch_attr})
    local LLerr = criterionLL:forward(f, batch_im)
    local KLDerr = criterionKLD:forward(enc_sampling:get(1).output, batch_im)

    samples_gt = batch_im:float():clone()
    samples_gen = f[1]:float():clone()

    valLB = valLB + (LLerr + KLDerr)
    valLL = valLL + LLerr
    valKL = valKL + KLDerr
  end

  if LBlist_val then
    LBlist_val = torch.cat(LBlist_val:float(), torch.Tensor(1, 1):fill(valLB/N_val):float(), 1)
  else
    LBlist_val = torch.Tensor(1,1):fill(valLB/N_val):float()
  end

  print(string.format('#### epoch (%d)\t train LB (LL, KL) = %g (%g, %g) ####',
    t, trainLB/N_train, trainLL/N_train, trainKL/N_train))
  print(string.format('#### epoch (%d)\t val LB (LL, KL) = %g (%g, %g) ####', 
    t, valLB/N_val, valLL/N_val, valKL/N_val))

  if t % 1 == 0 then
    
    local batch_im = torch.Tensor(32, 3, opts.scale, opts.scale)
    local batch_attr = torch.Tensor(32, opts.ydim)
    local batch_z = torch.Tensor(32, opts.zdim):normal(0,1)
    for i = 1, 32 do
      local idx = math.random(nval)
      local cur_im = valData[idx][1]:float():clone()
      local cur_attr = valData[idx][2]:float():clone()
      cur_im:mul(std)
      cur_im:add(mean)
      batch_im[i] = cur_im
      batch_attr[i] = cur_attr
    end
    batch_im:mul(2):add(-1)

    local f = decoder:forward({batch_z, batch_attr})

    to_plot = {}
    for i = 1, 32 do
      local res = f[1][i]:clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()
      local res = batch_im[i]:clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()
    end

    local formatted = image.toDisplayTensor({input=to_plot, nrow=8})
    formatted = formatted:double()
    formatted:mul(255)
    formatted = image.rotate(formatted, -math.pi/2):clone()
    formatted = formatted:byte()

    image.save(opts.modelPath .. string.format('/sample-%d.jpg', t), formatted)
  end

  if t % opts.saveFreq == 0 then
    collectgarbage()
    torch.save((opts.modelPath .. string.format('/net-epoch-%d.t7', t))
      , {encoder = encoder, decoder = decoder})
    torch.save((opts.modelPath .. '/state.t7'), state)
    torch.save((opts.modelPath .. '/statistic_train.t7'), LBlist_train)
    torch.save((opts.modelPath .. '/statistic_val.t7'), LBlist_val)
  end

end

