require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'
require 'image'
require 'pl'
require 'paths'

require 'layers.LinearWP'
require 'layers.LinearGaussian'
require 'layers.LinearMix'
require 'layers.LinearMix2'
require 'layers.Reparametrize'
require 'layers.GaussianCriterion'
require 'layers.KLDCriterion'
require 'utils.scaled_lfw'
optim_utils = require 'utils.adam_v2'

-- parse command-line options
opts = lapp[[
  --saveFreq      (default 20)        save every saveFreq epochs
  --modelString   (default 'arch_disCVAE')        reload pretrained network
  -p,--plot                         plot while training
  -t,--threads    (default 4)         number of threads
  -g,--gpu        (default -1)        gpu to run on (default cpu)
  --scale         (default 64)        scale of images to train on
  --zfdim         (default 192)
  --zbdim         (default 64)
  -y,--ydim       (default 73)
  --maxEpoch      (default 0)
  --batchSize     (default 32)
  --weightDecay   (default 0.004)
  --adam          (default 1)         lr mode
  --modelDir      (default 'models/')
  --alpha         (default 0.3)
]]

opts.modelName = string.format('LFW_%s_adam%d_bs%d_zdim(%d,%d)_wd%g_alpha%g', 
  opts.modelString, opts.adam, opts.batchSize, 
  opts.zfdim, opts.zbdim, opts.weightDecay, opts.alpha)

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
  print('scripts/' .. opts.modelString .. '.lua')

  lfwdiscvae_module = dofile('scripts/' .. opts.modelString .. '.lua')
  encoder_fg, encoder_bg, decoder_fg, decoder_bg = lfwdiscvae_module.create(opts)
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
    encoder_fg = loader.encoder_fg
    encoder_bg = loader.encoder_bg
    decoder_fg = loader.decoder_fg
    decoder_bg = loader.decoder_bg
    state = torch.load(opts.modelPath .. '/state.t7')
    LBlist_train = torch.load(opts.modelPath .. '/statistic_train.t7') 
    LBlist_val = torch.load(opts.modelPath .. '/statistic_val.t7')
    print(string.format('resuming from epoch %d', i))
    break
  end
end

local inputs = {nn.Identity()(), nn.Identity()(), nn.Identity()()}

local mu_sigma_fg = encoder_fg({inputs[1], inputs[3]})
sampler_fg = nn.Sequential()
sampler_fg:add(nn.Reparametrize(opts.zfdim))
local z_fg = sampler_fg(mu_sigma_fg)

local mu_sigma_bg = encoder_bg({inputs[2], inputs[3], z_fg})
sampler_bg = nn.Sequential()
sampler_bg:add(nn.Reparametrize(opts.zbdim))
local z_bg = sampler_bg(mu_sigma_bg)

-- decoder with foreground
local h_fg, h_gate = decoder_fg({z_fg, inputs[3]}):split(2)
--local output_fg = decoder_fg(h_fg)

-- decoder with image
local h_bg = decoder_bg(z_bg)
local h_mulgate_bg = nn.CMulTable()({h_bg, h_gate})
local h_im = nn.CAddTable()({h_fg, h_mulgate_bg})
--local ouput_im = decoder_im(h_im)

local outputs = {h_fg, h_im, h_gate}
discvae = nn.gModule(inputs, outputs)

parameters, gradients = discvae:getParameters()

-- print networks
if opts.gpu >= 0 then
  print('Copying model to gpu')
  encoder_fg:cuda()
  encoder_bg:cuda()
  decoder_fg:cuda()
  decoder_bg:cuda()
end

-- Optimization criteria
criterionLL0 = nn.SmoothL1Criterion()
criterionLL0.sizeAverage = false
criterionLL1 = nn.SmoothL1Criterion()
criterionLL1.sizeAverage = false
criterionKLD0 = nn.KLDCriterion()
criterionKLD1 = nn.KLDCriterion()
criterionBCE = nn.BCECriterion()
criterionBCE.sizeAverage = false

-- get/create dataset
ntrain = math.floor(9464 * 0.9)
nval = 9464 - ntrain

print('data preprocessing')
lfw.setScale(opts.scale)
trainData = lfw.loadTrainSet(1, ntrain)
--mean, std = image_utils.normalizeGlobal(trainData.data)
trainData:scaleData()

valData = lfw.loadTrainSet(ntrain + 1, ntrain + nval)
--image_utils.normalizeGlobal(valData.data, mean, std)
valData:scaleData()

ntrain = trainData:size()
nval = valData:size()
print(ntrain)
print(nval)

-- hyperparams
function getAdamParams(opts)
  config = {}
  if opts.adam == 1 then
    config.learningRate = 0.0003
    config.epsilon = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weightDecay = opts.weightDecay
  elseif opts.adam == 2 then
    config.learningRate = 0.0003
    config.epsilon = 1e-8
    config.beta1 = 0.5
    config.beta2 = 0.999
    config.weightDecay = opts.weightDecay
  elseif opts.adam == 3 then
    config.learningRate = 0.0001
    config.epsilon = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weightDecay = opts.weightDecay
  end
  return config
end

config = getAdamParams(opts)
print(config)

local blurK = torch.FloatTensor(5, 5):fill(1/25)

-- training loop
for t = epoch+1, opts.maxEpoch do
  local trainLB = 0
  local trainLL0 = 0
  local trainLL1 = 0
  local trainKL0 = 0
  local trainKL1 = 0
  local trainBCE = 0

  local valLB = 0
  local valLL0 = 0
  local valLL1 = 0
  local valKL0 = 0
  local valKL1 = 0
  local valBCE = 0

  local time = sys.clock()

  -- Make sure batches are always batchSize
  local N_train = ntrain - (ntrain % opts.batchSize)
  local N_val = nval - (nval % opts.batchSize)

  discvae:training()
  for i = 1, N_train, opts.batchSize do 
    local batch_im = torch.Tensor(opts.batchSize, 3, opts.scale, opts.scale)
    local batch_fg = torch.Tensor(opts.batchSize, 3, opts.scale, opts.scale)
    local batch_attr = torch.Tensor(opts.batchSize, opts.ydim)
    local batch_mask = torch.Tensor(opts.batchSize, 3, opts.scale, opts.scale)

    local k = 1
    for j = i, i+opts.batchSize-1 do
      local idx = math.random(ntrain)
      local cur_im = trainData[idx][1]:float():clone()
      local cur_mask = trainData[idx][2]:float():clone()
      -- flip
      local flip = math.random(2)-1
      if flip == 1 then
        cur_im = image.hflip(cur_im:float())
        cur_mask = image.hflip(cur_mask:float())
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
      
      local cur_fg = torch.cmul(cur_im, torch.repeatTensor(cur_mask, 3, 1, 1))
      cur_mask:mul(-1):add(1)

      local cur_attr = trainData[idx][3]:clone()

      batch_im[k] = cur_im
      batch_fg[k] = cur_fg
      batch_mask[k] = torch.repeatTensor(cur_mask, 3, 1, 1)
      batch_attr[k] = cur_attr
      k = k + 1
    end
    batch_im:mul(2):add(-1)
    batch_fg:mul(2):add(-1)

    local curLL0 = 0
    local curLL1 = 0
    local curKL0 = 0
    local curKL1 = 0
    local curBCE = 0

    local opfunc = function(x)
      collectgarbage()
      if x ~= parameters then
        parameters:copy(x)
      end

      discvae:zeroGradParameters()
      --print(decoder_fg:forward({torch.Tensor(32, opts.zfdim), torch.Tensor(32, opts.ydim)}))
      local f = discvae:forward({batch_fg, batch_im, batch_attr})
     
      local mu_sigma_fg = encoder_fg.output
      local mu_sigma_bg = encoder_bg.output
      local batch_zfg = sampler_fg.output
      local batch_zbg = sampler_bg.output
      
      local f0 = f[1]:clone()
      local LLerr0 = criterionLL0:forward(f0, batch_fg)
      local df0_dw = criterionLL0:backward(f0, batch_fg)
  
      local f1 = f[2]:clone()
      local LLerr1 = criterionLL1:forward(f1, batch_im)
      local df1_dw = criterionLL1:backward(f1, batch_im)

      local fm = f[3]:clone()
      local BCEerr = criterionBCE:forward(fm, batch_mask)
      local dfm_dw = criterionBCE:backward(fm, batch_mask):mul(opts.alpha)

      discvae:backward({batch_fg, batch_im, batch_attr}, {df0_dw, df1_dw, dfm_dw})

      local KLDerr0 = -criterionKLD0:forward(mu_sigma_fg, nil)
      local KLDerr1 = -criterionKLD1:forward(mu_sigma_bg, nil)

      local de0_dw = criterionKLD0:backward(mu_sigma_fg, nil)
      de0_dw[1]:mul(-1)
      de0_dw[2]:mul(-1)
      local de1_dw = criterionKLD1:backward(mu_sigma_bg, nil)
      de1_dw[1]:mul(-1)
      de1_dw[2]:mul(-1)

      encoder_fg:backward({batch_fg, batch_attr}, de0_dw)
      encoder_bg:backward({batch_im, batch_attr, batch_zfg}, de1_dw)
     
      curLL0 = LLerr0
      curLL1 = LLerr1
      curKL0 = KLDerr0
      curKL1 = KLDerr1
      curBCE = BCEerr

      local lowerbound = (LLerr0 + LLerr1 + KLDerr0 + KLDerr1 + BCEerr * opts.alpha) 
       
      return lowerbound, gradients
    end
    
    x, batchLB = optim_utils.adam_v2(opfunc, parameters, config, state)
    trainLB = trainLB + batchLB[1]
    trainLL0 = trainLL0 + curLL0
    trainLL1 = trainLL1 + curLL1
    trainKL0 = trainKL0 + curKL0
    trainKL1 = trainKL1 + curKL1
    trainBCE = trainBCE + curBCE
    print(string.format('Epoch %d [%d/%d]:\t LB (FG, IM, MA, KL_FG, KL_IM) = %g [%g %g %g], [%g %g]', 
      t, i, N_train, batchLB[1]/opts.batchSize, curLL0/opts.batchSize, 
      curLL1/opts.batchSize, curBCE/opts.batchSize, 
      curKL0/opts.batchSize, curKL1/opts.batchSize))

  end

  if LBlist_train then
    LBlist_train = torch.cat(LBlist_train:float(), torch.Tensor(1,1):fill(trainLB/N_train):float(), 1)
  else
    LBlist_train = torch.Tensor(1,1):fill(trainLB/N_train):float()
  end

  -- val set
  discvae:evaluate()
  for i = 1, N_val, opts.batchSize do
    local batch_im = torch.Tensor(opts.batchSize, 3, opts.scale, opts.scale)
    local batch_fg = torch.Tensor(opts.batchSize, 3, opts.scale, opts.scale)
    local batch_mask = torch.Tensor(opts.batchSize, 3, opts.scale, opts.scale)
    local batch_attr = torch.Tensor(opts.batchSize, opts.ydim)

    local k = 1
    for j = i, i+opts.batchSize-1 do
      local idx = math.random(nval)
      local cur_im = valData[idx][1]:float():clone()
      local cur_mask = valData[idx][2]:float():clone()
      local flip = math.random(2)-1
      if flip == 1 then
        cur_im = image.hflip(cur_im:float())
      end
      local cur_fg = torch.cmul(cur_im, torch.repeatTensor(cur_mask, 3, 1, 1))
      cur_mask:mul(-1):add(1)

      local cur_attr = valData[idx][3]:clone()
      
      batch_im[k] = cur_im
      batch_fg[k] = cur_fg
      batch_mask[k] = torch.repeatTensor(cur_mask, 3, 1, 1)
      batch_attr[k] = cur_attr
      k = k + 1
    end
    batch_im:mul(2):add(-1)
    batch_fg:mul(2):add(-1)

    discvae:zeroGradParameters()
    local f = discvae:forward({batch_fg, batch_im, batch_attr})
    
    -- forward
    local mu_sigma_fg = encoder_fg.output
    local mu_sigma_bg = encoder_bg.output

    local f0 = f[1]:clone()
    local LLerr0 = criterionLL0:forward(f0, batch_fg)

    local f1 = f[2]:clone()
    local LLerr1 = criterionLL1:forward(f1, batch_im)

    local fm = f[3]:clone()
    local BCEerr = criterionBCE:forward(fm, batch_mask)

    local KLDerr0 = -criterionKLD0:forward(mu_sigma_fg, nil)
    local KLDerr1 = -criterionKLD1:forward(mu_sigma_bg, nil)

    valLB = valLB + (LLerr0 + LLerr1 + KLDerr0 + KLDerr1 + BCEerr * opts.alpha)
    valLL0 = valLL0 + LLerr0
    valLL0 = valLL1 + LLerr1
    valKL0 = valKL0 + KLDerr0
    valKL1 = valKL1 + KLDerr1
    valBCE = valBCE + BCEerr
  end

  if LBlist_val then
    LBlist_val = torch.cat(LBlist_val:float(), torch.Tensor(1, 1):fill(valLB/N_val):float(), 1)
  else
    LBlist_val = torch.Tensor(1,1):fill(valLB/N_val):float()
  end

  print(string.format('#### epoch (%d)\t train LB (FG, IM, MA, KL_FG, KL_IM) = %g (%g, %g, %g, %g, %g) ####',
    t, trainLB/N_train, trainLL0/N_train, trainLL1/N_train, trainBCE/N_train,
      trainKL0/N_train, trainKL1/N_train))
  print(string.format('#### epoch (%d)\t val LB (FG, IM, MA, KL_FG, KL_IM) = %g (%g, %g, %g, %g, %g) ####', 
    t, valLB/N_val, valLL0/N_val, valLL1/N_val, valBCE/N_val, 
      valKL0/N_val, valKL1/N_val))

  -- Displaying the intermediate results
  if t % 1 == 0 then
    
    local batch_im = torch.Tensor(32, 3, opts.scale, opts.scale)
    local batch_fg = torch.Tensor(32, 3, opts.scale, opts.scale)
    local batch_attr = torch.Tensor(32, opts.ydim)
    local batch_z_fg = torch.Tensor(32, opts.zfdim):normal(0,1)
    local batch_z_bg = torch.Tensor(32, opts.zbdim):normal(0,1)
    for i = 1, 32 do
      local idx = math.random(nval)
      local cur_im = valData[idx][1]:float():clone()
      local cur_mask = valData[idx][2]:float():clone()
      local cur_attr = valData[idx][3]:float():clone()

      local cur_fg = torch.cmul(cur_im, torch.repeatTensor(cur_mask, 3, 1, 1))
      
      batch_im[i] = cur_im
      batch_fg[i] = cur_fg
      batch_attr[i] = cur_attr
    end
    --batch_im:mul(2):add(-1)

    local h_mix = decoder_fg:forward({batch_z_fg, batch_attr})
    local f0 = h_mix[1]
    local h_gate = h_mix[2]

    local h_bg = decoder_bg:forward(batch_z_bg)
    local h_mulgate_bg = nn.CMulTable():forward({h_bg, h_gate})
    local f1 = nn.CAddTable():forward({f0, h_mulgate_bg})
    
    to_plot = {}
    for i = 1, 32 do
      local res = f0[i]:clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)      
      to_plot[#to_plot+1] = res:clone()

      local res = f1[i]:clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()
      
      local res = h_gate[i]:clone()
      res = torch.squeeze(res)
      to_plot[#to_plot+1] = res:clone()

      local res = batch_im[i]:clone()
      res = torch.squeeze(res)
      to_plot[#to_plot+1] = res:clone()
    end

    local formatted = image.toDisplayTensor({input=to_plot, nrow=16})
    formatted = formatted:double()
    formatted:mul(255)
    formatted = formatted:byte()

    image.save(opts.modelPath .. string.format('/sample-%d.jpg', t), formatted)
  end

  -- Saving to files
  if t % opts.saveFreq == 0 then
    collectgarbage()
    torch.save((opts.modelPath .. string.format('/net-epoch-%d.t7', t))
      , {encoder_fg = encoder_fg, encoder_bg = encoder_bg, 
        decoder_fg = decoder_fg, decoder_bg = decoder_bg})
    torch.save((opts.modelPath .. '/state.t7'), state)
    torch.save((opts.modelPath .. '/statistic_train.t7'), LBlist_train)
    torch.save((opts.modelPath .. '/statistic_val.t7'), LBlist_val)
  end

end

