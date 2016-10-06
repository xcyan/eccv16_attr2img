local LFWdiscvae = {}

function LFWdiscvae.create(opts)
  local encoder_fg = LFWdiscvae.create_encoder_fg(opts)
  local encoder_bg = LFWdiscvae.create_encoder_bg(opts)
  local decoder_fg = LFWdiscvae.create_decoder_fg(opts)
  local decoder_bg = LFWdiscvae.create_decoder_bg(opts)
  return encoder_fg, encoder_bg, decoder_fg, decoder_bg
end

function LFWdiscvae.create_encoder_fg(opts)
  local encoderX = nn.Sequential()
  -- 64 x 64 --> 32 x 32
  encoderX:add(cudnn.SpatialConvolution(3, 64, 5, 5, 1, 1, 2, 2))
  encoderX:add(cudnn.ReLU())
  encoderX:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- 32 x 32 --> 16 x 16
  encoderX:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
  encoderX:add(nn.SpatialBatchNormalization(128))
  encoderX:add(cudnn.ReLU())
  encoderX:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 16 x 16 --> 8 x 8
  encoderX:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  encoderX:add(nn.SpatialBatchNormalization(256))
  encoderX:add(cudnn.ReLU())
  encoderX:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- 8 x 8 --> 4 x 4
  encoderX:add(cudnn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1))
  encoderX:add(nn.SpatialBatchNormalization(256))
  encoderX:add(cudnn.ReLU())

  -- 4 x 4 --> 1 x 1
  encoderX:add(cudnn.SpatialConvolution(256, 1024, 4, 4, 1, 1, 0, 0))
  encoderX:add(cudnn.ReLU())

  encoderX:add(nn.Reshape(1024))
  encoderX:add(nn.Linear(1024, 1024))
  encoderX:add(cudnn.ReLU())
  encoderX:add(nn.Dropout(0.5))

  local encoderY = nn.Sequential()
  encoderY:add(nn.Linear(opts.ydim, opts.zfdim))
  encoderY:add(cudnn.ReLU())

  local encoder = nn.Sequential()
  encoder:add(nn.ParallelTable():add(encoderX):add(encoderY))
  local enc_z0 = nn.LinearMix2(1024, opts.zfdim, opts.zfdim)
  encoder:add(enc_z0)
  
  return encoder
end


function LFWdiscvae.create_encoder_bg(opts)

  local encoderX = nn.Sequential()
  -- 64 x 64 --> 32 x 32
  encoderX:add(cudnn.SpatialConvolution(3, 64, 5, 5, 1, 1, 2, 2))
  encoderX:add(cudnn.ReLU())
  encoderX:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- 32 x 32 --> 16 x 16
  encoderX:add(cudnn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2))
  encoderX:add(nn.SpatialBatchNormalization(64))
  encoderX:add(cudnn.ReLU())
  encoderX:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 16 x 16 --> 8 x 8
  encoderX:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
  encoderX:add(nn.SpatialBatchNormalization(64))
  encoderX:add(cudnn.ReLU())
  encoderX:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

  -- 8 x 8 --> 4 x 4
  encoderX:add(cudnn.SpatialConvolution(64, 64, 3, 3, 2, 2, 1, 1))
  encoderX:add(nn.SpatialBatchNormalization(64))
  encoderX:add(cudnn.ReLU())

  -- 4 x 4 --> 1 x 1
  encoderX:add(cudnn.SpatialConvolution(64, 256, 4, 4, 1, 1, 0, 0))
  encoderX:add(cudnn.ReLU())

  encoderX:add(nn.Reshape(256))
  encoderX:add(nn.Linear(256, 256))
  encoderX:add(cudnn.ReLU())
  encoderX:add(nn.Dropout(0.5))
  
  local encoder = nn.Sequential()
  encoder:add(nn.ParallelTable():add(encoderX):add(nn.Copy()):add(nn.Copy()))
  encoder:add(nn.JoinTable(2))
  encoder:add(nn.Linear(256 + opts.ydim + opts.zfdim, 256))
  encoder:add(cudnn.ReLU())

  encoder:add(nn.LinearGaussian(256, opts.zbdim))

  return encoder
end

--build decoder for fg
function LFWdiscvae.create_decoder_fg(opts)

  local decoderY = nn.Sequential()
  decoderY:add(nn.Linear(opts.ydim, opts.zfdim*2))
  decoderY:add(cudnn.ReLU())

  local decoderFG = nn.Sequential()
  decoderFG:add(nn.ParallelTable():add(nn.Copy()):add(decoderY))
  decoderFG:add(nn.LinearMix(opts.zfdim, opts.zfdim*2, 256))
  decoderFG:add(cudnn.ReLU())
 
  decoderFG:add(nn.Linear(256, 256*8*8))
  decoderFG:add(nn.Reshape(256, 8, 8))
  decoderFG:add(cudnn.ReLU())
 
  -- 8 x 8 --> 8 x 8
  decoderFG:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
  decoderFG:add(nn.SpatialBatchNormalization(256))
  decoderFG:add(cudnn.ReLU())

  -- 8 x 8 --> 16 x 16
  decoderFG:add(nn.SpatialUpSamplingNearest(2))
  decoderFG:add(cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2))
  decoderFG:add(nn.SpatialBatchNormalization(256))
  decoderFG:add(cudnn.ReLU())

  -- 16 x 16 --> 32 x 32 (full conv)
  decoderFG:add(nn.SpatialUpSamplingNearest(2))
  decoderFG:add(nn.SpatialFullConvolution(256, 128, 5, 5, 1, 1, 2, 2))
  decoderFG:add(nn.SpatialBatchNormalization(128))
  decoderFG:add(cudnn.ReLU())
 
  -- 32 x 32 --> 64 x 64 (full conv)
  decoderFG:add(nn.SpatialUpSamplingNearest(2))
  decoderFG:add(nn.SpatialFullConvolution(128, 64, 5, 5, 1, 1, 2, 2))
  decoderFG:add(nn.SpatialBatchNormalization(64))
  decoderFG:add(cudnn.ReLU())

  local fea = nn.Sequential()
  fea:add(cudnn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2))
  fea:add(nn.Tanh())

  local gate = nn.Sequential()
  gate:add(cudnn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2))
  gate:add(nn.Sigmoid())

  decoderFG:add(nn.ConcatTable():add(fea):add(gate))
  return decoderFG
end

--build mixer for image
function LFWdiscvae.create_decoder_bg(opts)
  local decoderBG = nn.Sequential()

  decoderBG:add(nn.Linear(opts.zbdim, 256*2*2))
  decoderBG:add(nn.Reshape(256, 2, 2))
  decoderBG:add(cudnn.ReLU())

  -- 2 x 2 --> 4 x 4
  decoderBG:add(nn.SpatialUpSamplingNearest(2))
  decoderBG:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
  decoderBG:add(nn.SpatialBatchNormalization(128))
  decoderBG:add(cudnn.ReLU())

  -- 4 x 4 --> 8 x 8
  decoderBG:add(nn.SpatialUpSamplingNearest(2))
  decoderBG:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
  decoderBG:add(nn.SpatialBatchNormalization(128))
  decoderBG:add(cudnn.ReLU())

  -- 8 x 8 --> 16 x 16
  decoderBG:add(nn.SpatialUpSamplingNearest(2))
  decoderBG:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
  decoderBG:add(nn.SpatialBatchNormalization(64))
  decoderBG:add(cudnn.ReLU())

  decoderBG:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
  decoderBG:add(nn.SpatialBatchNormalization(64))
  decoderBG:add(cudnn.ReLU())

  -- 16 x 16 --> 32 x 32
  decoderBG:add(nn.SpatialUpSamplingNearest(2))
  decoderBG:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
  decoderBG:add(nn.SpatialBatchNormalization(64))
  decoderBG:add(cudnn.ReLU())

  decoderBG:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
  decoderBG:add(nn.SpatialBatchNormalization(64))
  decoderBG:add(cudnn.ReLU())

  -- 32 x 32 --> 64 x 64
  decoderBG:add(nn.SpatialUpSamplingNearest(2))
  decoderBG:add(cudnn.SpatialConvolution(64, 32, 5, 5, 1, 1, 2, 2))
  decoderBG:add(nn.SpatialBatchNormalization(32))
  decoderBG:add(cudnn.ReLU())

  -- 64 x 64 --> 64 x 64
  decoderBG:add(cudnn.SpatialConvolution(32, 3, 5, 5, 1, 1, 2, 2))
  decoderBG:add(nn.Tanh())

  return decoderBG
end

return LFWdiscvae
