require 'torch'
require 'nn'

local VAE = {}

function VAE.get_encoder(latent_variable_size)
  -- The Encoder
  local encoder = nn.Sequential()
  encoder:add(cudnn.SpatialConvolution(1, 32, 5, 5, 1, 1, 2, 2))
  encoder:add(nn.SpatialBatchNormalization(32))
  encoder:add(cudnn.SpatialMaxPooling(2,2))
  encoder:add(cudnn.ReLU(true))
  encoder:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
  encoder:add(nn.SpatialBatchNormalization(64))
  encoder:add(cudnn.SpatialMaxPooling(2,2))
  encoder:add(cudnn.ReLU(true))
  encoder:add(cudnn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2))
  encoder:add(nn.SpatialBatchNormalization(96))
  encoder:add(cudnn.ReLU(true))
  encoder:add(cudnn.SpatialMaxPooling(2,2))
  encoder:add(nn.Reshape(8*8*96))
  encoder:add(nn.Linear(8*8*96, 1024))
  encoder:add(cudnn.ReLU(true))
  --encoder:add(nn.Dropout())

  mean_logvar = nn.ConcatTable()
  mean_logvar:add(nn.Linear(1024, latent_variable_size))
  mean_logvar:add(nn.Linear(1024, latent_variable_size))

  encoder:add(mean_logvar)

  return encoder
end

function VAE.get_decoder(latent_variable_size, feature_size)
  -- The Decoder
  local decoder = nn.Sequential()
  decoder:add(nn.Linear(latent_variable_size, 1024))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.Linear(1024, 128*8*8))
  decoder:add(nn.Reshape(128, 8, 8))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(256))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(256))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(cudnn.SpatialConvolution(256, feature_size, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(128))
  decoder:add(cudnn.ReLU(true))
  two_outs = nn.ConcatTable()
  two_outs:add(nn.Identity())
  two_outs:add(cudnn.SpatialConvolution(feature_size, 1, 3, 3, 1, 1, 1, 1))
  decoder:add(two_outs)

  return decoder
end

return VAE
