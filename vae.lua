require 'torch'
require 'nn'

local VAE = {}

function VAE.get_encoder(latent_variable_size)
  -- The Encoder
  local encoder = nn.Sequential()
  encoder:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
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
  decoder:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(128))
  decoder:add(cudnn.ReLU(true))
  decoder:add(cudnn.SpatialConvolution(128, feature_size, 3, 3, 1, 1, 1, 1))
  decoder:add(cudnn.Sigmoid())

  return decoder
end

return VAE
