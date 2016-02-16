require 'torch'
require 'nn'

local VAE = {}

function VAE.get_encoder(latent_variable_size)
  -- The Encoder
  local encoder = nn.Sequential()
  encoder:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
  encoder:add(cudnn.SpatialMaxPooling(2,2))
  encoder:add(cudnn.ReLU(true))
  encoder:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
  encoder:add(cudnn.SpatialMaxPooling(2,2))
  encoder:add(cudnn.ReLU(true))
  encoder:add(cudnn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2))
  encoder:add(cudnn.ReLU(true))
  encoder:add(cudnn.SpatialMaxPooling(2,2))
  encoder:add(nn.Reshape(8*8*96))
  encoder:add(nn.Linear(8*8*96, 1024))
  encoder:add(cudnn.ReLU(true))
  --encoder:add(nn.Dropout())
  encoder:add(nn.Linear(1024, 500))
  encoder:add(cudnn.ReLU(true))
  --encoder:add(nn.Dropout())

  --local encoder = nn.Sequential()
  --encoder:add(nn.Reshape(16*16*3))
  --encoder:add(nn.Linear(16*16*3, 500))
  --encoder:add(nn.ReLU(true))

  mean_logvar = nn.ConcatTable()
  mean_logvar:add(nn.Linear(500, latent_variable_size))
  mean_logvar:add(nn.Linear(500, latent_variable_size))

  encoder:add(mean_logvar)

  return encoder
end

function VAE.get_decoder(latent_variable_size, feature_size)
  -- The Decoder
  local decoder = nn.Sequential()
  decoder:add(nn.Linear(latent_variable_size, 500))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.Linear(500, 1024))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.Linear(1024, 96*8*8))
  decoder:add(nn.Reshape(96, 8, 8))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(cudnn.SpatialConvolution(96, 64, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(64))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(cudnn.SpatialConvolution(64, 32, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(32))
  decoder:add(cudnn.ReLU(true))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(cudnn.SpatialConvolution(32, feature_size, 5, 5, 1, 1, 2, 2))
  decoder:add(cudnn.Sigmoid())

  --local decoder = nn.Sequential()
  --decoder:add(nn.Linear(latent_variable_size, 500))
  --decoder:add(nn.ReLU(true))
  --decoder:add(nn.Linear(500, feature_size*16*16))
  --decoder:add(nn.Reshape(feature_size, 16, 16))
  return decoder
end

-- Based on JoinTable module
local GSampler, parent = torch.class('nn.GSampler', 'nn.Module')

function GSampler:__init()
  parent.__init(self)
  self.gradInput = {}
end

function GSampler:updateOutput(input)
  self.eps = self.eps or input[1].new()
  self.eps:resizeAs(input[1]):copy(torch.randn(input[1]:size()))

  self.ouput = self.output or self.output.new()
  self.output:resizeAs(input[2]):copy(input[2])
  self.output:mul(0.5):exp():cmul(self.eps)

  self.output:add(input[1])

  return self.output
end

function GSampler:updateGradInput(input, gradOutput)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)

  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[2]:resizeAs(gradOutput):copy(input[2])

  self.gradInput[2]:mul(0.5):exp():mul(0.5):cmul(self.eps)
  self.gradInput[2]:cmul(gradOutput)

  return self.gradInput
end

-- Based on JoinTable module
local PatchExtractor, parent = torch.class('nn.PatchExtractor', 'nn.Module')

function PatchExtractor:__init(opt)
  parent.__init(self)
  self.patch_size = opt.patch_size
  self.feature_dim = opt.feature_dim
  self.im_batch_size = opt.im_batch_size
  self.pm_batch_size = opt.pm_batch_size
  self.batch_size = self.im_batch_size * self.pm_batch_size
  self.num_neighbors = opt.num_neighbors
  self.border_size = opt.border_size
  self.border = opt.border
  self.noise = opt.noise
  self.h_index = torch.Tensor(self.batch_size)
  self.w_index = torch.Tensor(self.batch_size)
  self.gradInput = {}
end

function PatchExtractor:updateOutput(input)
  local images = input[1]
  local features = input[2]
  local height = images:size(3)
  local width = images:size(4)
  local nChannels = images:size(2)
  local batch_size = self.batch_size
  local ps = self.patch_size
  local bs = self.border_size
  -- two potential schemes, initialize with a border of one pixel in both directions.
  local pixel_patches = torch.Tensor(batch_size, nChannels, ps+2, ps+2):type(images:type()):fill(self.border)
  local feature_patches = torch.Tensor(batch_size, self.feature_dim, ps, ps):type(images:type())
  for i=1,self.im_batch_size do
    for j=1,self.pm_batch_size do
      local h = torch.random(1, height-ps+1)
      local w = torch.random(1, width-ps+1)
      -- put the patch in the center.
      local idx = (i-1)*self.pm_batch_size + j
      pixel_patches[{idx,{},{2,ps+1},{2,ps+1}}] = images[{i, {}, {h, h+ps-1}, {w, w+ps-1}}]
      feature_patches[idx] = features[{i, {}, {h, h+ps-1}, {w, w+ps-1}}]
      self.h_index[idx] = h
      self.w_index[idx] = w
    end
  end
  -- prepare the targets
  local targets = pixel_patches[{{},{},{2,ps+1},{2,ps+1}}]
  targets = targets[{{},{},{bs+1,ps-bs},{bs+1, ps-bs}}]:clone()
  targets = targets:view(batch_size, nChannels, -1)
  targets = targets:permute(3, 1, 2):contiguous()

  -- prepare the inputs. -n1, left, n2, up, n3, right, n4 down.
  local n1, n2, n3, n4, pixel_inputs, feature_inputs
  n1 = pixel_patches[{{},{},{2,ps+1},{1,ps}}]:clone()
  n1 = n1:view(batch_size, nChannels, -1)
  n1 = n1:permute(3, 1, 2)
  n2 = pixel_patches[{{},{},{1,ps},{2,ps+1}}]:clone()
  n2 = n2:view(batch_size, nChannels, -1)
  n2 = n2:permute(3, 1, 2)
  n3 = pixel_patches[{{},{},{2,ps+1},{3,ps+2}}]:clone()
  n3 = n3:view(batch_size, nChannels, -1)
  n3 = n3:permute(3, 1, 2)
  if self.num_neighbors == 3 then
    pixel_inputs = torch.cat(torch.cat(n1, n2, 3), n3, 3)
    pixel_inputs = pixel_inputs:contiguous()
  elseif self.num_neighbors == 4 then
    n4 = pixel_patches[{{},{},{3,ps+2},{2,ps+1}}]:clone()
    n4 = n4:view(self.batch_size, nChannels, -1)
    n4 = n4:permute(3, 1, 2)
    pixel_inputs = torch.cat(torch.cat(torch.cat(n1, n2, 3), n3, 3), n4, 3)
  end
  feature_patches = feature_patches:view(batch_size, self.feature_dim, -1)
  local feature_inputs = feature_patches:permute(3, 1, 2)

  inputs = torch.cat(pixel_inputs, feature_inputs, 3)
  self.output = {targets, inputs}

  return self.output
end

function PatchExtractor:updateGradInput(input, gradOutput)
  local images = input[1]
  local features = input[2]
  local height = images:size(3)
  local width = images:size(4)
  local nChannels = images:size(2)
  local batch_size = self.batch_size
  local ps = self.patch_size
  local bs = self.border_size

  local grads = gradOutput[2]:permute(2, 3, 1)
  local grad_features = grads[{{}, {-self.feature_dim, -1} ,{}}]:clone()
  grad_features = grad_features:view(batch_size, self.feature_dim, ps, ps)
  self.gradInput[2] = torch.Tensor(features:size()):type(features:type()):fill(0)
  for i=1,self.im_batch_size do
    for j=1,self.pm_batch_size do
      local idx = (i-1)*self.pm_batch_size + j
      local h = self.h_index[idx]
      local w = self.w_index[idx]
      self.gradInput[2][{i, {}, {h, h+ps-1}, {w, w+ps-1}}]:add(grad_features[idx])
    end
  end
  -- no need to backward to pixels
  self.gradInput[1] = self.gradInput[1] or input[1].new()

  return self.gradInput
end

function VAE.ImageModel(opt)

  local encoder = VAE.get_encoder(opt.latent_variable_size)
  local sampler = nn.GSampler()
  local decoder = VAE.get_decoder(opt.latent_variable_size, opt.feature_size)
  --local decoder2 = decoder:clone('weight', 'bias', 'gradWeight', 'gradBias') -- sampler shares the decoder

  local input = nn.Identity()()
  local mean, log_var = encoder(input):split(2)
  local z = sampler({mean, log_var})
  local features, im_trainer
  features = decoder(z)
  im_trainer = nn.gModule({input},{features, mean, log_var})

  local features2, im_sampler
  local input2 = nn.Identity()()
  features2 = decoder(input2)
  im_sampler = nn.gModule({input2},{features2})

  return {im_trainer, im_sampler}
end

return VAE
