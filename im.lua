require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local VAE = require 'vae'

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

-------------------------------------------------------------------------------
-- IMAGE VAE Model
-------------------------------------------------------------------------------
local layer, parent = torch.class('nn.VAEModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)
  self.encoder = VAE.get_encoder(opt.latent_variable_size)
  self.sampler = nn.GSampler()
  self.decoder = VAE.get_decoder(opt.latent_variable_size, opt.feature_size)
end

function layer:getModulesList()
  return {self.encoder, self.sampler, self.decoder}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.encoder:parameters()
  local p2,g2 = self.decoder:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  self.encoder:training()
  self.sampler:training()
  self.decoder:training()
end

function layer:evaluate()
  self.encoder:evaluate()
  self.sampler:evaluate()
  self.decoder:evaluate()
end

function layer:updateOutput(input)

  self.mean, self.var_log = unpack(self.encoder:forward(input))

  self.z = self.sampler:forward({self.mean, self.var_log})

  local features, pred = unpack(self.decoder:forward(self.z))

  self.output = {features, pred, self.mean, self.var_log}

  return self.output
end

function layer:updateGradInput(input, gradOutput)

  local dz = self.decoder:backward(self.z, {gradOutput[1], gradOutput[2]})

  local dmean, dvar_log = unpack(self.sampler:backward({self.mean, self.var_log}, dz))
  dmean:add(gradOutput[3])
  dvar_log:add(gradOutput[4])

  self.gradInput = self.encoder:backward(input, {dmean, dvar_log})

  return self.gradInput
end
