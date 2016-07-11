require 'nn'
require 'gmms'
require 'pm'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'lstm'
local mvn = require 'misc.mvn'

-------------------------------------------------------------------------------
-- COLOR Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.RGBModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.pixel_size = 1 -- must be 1
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.num_mixtures = utils.getopt(opt, 'num_mixtures')
  self.encoding_size = utils.getopt(opt, 'encoding_size')
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Pixel Model
  self.seq_length = 3 -- three channel must be 3
  self.mult_in = utils.getopt(opt, 'mult_in')
  self.output_size = self.num_mixtures * (1+1+0+1)
  -- create the core lstm network.
  -- mult_in for multiple input to deep layer connections.
  self.core = LSTM.lstm(self.encoding_size, self.output_size, self.rnn_size, self.num_layers, dropout, self.mult_in)
  self.lookup_matrix = nn.Linear(1, self.encoding_size)
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  -- one for the core and one for the hidden, per layer
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
  self.output:resize(self.seq_length, batch_size, self.output_size)
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the ColorModel')
  self.clones = {self.core}
  self.lookup_matrices = {self.lookup_matrix}
  for t=2,self.seq_length do
    self.clones[t] = self.core:sharedClone(true, true)
  end
  for t=2,self.seq_length-1 do
    self.lookup_matrices[t] = self.lookup_matrix:sharedClone(true, true)
  end
  collectgarbage()
end

function layer:getModulesList()
  return {self.core, self.lookup_matrix}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_matrix:parameters()

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
  --if self.clones == nil then self:createClones() end -- create these lazily if needed
  if self.clones ~= nil then
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_matrices) do v:training() end
  end
end

function layer:evaluate()
  --if self.clones == nil then self:createClones() end -- create these lazily if needed
  if self.clones ~= nil then
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_matrices) do v:evaluate() end
  end
end

function layer:sample(features, temperature, gt)
  if self.gmms == nil then self.gmms = nn.PixelModelCriterion(1, self.num_mixtures) end
  if self.clones == nil then self:createClones() end
  local batch_size = features:size(1)
  self:_createInitState(batch_size)
  self._states = {[0] = self.init_state}
  self._inputs = {}
  local pixels = gt:clone():fill(0)
  local losses = 0
  local train_losses = 0
  -- loop through each timestep
  for t=1,self.seq_length do
    local xt
    if t == 1 then
      xt = features
    else
      xt = self.lookup_matrices[t-1]:forward(pixels[t-1])
    end
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t] }
    self._inputs[t] = {xt,table.unpack(self._states[t-1])}
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = self.clones[t]:forward(self._inputs[t])
    -- save the state
    self._states[t] = {}
    for i=1,self.num_state do table.insert(self._states[t], lsts[i]) end
    local gmms = lsts[#lsts]
    local pixel, loss, train_loss = self.gmms:sample(gmms, temperature, gt[t])
    pixels[t] = pixel
    losses = losses + loss
    train_losses = train_losses + train_loss
  end
  losses = losses / self.seq_length
  train_losses = train_losses / self.seq_length
  return pixels, losses, train_losses
end
--[[
Implements the FORWARD of the PixelModel module
input: pixel input sequence
  torch.FloatTensor of size DxNx(M+1)
  where M = opt.pixel_size and D = opt.seq_length and N = batch size
output:
  returns a DxNxG Tensor giving Mixture of Gaussian encodings
  where G is the encoding length specifying (mean, variance, covariance, end-token)
--]]
function layer:updateOutput(input)
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass
  local features = input[1]
  local pixels = input[2]

  assert(pixels:size(1) == self.seq_length)
  local batch_size = pixels:size(2)
  self:_createInitState(batch_size)

  self._states = {[0] = self.init_state}
  self._inputs = {}
  -- loop through each timestep
  for t=1,self.seq_length do
    local xt
    if t == 1 then
      xt = features
    else
      xt = self.lookup_matrices[t-1]:forward(pixels[t-1])
    end
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t] }
    self._inputs[t] = {xt,table.unpack(self._states[t-1])}
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = self.clones[t]:forward(self._inputs[t])
    -- save the state
    self._states[t] = {}
    for i=1,self.num_state do table.insert(self._states[t], lsts[i]) end
    self.output[t] = lsts[#lsts]
  end
  return self.output
end

--[[
Implements BACKWARD of the PixelModel module
input:
  input is ignored, we assume every backward call is preceded by a forward call.
  gradOutput is an DxNx(M+1) Tensor.

output:
  returns gradInput of DxNx(M+1) Tensor.
  where M = opt.pixel_size and D = opt.seq_length and N = batch size
--]]
function layer:updateGradInput(input, gradOutput)
  local pixels = input[2]
  local dfeatures

  local batch_size = gradOutput:size(2)
  -- initialize the gradient of states all to zeros.
  -- this works when init_state is all zeros
  local _dstates = {}
  for t=self.seq_length,1,-1 do

    -- concat state gradients and output vector gradients at time step t
    if _dstates[t] == nil then _dstates[t] = self.init_state end
    local douts = {}
    for k=1,#_dstates[t] do table.insert(douts, _dstates[t][k]) end
    table.insert(douts, gradOutput[t])
    -- backward LSTMs
    local dinputs = self.clones[t]:backward(self._inputs[t], douts)

    -- split the gradient to pixel and to state
    local dxt = dinputs[1] -- first element is the input pixel vector
    _dstates[t-1] = {}
    for k=2,self.num_state+1 do table.insert(_dstates[t-1], dinputs[k]) end

    -- copy to _dstates[t-1]
    if t == 1 then
      dfeatures = dxt
    else
      self.lookup_matrices[t-1]:backward(pixels[t-1], dxt)
    end
  end

  -- don't prop to gt pixels
  self.gradInput = {dfeatures, nil}
  return self.gradInput
end
