require 'nn'
require 'gmms'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'lstm'
local mvn = require 'misc.mvn'

-------------------------------------------------------------------------------
-- PIXEL Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.PixelModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.pixel_size = utils.getopt(opt, 'pixel_size') -- required
  assert(self.pixel_size == 1 or self.pixel_size == 3, 'image can only have either 1 or 3 channels')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 3)
  self.num_mixtures = utils.getopt(opt, 'num_mixtures')
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Pixel Model
  self.recurrent_stride = utils.getopt(opt, 'recurrent_stride')
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.mult_in = utils.getopt(opt, 'mult_in')
  self.num_neighbors = utils.getopt(opt, 'num_neighbors')
  if self.pixel_size == 3 then
    self.output_size = self.num_mixtures * (3+3+3+1)
  else
    self.output_size = self.num_mixtures * (1+1+0+1)
  end
  -- create the core lstm network.
  -- mult_in for multiple input to deep layer connections.
  self.core = LSTM.lstm2d(self.pixel_size*self.num_neighbors, self.output_size, self.rnn_size, self.num_layers, dropout, self.mult_in)
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
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the PixelModel')
  self.clones = {self.core}
  for t=2,self.seq_length do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end

function layer:getModulesList()
  return {self.core}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
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

  assert(input:size(1) == self.seq_length)
  local batch_size = input:size(2)
  -- output is a table, indexed by the seq index.
  self.output = torch.Tensor(self.seq_length, batch_size, self.output_size):type(input:type())

  self:_createInitState(batch_size)

  self._states = {[0] = self.init_state}
  self._inputs = {}
  -- loop through each timestep
  for t=1,self.seq_length do
    local h = math.floor((t-1) / self.recurrent_stride + 1)
    local w = (t-1) % self.recurrent_stride + 1
    local t_w = t - 1
    if w == 1 then t_w = 0 end
    local t_h = t - self.recurrent_stride
    if h == 1 then t_h = 0 end
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t] }
    self._inputs[t] = {input[t],unpack(self._states[t_w])}
    for i,v in ipairs(self._states[t_h]) do table.insert(self._inputs[t], v) end
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

  local batch_size = gradOutput:size(1)
  self.gradInput:resizeAs(input)

  -- initialize the gradient of states all to zeros.
  -- this works when init_state is all zeros
  local _dstates = {}
  for t=self.seq_length,1,-1 do
    local h = math.floor((t-1) / self.recurrent_stride + 1)
    local w = (t-1) % self.recurrent_stride + 1
    local t_w = t - 1
    if w == 1 then t_w = 0 end
    local t_h = t - self.recurrent_stride
    if h == 1 then t_h = 0 end

    -- concat state gradients and output vector gradients at time step t
    if _dstates[t] == nil then _dstates[t] = self.init_state end
    local douts = {}
    for k=1,#_dstates[t] do table.insert(douts, _dstates[t][k]) end
    table.insert(douts, gradOutput[t])
    -- backward LSTMs
    local dinputs = self.clones[t]:backward(self._inputs[t], douts)

    -- split the gradient to pixel and to state
    self.gradInput[t] = dinputs[1] -- first element is the input pixel vector
    -- copy to _dstates[t,t-1]
    if t_w > 0 then
      if _dstates[t_w] == nil then
        _dstates[t_w] = {}
        for k=2,self.num_state+1 do table.insert(_dstates[t_w], dinputs[k]) end
      else
        for k=2,self.num_state+1 do _dstates[t_w][k-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t-1, t]
    if t_h > 0 then
      if _dstates[t_h] == nil then
        _dstates[t_h] = {}
        for k=self.num_state+2,2*self.num_state+1 do table.insert(_dstates[t_h], dinputs[k]) end
      else
        -- this is unnecessary, just keep it for cleanness
        for k=self.num_state+2,2*self.num_state+1 do _dstates[t_h][k-self.num_state-1]:add(dinputs[k]) end
      end
    end
  end

  return self.gradInput
end

-------------------------------------------------------------------------------
-- PIXEL Model core for 3 Neighbor Case
-- The generation sequence will be zigzag shape in 2 dimensional space.
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.PixelModel3N', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.pixel_size = utils.getopt(opt, 'pixel_size') -- required
  assert(self.pixel_size == 1 or self.pixel_size == 3, 'image can only have either 1 or 3 channels')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 3)
  self.num_mixtures = utils.getopt(opt, 'num_mixtures')
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Pixel Model
  self.recurrent_stride = utils.getopt(opt, 'recurrent_stride')
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.mult_in = utils.getopt(opt, 'mult_in')
  self.num_neighbors = utils.getopt(opt, 'num_neighbors')
  self.border_init = utils.getopt(opt, 'border_init')
  if self.pixel_size == 3 then
    self.output_size = self.num_mixtures * (3+3+3+1)
  else
    self.output_size = self.num_mixtures * (1+1+0+1)
  end
  -- create the core lstm network.
  -- mult_in for multiple input to deep layer connections.
  self.core = LSTM.lstm3d(self.pixel_size*self.num_neighbors, self.output_size, self.rnn_size, self.num_layers, dropout, self.mult_in)
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
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the PixelModel')
  self.clones = {self.core}
  for t=2,self.seq_length do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end

function layer:getModulesList()
  return {self.core}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
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

  assert(input:size(1) == self.seq_length)
  local batch_size = input:size(2)
  -- output is a table, indexed by the seq index.
  self.output = torch.Tensor(self.seq_length, batch_size, self.output_size):type(input:type())

  self:_createInitState(batch_size)

  self._states = {[0] = self.init_state}
  self._inputs = {}
  -- loop through each timestep
  for t=1,self.seq_length do
    local h = math.floor((t-1) / self.recurrent_stride + 1)
    local w = (t-1) % self.recurrent_stride + 1 -- not coordinate, count left from 1 row, and right from the second row
    local pu = t + 1 - 2 * w -- up
    if h == 1 then pu = 0 end
    local pl = t - 1 -- left
    if w == 1 or h % 2 == 0 then pl = 0 end
    local pr = t - 1 -- right
    if w == 1 or h % 2 == 1 then pr = 0 end
    local pi = t
    if h % 2 == 0 then pi = pu + self.recurrent_stride end
    -- prepare the input border
    if self.border_init == 0 then
      if pl == 0 then input[{pi, {}, {1, self.pixel_size}}] = 0 end
      if pr == 0 then input[{pi, {}, {2*self.pixel_size+1, 3*self.pixel_size}}] = 0 end
    else
      if pl == 0 then input[{pi, {}, {1, self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
      if pr == 0 then input[{pi, {}, {2*self.pixel_size+1, 3*self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
    end
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t], states[t, t+1]}
    self._inputs[t] = {input[pi],unpack(self._states[pl])}
    for i,v in ipairs(self._states[pu]) do table.insert(self._inputs[t], v) end
    for i,v in ipairs(self._states[pr]) do table.insert(self._inputs[t], v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = self.clones[t]:forward(self._inputs[t])
    -- save the state
    self._states[t] = {}
    for i=1,self.num_state do table.insert(self._states[t], lsts[i]) end
    self.output[pi] = lsts[#lsts]
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

  local batch_size = gradOutput:size(1)
  self.gradInput:resizeAs(input)

  -- initialize the gradient of states all to zeros.
  -- this works when init_state is all zeros
  local _dstates = {}
  for t=self.seq_length,1,-1 do
    local h = math.floor((t-1) / self.recurrent_stride + 1)
    local w = (t-1) % self.recurrent_stride + 1 -- not coordinate, count left from 1 row, and right from the second row
    local pu = t + 1 - 2 * w -- up
    if h == 1 then pu = 0 end
    local pl = t - 1 -- left
    if w == 1 or h % 2 == 0 then pl = 0 end
    local pr = t - 1 -- right
    if w == 1 or h % 2 == 1 then pr = 0 end
    local pi = t
    if h % 2 == 0 then pi = pu + self.recurrent_stride end
    -- concat state gradients and output vector gradients at time step t
    if _dstates[t] == nil then _dstates[t] = self.init_state end
    local douts = {}
    for k=1,#_dstates[t] do table.insert(douts, _dstates[t][k]) end
    table.insert(douts, gradOutput[pi])
    -- backward LSTMs
    local dinputs = self.clones[t]:backward(self._inputs[t], douts)

    -- split the gradient to pixel and to state
    self.gradInput[pi] = dinputs[1] -- first element is the input pixel vector
    -- copy to _dstates[t,t-1]
    if pl > 0 then
      if _dstates[pl] == nil then
        _dstates[pl] = {}
        for k=2,self.num_state+1 do table.insert(_dstates[pl], dinputs[k]) end
      else
        for k=2,self.num_state+1 do _dstates[pl][k-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t-1, t]
    if pu > 0 then
      if _dstates[pu] == nil then
        _dstates[pu] = {}
        for k=self.num_state+2,2*self.num_state+1 do table.insert(_dstates[pu], dinputs[k]) end
      else
        -- this is unnecessary, just keep it for cleanness
        for k=self.num_state+2,2*self.num_state+1 do _dstates[pu][k-self.num_state-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t, t+1]
    if pr > 0 then
      if _dstates[pr] == nil then
        _dstates[pr] = {}
        for k=2*self.num_state+2,3*self.num_state+1 do table.insert(_dstates[pr], dinputs[k]) end
      else
        for k=2*self.num_state+2,3*self.num_state+1 do _dstates[pr][k-2*self.num_state-1]:add(dinputs[k]) end
      end
    end
  end

  return self.gradInput
end

-------------------------------------------------------------------------------
-- PIXEL Model core for 4 Neighbor Case
-- The sequence genrates each pixel twice, forward and backward. Each sequence
-- again is a zigzag shape.
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.PixelModel4N', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.pixel_size = utils.getopt(opt, 'pixel_size') -- required
  assert(self.pixel_size == 1 or self.pixel_size == 3, 'image can only have either 1 or 3 channels')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 3)
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Pixel Model
  self.recurrent_stride = utils.getopt(opt, 'recurrent_stride')
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.mult_in = utils.getopt(opt, 'mult_in')
  self.num_neighbors = utils.getopt(opt, 'num_neighbors')
  self.border_init = utils.getopt(opt, 'border_init')
  self.output_size = self.pixel_size -- for euclidean loss
  -- create the core lstm network.
  -- mult_in for multiple input to deep layer connections.
  self.core = LSTM.lstm4d(self.pixel_size*4*2, self.output_size, self.rnn_size, self.num_layers, dropout, self.mult_in)
  self:_createInitState(1) -- will be lazily resized later during forward passes
  self:_buildIndex()
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
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the PixelModel')
  self.clones = {self.core}
  for t=2,2*self.seq_length do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end

function layer:getModulesList()
  return {self.core}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
end

function layer:_buildIndex()
  self._Findex = torch.Tensor(self.seq_length, 5) -- left, up, right, down, target
  self._Bindex = torch.Tensor(self.seq_length, 5) -- left, up, right, down, target
  local sl = self.seq_length
  for t=1,sl do
    local h = math.floor((t-1) / self.recurrent_stride + 1)
    local w = (t-1) % self.recurrent_stride + 1 -- not coordinate, count left from 1 row, and right from the second row
    local pu = t + 1 - 2 * w -- up
    if h == 1 then pu = 0 end
    local pl = t - 1 -- left
    if w == 1 or h % 2 == 0 then pl = 0 end
    local pr = t - 1 -- right
    if w == 1 or h % 2 == 1 then pr = 0 end
    local pd = 0 -- down, first path will not have available pixel downwards. And it is initialized properly in data loader.
    local pi = t
    if h % 2 == 0 then pi = pu + self.recurrent_stride end
    self._Findex[{t, 1}] = pl
    self._Findex[{t, 2}] = pu
    self._Findex[{t, 3}] = pr
    self._Findex[{t, 4}] = pd
    self._Findex[{t, 5}] = pi
  end
  for t=sl,1,-1 do
    local h = math.floor((t-1) / self.recurrent_stride + 1)
    local w = self.recurrent_stride - (t-1) % self.recurrent_stride  -- not coordinate, count right from 1 row, and left from the second row
    local pd = t + 2 * w - 1 -- down
    local pu = pd - 2 * self.recurrent_stride -- upward pixel always in the forwad table
    pd = pd + sl -- downward pixel always in the backward table
    if h == self.recurrent_stride or pd > 2 * sl then pd = 0 end
    if h == 1 then pu = 0 end
    local pl -- left
    if h % 2 == 0 then pl = t + 1 + sl if w == 1 or pl > 2 * sl then pl = 0 end end
    if h % 2 == 1 then pl = t - 1 if w == self.recurrent_stride then pl = 0 end end
    local pr -- right
    if h % 2 == 1 then pr = t + 1 + sl if w == 1 or pr > 2 * sl then pr = 0 end end
    if h % 2 == 0 then pr = t - 1 if w == self.recurrent_stride then pr = 0 end end
    local pi = t + sl
    if h % 2 == 0 then pi = pd - self.recurrent_stride end
    self._Bindex[{t, 1}] = pl
    self._Bindex[{t, 2}] = pu
    self._Bindex[{t, 3}] = pr
    self._Bindex[{t, 4}] = pd
    self._Bindex[{t, 5}] = pi
  end
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

  local sl = self.seq_length
  assert(input:size(1) == sl)
  local batch_size = input:size(2)
  -- output is a table, indexed by the seq index.
  input = torch.repeatTensor(input, 2, 1, 1)
  self.output = torch.Tensor(2*sl, batch_size, self.output_size):type(input:type())

  self:_createInitState(batch_size)

  self._states = {[0] = self.init_state}
  self._inputs = {}
  -- forward loop through the image pixels
  for t=1,sl do
    local pl = self._Findex[{t, 1}]
    local pu = self._Findex[{t, 2}]
    local pr = self._Findex[{t, 3}]
    local pd = self._Findex[{t, 4}]
    local pi = self._Findex[{t, 5}]
    -- prepare the input border
    if self.border_init == 0 then
      if pl == 0 then input[{pi, {}, {1, self.pixel_size}}] = 0 end
      if pu == 0 then input[{pi, {}, {1*self.pixel_size+1, 2*self.pixel_size}}] = 0 end
      if pr == 0 then input[{pi, {}, {2*self.pixel_size+1, 3*self.pixel_size}}] = 0 end
      if pd == 0 then input[{pi, {}, {3*self.pixel_size+1, 4*self.pixel_size}}] = 0 end
    else
      if pl == 0 then input[{pi, {}, {1, self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
      if pu == 0 then input[{pi, {}, {1*self.pixel_size+1, 2*self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
      if pr == 0 then input[{pi, {}, {2*self.pixel_size+1, 3*self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
      if pd == 0 then input[{pi, {}, {3*self.pixel_size+1, 4*self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
    end
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t], states[t, t+1]}
    self._inputs[t] = {input[pi],unpack(self._states[pl])}
    for i,v in ipairs(self._states[pu]) do table.insert(self._inputs[t], v) end
    for i,v in ipairs(self._states[pr]) do table.insert(self._inputs[t], v) end
    for i,v in ipairs(self._states[pd]) do table.insert(self._inputs[t], v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = self.clones[t]:forward(self._inputs[t])
    -- save the state
    self._states[t] = {}
    for i=1,self.num_state do table.insert(self._states[t], lsts[i]) end
    self.output[pi] = lsts[#lsts]
  end
  -- backward loop through the image pixels
  -- states in all four directions will be available
  for t=sl,1,-1 do
    local pl = self._Bindex[{t, 1}]
    local pu = self._Bindex[{t, 2}]
    local pr = self._Bindex[{t, 3}]
    local pd = self._Bindex[{t, 4}]
    local pi = self._Bindex[{t, 5}]
    -- prepare the input border
    if self.border_init == 0 then
      if pl == 0 then input[{pi, {}, {1, self.pixel_size}}] = 0 end
      if pu == 0 then input[{pi, {}, {1*self.pixel_size+1, 2*self.pixel_size}}] = 0 end
      if pr == 0 then input[{pi, {}, {2*self.pixel_size+1, 3*self.pixel_size}}] = 0 end
      if pd == 0 then input[{pi, {}, {3*self.pixel_size+1, 4*self.pixel_size}}] = 0 end
    else
      if pl == 0 then input[{pi, {}, {1, self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
      if pu == 0 then input[{pi, {}, {1*self.pixel_size+1, 2*self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
      if pr == 0 then input[{pi, {}, {2*self.pixel_size+1, 3*self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
      if pd == 0 then input[{pi, {}, {3*self.pixel_size+1, 4*self.pixel_size}}] = torch.rand(batch_size, self.pixel_size) end
    end
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t], states[t, t+1]}
    self._inputs[t+sl] = {input[pi],unpack(self._states[pl])}
    for i,v in ipairs(self._states[pu]) do table.insert(self._inputs[t+sl], v) end
    for i,v in ipairs(self._states[pr]) do table.insert(self._inputs[t+sl], v) end
    for i,v in ipairs(self._states[pd]) do table.insert(self._inputs[t+sl], v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = self.clones[t+sl]:forward(self._inputs[t+sl])
    -- save the state
    self._states[t+sl] = {}
    for i=1,self.num_state do table.insert(self._states[t+sl], lsts[i]) end
    self.output[pi] = lsts[#lsts]
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

  local sl = self.seq_length
  local batch_size = gradOutput:size(1)
  self.gradInput:resizeAs(input)
  local dgradInput = torch.repeatTensor(self.gradInput, 2, 1, 1)

  -- initialize the gradient of states all to zeros.
  -- this works when init_state is all zeros
  local _dstates = {}
  -- the backward table
  for t=1,sl do
    local pl = self._Bindex[{t, 1}]
    local pu = self._Bindex[{t, 2}]
    local pr = self._Bindex[{t, 3}]
    local pd = self._Bindex[{t, 4}]
    local pi = self._Bindex[{t, 5}]
    -- concat state gradients and output vector gradients at time step t
    if _dstates[t+sl] == nil then _dstates[t+sl] = self.init_state end
    local douts = {}
    for k=1,#_dstates[t+sl] do table.insert(douts, _dstates[t+sl][k]) end
    table.insert(douts, gradOutput[pi])
    -- backward LSTMs
    local dinputs = self.clones[t+sl]:backward(self._inputs[t+sl], douts)

    -- split the gradient to pixel and to state
    dgradInput[pi] = dinputs[1] -- first element is the input pixel vector
    -- copy to _dstates[t,t-1]
    if pl > 0 then
      if _dstates[pl] == nil then
        _dstates[pl] = {}
        for k=2,self.num_state+1 do table.insert(_dstates[pl], dinputs[k]) end
      else
        for k=2,self.num_state+1 do _dstates[pl][k-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t-1, t]
    if pu > 0 then
      if _dstates[pu] == nil then
        _dstates[pu] = {}
        for k=self.num_state+2,2*self.num_state+1 do table.insert(_dstates[pu], dinputs[k]) end
      else
        -- this is unnecessary, just keep it for cleanness
        for k=self.num_state+2,2*self.num_state+1 do _dstates[pu][k-self.num_state-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t, t+1]
    if pr > 0 then
      if _dstates[pr] == nil then
        _dstates[pr] = {}
        for k=2*self.num_state+2,3*self.num_state+1 do table.insert(_dstates[pr], dinputs[k]) end
      else
        for k=2*self.num_state+2,3*self.num_state+1 do _dstates[pr][k-2*self.num_state-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t+1, t]
    if pd > 0 then
      if _dstates[pd] == nil then
        _dstates[pd] = {}
        for k=3*self.num_state+2,4*self.num_state+1 do table.insert(_dstates[pd], dinputs[k]) end
      else
        for k=3*self.num_state+2,4*self.num_state+1 do _dstates[pd][k-3*self.num_state-1]:add(dinputs[k]) end
      end
    end
  end
  -- the forward table
  for t=sl,1,-1 do
    local pl = self._Findex[{t, 1}]
    local pu = self._Findex[{t, 2}]
    local pr = self._Findex[{t, 3}]
    local pd = self._Findex[{t, 4}]
    local pi = self._Findex[{t, 5}]
    -- concat state gradients and output vector gradients at time step t
    if _dstates[t] == nil then _dstates[t] = self.init_state end
    local douts = {}
    for k=1,#_dstates[t] do table.insert(douts, _dstates[t][k]) end
    table.insert(douts, gradOutput[pi])
    -- backward LSTMs
    local dinputs = self.clones[t]:backward(self._inputs[t], douts)

    -- split the gradient to pixel and to state
    dgradInput[pi] = dinputs[1] -- first element is the input pixel vector
    -- copy to _dstates[t,t-1]
    if pl > 0 then
      if _dstates[pl] == nil then
        _dstates[pl] = {}
        for k=2,self.num_state+1 do table.insert(_dstates[pl], dinputs[k]) end
      else
        for k=2,self.num_state+1 do _dstates[pl][k-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t-1, t]
    if pu > 0 then
      if _dstates[pu] == nil then
        _dstates[pu] = {}
        for k=self.num_state+2,2*self.num_state+1 do table.insert(_dstates[pu], dinputs[k]) end
      else
        -- this is unnecessary, just keep it for cleanness
        for k=self.num_state+2,2*self.num_state+1 do _dstates[pu][k-self.num_state-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t, t+1]
    if pr > 0 then
      if _dstates[pr] == nil then
        _dstates[pr] = {}
        for k=2*self.num_state+2,3*self.num_state+1 do table.insert(_dstates[pr], dinputs[k]) end
      else
        for k=2*self.num_state+2,3*self.num_state+1 do _dstates[pr][k-2*self.num_state-1]:add(dinputs[k]) end
      end
    end
    -- copy to _dstates[t+1, t]
    -- will never have downward pixel in this case.
  end
  self.gradInput = torch.add(dgradInput:narrow(1,1,sl), dgradInput:narrow(1,sl+1,sl))
  return self.gradInput
end
