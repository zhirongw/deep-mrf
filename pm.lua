require 'nn'
require 'gmm_decoder'
require 'distributions'
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
  if self.pixel_size == 3 then
    self.output_size = self.num_mixtures * (3+3+3+1)
  else
    self.output_size = self.num_mixtures * (1+1+0+1)
  end
  -- create the core lstm network.
  -- mult_in for multiple input to deep layer connections.
  self.core = LSTM.lstm2d(self.pixel_size, self.output_size, self.rnn_size, self.num_layers, dropout, self.mult_in)
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
  return {self.core, self.gmm}
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
  --for k,v in pairs(self.gmms) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  -- for k,v in pairs(self.gmms) do v:evaluate() end
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
  self.output = torch.Tensor(batch_size, self.output_size):type(input:type())

  self:_createInitState(batch_size)

  self._states = {[0] = self.init_state}
  self._inputs = {}
  -- self._gmm_encodings = {}
  -- loop through each timestep
  for t=1,self.seq_length do
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t] }
    self._inputs[t] = {input[t],unpack(self._states[t-1])}
    local t_stride = t - self.recurrent_stride
    if t_stride < 1 then t_stride = 0 end
    -- insert the states[t-1,t]
    for i,v in ipairs(self._states[t_stride]) do table.insert(self._inputs[t], v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = self.clones[t]:forward(self._inputs[t])
    -- save the state
    self._states[t] = {}
    for i=1,self.num_state do table.insert(self._states[t], lsts[i]) end
    -- only predicting the very last pixel
    if t == self.seq_length then
      self.output = lsts[#lsts]
    end
    -- self.output[t] = lsts[#lsts]
    -- inputs to Mixture of Gaussian encodings
    -- table.insert(self._gmm_encodings, lsts[#lsts])
    -- table.insert(self.output, self.gmms[t]:forward(lsts[#lsts])) -- last element is the output vector
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
  local _dstates = {[self.seq_length] = self.init_state}
  local _doutputstate = torch.zeros(batch_size, self.output_size):type(input:type())
  for t=self.seq_length,1,-1 do
    --local dgmm_encodings = self.gmms[t]:backward(self._gmm_encodings[t], gradOutput[t])
    -- concat state gradients and output vector gradients at time step t
    local douts = {}
    for k=1,#_dstates[t] do table.insert(douts, _dstates[t][k]) end
    if t == self.seq_length then
      table.insert(douts, gradOutput)
    else
      table.insert(douts, _doutputstate)
    end
    -- backward LSTMs
    local dinputs = self.clones[t]:backward(self._inputs[t], douts)

    -- split the gradient to pixel and to state
    self.gradInput[t] = dinputs[1] -- first element is the input pixel vector
    -- copy to _dstates[t,t-1]
    if _dstates[t-1] == nil then
      _dstates[t-1] = {}
      for k=2,self.num_state+1 do table.insert(_dstates[t-1], dinputs[k]) end
    else
      for k=2,self.num_state+1 do _dstates[t-1][k-1]:add(dinputs[k]) end
    end
    -- copy to _dstates[t-1, t]
    local t_stride = t - self.recurrent_stride
    if t_stride > 0 then
      if _dstates[t_stride] == nil then
        _dstates[t_stride] = {}
        for k=self.num_state+2,2*self.num_state+1 do table.insert(_dstates[t_stride], dinputs[k]) end
      else
        -- this is unnecessary, just keep it for cleanness
        for k=self.num_state+2,2*self.num_state+1 do _dstates[t_stride][k-self.num_state-1]:add(dinputs[k]) end
      end
    end
  end

  return self.gradInput
end

--[[
runs the model forward in sampling mode to generate an image.
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxNx(M+1) FloatTensor
where D is sequence length and N is batch (so columns are sequences),
and M is the number of image channels (3 for most RGB images)
--]]
function layer:sample(opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  local batch_size = utils.getopt(opt, 'batch_size', 1)
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- indirection for beam search

  self:_createInitState(batch_size)
  local state = self.init_state

  -- we will write output predictions into tensor seq
  local seq = torch.FloatTensor(self.seq_length, batch_size, self.pixel_size+1):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  -- feed in the random pixels
  local xt = torch.Tensor(batch_size, self.pixel_size+1):uniform(0,1)
  xt:select(2,self.pixel_size+1):fill(0)
  for t=1,self.seq_length do
    local sampleLogprobs
    -- sample from the distribution of previous predictions
    local inputs = {xt,unpack(state)}
    local out = self.core:forward(inputs)
    local g_encodings = out[self.num_state+1] -- last element is the output vector
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end

    local g_mean = g_encodings:narrow(1,1,self.num_mixture*self.pixel_size)
    local g_var = g_encodings:narrow(1,self.num_mixture*self.pixel_size+1,self.num_mixture*self.pixel_size)
    g_var = torch.exp(g_var_)
    local g_cov
    if pixel_size == 3 then g_cov = g_encodings:narrow(1,2*self.num_mixture*self.pixel_size+1,self.num_mixture*self.pixel_size) end
    if pixel_size == 3 then g_cov = torch.tanh(g_cov) end
    local g_w = g_encodings:narrow(1,3*self.num_mixture*self.pixel_size+1,self.num_mixture)
    local g_w = nn.SoftMax()(g_w)

    local prob_prev
    if temperature == 1.0 then
      prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
    else
      -- scale logprobs by temperature
      prob_prev = torch.exp(torch.div(logprobs, temperature))
    end
    it = torch.multinomial(prob_prev, 1)
    sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
    it = it:view(-1):long() -- and flatten indices for downstream processing
    t = self.lookup_table:forward(it)

    if t > 1 then
      seq[t-1] = xt -- record the samples
      seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
Implements beam search.
]]--
function layer:sample_beam(imgs, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size, feat_dim = imgs:size(1), imgs:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do

    -- create initial states for all beams
    self:_createInitState(beam_size)
    local state = self.init_state

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    for t=1,self.seq_length+2 do

      local xt, it, sampleLogprobs
      local new_state
      if t == 1 then
        -- feed in the images
        local imgk = imgs[{ {k,k} }]:expand(beam_size, feat_dim) -- k'th image feature expanded out
        xt = imgk
      elseif t == 2 then
        -- feed in the start tokens
        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        xt = self.lookup_table:forward(it)
      else
        --[[
          perform a beam merge. that is,
          for every previous beam we now many new possibilities to branch out
          we need to resort our beams to maintain the loop invariant of keeping
          the top beam_size most likely sequences.
        ]]--
        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 3 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 3 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-3}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-3}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 3 then
            beam_seq[{ {1,t-3}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-3}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-2, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-2, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+2 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(),
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end

        -- encode as vectors
        it = beam_seq[t-2]
        xt = self.lookup_table:forward(it)
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

      local inputs = {xt,unpack(state)}
      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end



-------------------------------------------------------------------------------
-- Pixel Model Mixture of Gaussian Density Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.PixelModelCriterion', 'nn.Criterion')
function crit:__init(pixel_size, num_mixtures)
  parent.__init(self)
  self.pixel_size = pixel_size
  self.num_mixtures = num_mixtures
    if self.pixel_size == 3 then
    self.output_size = self.num_mixtures * (3+3+3+1)
  else
    self.output_size = self.num_mixtures * (1+1+0+1)
  end
  if pixel_size == 3 then self.var_mm = nn.MM() end
  self.w_softmax = nn.SoftMax()
  self.var_exp = nn.Exp()
end

--[[
-- this is an optimized version of the gmm loss, though looks ugly though
inputs:
  input is a Tensor of size DxNx(G), encodings of the gmms
  target is a Tensor of size DxNx(M+1).
  where, D is the sequence length, N is the batch size, M is the pixel channels,
Criterion:
  Mixture of Gaussian, Log probability.
The way we infer the target
in this criterion is as follows:
- at first time step the output is ignored (loss = 0). It's the image tick
- the label sequence "seq" is shifted by one to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, target)
  local N_ = input:size(1)
  local N,Mp1= target:size(1), target:size(2)
  local ps = Mp1 -- pixel size
  assert(N == N_, 'input Tensor should have the same batch size as the target')
  assert(ps == self.pixel_size, 'input dimensions of pixel do not match')
  local nm = self.num_mixtures

  -- decode the gmms first
  -- mean undertake no changes
  local g_mean_input = input:narrow(2,1,nm*ps):clone()
  local g_mean = g_mean_input:view(N, nm, ps)
  -- we use 6 numbers to denote the cholesky depositions
  local g_var_input = input:narrow(2, nm*ps+1, nm*ps):clone()
  g_var_input = g_var_input:view(-1, ps)
  local g_var = self.var_exp:forward(g_var_input)

  local g_cov_input
  local g_clk
  local g_clk_T
  local g_sigma
  p = 2
  if ps == 3 then
    g_cov_input = input:narrow(2, p*nm*ps+1, nm*ps):clone()
    g_cov_input = g_cov_input:view(-1, 3)
    p = p + 1
    g_clk = torch.Tensor(N*nm, 3, 3):fill(0):type(g_var_input:type())
    g_clk_T = torch.Tensor(N*nm, 3, 3):fill(0):type(g_var_input:type())
    g_clk[{{}, 1, 1}] = g_var[{{}, 1}]
    g_clk[{{}, 2, 2}] = g_var[{{}, 2}]
    g_clk[{{}, 3, 3}] = g_var[{{}, 3}]
    g_clk[{{}, 2, 1}] = g_cov_input[{{}, 1}]
    g_clk[{{}, 3, 1}] = g_cov_input[{{}, 2}]
    g_clk[{{}, 3, 2}] = g_cov_input[{{}, 3}]
    g_clk_T[{{}, 1, 1}] = g_var[{{}, 1}]
    g_clk_T[{{}, 2, 2}] = g_var[{{}, 2}]
    g_clk_T[{{}, 3, 3}] = g_var[{{}, 3}]
    g_clk_T[{{}, 1, 2}] = g_cov_input[{{}, 1}]
    g_clk_T[{{}, 1, 3}] = g_cov_input[{{}, 2}]
    g_clk_T[{{}, 2, 3}] = g_cov_input[{{}, 3}]
    g_sigma = self.var_mm:forward({g_clk, g_clk_T})
    g_clk = g_clk:view(N, nm, ps, ps)
    g_clk_T = g_clk_T:view(N, nm, ps, ps)
  else
    g_sigma = torch.cmul(g_var_input, g_var_input)
  end
  g_sigma = g_sigma:view(N, nm, ps, ps)
  -- weights coeffs is taken care of at final loss, for computation efficiency and stability
  local g_w_input = input:narrow(2,p*nm*ps+1, nm):clone()
  local g_w = self.w_softmax:forward(g_w_input:view(-1, nm))
  g_w = g_w:view(N, nm)
  -- border is a single scalar indicator
  -- local borders = input[{{}, {}, -1}]:clone()

  -- do the loss the gradients
  local loss1 = 0 -- loss of pixels, Mixture of Gaussians
  local grad_g_mean = torch.Tensor(g_mean:size()):type(g_mean:type())
  local grad_g_sigma = torch.Tensor(g_sigma:size()):type(g_sigma:type())
  local grad_g_w = torch.Tensor(g_w:size()):type(g_w:type())

  local target_x, g_mean_x, g_clk_x, g_w_x, g_sigma_x
  -- some part of the computation can only happen on CPU, cache those variables
  if input:type() == 'torch.CudaTensor' then
    g_sigma_x = g_sigma:float()
    g_clk_x = g_clk:float()
    g_w_x = g_w:float()
    g_mean_x = g_mean:float()
    target_x = target:float()
  else
    g_sigma_x = g_sigma
    g_clk_x = g_clk
    g_w_x = g_w
    g_mean_x = g_mean
    target_x = target
  end
  for b=1,N do -- iterate over batches
    -- Start of CPU. this is the only place that should take place in CPU
    local target_pixel_ = target_x[b]:narrow(1,1,ps)
    -- can we vectorize this? Now constrains by the MVN.PDF
    local g_rpb_ = torch.zeros(nm)
    local g_sigma_inv_ = torch.Tensor(g_sigma[b]:size())
    for g=1,nm do -- iterate over mixtures
      g_rpb_[g] = mvn.pdf(target_pixel_, g_mean_x[{b,g,{}}], g_clk_x[{b,g,{},{}}]) * g_w_x[{b,g}]
      g_sigma_inv_[g] = torch.inverse(g_sigma_x[{b,g,{},{}}])
    end
    local pdf = torch.sum(g_rpb_)
    loss1 = loss1 - torch.log(pdf)
    if input:type() == 'torch.CudaTensor' then
      g_rpb_ = g_rpb_:cuda()
      g_sigma_inv_ = g_sigma_inv_:cuda()
    end
    -- End of CPU

    -- normalize the responsibilities for backprop
    g_rpb_:div(pdf)

    -- gradient of weight is tricky, making it efficient together with softmax
    grad_g_w[{b, {}}] = - g_rpb_
    -- gradient of mean
    local g_mean_diff_ = torch.add(torch.repeatTensor(target[b]:narrow(1,1,ps),nm,1), -1, g_mean[b])
    local mean_left_ = g_mean_diff_:view(nm, ps, 1)
    local mean_right_ = g_mean_diff_:view(nm, 1, ps)
    g_rpb_ = torch.repeatTensor(g_rpb_:view(-1,1), 1, ps)
    grad_g_mean[{b, {}, {}}] = - torch.cmul(g_rpb_, torch.bmm(g_sigma_inv_, mean_left_))
    -- gradient of covariance matrix
    g_rpb_ = torch.repeatTensor(g_rpb_:view(-1,ps,1), 1, 1, ps)
    local g_temp_ = torch.bmm(torch.bmm(g_sigma_inv_, torch.bmm(mean_left_, mean_right_)), g_sigma_inv_) - g_sigma_inv_
    grad_g_sigma[{b, {}, {}, {}}] = - torch.cmul(g_rpb_, g_temp_) * 0.5
  end

  -- back prop encodings
  -- mean undertake no changes
  grad_g_mean = grad_g_mean:view(N, -1)
  -- gradient of weight is tricky, making it efficient together with softmax
  grad_g_w:add(g_w)
  grad_g_w = grad_g_w:view(N, -1)
  -- gradient of the var, and cov
  local grad_g_var
  local grad_g_cov
  g_clk = g_clk:view(-1, ps, ps)
  g_clk_T = g_clk_T:view(-1, ps, ps)
  if ps == 3 then
    grad_g_sigma = grad_g_sigma:view(-1, 3, 3)
    local grad_g_clk = self.var_mm:backward({g_clk, g_clk_T}, grad_g_sigma)
    grad_g_clk = grad_g_clk[1]:mul(2)
    grad_g_var = torch.Tensor(N*nm, ps):type(grad_g_clk:type())
    grad_g_cov = torch.Tensor(N*nm, ps):type(grad_g_clk:type())
    grad_g_var[{{}, 1}] = grad_g_clk[{{},1,1}]
    grad_g_var[{{}, 2}] = grad_g_clk[{{},2,2}]
    grad_g_var[{{}, 3}] = grad_g_clk[{{},3,3}]
    grad_g_var = self.var_exp:backward(g_var_input, grad_g_var)
    grad_g_var = grad_g_var:view(N, -1)
    grad_g_cov[{{}, 1}] = grad_g_clk[{{},2,1}]
    grad_g_cov[{{}, 2}] = grad_g_clk[{{},3,1}]
    grad_g_cov[{{}, 3}] = grad_g_clk[{{},3,2}]
    grad_g_cov = grad_g_cov:view(N, -1)
  else
    grad_g_var = torch.cmul(g_var_input, grad_g_sigma):mul(2)
    grad_g_var = self.var_exp:backward(g_var_input, grad_g_var)
    grad_g_var = grad_g_var:view(N, -1)
  end

  grad_g_mean:div(N)
  grad_g_var:div(N)
  if ps == 3 then grad_g_cov:div(N) end
  grad_g_w:div(N)
  -- grad_borders:div(D*N)

  -- concat to gradInput
  if self.pixel_size == 3 then
    -- torch does not allow us to concat more than 2 tensors for FloatTensors
    self.gradInput = torch.cat(torch.cat(grad_g_mean, grad_g_var), torch.cat(grad_g_cov, grad_g_w))
  else
    self.gradInput = torch.cat(torch.cat(grad_g_mean, grad_g_var), grad_g_w)
  end
  -- return the loss
  self.output = (loss1) / (N)
  return self.output
end

function crit:updateGradInput(input, target)
  -- just return it
  return self.gradInput
end
