require 'nn'
require 'gmm_decoder'
require 'distributions'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'lstm'

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
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.recurrent_stride = utils.getopt(opt, 'recurrent_stride')
  if self.pixel_size == 3 then
    self.output_size = self.num_mixtures * (3+3+3+1) + 1
  else
    self.output_size = self.num_mixtures * (1+1+0+1) + 1
  end
  -- create the core lstm network.
  -- note +1 for addition end tokens, true for multiple input to deep layer connections.
  self.core = LSTM.lstm2d(self.pixel_size+1, self.output_size, self.rnn_size, self.num_layers, dropout, true)
  -- decoding the output to gaussian mixture parameters
  self.gmm = nn.GMMDecoder(self.pixel_size, self.num_mixtures)
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
  self.gmms = {self.gmm}
  for t=2,self.seq_length do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.gmms[t] = self.gmm:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end

function layer:getModulesList()
  return {self.core, self.gmm}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.gmm:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  assert(p2 == nil, 'GMM decoder should have no params')
  for k,v in pairs(p2) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  assert(g2 == nil, 'GMM decoder should have no params')
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.gmms) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.gmms) do v:evaluate() end
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
  self.output = {}

  self:_createInitState(batch_size)

  self._states = {[0] = self.init_state}
  self._inputs = {}
  self._gmm_encodings = {}
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
    -- inputs to Mixture of Gaussian encodings
    table.insert(self._gmm_encodings, lsts[#lsts])
    table.insert(self.output, self.gmms[t]:forward(lsts[#lsts])) -- last element is the output vector
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

  assert(#gradOutput == self.seq_length)
  local batch_size = gradOutput[1][1]:size(1)
  self.gradInput:resizeAs(input)

  -- initialize the gradient of states all to zeros.
  -- this works when init_state is all zeros
  local _dstates = {[self.seq_length] = self.init_state}
  for t=self.seq_length,1,-1 do
    local dgmm_encodings = self.gmms[t]:backward(self._gmm_encodings[t], gradOutput[t])
    -- concat state gradients and output vector gradients at time step t
    local douts = {}
    for k=1,#_dstates[t] do table.insert(douts, _dstates[t][k]) end
    table.insert(douts, dgmm_encodings)
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
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- indirection for beam search

  local batch_size = imgs:size(1)
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
function crit:__init(num_mixtures)
  parent.__init(self)
  self.num_mixtures = num_mixtures
end

--[[
inputs:
  input is a table of FloatTensor of size D. Each element is also a table of size 4, specifyin the params of the gaussians
  The input contains  (mean, cov matrix, weights) + end_token
  target is a FloatTensor of size DxNx(M+1).
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
  local D_ = #input
  local N_ = input[1][1]:size(1)
  local D,N,Mp1= target:size(1), target:size(2), target:size(3)
  local pixel_size = Mp1 - 1
  assert(D == D_, 'input Tensor should have the same sequence length as the target')
  assert(N == N_, 'input Tensor should have the same batch size as the target')

  local loss1 = 0 -- loss of pixels, Mixture of Gaussians
  local loss2 = 0 -- loss of boundary, Sigmoid Loss
  for t=1,D do --iterate over sequence time
    local target_ = target[t]
    local gmm_params_ = input[t]
    local gmm_mean_ = gmm_params_[1]
    local gmm_cov_ = gmm_params_[2]
    local gmm_weight_ = gmm_params_[3]
    local border_ = gmm_params_[4]
    for b=1,N do -- iterate over batches
      local target_pixel_ = target_[b]:narrow(1,1,pixel_size)
      local target_border_ = target_[{b,Mp1}]
      -- can we vectorize this? Now constrains by the MVN.PDF
      local mvnloss = 0
      for g=1,self.num_mixtures do -- iterate over mixtures
        local pdf = distributions.mvn.pdf(target_pixel_, gmm_mean_[{b,g,{}}], gmm_cov_[{b,g,{},{}}]) * gmm_weight_[{b,g}]
        mvnloss = mvnloss + pdf
      end
      loss1 = loss1 - torch.log(mvnloss)
      if target_border_ >= 0.5 then
        loss2 = loss2 - torch.log((1/(1+torch.exp(-border_[b]))))
      else
        loss2 = loss2 - torch.log(1-1/(1+torch.exp(-border_[b])))
      end
    end
  end

  self.output = (loss1 + loss2) / (D * N)
  return self.output
end

function crit:updateGradInput(input, target)
  self.gradInput = {}
  local D,N,Mp1= target:size(1), target:size(2), target:size(3)
  local pixel_size = Mp1 - 1

  for t=1,D do -- iterate over sequence time
    local target_ = target[t]
    local gmm_params_ = input[t]
    local gmm_mean_ = gmm_params_[1]
    local gmm_cov_ = gmm_params_[2]
    local gmm_weight_ = gmm_params_[3]
    local border_ = gmm_params_[4]
    -- pre-allocate spaces
    local grad_ = {}
    local grad_gmm_mean_ = torch.Tensor(gmm_mean_:size()):type(gmm_mean_:type())
    local grad_gmm_cov_ = torch.Tensor(gmm_cov_:size()):type(gmm_cov_:type())
    local grad_gmm_weight_ = torch.Tensor(gmm_weight_:size()):type(gmm_weight_:type())
    local grad_border_ = torch.Tensor(border_:size()):type(border_:type())
    for b=1,N do --iterate over batches
      local target_pixel_ = target_[b]:narrow(1,1,pixel_size)
      local target_border_ = target_[{b,Mp1}]
      -- calcuate GMM responsibilities
      local gmm_rpb = torch.zeros(self.num_mixtures):type(gmm_mean_:type())
      local gmm_cov_inv_ = torch.Tensor(gmm_cov_[b]:size()):type(gmm_cov_:type())
      for g=1,self.num_mixtures do -- iterate over mixtures
        gmm_rpb[g] = distributions.mvn.pdf(target_pixel_, gmm_mean_[{b,g,{}}], gmm_cov_[{b,g,{},{}}]) * gmm_weight_[{b,g}]
        gmm_cov_inv_[g] = torch.inverse(gmm_cov_[{b,g}])
      end
      gmm_rpb = gmm_rpb / torch.sum(gmm_rpb)
      local gmm_mean_diff_ = torch.repeatTensor(target_[b]:narrow(1,1,pixel_size),self.num_mixtures,1) - gmm_mean_[b]
      local mean_left_ = gmm_mean_diff_:view(self.num_mixtures, pixel_size, 1)
      local mean_right_ = gmm_mean_diff_:view(self.num_mixtures, 1, pixel_size)

      -- gradients
      -- gradient of weight is easy, may have numerical instabilities
      grad_gmm_weight_[b] = - torch.cdiv(gmm_rpb, gmm_weight_[b])
      -- gradient of mean
      gmm_rpb = torch.repeatTensor(gmm_rpb:view(-1,1), 1, pixel_size)
      grad_gmm_mean_[b] = - torch.cmul(gmm_rpb, torch.bmm(gmm_cov_inv_, mean_left_))
      -- gradient of covariance matrix
      gmm_rpb = torch.repeatTensor(gmm_rpb:view(-1,pixel_size,1), 1, 1, pixel_size)
      local gmm_temp_ = torch.bmm(torch.bmm(gmm_cov_inv_, torch.bmm(mean_left_, mean_right_)), gmm_cov_inv_) - gmm_cov_inv_
      grad_gmm_cov_[b] = - torch.cmul(gmm_rpb, gmm_temp_) * 0.5
      -- gradient of border is easy
      grad_border_[b] =  1/(1+torch.exp(-border_[b])) - target_border_
    end
    grad_gmm_mean_:div(D*N)
    grad_gmm_cov_:div(D*N)
    grad_gmm_weight_:div(D*N)
    grad_border_:div(D*N)
    table.insert(grad_, grad_gmm_mean_)
    --table.insert(grad_, torch.Tensor(gmm_mean_:size()):fill(0))
    table.insert(grad_, grad_gmm_cov_)
    --table.insert(grad_, torch.Tensor(gmm_cov_:size()):fill(0))
    table.insert(grad_, grad_gmm_weight_)
    --table.insert(grad_, torch.Tensor(gmm_weight_:size()):fill(0))
    table.insert(grad_, grad_border_)
    --table.insert(grad_, torch.Tensor(border_:size()):fill(0))
    table.insert(self.gradInput, grad_)
  end
  return self.gradInput
end
