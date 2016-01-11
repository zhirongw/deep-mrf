--[[
Courtesy of Karpathy
Unit tests for the PixelModel implementation, making sure
that nothing crashes, that we can overfit a small dataset
and that everything gradient checks.
--]]

require 'torch'
require 'pm'

local gradcheck = require 'misc.gradcheck'

local tests = {}
local tester = torch.Tester()

-- validates the size and dimensions of a given
-- tensor a to be size given in table sz
function tester:assertTensorSizeEq(a, sz)
  tester:asserteq(a:nDimension(), #sz)
  for i=1,#sz do
    tester:asserteq(a:size(i), sz[i])
  end
end

-- Test the API of the Pixel Model
local function forwardApiTestFactory(dtype)
  if dtype == 'torch.CudaTensor' then
    require 'cutorch'
    require 'cunn'
  end
  local function f()
    -- create PixelModel instance
    local opt = {}
    opt.pixel_size = 3
    opt.num_mixtures = 2
    opt.recurrent_stride = 3
    opt.rnn_size = 8
    opt.num_layers = 2
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 10
    local pm = nn.PixelModel(opt)
    local crit = nn.PixelModelCriterion(opt.num_mixtures)
    pm:type(dtype) --type function set the module parameters
    crit:type(dtype)

    -- construct some input to feed in
    local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
    local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
    local seq = torch.cat(pixels, borders, 3):type(dtype)
    -- forward
    local output = pm:forward(seq)
    -- returned output is a table with each element is a sequence
    tester:asserteq(#output, opt.seq_length)
    -- for each sequnece, it's also a table, with 4 tensors corresponding to gmm encodings
    tester:asserteq(#output[1], 4)
    -- the first is the mean
    tester:assertTensorSizeEq(output[1][1], {opt.batch_size, opt.num_mixtures, opt.pixel_size})
    -- the second is the covariance
    tester:assertTensorSizeEq(output[1][2], {opt.batch_size, opt.num_mixtures, opt.pixel_size, opt.pixel_size})
    tester:assertTensorEq(output[1][2][{{},{},1,2}], output[1][2][{{},{},2,1}], 1e-6) -- should be symmetric
    tester:assertTensorEq(output[1][2][{{},{},1,3}], output[1][2][{{},{},3,1}], 1e-6)
    tester:assertTensorEq(output[1][2][{{},{},2,3}], output[1][2][{{},{},3,2}], 1e-6)
    -- the third is the gmm weights
    tester:assertTensorSizeEq(output[1][3], {opt.batch_size, opt.num_mixtures})
    tester:assertTensorEq(torch.sum(output[1][3], 2), 1, 1e-6) -- weights should be summed to 1
    -- the fourth is the border
    tester:assertTensorSizeEq(output[1][4], {opt.batch_size})

    local loss = crit:forward(output, seq)
    local gradOutput = crit:backward(output, seq)
    -- this should be the same size as the output
    tester:asserteq(#gradOutput, opt.seq_length)
    tester:asserteq(#gradOutput[1], 4)
    -- the first is the mean
    tester:assertTensorSizeEq(gradOutput[1][1], {opt.batch_size, opt.num_mixtures, opt.pixel_size})
    -- the second is the covariance
    tester:assertTensorSizeEq(gradOutput[1][2], {opt.batch_size, opt.num_mixtures, opt.pixel_size, opt.pixel_size})
    tester:assertTensorEq(gradOutput[1][2][{{},{},1,2}], gradOutput[1][2][{{},{},2,1}], 1e-6) -- should be symmetric
    tester:assertTensorEq(gradOutput[1][2][{{},{},1,3}], gradOutput[1][2][{{},{},3,1}], 1e-6)
    tester:assertTensorEq(gradOutput[1][2][{{},{},2,3}], gradOutput[1][2][{{},{},3,2}], 1e-6)
    -- the third is the gmm weights
    tester:assertTensorSizeEq(gradOutput[1][3], {opt.batch_size, opt.num_mixtures})
    -- the fourth is the border
    tester:assertTensorSizeEq(gradOutput[1][4], {opt.batch_size})

    local gradInput = pm:backward(seq, gradOutput)
    tester:assertTensorSizeEq(gradInput, {opt.seq_length, opt.batch_size, opt.pixel_size+1})
  end
  return f
end

-- test just the language model alone (without the criterion)
local function gradCheckPM()

  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.pixel_size = 3
  opt.num_mixtures = 2
  opt.recurrent_stride = 3
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 10
  local pm = nn.PixelModel(opt)
  pm:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  local seq = torch.cat(pixels, borders, 3):type(dtype)

  -- evaluate the analytic gradient
  local output = pm:forward(seq)
  local w1 = torch.randn(opt.batch_size, opt.num_mixtures, opt.pixel_size)
  local w2 = torch.randn(opt.batch_size, opt.num_mixtures, opt.pixel_size, opt.pixel_size)
  local w3 = torch.randn(opt.batch_size, opt.num_mixtures)
  local w4 = torch.randn(opt.batch_size)
  -- generate random weighted sum criterion
  local loss = 0
  local gradOutput = {}
  for t=1,#output do
    loss = torch.sum(torch.cmul(output[t][1], w1)) + loss
    loss = torch.sum(torch.cmul(output[t][2], w2)) + loss
    loss = torch.sum(torch.cmul(output[t][3], w3)) + loss
    loss = torch.sum(torch.cmul(output[t][4], w4)) + loss
    table.insert(gradOutput, {torch.add(w1,t),torch.add(w2,t),torch.add(w3,t),torch.add(w4,t)})
  end
  local gradInput = pm:backward(seq, gradOutput)

  -- create a loss function wrapper
  local function f(x)
    local output = pm:forward(x)
    local loss = 0
    for t=1,#output do
      loss = torch.sum(torch.cmul(output[t][1], torch.add(w1,t))) + loss
      loss = torch.sum(torch.cmul(output[t][2], torch.add(w2,t))) + loss
      loss = torch.sum(torch.cmul(output[t][3], torch.add(w3,t))) + loss
      loss = torch.sum(torch.cmul(output[t][4], torch.add(w4,t))) + loss
    end
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, seq, 1, 1e-6)

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)
end

local function gradCheckCrit()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.pixel_size = 3
  opt.num_mixtures = 2
  opt.recurrent_stride = 3
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 2
  local crit = nn.PixelModelCriterion(opt.num_mixtures)
  crit:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  local seq = torch.cat(pixels, borders, 3):type(dtype)

  local gmms = {}
  for t = 1, opt.seq_length do
    local gmm = {}
    local gmms_mean = torch.rand(opt.batch_size, opt.num_mixtures, opt.pixel_size)
    local gmms_var = torch.repeatTensor(torch.eye(opt.pixel_size):view(1,1,3,3), opt.batch_size, opt.num_mixtures, 1, 1)
    local gmms_weight = torch.ones(opt.batch_size, opt.num_mixtures)
    local gmms_sum = torch.repeatTensor(torch.sum(gmms_weight, 2), 1, opt.num_mixtures)
    gmms_weight:cdiv(gmms_sum)
    local border = torch.zeros(opt.batch_size)
    table.insert(gmm, gmms_mean)
    table.insert(gmm, gmms_var)
    table.insert(gmm, gmms_weight)
    table.insert(gmm, border)
    table.insert(gmms, gmm)
  end

  local gm = gmms[7][1]:clone()
  local gv = gmms[7][2]:clone()
  local gw = gmms[7][3]:clone()
  local gb = gmms[7][4]:clone()
  tester:assertTensorEq(gm, gmms[7][1], 1e-4)
  tester:assertTensorEq(gv, gmms[7][2], 1e-4)
  tester:assertTensorEq(gw, gmms[7][3], 1e-4)
  tester:assertTensorEq(gb, gmms[7][4], 1e-4)
  -- evaluate the analytic gradient
  local loss = crit:forward(gmms, seq)
  local gradInput = crit:backward(gmms, seq)
  tester:assertTensorEq(gm, gmms[7][1], 1e-4)
  tester:assertTensorEq(gv, gmms[7][2], 1e-4)
  tester:assertTensorEq(gw, gmms[7][3], 1e-4)
  tester:assertTensorEq(gb, gmms[7][4], 1e-4)

  print('----------')
  -- create a loss function wrapper
  local function f(x)
    local loss = crit:forward(x, seq)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient_crit(f, gmms, 1, 1e-6)

  print(gradInput[1][2])
  print(gradInput_num[1][2])
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end
  for t=1,opt.seq_length do
    tester:assertTensorEq(gradInput[t][1], gradInput_num[t][1], 1e-4)
    tester:assertlt(gradcheck.relative_error(gradInput[t][1], gradInput_num[t][1], 1e-8), 5e-4)
    tester:assertTensorEq(gradInput[t][3], gradInput_num[t][3], 1e-4)
    tester:assertlt(gradcheck.relative_error(gradInput[t][3], gradInput_num[t][3], 1e-8), 5e-4)
    tester:assertTensorEq(gradInput[t][4], gradInput_num[t][4], 1e-4)
    tester:assertlt(gradcheck.relative_error(gradInput[t][4], gradInput_num[t][4], 1e-8), 5e-4)
  end
end

local function gradCheck()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.pixel_size = 3
  opt.num_mixtures = 2
  opt.recurrent_stride = 3
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 2
  local pm = nn.PixelModel(opt)
  local crit = nn.PixelModelCriterion(opt.num_mixtures)
  pm:type(dtype)
  crit:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  local seq = torch.cat(pixels, borders, 3):type(dtype)

  local seq_cp = seq:clone()
  tester:assertTensorEq(seq_cp, seq, 1e-4)
  -- evaluate the analytic gradient
  local output = pm:forward(seq)
  local loss = crit:forward(output, seq)
  local gradOutput = crit:backward(output, seq)
  local gradInput = pm:backward(seq, gradOutput)
  tester:assertTensorEq(seq_cp, seq, 1e-4)

  -- create a loss function wrapper
  local function f(x)
    local output = pm:forward(x)
    local loss = crit:forward(output, x)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, seq, 1, 1e-6)

  print(gradInput)
  print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end

--[[
local function overfit()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.pixel_size = 3
  opt.output_size = 10
  opt.rnn_size = 24
  opt.num_layers = 1
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local pm = nn.PixelModel(opt)
  local crit = nn.PixelModelCriterion()
  pm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.pixel_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local params, grad_params = pm:getParameters()
  print('number of parameters:', params:nElement(), grad_params:nElement())
  local lstm_params = 4*(opt.input_encoding_size + opt.rnn_size)*opt.rnn_size + opt.rnn_size*4*2
  local output_params = opt.rnn_size * (opt.pixel_size + 1) + opt.pixel_size+1
  local table_params = (opt.pixel_size + 1) * opt.input_encoding_size
  local expected_params = lstm_params + output_params + table_params
  print('expected:', expected_params)

  local function lossFun()
    grad_params:zero()
    local output = pm:forward{imgs, seq}
    local loss = crit:forward(output, seq)
    local gradOutput = crit:backward(output, seq)
    pm:backward({imgs, seq}, gradOutput)
    return loss
  end

  local loss
  local grad_cache = grad_params:clone():fill(1e-8)
  print('trying to overfit the language model on toy data:')
  for t=1,30 do
    loss = lossFun()
    -- test that initial loss makes sense
    if t == 1 then tester:assertlt(math.abs(math.log(opt.pixel_size+1) - loss), 0.1) end
    grad_cache:addcmul(1, grad_params, grad_params)
    params:addcdiv(-1e-1, grad_params, torch.sqrt(grad_cache)) -- adagrad update
    print(string.format('iteration %d/30: loss %f', t, loss))
  end
  -- holy crap adagrad destroys the loss function!

  tester:assertlt(loss, 0.2)
end

-- check that we can call :sample() and that correct-looking things happen
local function sample()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.pixel_size = 3
  opt.output_size = 10
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local pm = nn.PixelModel(opt)

  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)
  local seq = pm:sample(imgs)

  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 1)
  tester:assertle(torch.max(seq), opt.pixel_size+1)
  print('\nsampled sequence:')
  print(seq)
end


-- check that we can call :sample_beam() and that correct-looking things happen
-- these are not very exhaustive tests and basic sanity checks
local function sample_beam()
  local dtype = 'torch.DoubleTensor'
  torch.manualSeed(1)

  local opt = {}
  opt.pixel_size = 3
  opt.output_size = 10
  opt.rnn_size = 8
  opt.num_layers = 1
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local pm = nn.PixelModel(opt)

  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local seq_vanilla, logprobs_vanilla = pm:sample(imgs)
  local seq, logprobs = pm:sample(imgs, {beam_size = 1})

  -- check some basic I/O, types, etc.
  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 0)
  tester:assertle(torch.max(seq), opt.pixel_size+1)

  -- doing beam search with beam size 1 should return exactly what we had before
  print('')
  print('vanilla sampling:')
  print(seq_vanilla)
  print('beam search sampling with beam size 1:')
  print(seq)
  tester:assertTensorEq(seq_vanilla, seq, 0) -- these are LongTensors, expect exact match
  tester:assertTensorEq(logprobs_vanilla, logprobs, 1e-6) -- logprobs too

  -- doing beam search with higher beam size should yield higher likelihood sequences
  local seq2, logprobs2 = pm:sample(imgs, {beam_size = 8})
  local logsum = torch.sum(logprobs, 1)
  local logsum2 = torch.sum(logprobs2, 1)
  print('')
  print('beam search sampling with beam size 1:')
  print(seq)
  print('beam search sampling with beam size 8:')
  print(seq2)
  print('logprobs:')
  print(logsum)
  print(logsum2)

  -- the logprobs should always be >=, since beam_search is better argmax inference
  tester:assert(torch.all(torch.gt(logsum2, logsum)))
end

--]]

--tests.doubleApiForwardTest = forwardApiTestFactory('torch.DoubleTensor')
--tests.floatApiForwardTest = forwardApiTestFactory('torch.FloatTensor')
-- tests.cudaApiForwardTest = forwardApiTestFactory('torch.CudaTensor')
-- tests.gradCheckPM = gradCheckPM
-- tests.gradCheckCrit = gradCheckCrit
tests.gradCheck = gradCheck
--tests.overfit = overfit
--tests.sample = sample
--tests.sample_beam = sample_beam

tester:add(tests)
tester:run()
