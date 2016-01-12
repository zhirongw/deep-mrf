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
  opt.mult_in = true
  local pm = nn.PixelModel(opt)
  pm:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  --local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  --local seq = torch.cat(pixels, borders, 3):type(dtype)

  -- evaluate the analytic gradient
  local output = pm:forward(pixels)
  local w = torch.randn(output:size())
  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output, w))
  local gradOutput = w
  local gradInput = pm:backward(pixels, gradOutput)

  -- create a loss function wrapper
  local function f(x)
    local output = pm:forward(x)
    local loss = torch.sum(torch.cmul(output, w))
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, pixels, 1, 1e-6)

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
  opt.mult_in = true
  local crit = nn.PixelModelCriterion(opt.pixel_size, opt.num_mixtures)
  crit:type(dtype)

  local target = torch.rand(opt.batch_size, opt.pixel_size)
  --local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  --local seq = torch.cat(pixels, borders, 3):type(dtype)

  local gmms = torch.rand(opt.batch_size, crit.output_size)
  -- evaluate the analytic gradient
  local loss = crit:forward(gmms, target)
  local gradInput = crit:backward(gmms, target)

  -- create a loss function wrapper
  local function f(x)
    local loss = crit:forward(x, target)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, gmms, 1, 1e-6)

  print(gradInput)
  print(gradInput_num)
  --local g = gradInput:view(-1)
  --local gn = gradInput_num:view(-1)
  --for i=1,g:nElement() do
  --  local r = gradcheck.relative_error(g[i],gn[i])
  --  print(i, g[i], gn[i], r)
  --end
  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
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
  opt.mult_in = true
  local pm = nn.PixelModel(opt)
  local crit = nn.PixelModelCriterion(opt.pixel_size, opt.num_mixtures)
  pm:type(dtype)
  crit:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  local target = torch.rand(opt.batch_size, opt.pixel_size)
  --local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  --local seq = torch.cat(pixels, borders, 3):type(dtype)

  -- evaluate the analytic gradient
  local output = pm:forward(pixels)
  local loss = crit:forward(output, target)
  local gradOutput = crit:backward(output, target)
  local gradInput = pm:backward(pixels, gradOutput)

  -- create a loss function wrapper
  local function f(x)
    local output = pm:forward(x)
    local loss = crit:forward(output, target)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, pixels, 1, 1e-6)

  print(gradInput)
  print(gradInput_num)
  --local g = gradInput:view(-1)
  --local gn = gradInput_num:view(-1)
  --for i=1,g:nElement() do
  --local r = gradcheck.relative_error(g[i],gn[i])
  --  print(i, g[i], gn[i], r)
  --end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end

local function overfit()
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
  opt.mult_in = true
  local pm = nn.PixelModel(opt)
  local crit = nn.PixelModelCriterion(opt.pixel_size, opt.num_mixtures)
  pm:type(dtype)
  crit:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  --local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  --local seq = torch.cat(pixels, borders, 3):type(dtype)

  local params, grad_params = pm:getParameters()
  local function lossFun()
    grad_params:zero()
    local output = pm:forward(pixels)
    local loss = crit:forward(output, pixels)
    local gradOutput = crit:backward(output, pixels)
    pm:backward(pixels, gradOutput)
    return loss
  end

  local loss
  local grad_cache = grad_params:clone():fill(1e-8)
  print('trying to overfit the language model on toy data:')
  for t=1,300 do
    loss = lossFun()
    -- test that initial loss makes sense
    if t == 1 then tester:assertlt(math.abs(math.log(opt.pixel_size+1) - loss), 0.1) end
    grad_cache:addcmul(1, grad_params, grad_params)
    params:addcdiv(-1e-1, grad_params, torch.sqrt(grad_cache)) -- adagrad update
    print(string.format('iteration %d/300: loss %f', t, loss))
  end
  -- holy crap adagrad destroys the loss function!

  tester:assertlt(loss, 0.2)
end

-- check that we can call :sample() and that correct-looking things happen
local function sample()
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
  opt.mult_in = true
  local pm = nn.PixelModel(opt)
  pm:type(dtype)

  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)
  local seq = pm:sample(imgs)

  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 1)
  tester:assertle(torch.max(seq), opt.pixel_size+1)
  print('\nsampled sequence:')
  print(seq)
end


--[[
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
--tests.gradCheckCrit = gradCheckCrit
tests.gradCheck = gradCheck
--tests.overfit = overfit
--tests.sample = sample
--tests.sample_beam = sample_beam

tester:add(tests)
tester:run()
