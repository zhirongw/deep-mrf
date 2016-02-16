--[[
Courtesy of Karpathy
Unit tests for the PixelModel implementation, making sure
that nothing crashes, that we can overfit a small dataset
and that everything gradient checks.
--]]

require 'torch'
require 'pm'
require 'kld'
require 'cudnn'

local gradcheck = require 'misc.gradcheck'
local VAE = require 'vae'

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
  opt.pixel_size = 1
  opt.num_mixtures = 2
  opt.recurrent_stride = 3
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 9
  opt.batch_size = 2
  opt.mult_in = false
  opt.num_neighbors = 4
  opt.border_init = 0
  local pm = nn.PixelModel4N(opt)
  pm:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size*(opt.num_neighbors+1))

  -- evaluate the analytic gradient
  local output = pm:forward(pixels)
  local w = torch.randn(output:size())
  local ww = w:clone()
  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output, w))
  local gradOutput = w
  local gradInput = pm:backward(pixels, gradOutput)

  -- create a loss function wrapper
  local function f(x)
    local output = pm:forward(x)
    local loss = torch.sum(torch.cmul(output, ww))
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

local function gradCheckIMPM()

  local dtype = 'torch.DoubleTensor'
  local imOpt = {}
  imOpt.latent_variable_size = 3
  imOpt.feature_size = 2
  local im_model = VAE.ImageModel(imOpt)
  local im = im_model[1]
  local im_sampler  = im_model[2]
  im:type(dtype)
  im_sampler:type(dtype)
  local opt = {}
  opt.pixel_size = 3
  opt.num_mixtures = 2
  opt.recurrent_stride = 3
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 9
  opt.batch_size = 2
  opt.mult_in = false
  opt.output_back = true
  opt.feature_dim = imOpt.feature_size
  opt.num_neighbors = 4
  opt.border_init = 0
  local pm = nn.PixelModel4N(opt)
  pm:type(dtype)
  local psOpt = {}
  psOpt.patch_size = opt.recurrent_stride
  psOpt.feature_dim = imOpt.feature_size
  psOpt.im_batch_size = 1
  psOpt.pm_batch_size = opt.batch_size
  psOpt.num_neighbors = 4
  psOpt.border_size = 0
  psOpt.border = opt.border_init
  psOpt.noise = 0
  local patch_extractor = nn.PatchExtractor(psOpt)
  patch_extractor:type(dtype)
  local imcrit = nn.KLDCriterion()

  local images = torch.rand(psOpt.im_batch_size, opt.pixel_size, 16, 16)
  local dummy = torch.rand(psOpt.im_batch_size, opt.pixel_size, 16, 16)

  -- evaluate the analytic gradient
  local features, mean, log_var = unpack(im:forward(images))
  local loss1 = imcrit:forward(mean, log_var)
  local targets, patches = unpack(patch_extractor:forward({dummy, features}))
  local pred = pm:forward(patches)

  local w_pred = torch.randn(pred:size())
  local ww = w_pred:clone()
  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(pred, w_pred))
  local gradOutput = w_pred
  local dpatches = pm:backward(patches, w_pred)
  local x, dfeatures = unpack(patch_extractor:backward({dummy, features},{x, dpatches}))
  local dmean, dlog_var = unpack(imcrit:backward(mean, log_var))
  local gradInput = im:backward(images, {dfeatures, dmean, dlog_var})

  -- create a loss function wrapper
  local function f(x)
    local features, mean, log_var = unpack(im:forward(x))
    local targets, patches = unpack(patch_extractor:forward({dummy, features}))
    local pred = pm:forward(patches)
    local loss1 = imcrit:forward(mean, log_var)
    local loss = torch.sum(torch.cmul(pred, ww))
    return loss + loss1
  end

  local gradInput_num = gradcheck.numeric_gradient(f, images, 1, 1e-6)

  print(gradInput)
  print(gradInput_num)
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
  opt.seq_length = 9
  opt.batch_size = 2
  opt.mult_in = true
  opt.num_neighbors = 4
  opt.border_init = 0
  local crit = nn.PixelModelCriterion(opt.pixel_size, opt.num_mixtures, {policy='linear', val=0.9})
  crit:type(dtype)

  local gmms = torch.rand(opt.seq_length, opt.batch_size, crit.output_size)
  local targets = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  --local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  --local seq = torch.cat(pixels, borders, 3):type(dtype)

  -- evaluate the analytic gradient
  local loss = crit:forward(gmms, targets)
  local gradInput = crit:backward(gmms, targets)

  -- create a loss function wrapper
  local function f(x)
    local loss = crit:forward(x, targets)
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
  opt.num_mixtures = 5
  opt.recurrent_stride = 3
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 9
  opt.batch_size = 2
  opt.mult_in = true
  opt.num_neighbors = 4
  opt.border_init = 0
  local pm = nn.PixelModel4N(opt)
  local crit = nn.MSECriterion()
  pm:type(dtype)
  crit:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size*(opt.num_neighbors+1))
  local targets = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  --local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  --local seq = torch.cat(pixels, borders, 3):type(dtype)

  -- evaluate the analytic gradient
  local output = pm:forward(pixels)
  local loss = crit:forward(output, targets)
  local gradOutput = crit:backward(output, targets)
  local gradInput = pm:backward(pixels, gradOutput)

  -- create a loss function wrapper
  local function f(x)
    local output = pm:forward(x)
    local loss = crit:forward(output, targets)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, pixels, 1, 1e-6)

  --print(gradInput)
  --print(gradInput_num)
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
  local targets = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)

  local params, grad_params = pm:getParameters()
  local function lossFun()
    grad_params:zero()
    local output = pm:forward(pixels)
    local loss = crit:forward(output, targets)
    local gradOutput = crit:backward(output, targets)
    pm:backward(pixels, gradOutput)
    return loss
  end

  local loss
  local grad_cache = grad_params:clone():fill(1e-8)
  print('trying to overfit the pixel model on toy data:')
  for t=1,5000 do
    loss = lossFun()
    -- test that initial loss makes sense
    grad_cache:addcmul(1, grad_params, grad_params)
    params:addcdiv(-1e-3, grad_params, torch.sqrt(grad_cache)) -- adagrad update
    print(string.format('iteration %d/5000: loss %f', t, loss))
  end
  -- holy crap adagrad destroys the loss function!

  tester:assertlt(loss, 0.2)
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
--tests.gradCheckPM = gradCheckPM
tests.gradCheckIMPM = gradCheckIMPM
-- tests.gradCheckCrit = gradCheckCrit
--tests.gradCheck = gradCheck
-- tests.overfit = overfit
--tests.sample = sample
--tests.sample_beam = sample_beam

tester:add(tests)
tester:run()
