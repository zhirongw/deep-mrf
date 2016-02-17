require 'torch'
require 'nn'
require 'nngraph'
require 'image'
-- local imports
require 'pm'
require 'gmms'
local matio = require 'matio'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Sampling an Image from a Pixel Model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','','path to model to evaluate')
-- Sampling options
cmd:option('-batch_size', 1, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
-- For evaluation on a folder of images:
cmd:option('-test_path', '../../pixel/SR/train_ycb.mat', 'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '', 'In case the image paths have to be preprended with a root path to an image folder')
-- For evaluation on MSCOCO images from some split:
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
local batch_size = opt.batch_size
if opt.batch_size == 0 then batch_size = checkpoint.opt.batch_size end

-- change it to evaluation mode
local protos = checkpoint.protos
local patch_size = checkpoint.opt['patch_size']
local border = checkpoint.opt['border_init']
local shift = checkpoint.opt['input_shift']
if shift == nil then shift = 0 end
crit = nn.MSECriterion()
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end
local pm = protos.pm
pm.output_back = true
pm.core:evaluate()
print('The loaded model is trained on patch size with: ', patch_size)
print('Number of neighbors used: ', pm.num_neighbors)

-- prepare the empty states
local init_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(batch_size, pm.rnn_size):double()
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone()) -- for lstm c
    table.insert(init_state, h_init:clone()) -- for lstm h
end

batch_size = 1 -- now, We will force to evaluate one image per time.
-------------------------------------------------------------------------------

local function sample3n(lowres, gt)
  local states = {[0]=init_state}
  local height = lowres:size(2)
  local width = lowres:size(3)
  pm.recurrent_stride = width
  pm.seq_length = height * width

  if pm.pixel_size == 1 then lowres = lowres[1]:clone() end
  lowres_input = lowres:view(batch_size, pm.pixel_size, height, width)
  lowres_input = lowres_input:cuda()

  images = torch.Tensor(batch_size, pm.pixel_size, height, width):cuda()
  if pm.pixel_size == 1 then gt = gt[1]:clone() end
  gt = gt:view(batch_size, pm.pixel_size, height, width)
  gt = gt:cuda()

  local loss_sum = 0
  local pixel
  -- loop through each timestep
  for h=1,height do
    for w=1,width do
      local ww = w -- actual coordinate
      if h % 2 == 0 then ww = pm.recurrent_stride + 1 - w end

      local pixel_left, pixel_up, pixel_right
      local pl, pr, pu
      if ww == 1 or h % 2 == 0 then
        pixel_left = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
        pl = 0
      else
        pixel_left = images[{{}, {}, h, ww-1}]
        pl = ww - 1
      end
      if ww == pm.recurrent_stride or h % 2 == 1 then
        pixel_right = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
        pr = 0
      else
        pixel_right = images[{{}, {}, h, ww+1}]
        pr = ww + 1
      end
      if h == 1 then
        pixel_up = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
        pu = 0
      else
        pixel_up = images[{{}, {}, h-1, ww}]
        pu = ww
      end

      -- inputs to LSTM, {input, states[t, t-1], states[t-1, t], states[t, t+1] }
      -- Need to fix this for the new model
      local highres_input = torch.cat(torch.cat(pixel_left, pixel_up, 2), pixel_right, 2)
      local inputs = {torch.cat(highres_input, lowres_input[{{},{},h,ww}]), unpack(states[pl])}
      -- insert the states[t-1,t]
      for i,v in ipairs(states[pu]) do table.insert(inputs, v) end
      -- insert the states[t,t+1]
      for i,v in ipairs(states[pr]) do table.insert(inputs, v) end
      -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
      local lsts = pm.core:forward(inputs)

      -- save the state
      states[ww] = {}
      for i=1,pm.num_state do table.insert(states[ww], lsts[i]:clone()) end
      pixel = lsts[#lsts]

      -- sampling
      local train_pixel = gt[{{}, {}, h, ww}]:clone()
      loss = crit:forward(pixel, train_pixel)
      --pixel = train_pixel
      images[{{},{},h,ww}] = pixel
      loss_sum = loss_sum + loss
    end
    collectgarbage()
  end

  loss_sum = loss_sum / pm.seq_length
  print('loss: ', loss_sum)
  -- output the sampled images
  local images_cpu = images:float():view(batch_size, pm.pixel_size, height, width)
  return images_cpu
end

-- we need to cache the states of all the pixels, and go though the image twice.
local function sample4n(lowres, gt)
  local states = {[0]=init_state}

  local h = lowres:size(2)
  local w = lowres:size(3)
  pm.recurrent_stride = w
  pm.seq_length = h * w
  pm:_buildIndex()

  if pm.pixel_size == 1 then lowres = lowres[1]:clone() end
  lowres_input = lowres:view(batch_size, pm.pixel_size, h*w)
  lowres_input = lowres_input:permute(3, 1, 2)
  lowres_input = lowres_input:cuda()
  lowres_input = torch.repeatTensor(lowres_input, 2, 1, 1)

  images = torch.Tensor(batch_size, pm.pixel_size, h*w):cuda()
  images = torch.repeatTensor(images, 1, 1, 2)
  if pm.pixel_size == 1 then gt = gt[1]:clone() end
  gt = gt:view(batch_size, pm.pixel_size, h*w)
  gt = torch.repeatTensor(gt, 1, 1, 2)
  gt = gt:cuda()

  -------------------------------The Forward Pass -----------------------------
  local loss_sum_f = 0
  local pixel
  -- loop through each timestep
  for t=1,pm.seq_length do
    local pixel_left, pixel_up, pixel_right, pixel_down
    local pl, pr, pu, pd, pi
    pl = pm._Findex[{t, 1}]
    pu = pm._Findex[{t, 2}]
    pr = pm._Findex[{t, 3}]
    pd = pm._Findex[{t, 4}]
    pi = pm._Findex[{t, 5}]

    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t], states[t, t+1] }
    -- first sweep, no seq info.
    local highres_input = torch.Tensor(batch_size, pm.pixel_size * 4):fill(border):cuda()
    local pixel_input = torch.cat(highres_input, lowres_input[pi])
    local inputs = {pixel_input, unpack(states[pl])}
    for i,v in ipairs(states[pu]) do table.insert(inputs, v) end
    for i,v in ipairs(states[pr]) do table.insert(inputs, v) end
    for i,v in ipairs(states[pd]) do table.insert(inputs, v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = pm.core:forward(inputs)

    -- save the state
    states[t] = {}
    for i=1,pm.num_state do table.insert(states[t], lsts[i]:clone()) end
    pixel = lsts[#lsts]
    --print(gmms)
    -- sampling
    local train_pixel = gt[{{}, {}, pi}]:clone()
    loss = crit:forward(pixel, train_pixel)
    --pixel = train_pixel
    images[{{},{},pi}] = pixel
    loss_sum_f = loss_sum_f + loss
  end
  collectgarbage()

  loss_sum_f = loss_sum_f / pm.seq_length
  print('forward loss: ', loss_sum_f)

  -------------------------------The Backward Pass -----------------------------
  local loss_sum_b = 0
  local pixel
  -- loop through each timestep
  for t=pm.seq_length,1,-1 do
    local pixel_left, pixel_up, pixel_right, pixel_down
    local pl, pr, pu, pd, pi
    pl = pm._Bindex[{t, 1}]
    pu = pm._Bindex[{t, 2}]
    pr = pm._Bindex[{t, 3}]
    pd = pm._Bindex[{t, 4}]
    pi = pm._Bindex[{t, 5}]
    if pl == 0 then
      pixel_left = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
    elseif pl > pm.seq_length then
      pixel_left = images[{{}, {}, pm._Bindex[{pl-pm.seq_length, 5}]}]
    else
      if pm.output_back then
        pixel_left = images[{{}, {}, pm._Findex[{pl, 5}]}]
      else
        pixel_left = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
      end
    end
    if pu == 0 then
      pixel_up = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
    elseif pu > pm.seq_length then
      pixel_up = images[{{}, {}, pm._Bindex[{pu-pm.seq_length, 5}]}]
    else
      if pm.output_back then
        pixel_up = images[{{}, {}, pm._Findex[{pu, 5}]}]
      else
        pixel_up = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
      end
    end
    if pr == 0 then
      pixel_right = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
    elseif pr > pm.seq_length then
      pixel_right = images[{{}, {}, pm._Bindex[{pr-pm.seq_length, 5}]}]
    else
      if pm.output_back then
        pixel_right = images[{{}, {}, pm._Findex[{pr, 5}]}]
      else
        pixel_right = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
      end
    end
    if pd == 0 then
      pixel_down = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
    elseif pd > pm.seq_length then
      pixel_down = images[{{}, {}, pm._Bindex[{pd-pm.seq_length, 5}]}]
    else
      if pm.output_back then
        pixel_down = images[{{}, {}, pm._Findex[{pd, 5}]}]
      else
        pixel_down = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
      end
    end
    local highres_input = torch.cat(torch.cat(torch.cat(pixel_left, pixel_up, 2), pixel_right, 2), pixel_down, 2)
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t], states[t, t+1] }
    -- Need to fix this for the new model
    local pixel_input = torch.cat(highres_input, lowres_input[pi])
    local inputs = {pixel_input, unpack(states[pl])}
    for i,v in ipairs(states[pu]) do table.insert(inputs, v) end
    for i,v in ipairs(states[pr]) do table.insert(inputs, v) end
    for i,v in ipairs(states[pd]) do table.insert(inputs, v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = pm.core:forward(inputs)

    -- save the state
    states[t+pm.seq_length] = {}
    for i=1,pm.num_state do table.insert(states[t+pm.seq_length], lsts[i]:clone()) end
    pixel = lsts[#lsts]
    --pixel:clamp(16/255+shift,235/255+shift)
    --print(gmms)

    -- sampling
    local train_pixel = gt[{{}, {}, pi}]:clone()
    loss = crit:forward(pixel, train_pixel)
    --pixel = train_pixel
    images[{{},{},pi}] = pixel
    loss_sum_b = loss_sum_b + loss
  end
  collectgarbage()

  loss_sum_b = loss_sum_b / pm.seq_length
  print('backward loss: ', loss_sum_b)
  print('Our PSNR: ', math.log10(1 / math.sqrt(loss_sum_b))*20)
  --print('overall loss: ', (loss_sum_f + loss_sum_b) / 2)
  -- output the sampled images
  local images_cpu = images:float():view(batch_size, pm.pixel_size, 2, h, w)
  return images_cpu
end

local data = matio.load(opt.test_path, 'data')
local N = data['num'][{1,1}]
for i=1,N do
  print(i)
  local highres = data['highres'][i]:type('torch.FloatTensor'):div(255)
  local lowres = data['lowres'][i]:type('torch.FloatTensor')
  local imgdiff = torch.add(highres[1], -1, lowres[1])
  local diff = torch.mean(imgdiff:cmul(imgdiff))
  print('bicubic loss: ', diff)
  print('bicubic PSNR: ', math.log10(1 / math.sqrt(diff))*20)
  local lowres_rgb = utils.ycbcr2rgb(lowres)
  lowres_rgb = lowres_rgb:clamp(0,1):mul(255):type('torch.ByteTensor')
  image.save(string.format('SR/%d_lowres.png', i), lowres_rgb)
  highres:add(shift)
  lowres:add(shift)
  if pm.num_neighbors == 3 then
    local out = sample3n(lowres, highres)
    out:add(-shift)
    local pred
    if pm.pixel_size == 1 then
      pred = lowres:add(-shift)
      pred[1] = out[1]
    else
      pred = out
    end
    local pred_rgb = utils.ycbcr2rgb(pred)
    pred_rgb = pred_rgb:clamp(0,1):mul(255):type('torch.ByteTensor')
    image.save("SR/"..i.."_pred.png", pred_rgb)

  elseif pm.num_neighbors == 4 then
    local out = sample4n(lowres, highres)
    out:add(-shift)
    local pred
    if pm.pixel_size == 1 then
      local pred_rgb
      pred = lowres:clone():add(-shift)
      pred[1] = out[{1,{},1,{},{}}]
      pred_rgb = utils.ycbcr2rgb(pred)
      pred_rgb = pred_rgb:clamp(0,1):mul(255):type('torch.ByteTensor')
      image.save("SR/"..i.."_f.png", pred_rgb)
      pred = lowres:clone():add(-shift)
      pred[1] = out[{1,{},2,{},{}}]
      pred_rgb = utils.ycbcr2rgb(pred)
      pred_rgb = pred_rgb:clamp(0,1):mul(255):type('torch.ByteTensor')
      image.save("SR/"..i.."_b.png", pred_rgb)
    else
      pred = out[{1,{},1,{},{}}]
      pred_rgb = utils.ycbcr2rgb(pred)
      pred_rgb = pred_rgb:clamp(0,1):mul(255):type('torch.ByteTensor')
      image.save("SR/"..i.."_f.png", pred_rgb)
      pred = out[{1,{},2,{},{}}]
      pred_rgb = utils.ycbcr2rgb(pred)
      pred_rgb = pred_rgb:clamp(0,1):mul(255):type('torch.ByteTensor')
      image.save("SR/"..i.."_b.png", pred_rgb)
    end
  end
end
