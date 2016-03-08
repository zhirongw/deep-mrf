require 'torch'
require 'nn'
require 'nngraph'
require 'image'
-- local imports
require 'pm'
require 'rgb'
require 'gmms'
local utils = require 'misc.utils'
--require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
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
cmd:option('-img_size', 256, 'size of the sampled image')
-- Sampling options
cmd:option('-batch_size', 1, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
-- For evaluation on a folder of images:
cmd:option('-image_folder', '', 'If this is nonempty then will predict on the images in this folder path')
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
local temperature = opt.temperature

-- change it to evaluation mode
local protos = checkpoint.protos
local patch_size = checkpoint.opt['patch_size']
local border = checkpoint.opt['border_init']
local shift = checkpoint.opt['input_shift']
if shift == nil then shift = 0 end
protos.pm.recurrent_stride = patch_size + opt.img_size
protos.pm.seq_length = protos.pm.recurrent_stride * protos.pm.recurrent_stride
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end
local pm = protos.pm
local rgb = protos.rgb
pm.core:evaluate()
rgb.core:evaluate()
rgb.lookup_matrix:evaluate()
print('The loaded model is trained on patch size with: ', patch_size)
print('Number of neighbors used: ', pm.num_neighbors)
print('Number of mixtures used: ', rgb.num_mixtures)
print('rnn size: ', pm.rnn_size)

-- prepare the empty states
local init_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(batch_size, pm.rnn_size):double()
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone()) -- for lstm c
    table.insert(init_state, h_init:clone()) -- for lstm h
end

local states = {[0] = init_state}
local images = torch.Tensor(batch_size, pm.pixel_size, pm.recurrent_stride, pm.recurrent_stride):cuda()
------------------ debug ------------------------
local img = image.load('imgs/D1.png', pm.pixel_size, 'float')
if pm.pixel_size == 1 then
  img = img:resize(pm.pixel_size, 1, img:size(1), img:size(2))
else
  img = img:resize(pm.pixel_size, 1, img:size(2), img:size(3))
end
--img = image.scale(img, 256, 256):resize(1, pm.pixel_size, 256, 256)
img = img[{{}, {}, {1, pm.recurrent_stride},{1, pm.recurrent_stride}}]:contiguous()
img = torch.repeatTensor(img, batch_size, 1, 1, 1)
img = img:cuda()
-------------------------------------------------

local function sample2n()
  local loss_sum = 0
  local train_loss_sum = 0

  local pixel
  local features
  -- loop through each timestep
  for h=1,pm.recurrent_stride do
    for w=1,pm.recurrent_stride do
      local pixel_left, pixel_up
      if w == 1 then
        if border == 0 then
          pixel_left = torch.zeros(batch_size, pm.pixel_size):cuda()
        else
          pixel_left = torch.rand(batch_size, pm.pixel_size):cuda()
        end
      else
        pixel_left = images[{{}, {}, h, w-1}]
      end
      if h == 1 then
        if border == 0 then
          pixel_up = torch.zeros(batch_size, pm.pixel_size):cuda()
        else
          pixel_up = torch.rand(batch_size, pm.pixel_size):cuda()
        end
      else
        pixel_up = images[{{}, {}, h-1,w}]
      end

      -- inputs to LSTM, {input, states[t, t-1], states[t-1, t] }
      -- Need to fix this for the new model
      local inputs = {torch.cat(pixel_left, pixel_up, 2), unpack(states[w-1])}
      local prev_w = w
      if states[w] == nil then prev_w = 0 end
      -- insert the states[t-1,t]
      for i,v in ipairs(states[prev_w]) do table.insert(inputs, v) end
      -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
      local lsts = pm.core:forward(inputs)

      -- save the state
      states[w] = {}
      for i=1,pm.num_state do table.insert(states[w], lsts[i]:clone()) end
      features = lsts[#lsts]

      -- sampling
      local train_pixel = img[{{}, {}, h, w}]:clone()
      train_pixel = train_pixel:view(3, -1, 1)
      pixel, loss, train_loss = rgb:sample(features, temperature, train_pixel)
      --pixel = train_pixel
      images[{{},{},h,w}] = pixel
      loss_sum = loss_sum + loss
      train_loss_sum = train_loss_sum + train_loss
    end
    collectgarbage()
  end

  loss_sum = loss_sum / (pm.recurrent_stride * pm.recurrent_stride)
  train_loss_sum = train_loss_sum / (pm.recurrent_stride * pm.recurrent_stride)
  print('testing loss: ', loss_sum)
  print('training loss: ', train_loss_sum)

  return images
end

local function sample3n()
  local loss_sum = 0
  local train_loss_sum = 0

  local pixel
  local features
  -- loop through each timestep
  for h=1,pm.recurrent_stride do
    for w=1,pm.recurrent_stride do
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
      local inputs = {torch.cat(torch.cat(pixel_left, pixel_up, 2), pixel_right, 2), unpack(states[pl])}
      -- insert the states[t-1,t]
      for i,v in ipairs(states[pu]) do table.insert(inputs, v) end
      -- insert the states[t,t+1]
      for i,v in ipairs(states[pr]) do table.insert(inputs, v) end
      -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
      local lsts = pm.core:forward(inputs)

      -- save the state
      states[ww] = {}
      for i=1,pm.num_state do table.insert(states[ww], lsts[i]:clone()) end
      features = lsts[#lsts]

      -- sampling
      local train_pixel = img[{{}, {}, h, ww}]:clone():add(shift)
      train_pixel = train_pixel:view(3, -1, 1)
      pixel, loss, train_loss = rgb:sample(features, temperature, train_pixel)
      --if h < patch_size then pixel = train_pixel end
      images[{{},{},h,ww}] = pixel:permute(2,1,3):contiguous()
      loss_sum = loss_sum + loss
      train_loss_sum = train_loss_sum + train_loss
    end
    collectgarbage()
  end
  loss_sum = loss_sum / (pm.recurrent_stride * pm.recurrent_stride)
  train_loss_sum = train_loss_sum / (pm.recurrent_stride * pm.recurrent_stride)
  print('testing loss: ', loss_sum)
  print('training loss: ', train_loss_sum)

  return images
end

-- we need to cache the states of all the pixels, and go though the image twice.
local function sample4n()
  images = images:view(batch_size, pm.pixel_size, -1)
  images = torch.repeatTensor(images, 1, 1, 2)
  img = img:view(pm.pixel_size, batch_size, -1)
  img = torch.repeatTensor(img, 1, 1, 2)
  pm:_buildIndex()

  -------------------------------The Forward Pass -----------------------------
  local loss_sum_f = 0
  local train_loss_sum_f = 0

  local pixel
  local gmms
  -- loop through each timestep
  for t=1,pm.seq_length do
    local pixel_left, pixel_up, pixel_right, pixel_down
    local pl, pr, pu, pd, pi
    pl = pm._Findex[{t, 1}]
    pu = pm._Findex[{t, 2}]
    pr = pm._Findex[{t, 3}]
    pd = pm._Findex[{t, 4}]
    pi = pm._Findex[{t, 5}]
    if pl == 0 then
      pixel_left = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
    else
      pixel_left = images[{{}, {}, pm._Findex[{pl, 5}]}]
    end
    if pu == 0 then
      pixel_up = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
    else
      pixel_up = images[{{}, {}, pm._Findex[{pu, 5}]}]
    end
    if pr == 0 then
      pixel_right = torch.Tensor(batch_size, pm.pixel_size):fill(border):cuda()
    else
      pixel_right = images[{{}, {}, pm._Findex[{pr, 5}]}]
    end
    if pd == 0 then
      pixel_down = torch.zeros(batch_size, pm.pixel_size):fill(border):cuda()
    else
      pixel_down = images[{{}, {}, pm._Findex[{pd, 5}]}]
    end
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t], states[t, t+1] }
    -- Need to fix this for the new model
    -- local input_pixel = torch.cat(torch.cat(torch.cat(pixel_left, pixel_up, 2), pixel_right, 2), pixel_down, 2)
    local input_pixel = torch.Tensor(batch_size, pm.pixel_size * 4):fill(border):cuda()
    local inputs = {input_pixel, unpack(states[pl])}
    for i,v in ipairs(states[pu]) do table.insert(inputs, v) end
    for i,v in ipairs(states[pr]) do table.insert(inputs, v) end
    for i,v in ipairs(states[pd]) do table.insert(inputs, v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = pm.core:forward(inputs)

    -- save the state
    states[t] = {}
    for i=1,pm.num_state do table.insert(states[t], lsts[i]:clone()) end
    gmms = lsts[#lsts]
    --print(gmms)
    -- sampling
    local train_pixel = img[{{}, {}, pi}]:clone()
    pixel, loss, train_loss = crit:sample(gmms, temperature, train_pixel)
    --pixel = train_pixel
    images[{{},{},pi}] = pixel:permute(2,1,3):contiguous()
    loss_sum_f = loss_sum_f + loss
    train_loss_sum_f = train_loss_sum_f + train_loss
  end
  collectgarbage()
  loss_sum_f = loss_sum_f / (pm.recurrent_stride * pm.recurrent_stride)
  train_loss_sum_f = train_loss_sum_f / (pm.recurrent_stride * pm.recurrent_stride)
  print('forward testing loss: ', loss_sum_f)
  print('forward training loss: ', train_loss_sum_f)

  -------------------------------The Backward Pass -----------------------------
  local loss_sum_b = 0
  local train_loss_sum_b = 0

  local pixel
  local gmms
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
      pixel_left = torch.zeros(batch_size, pm.pixel_size):fill(border):cuda()
    elseif pl > pm.seq_length then
      pixel_left = images[{{}, {}, pm._Bindex[{pl-pm.seq_length, 5}]}]
    else
      if pm.output_back then
        pixel_left = images[{{}, {}, pm._Findex[{pl, 5}]}]
      else
        pixel_left = torch.zeros(batch_size, pm.pixel_size):fill(border):cuda()
      end
    end
    if pu == 0 then
      pixel_up = torch.zeros(batch_size, pm.pixel_size):fill(border):cuda()
    elseif pu > pm.seq_length then
      pixel_up = images[{{}, {}, pm._Bindex[{pu-pm.seq_length, 5}]}]
    else
      if pm.output_back then
        pixel_up = images[{{}, {}, pm._Findex[{pu, 5}]}]
      else
        pixel_up = torch.zeros(batch_size, pm.pixel_size):fill(border):cuda()
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
    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t], states[t, t+1] }
    -- Need to fix this for the new model
    local input_pixel = torch.cat(torch.cat(torch.cat(pixel_left, pixel_up, 2), pixel_right, 2), pixel_down, 2)
    local inputs = {input_pixel, unpack(states[pl])}
    for i,v in ipairs(states[pu]) do table.insert(inputs, v) end
    for i,v in ipairs(states[pr]) do table.insert(inputs, v) end
    for i,v in ipairs(states[pd]) do table.insert(inputs, v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = pm.core:forward(inputs)

    -- save the state
    states[t+pm.seq_length] = {}
    for i=1,pm.num_state do table.insert(states[t+pm.seq_length], lsts[i]:clone()) end
    gmms = lsts[#lsts]
    --print(gmms)

    -- sampling
    local train_pixel = img[{{}, {}, pi}]:clone()
    pixel, loss, train_loss = crit:sample(gmms, temperature, train_pixel)
    --pixel = train_pixel
    images[{{},{},pi}] = pixel
    loss_sum_b = loss_sum_b + loss
    train_loss_sum_b = train_loss_sum_b + train_loss
  end
  collectgarbage()

  loss_sum_b = loss_sum_b / (pm.recurrent_stride * pm.recurrent_stride)
  train_loss_sum_b = train_loss_sum_b / (pm.recurrent_stride * pm.recurrent_stride)
  print('backward testing loss: ', loss_sum_b)
  print('backward training loss: ', train_loss_sum_b)
  print('testing loss: ', (loss_sum_f + loss_sum_b) / 2)
  print('training loss: ', (train_loss_sum_f + train_loss_sum_b) / 2)
end

local images
if pm.num_neighbors == 2 then
  images = sample2n()
elseif pm.num_neighbors == 3 then
  images = sample3n()
elseif pm.num_neighbors == 4 then
  images = sample4n()
else
  print('not implemented')
end

-- output the sampled images
--local images_cpu = images:narrow(3, pm.seq_length+1, pm.seq_length)
local images_cpu = images
images_cpu = images_cpu:float():view(batch_size, pm.pixel_size, pm.recurrent_stride, pm.recurrent_stride)
images_cpu = images_cpu[{{}, {}, {patch_size+1, pm.recurrent_stride},{patch_size+1, pm.recurrent_stride}}]
--images_cpu = images_cpu:clamp(0,1):mul(255):type('torch.ByteTensor')
for i=1,batch_size do
  local filename = path.join('samples', i .. '_b.png')
  local im = images_cpu[i]:clone()
  if pm.pixel_size == 3 then
    --local h = im:size(2)
    --local w = im:size(3)
    --im = im:view(3, -1)
    --im = torch.mm(transform.affine, im)
    --im = torch.add(im, torch.repeatTensor(transform.mu, 1, h*w))
    --im = im:view(3, h, w)
    --im = image.yuv2rgb(im)
    im:add(-shift)
  else
    im:add(-shift)
  end
  im = im:clamp(0,1):mul(255):type('torch.ByteTensor')
  image.save(filename, im)
end
