require 'torch'
require 'nn'
require 'nngraph'
require 'image'
-- local imports
require 'pm'
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

-- change it to evaluation mode
local protos = checkpoint.protos
local patch_size = checkpoint.opt['patch_size']
protos.pm.recurrent_stride = patch_size + opt.img_size - 1
protos.pm.seq_length = protos.pm.recurrent_stride * protos.pm.recurrent_stride
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end
local pm = protos.pm
print('The loaded model is trained on patch size with: ', patch_size)

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
local images = torch.Tensor(pm.seq_length, batch_size, pm.pixel_size):cuda()
------------------ debug ------------------------
local img = image.load('imgs/D1.png', pm.pixel_size, 'float')
img = image.scale(img, 256, 256):resize(1, pm.pixel_size, 256, 256)
img = torch.repeatTensor(img, batch_size, 1, 1, 1)
img = img:cuda()

local loss_sum = 0
local train_loss_sum = 0
-- random seed the zero-th pixel
-- local pixel = torch.rand(batch_size, pm.pixel_size):cuda()
local pixel
local gmms
-- loop through each timestep
for h=1,pm.recurrent_stride do
  for w=1,pm.recurrent_stride do
    if w < patch_size or h < patch_size then
      pixel = img[{{}, {}, h, w}]
      images[(h-1)*pm.recurrent_stride+w] = pixel
    else
      local train_pixel = img[{{}, {}, h, w}]
      pixel, loss, train_loss = pm:sample(gmms, train_pixel)
      images[(h-1)*pm.recurrent_stride+w] = pixel
      loss_sum = loss_sum + loss
      train_loss_sum = train_loss_sum + train_loss
      print(train_loss .. '.....' .. loss)
      if w == patch_size and h == patch_size then
        print(train_loss .. '.....' .. loss)
        print(train_pixel)
        print(pixel)
        print(gmms)
        --exit()
        end
    end

    -- inputs to LSTM, {input, states[t, t-1], states[t-1, t] }
    local inputs = {}
    inputs = {pixel, unpack(states[w-1])}
    local prev_w = w
    if states[w] == nil then prev_w = 0 end
    -- insert the states[t-1,t]
    for i,v in ipairs(states[prev_w]) do table.insert(inputs, v) end
    -- forward the network outputs, {next_c, next_h, next_c, next_h ..., output_vec}
    local lsts = pm.core:forward(inputs)
    -- save the state
    states[w] = {}
    for i=1,pm.num_state do table.insert(states[w], lsts[i]) end
    gmms = lsts[#lsts]
    ------------------------ debug ------------------------
  end
  collectgarbage()
end

-- output the sampled images
local images_cpu = images:float()
images_cpu = images_cpu:permute(2,3,1):clone()
images_cpu = images_cpu:view(batch_size, pm.pixel_size, pm.recurrent_stride, pm.recurrent_stride)
--images_cpu = images_cpu[{{}, {}, {patch_size, pm.recurrent_stride},{patch_size, pm.recurrent_stride}}]
images_cpu = images_cpu:clamp(0,1):mul(255):type('torch.ByteTensor')
for i=1,batch_size do
  local filename = path.join('samples', i .. '.png')
  image.save(filename, images_cpu[{i,1,{},{}}])
end

loss_sum = loss_sum / (opt.img_size * opt.img_size)
train_loss_sum = train_loss_sum / (opt.img_size * opt.img_size)
print('testing loss: ', loss_sum)
print('training loss: ', train_loss_sum)
