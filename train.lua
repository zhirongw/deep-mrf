require 'torch'
require 'nn'
require 'nngraph'
-- local imports
require 'pm'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'misc.DataLoaderRaw'
--require 'misc.DataLoader'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Generating textures & images pixels by pixels')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-train_path','../../pixel/SR/train_ycb.mat','path to the training data')
cmd:option('-val_path','../../pixel/SR/val_ycb.mat','path to the val data')
cmd:option('-color', 1, 'whether the input image is color image or grayscale image')
--cmd:option('-input_h5','coco/data.h5','path to the h5file containing the preprocessed dataset')
--cmd:option('-input_json','coco/data.json','path to the json file containing additional info and vocab')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size',200,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-num_layers',2,'number of layers in stacked RNN/LSTMs')
cmd:option('-patch_size',15,'size of the neighbor patch that a pixel is conditioned on')
cmd:option('-border_size',0,'size of the border to ignore, i.e. only the center label will be supervised')
cmd:option('-border_init', 1, 'ways to initialize the border, 0 for zeros, 1 for random.')
cmd:option('-num_neighbors', 4, 'Number of neighbors in the pixel model.')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',32,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_pm', 0.5, 'strength of dropout in the Pixel Model')
cmd:option('-mult_in', true, 'An extension of the LSTM architecture')
cmd:option('-loss_policy', 'exp', 'loss decay policy for spatial patch') -- exp for exponential decay, and linear for linear decay
cmd:option('-loss_decay', 0.9, 'loss decay rate for spatial patch')
-- Optimization: for the Pixel Model
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',1e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 500, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.90,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 2000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'SR/models', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
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
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoaderRaw{folder_path = opt.train_path, color = opt.color}
local val_loader = DataLoaderRaw{folder_path = opt.val_path, color = opt.color}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}
local iter = 0
if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  local pm_modules = protos.pm:getModulesList()
  for k,v in pairs(pm_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.MSECriterion()
  iter = loaded_checkpoint.iter
else
  -- create protos from scratch
  -- intialize pixel model
  local pmOpt = {}
  pmOpt.pixel_size = loader:getChannelSize()
  pmOpt.rnn_size = opt.rnn_size
  pmOpt.num_layers = opt.num_layers
  pmOpt.dropout = opt.drop_prob_pm
  pmOpt.batch_size = opt.batch_size
  pmOpt.recurrent_stride = opt.patch_size
  pmOpt.seq_length = opt.patch_size * opt.patch_size
  pmOpt.mult_in = opt.mult_in
  pmOpt.num_neighbors = opt.num_neighbors
  pmOpt.border_init = opt.border_init
  if pmOpt.num_neighbors == 3 then
    protos.pm = nn.PixelModel3N(pmOpt)
  elseif pmOpt.num_neighbors == 4 then
    protos.pm = nn.PixelModel4N(pmOpt)
  else
    print('undefined')
  end
  -- criterion for the pixel model
  protos.crit = nn.MSECriterion()
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

print('Training a 2D LSTM with number of layers: ', opt.num_layers)
print('Hidden nodes in each layer: ', opt.rnn_size)
print('The input image local patch size: ', opt.patch_size)
print('Ignoring the border of size: ', opt.border_size)
print('Input channel dimension: ', loader:getChannelSize())
print('Training batch size: ', opt.batch_size)
-- flatten and prepare all model parameters to a single vector.
local params, grad_params = protos.pm:getParameters()
print('total number of parameters in PM: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_pm = protos.pm:clone()
thin_pm.core:share(protos.pm.core, 'weight', 'bias') -- TODO: we are assuming that PM has specific members! figure out clean way to get rid of, not modular.
-- sanitize all modules of gradient storage so that we dont save big checkpoints
local pm_modules = thin_pm:getModulesList()
for k,v in pairs(pm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.pm:createClones()

collectgarbage() -- "yeah, sure why not"
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(n)
  protos.pm:evaluate()
  --loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local loss_sum = 0
  local i = 0
  while i < n do

    -- fetch a batch of data
    local data = val_loader:getBatch{batch_size = opt.batch_size, num_neighbors = opt.num_neighbors,
                                patch_size = opt.patch_size, border_size = opt.border_size, gpu = opt.gpuid,
                                border = opt.border_init}

    -- forward the model to get loss
    local pred = protos.pm:forward(data.pixels)
    --print(gmms)
    pred = pred:view(opt.patch_size, opt.patch_size, opt.num_neighbors-2, opt.batch_size, -1)
    local lpred = pred[{{opt.border_size+1,opt.patch_size-opt.border_size},{opt.border_size+1,opt.patch_size-opt.border_size},{},{},{}}]
    local loss = protos.crit:forward(lpred, data.targets)
    loss_sum = loss_sum + loss

    i = i + 1
    if i % 10 == 0 then collectgarbage() end
  end

  return loss_sum/n
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  protos.pm:training()
  grad_params:zero()

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  --local timer = torch.Timer()
  local data = loader:getBatch{batch_size = opt.batch_size, num_neighbors = opt.num_neighbors,
                              patch_size = opt.patch_size, border_size = opt.border_size, gpu = opt.gpuid,
                              border = opt.border_init}

  -- forward the pixel model
  local pred = protos.pm:forward(data.pixels)
  --print('Forward time: ' .. timer:time().real .. ' seconds')
  -- forward the pixel model criterion
  pred = pred:view(opt.patch_size, opt.patch_size, opt.num_neighbors-2, opt.batch_size, -1)
  local lpred = pred[{{opt.border_size+1,opt.patch_size-opt.border_size},{opt.border_size+1,opt.patch_size-opt.border_size},{},{},{}}]
  local loss = protos.crit:forward(lpred, data.targets)

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlpred = protos.crit:backward(lpred, data.targets)
  local dpred = torch.Tensor(pred:size()):type(pred:type()):fill(0)
  dpred[{{opt.border_size+1,opt.patch_size-opt.border_size},{opt.border_size+1,opt.patch_size-opt.border_size},{},{},{}}] = dlpred
  dpred = dpred:view(-1, opt.batch_size, protos.pm.pixel_size)
  --print('Criterion time: ' .. timer:time().real .. ' seconds')
  -- backprop pixel model
  local dpixels = protos.pm:backward(data.pixels, dpred)
  --print('Backward time: ' .. timer:time().real .. ' seconds')

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -----------------------------------------------------------------------------
  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local loss_history = {}
local val_loss_history = {}
local best_score
while true do
  iter = iter + 1

  -- decay the learning rate
  local learning_rate = opt.learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f. LR: %f', iter, losses.total_loss, learning_rate))

  -- save checkpoint once in a while (or on final iteration)
  if ((opt.save_checkpoint_every > 0 and iter % opt.save_checkpoint_every == 0) or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss = eval_split(14)
    print('validation loss: ', val_loss)
    -- val_loss_history[iter] = val_loss

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id .. iter)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    -- checkpoint.val_loss_history = val_loss_history
    -- checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    -- include the protos (which have weights) and save to file
    local save_protos = {}
    save_protos.pm = thin_pm -- these are shared clones, and point to correct param storage
    checkpoint.protos = save_protos
    torch.save(checkpoint_path .. '.t7', checkpoint)
    print('wrote checkpoint to ' .. checkpoint_path .. '.t7')

    -- utils.write_json(checkpoint_path .. '.json', checkpoint)
    -- print('wrote json checkpoint to ' .. checkpoint_path .. '.json')
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- stopping criterions
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
