require 'torch'
require 'nn'
require 'nngraph'
-- local imports
require 'pm'
require 'im'
require 'kld'
local VAE = require 'vae'
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

cmd:option('-phase', 1,'phase 1: learning features, phase 2: learning pixels')
-- Data input settings
cmd:option('-train_path','G/images/2','path to the training data')
cmd:option('-val_path','G/images/2','path to the val data')
cmd:option('-color', 1, 'whether the input image is color image or grayscale image')
--cmd:option('-input_h5','coco/data.h5','path to the h5file containing the preprocessed dataset')
--cmd:option('-input_json','coco/data.json','path to the json file containing additional info and vocab')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Image Model settings
cmd:option('-image_size',72,'resize the image to')
cmd:option('-crop_size',64,'the actually size feeds into the network')
cmd:option('-latent_size',500,'size of top latent representations of VAE')
cmd:option('-feature_size',30,'size of pixel features from VAE')

-- Pixel Model settings
cmd:option('-rnn_size',200,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-num_layers',2,'number of layers in stacked RNN/LSTMs')
cmd:option('-patch_size',16,'size of the neighbor patch that a pixel is conditioned on')
cmd:option('-num_mixtures',20,'number of mixtures used for encoding pixel output')
cmd:option('-border_size',0,'size of the border to ignore, i.e. only the center label will be supervised')
cmd:option('-border_init', 0, 'value to init pixels on the border.')
cmd:option('-input_shift', -0.5, 'shift the input by a constant, should get better performance.')
cmd:option('-num_neighbors', 4, 'Number of neighbors in the pixel model.')
cmd:option('-noise', 0, 'Input perturbation noise.')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-im_batch_size',1,'number of images per batch')
cmd:option('-pm_batch_size', 32,'number of patches per image per batch')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_pm', 0.5, 'strength of dropout in the Pixel Model')
cmd:option('-mult_in', true, 'An extension of the LSTM architecture')
cmd:option('-output_back', true, 'For 4D model, feed the output of the first sweep to the next sweep')
cmd:option('-grad_norm', false, 'whether to normalize the gradients for each direction')
cmd:option('-loss_policy', 'const', 'loss decay policy for spatial patch') -- exp for exponential decay, and linear for linear decay
cmd:option('-loss_decay', 0.9, 'loss decay rate for spatial patch')

-- Optimization: for the Pixel Model
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',1e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 1000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.95,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the VAE
cmd:option('-vae_optim','rmsprop','optimization to use for VAE')
cmd:option('-vae_optim_alpha',0.90,'alpha for momentum of VAE')
cmd:option('-vae_optim_beta',0.999,'beta for momentum of VAE')
cmd:option('-vae_learning_rate',1e-4,'learning rate for the VAE')
cmd:option('-vae_weight_decay', 0, 'L2 weight decay just for the VAE')
cmd:option('-finetune_vae_after', 0, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 5000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '/home/zhirongw/pixel/VAE/models', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 3, 'random number generator seed to use')
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

if opt.phase == 1 then opt.learning_rate = 0 else opt.vae_learning_rate = 0 end
if opt.phase == 1 then opt.patch_size = 1 end
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoaderRaw{folder_path = opt.train_path, img_size = opt.image_size, shift = opt.input_shift, color = opt.color}
local val_loader = DataLoaderRaw{folder_path = opt.val_path, img_size = opt.image_size, shift = opt.input_shift, color = opt.color}

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
  local im_modules = protos.im:getModulesList()
  for k,v in pairs(im_modules) do net_utils.unsanitize_gradients(v) end
  local pm_modules = protos.pm:getModulesList()
  for k,v in pairs(pm_modules) do net_utils.unsanitize_gradients(v) end
  -- initialize the pm
  protos.pm.batch_size = opt.batch_size
  protos.pm.recurrent_stride = opt.patch_size
  protos.pm.seq_length = opt.patch_size * opt.patch_size
  protos.pm:_buildIndex()
  -- initialize the patch extractor
  local psOpt = {}
  psOpt.patch_size = opt.patch_size
  psOpt.feature_dim = opt.feature_size
  psOpt.im_batch_size = opt.im_batch_size
  psOpt.pm_batch_size = opt.pm_batch_size
  psOpt.num_neighbors = opt.num_neighbors
  psOpt.border_size = opt.border_size
  psOpt.border = opt.border_init
  psOpt.noise = opt.noise
  protos.patch_extractor = nn.PatchExtractor(psOpt)

  protos.imcrit = nn.KLDCriterion()
  if opt.phase == 1 then
    protos.pmcrit = nn.MSECriterion()
  else
    protos.pmcrit = nn.MSECriterion()
  --protos.pmcrit = nn.PixelModelCriterion(protos.pm.pixel_size, opt.num_mixtures,
  --              {policy=opt.loss_policy, val=opt.loss_decay})
  end
  iter = loaded_checkpoint.iter
else
  -- create protos from scratch
  -- initialize image VAE
  local imOpt = {}
  imOpt.latent_variable_size = opt.latent_size
  imOpt.feature_size = opt.feature_size
  protos.im = nn.VAEModel(imOpt)
  -- initialize pixel model
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
  pmOpt.output_back = opt.output_back
  pmOpt.feature_dim = opt.feature_size
  if pmOpt.num_neighbors == 3 then
    protos.pm = nn.PixelModel3N(pmOpt)
  elseif pmOpt.num_neighbors == 4 then
    protos.pm = nn.PixelModel4N(pmOpt)
  else
    print('undefined')
  end
  -- initialize the patch extractor
  local psOpt = {}
  psOpt.patch_size = opt.patch_size
  psOpt.feature_dim = opt.feature_size
  psOpt.im_batch_size = opt.im_batch_size
  psOpt.pm_batch_size = opt.pm_batch_size
  psOpt.num_neighbors = opt.num_neighbors
  psOpt.border_size = opt.border_size
  psOpt.border = opt.border_init
  psOpt.noise = opt.noise
  protos.patch_extractor = nn.PatchExtractor(psOpt)
  -- criterion for the pixel model
  protos.imcrit = nn.KLDCriterion()
  if opt.phase == 1 then
    protos.pmcrit = nn.MSECriterion()
  else
    protos.pmcrit = nn.MSECriterion()
  --protos.pmcrit = nn.PixelModelCriterion(protos.pm.pixel_size, opt.num_mixtures,
  --              {policy=opt.loss_policy, val=opt.loss_decay})
  end
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
print('Input pixel shift: ', opt.input_shift)
print('Input border init: ', opt.border_init)
print('Training im batch size: ', opt.im_batch_size)
print('Training pm batch size: ', opt.pm_batch_size)
-- flatten and prepare all model parameters to a single vector.
local params, grad_params = protos.pm:getParameters()
local vae_params, vae_grad_params = protos.im:getParameters()
print('total number of parameters in PM: ', params:nElement())
print('total number of parameters in VAE: ', vae_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(vae_params:nElement() == vae_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_pm = protos.pm:clone()
thin_pm.core:share(protos.pm.core, 'weight', 'bias') -- TODO: we are assuming that PM has specific members! figure out clean way to get rid of, not modular.
local thin_im = protos.im:clone()
thin_im.encoder:share(protos.im.encoder, 'weight', 'bias', 'running_mean', 'running_var') -- TODO: we are assuming that IM has specific members! figure out clean way to get rid of, not modular.
thin_im.decoder:share(protos.im.decoder, 'weight', 'bias', 'running_mean', 'running_var') -- TODO: we are assuming that IM has specific members! figure out clean way to get rid of, not modular.
-- sanitize all modules of gradient storage so that we dont save big checkpoints
local pm_modules = thin_pm:getModulesList()
for k,v in pairs(pm_modules) do net_utils.sanitize_gradients(v) end
local im_modules = thin_im:getModulesList()
for k,v in pairs(im_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.pm:createClones()

collectgarbage() -- "yeah, sure why not"

batch_size = opt.im_batch_size * opt.pm_batch_size
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(n)
  protos.im:evaluate()
  protos.pm:evaluate()
  --loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local loss1_sum = 0
  local loss2_sum = 0
  local i = 0
  while i < n do

    -- fetch a batch of data
    local images = val_loader:getBatch{batch_size = opt.im_batch_size,
                                crop_size = opt.crop_size, gpu = opt.gpuid}

    -- 1. forward the image model
    local features, dummy, mean, log_var = unpack(protos.im:forward(images))
    -- 2. extract patches
    local targets, patches = unpack(protos.patch_extractor:forward({images, features}))
    -- 3. forward the pixel model
    local pred = protos.pm:forward(patches)
    --print(gmms)
    pred = pred:view(opt.patch_size, opt.patch_size, batch_size, -1)
    local lpred = pred[{{opt.border_size+1,opt.patch_size-opt.border_size},{opt.border_size+1,opt.patch_size-opt.border_size},{},{}}]

    local loss1 = protos.imcrit:forward(mean, log_var)
    local loss2 = protos.pmcrit:forward(lpred, targets)
    loss1_sum = loss1_sum + loss1
    loss2_sum = loss2_sum + loss2

    i = i + 1
    if i % 10 == 0 then collectgarbage() end
  end

  return {kld_loss = loss1_sum/n, pixel_loss = loss2_sum/n}
end

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_splitPre(n)
  protos.im:evaluate()
  protos.pm:evaluate()
  --loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local loss1_sum = 0
  local loss2_sum = 0
  local i = 0
  while i < n do

    -- fetch a batch of data
    local images = val_loader:getBatch{batch_size = opt.im_batch_size,
                                crop_size = opt.crop_size, gpu = opt.gpuid}

    -- 1. forward the image model
    local features, pred, mean, log_var = unpack(protos.im:forward(images))
    local loss1 = protos.imcrit:forward(mean, log_var)
    local loss2 = protos.pmcrit:forward(pred, images)
    loss1_sum = loss1_sum + loss1
    loss2_sum = loss2_sum + loss2

    i = i + 1
    if i % 10 == 0 then collectgarbage() end

    pred = pred:add(-opt.input_shift)
    images:add(-opt.input_shift)
    for t=1,opt.im_batch_size do
      local im = pred[t]:clamp(0,1):mul(255):type('torch.ByteTensor')
      image.save("G/"..t.."_gen.png", im)
      local pic = images[t]:clamp(0,1):mul(255):type('torch.ByteTensor')
      image.save("G/"..t.."_in.png", pic)
    end

  end

  return {kld_loss = loss1_sum/n, pixel_loss = loss2_sum/n}
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  protos.im:training()
  protos.pm:training()
  vae_grad_params:zero()
  grad_params:zero()
  print(params[1])
  print(vae_params[1])
  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  --local timer = torch.Timer()
  local images = loader:getBatch{batch_size = opt.im_batch_size,
                              crop_size = opt.crop_size, gpu = opt.gpuid}
  -- 1. forward the image model
  local features, dummy, mean, log_var = unpack(protos.im:forward(images))
  -- 2. extract patches
  local targets, patches = unpack(protos.patch_extractor:forward({images, features}))
  -- 3. forward the pixel model
  local pred = protos.pm:forward(patches)
  --print('Forward time: ' .. timer:time().real .. ' seconds')
  -- forward the pixel model criterion
  pred = pred:view(opt.patch_size, opt.patch_size, batch_size, -1)
  local lpred = pred[{{opt.border_size+1,opt.patch_size-opt.border_size},{opt.border_size+1,opt.patch_size-opt.border_size},{},{}}]

  -----------------------------------------------------------------------------
  -- Loss
  -----------------------------------------------------------------------------
  local loss1 = protos.imcrit:forward(mean, log_var)
  local loss2 = protos.pmcrit:forward(lpred, targets)
  local dmean, dlog_var = unpack(protos.imcrit:backward(mean, log_var))
  local dlpred = protos.pmcrit:backward(lpred, targets)
  --print('Criterion time: ' .. timer:time().real .. ' seconds')

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dpred = torch.Tensor(pred:size()):type(pred:type()):fill(0)
  dpred[{{opt.border_size+1,opt.patch_size-opt.border_size},{opt.border_size+1,opt.patch_size-opt.border_size},{},{}}] = dlpred
  dpred = dpred:view(-1, batch_size, protos.pm.pixel_size)
  -- 3. backprop pixel model
  local dpatches = protos.pm:backward(patches, dpred)
  -- 2. backprop patch extractor
  local x, dfeatures = unpack(protos.patch_extractor:backward({images, features},{x, dpatches}))
  -- 1. backprop image model
  local dimages = protos.im:backward(images, {dfeatures, torch.zeros(dummy:size()):type(dummy:type()), dmean, dlog_var})
  --print('Backward time: ' .. timer:time().real .. ' seconds')
  -- normalize the gradients for different directions
  if opt.grad_norm then
    protos.pm:norm_grad(grad_params)
  end
  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  -- apply L2 regularization
  if opt.vae_weight_decay > 0 then
    vae_grad_params:add(opt.vae_weight_decay, vae_params)
  end
  vae_grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -----------------------------------------------------------------------------
  -- and lets get out!
  local losses = { kld_loss = loss1, pixel_loss = loss2 }
  return losses
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFunPre()
  protos.im:training()
  protos.pm:training()
  vae_grad_params:zero()
  grad_params:zero()
  print(vae_params[1])
  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  --local timer = torch.Timer()
  local images = loader:getBatch{batch_size = opt.im_batch_size,
                              crop_size = opt.crop_size, gpu = opt.gpuid}
  -- forward the image model
  local features, pred, mean, log_var = unpack(protos.im:forward(images))
  -----------------------------------------------------------------------------
  -- Loss
  -----------------------------------------------------------------------------
  local loss1 = protos.imcrit:forward(mean, log_var)
  local loss2 = protos.pmcrit:forward(pred, images)
  local dmean, dlog_var = unpack(protos.imcrit:backward(mean, log_var))
  local dpred = protos.pmcrit:backward(pred, images)
  --print('Criterion time: ' .. timer:time().real .. ' seconds')
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop image model
  local dfeatures = torch.zeros(features:size()):type(features:type())
  local dimages = protos.im:backward(images, {dfeatures, dpred, dmean, dlog_var})
  --print('Backward time: ' .. timer:time().real .. ' seconds')
  -- normalize the gradients for different directions
  if opt.grad_norm then
    protos.pm:norm_grad(grad_params)
  end
  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  -- apply L2 regularization
  if opt.vae_weight_decay > 0 then
    vae_grad_params:add(opt.vae_weight_decay, vae_params)
  end
  vae_grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -----------------------------------------------------------------------------
  -- and lets get out!
  local losses = { kld_loss = loss1, pixel_loss = loss2 }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local vae_optim_state = {}
local loss_history = {}
local val_loss_history = {}
local best_score
while true do
  iter = iter + 1

  -- decay the learning rate
  local learning_rate = opt.learning_rate
  local vae_learning_rate = opt.vae_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    vae_learning_rate = vae_learning_rate * decay_factor
  end

  -- eval loss/gradient
  local losses
  if opt.phase == 1 then losses = lossFunPre() else losses = lossFun() end
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses end
  print(string.format('iter %d: %f = %f + %f. LR: %f', iter, losses.kld_loss+losses.pixel_loss, losses.pixel_loss, losses.kld_loss, learning_rate))

  -- save checkpoint once in a while (or on final iteration)
  if ((opt.save_checkpoint_every > 0 and iter % opt.save_checkpoint_every == 0) or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss
    if opt.phase == 1 then val_loss = eval_splitPre(1) else val_loss = eval_split(1) end
    print(string.format('validation loss: %f = %f + %f.', val_loss.kld_loss+val_loss.pixel_loss, val_loss.pixel_loss, val_loss.kld_loss))
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
    save_protos.im = thin_im -- these are shared clones, and point to correct param storage
    checkpoint.protos = save_protos
    torch.save(checkpoint_path .. '.t7', checkpoint)
    print('wrote training checkpoint to ' .. checkpoint_path .. '.t7')

    -- utils.write_json(checkpoint_path .. '.json', checkpoint)
    -- print('wrote json checkpoint to ' .. checkpoint_path .. '.json')
  end

  if opt.learning_rate > 0 then
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
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.vae_learning_rate > 0 and opt.finetune_vae_after >= 0 and iter >= opt.finetune_vae_after then
    if opt.vae_optim == 'rmsprop' then
      rmsprop(vae_params, vae_grad_params, vae_learning_rate, opt.vae_optim_alpha,opt.optim_epsilon, vae_optim_state)
    elseif opt.vae_optim == 'adagrad' then
      adagrad(vae_params, vae_grad_params, vae_learning_rate, opt.optim_epsilon, vae_optim_state)
    elseif opt.vae_optim == 'sgd' then
      sgd(vae_params, vae_grad_params, vae_learning_rate)
    elseif opt.vae_optim == 'sgdm' then
      sgdm(vae_params, vae_grad_params, vae_learning_rate, opt.vae_optim_alpha, vae_optim_state)
    elseif opt.vae_optim == 'adam' then
      adam(vae_params, vae_grad_params, vae_learning_rate, opt.vae_optim_alpha, opt.vae_optim_beta, opt.optim_epsilon, vae_optim_state)
    else
      error('bad option for opt.vae_optim')
    end
  end

  -- stopping criterions
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.kld_loss + losses.pixel_loss end
  if losses.kld_loss+losses.pixel_loss > loss0 * 2000 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
