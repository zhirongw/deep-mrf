require 'torch'
require 'nn'
require 'nngraph'
-- local imports
require 'pm'
require 'im'
require 'kld'
require 'rgb'
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

--cmd:option('-phase', 2,'phase 1: learning features, phase 2: learning pixels')
-- Data input settings
cmd:option('-train_path','G/images/valley','path to the training data')
cmd:option('-val_path','G/images/valley','path to the val data')
cmd:option('-color', 1, 'whether the input image is color image or grayscale image')
--cmd:option('-input_h5','coco/data.h5','path to the h5file containing the preprocessed dataset')
--cmd:option('-input_json','coco/data.json','path to the json file containing additional info and vocab')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Image Model settings
cmd:option('-image_size',64,'resize the image to')
cmd:option('-crop_size',64,'the actually size feeds into the network')
cmd:option('-latent_size',20,'size of top latent representations of VAE')
cmd:option('-feature_size',30,'size of pixel features from VAE')

-- Pixel Model settings
cmd:option('-rnn_size',200,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rgb_rnn_size',200,'size of the rgb rnn')
cmd:option('-num_layers',2,'number of layers in stacked RNN/LSTMs')
cmd:option('-rgb_num_layers',1,'number of layers in rgb RNN/LSTMs')
cmd:option('-rgb_encoding_size',100,'output of the pixel rnn encoding RGB infos')
cmd:option('-patch_size',16,'size of the neighbor patch that a pixel is conditioned on')
cmd:option('-num_mixtures',10,'number of mixtures used for encoding pixel output')
cmd:option('-border_size',0,'size of the border to ignore, i.e. only the center label will be supervised')
cmd:option('-border_init', 0, 'value to init pixels on the border.')
cmd:option('-input_shift', -0.5, 'shift the input by a constant, should get better performance.')
cmd:option('-num_neighbors', 4, 'Number of neighbors in the pixel model.')
cmd:option('-noise', 0, 'Input perturbation noise.')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-im_batch_size', 4,'number of images per batch')
cmd:option('-pm_batch_size', 16,'number of patches per image per batch')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_pm', 0, 'strength of dropout in the Pixel Model')
cmd:option('-mult_in', true, 'An extension of the LSTM architecture')
cmd:option('-output_back', false, 'For 4D model, feed the output of the first sweep to the next sweep')
cmd:option('-grad_norm', false, 'whether to normalize the gradients for each direction')
cmd:option('-loss_policy', 'const', 'loss decay policy for spatial patch') -- exp for exponential decay, and linear for linear decay
cmd:option('-loss_decay', 0.9, 'loss decay rate for spatial patch')

-- Optimization: for the Pixel Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 1000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.90,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the VAE
cmd:option('-vae_optim','adam','optimization to use for VAE')
cmd:option('-vae_optim_alpha',0.90,'alpha for momentum of VAE')
cmd:option('-vae_optim_beta',0.999,'beta for momentum of VAE')
cmd:option('-weight_decay', 0, 'L2 weight decay')
cmd:option('-finetune_vae_after', 0, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 2000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '/home/zhirongw/pixel/VAE/models', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 3, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-ngpu', 1, 'number of gpus to run')

cmd:text()

torch.setheaptracking(true)
-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  local DPT = nn.DataParallelTable
  function DPT:collectgarbage()
    self.impl:exec(function(m, i)
        collectgarbage()
    end)
  end

  if opt.backend == 'cudnn' then require 'cudnn' end
  if opt.ngpu == 1 then
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
  end
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoaderRaw{folder_path = opt.train_path, img_size = opt.image_size, shift = opt.input_shift, color = opt.color}
local val_loader = DataLoaderRaw{folder_path = opt.val_path, img_size = opt.image_size, shift = opt.input_shift, color = opt.color}
opt.data_info = loader:getChannelScale()

-------------------------------------------------------------------------------
-- Definition of the joint model

-------------------------------------------------------------------------------
local function VAEMRF(opt)
  -- params for VAE
  local imOpt = {}
  imOpt.latent_variable_size = opt.latent_size
  imOpt.feature_size = opt.feature_size
  -- params for MRF
  local pmOpt = {}
  pmOpt.pixel_size = loader:getChannelSize()
  pmOpt.rnn_size = opt.rnn_size
  pmOpt.num_layers = opt.num_layers
  pmOpt.dropout = opt.drop_prob_pm
  pmOpt.batch_size = opt.batch_size
  pmOpt.encoding_size = opt.rgb_encoding_size
  pmOpt.recurrent_stride = opt.patch_size
  pmOpt.seq_length = opt.patch_size * opt.patch_size
  pmOpt.mult_in = opt.mult_in
  pmOpt.num_neighbors = opt.num_neighbors
  pmOpt.border_init = opt.border_init
  pmOpt.output_back = opt.output_back
  pmOpt.feature_dim = opt.feature_size
  -- params for the patch extractor
  local psOpt = {}
  psOpt.patch_size = opt.patch_size
  psOpt.feature_dim = opt.feature_size
  psOpt.im_batch_size = opt.im_batch_size
  psOpt.pm_batch_size = opt.pm_batch_size
  psOpt.num_neighbors = opt.num_neighbors
  psOpt.border_size = opt.border_size
  psOpt.border = opt.border_init
  psOpt.noise = opt.noise
  -- params for the rgb model
  local rgbOpt = {}
  rgbOpt.rnn_size = opt.rgb_rnn_size
  rgbOpt.num_layers = opt.rgb_num_layers
  rgbOpt.num_mixtures = opt.num_mixtures
  rgbOpt.seq_length = 3
  rgbOpt.mult_in = false
  rgbOpt.encoding_size = opt.rgb_encoding_size

  ----------------------------------------------------------------------
  local protos = {}
  local input = nn.Identity()()
  local VAE_output = nn.VAEModel(imOpt)(input)
  local features = nn.SelectTable(1)(VAE_output)
  local mean = nn.SelectTable(2)(VAE_output)
  local log_var = nn.SelectTable(3)(VAE_output)
  local PE = nn.PatchExtractor(psOpt)({input, features})
  local targets = nn.SelectTable(1)(PE)
  local patches = nn.SelectTable(2)(PE)
  local rgb_features
  if pmOpt.num_neighbors == 3 then
    rgb_features = nn.PixelModel3N(pmOpt)(patches)
  elseif pmOpt.num_neighbors == 4 then
    rgb_features = nn.PixelModel4N(pmOpt)(patches)
  else
    error('undefined deep mrf')
  end
  rgb_features = nn.View(-1, opt.rgb_encoding_size)(rgb_features)
  local gmms = nn.RGBModel(rgbOpt)({rgb_features, targets})

  local model = nn.gModule({input},{gmms, targets, mean, log_var})

  if opt.ngpu > 1 then
    local gpus = torch.range(1, opt.ngpu):totable()
    local fastest, benchmark = cudnn.fastest, cudnn.benchmark

    local dpt = nn.DataParallelTable(1, true, true)
       :add(model, gpus)
       :threads(function()
          require 'nngraph'
          require 'im' require 'pm' require 'rgb'
          local cudnn = require 'cudnn'
          cudnn.fastest, cudnn.benchmark = fastest, benchmark
       end)
    dpt.gradInput = nil

    model = dpt:cuda()
   end

   return model
end
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
  -- initialize pixel model
  local pmOpt = {}
  pmOpt.pixel_size = loader:getChannelSize()
  pmOpt.rnn_size = opt.rnn_size
  pmOpt.num_layers = opt.num_layers
  pmOpt.dropout = opt.drop_prob_pm
  pmOpt.batch_size = opt.batch_size
  pmOpt.encoding_size = opt.rgb_encoding_size
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

  local rgbOpt = {}
  rgbOpt.rnn_size = opt.rgb_rnn_size
  rgbOpt.num_layers = opt.rgb_num_layers
  rgbOpt.num_mixtures = opt.num_mixtures
  rgbOpt.seq_length = 3
  rgbOpt.mult_in = false
  rgbOpt.encoding_size = opt.rgb_encoding_size
  protos.rgb = nn.RGBModel(rgbOpt)

  iter = loaded_checkpoint.iter
else
  -- create protos from scratch
  protos.model = VAEMRF(opt)
end

-- Criterions
protos.imcrit = nn.KLDCriterion()
protos.pmcrit = nn.PixelModelCriterion(1, opt.num_mixtures,
  {policy=opt.loss_policy, val=opt.loss_decay})

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

print('Training a 2D LSTM with number of layers: ', opt.num_layers)
print('Hidden nodes in each layer: ', opt.rnn_size)
print('Pixel encoding size: ', opt.rgb_encoding_size)
print('RGB rnn number of layers: ', opt.rgb_num_layers)
print('RGB rnn size: ', opt.rgb_rnn_size)
print('VAE latent size: ', opt.latent_size)
print('context feature size: ', opt.feature_size)
print('number of mixtures: ', opt.num_mixtures)
print('The input image local patch size: ', opt.patch_size)
print('Ignoring the border of size: ', opt.border_size)
print('Input channel dimension: ', loader:getChannelSize())
print('Input pixel shift: ', opt.input_shift)
print('Input border init: ', opt.border_init)
print('Training im batch size: ', opt.im_batch_size)
print('Training pm batch size: ', opt.pm_batch_size)
print('Save checkpoint to: ', opt.checkpoint_path)

-- flatten and prepare all model parameters to a single vector.
local params, grad_params = protos.model:getParameters()
print('total number of parameters in VAEMRF: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

protos.model:training()
batch_size = opt.im_batch_size * opt.pm_batch_size
collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(n)
  protos.model:evaluate()
  --loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local loss1_sum = 0
  local loss2_sum = 0
  local i = 0
  while i < n do

    -- fetch a batch of data
    local images = val_loader:getBatch{batch_size = opt.im_batch_size,
                                crop_size = opt.crop_size, gpu = opt.gpuid}

    local gmms, targets, mean, log_var = table.unpack(protos.model:forward(images))
    --print(gmms)
    --pred = pred:view(opt.patch_size, opt.patch_size, batch_size, -1)
    --local lpred = pred[{{opt.border_size+1,opt.patch_size-opt.border_size},{opt.border_size+1,opt.patch_size-opt.border_size},{},{}}]
    local loss1 = protos.imcrit:forward(mean, log_var)
    local loss2 = protos.pmcrit:forward(gmms, targets)
    loss1_sum = loss1_sum + loss1
    loss2_sum = loss2_sum + loss2

    i = i + 1
    if i % 10 == 0 then collectgarbage() end
  end
  protos.model:training()
  return {kld_loss = loss1_sum/n, pixel_loss = loss2_sum/n}
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  grad_params:zero()
  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  --local timer = torch.Timer()
  local images = loader:getBatch{batch_size = opt.im_batch_size,
                              crop_size = opt.crop_size, gpu = opt.gpuid}
  local gmms, targets, mean, log_var = table.unpack(protos.model:forward(images))
  --print('Forward time: ' .. timer:time().real .. ' seconds')
  -----------------------------------------------------------------------------
  -- Loss
  -----------------------------------------------------------------------------
  local loss1 = protos.imcrit:forward(mean, log_var)
  local loss2 = protos.pmcrit:forward(gmms, targets)
  local dmean, dlog_var = table.unpack(protos.imcrit:backward(mean, log_var))
  local dgmms = protos.pmcrit:backward(gmms, targets)
  --print('Criterion time: ' .. timer:time().real .. ' seconds')
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  protos.model:backward(images, {dgmms, torch.zeros(targets:size()):cuda(), dmean, dlog_var})
  --print('Backward time: ' .. timer:time().real .. ' seconds')
  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  -- apply L2 regularization
  if opt.weight_decay > 0 then
    grad_params:add(opt.weight_decay, params)
  end
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
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses end
  print(string.format('iter %d: %f = %f + %f. LR: %f', iter, losses.kld_loss+losses.pixel_loss, losses.pixel_loss, losses.kld_loss, learning_rate))

  -- save checkpoint once in a while (or on final iteration)
  if ((opt.save_checkpoint_every > 0 and iter % opt.save_checkpoint_every == 0) or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss = eval_split(1)
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
    local save_model = protos.model
    if torch.type(save_model) == 'nn.DataParallelTable' then
      save_model = save_model:get(1)
    end
    save_protos.model = save_model -- these are shared clones, and point to correct param storage
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

  -- stopping criterions
  if iter % 10 == 0 then collectgarbage() if opt.ngpu>1 then protos.model:collectgarbage() end end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.kld_loss + losses.pixel_loss end
  if losses.kld_loss+losses.pixel_loss > loss0 * 2000 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
