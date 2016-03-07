require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'lstm'
local mvn = require 'misc.mvn'

-------------------------------------------------------------------------------
-- Pixel Model Mixture of Gaussian Density Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.PixelModelCriterion', 'nn.Criterion')
function crit:__init(pixel_size, num_mixtures, opt)
  parent.__init(self)
  self.pixel_size = pixel_size
  self.num_mixtures = num_mixtures
  self.output_size = self.num_mixtures * (1+1+0+1)
  self.w_softmax = nn.SoftMax()
  self.var_exp = nn.Exp()
  self.opt = opt
end

-- only runs in the first batch
function crit:_createLossWeights(input)
    local D = input:size(1)
    local N = input:size(2)
    local nm = self.num_mixtures
    -- here we assume that D is a square number
    local L = math.sqrt(D)
    if self.opt.policy == 'exp' then
      local w = torch.cpow(torch.Tensor(L):fill(self.opt.val), torch.range(L-1,0,-1))
      w = w:resize(L, 1, 1, 1)
      w = torch.repeatTensor(w, 1, L, N, nm)
      self.LW = w:view(D, N, nm)
    elseif self.opt.policy == 'linear' then
      local w = torch.range(1, L):mul(1/L)
      w = w:resize(L, 1, 1, 1)
      w = torch.repeatTensor(w, 1, L, N, nm)
      self.LW = w:view(D, N, nm)
    else -- constant
      self.LW = torch.Tensor(D, N, nm):fill(1.0)
    end
    self.LW = self.LW:type(input:type())
end
--[[
-- this is an optimized version of the gmm loss, though looks ugly though
inputs:
  input is a Tensor of size DxNx(G), encodings of the gmms
  target is a Tensor of size DxNx(M+1).
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
  local D_ = input:size(1)
  local N_ = input:size(2)
  local D,N,Mp1= target:size(1), target:size(2), target:size(3)
  local ps = Mp1 -- pixel size
  assert(D == D_, 'input Tensor should have the same sequence length as the target')
  assert(N == N_, 'input Tensor should have the same batch size as the target')
  assert(ps == self.pixel_size, 'input dimensions of pixel do not match')
  local nm = self.num_mixtures
  if self.LW == nil or self.LW:size(2) ~= D then self:_createLossWeights(input) end

  -- decode the gmms first
  -- mean undertake no changes
  local g_mean_input = input:narrow(3,1,nm*ps):clone()
  local g_mean = g_mean_input:view(D, N, nm, ps)
  local g_mean_diff = torch.repeatTensor(target:view(D, N, 1, ps), 1,1,nm,1):add(-1, g_mean)
  g_mean_diff = g_mean_diff:view(-1, ps, 1)
  -- we use 6 numbers to denote the cholesky depositions
  local g_var_input = input:narrow(3, nm*ps+1, nm*ps):clone()
  g_var_input = g_var_input:view(-1, ps)
  local g_var = self.var_exp:forward(g_var_input)

  local g_cov_input
  local g_clk
  local g_clk_T
  local g_sigma
  local g_sigma_inv
  p = 2
  if ps == 3 then
    g_cov_input = input:narrow(3, p*nm*ps+1, nm*ps):clone()
    g_cov_input = g_cov_input:view(-1, 3)
    g_cov_input = g_cov_input:mul(0.000)
    p = p + 1
    g_clk = torch.Tensor(D*N*nm, 3, 3):fill(0):type(g_var_input:type())
    g_clk[{{}, 1, 1}] = g_var[{{}, 1}]
    g_clk[{{}, 2, 2}] = g_var[{{}, 2}]
    g_clk[{{}, 3, 3}] = g_var[{{}, 3}]
    g_clk[{{}, 2, 1}] = g_cov_input[{{}, 1}]
    g_clk[{{}, 3, 1}] = g_cov_input[{{}, 2}]
    g_clk[{{}, 3, 2}] = g_cov_input[{{}, 3}]
    g_clk_T = g_clk:transpose(2,3)
    g_sigma = self.var_mm:forward({g_clk, g_clk_T})
  else
    g_clk = g_var
    g_clk_T = g_var
    g_sigma = torch.cmul(g_clk, g_clk_T):view(-1,1,1)
  end
  -- weights coeffs is taken care of at final loss, for computation efficiency and stability
  local g_w_input = input:narrow(3,p*nm*ps+1, nm):clone()
  local g_w = self.w_softmax:forward(g_w_input:view(-1, nm))

  local g_rpb
  if ps == 1 then
    g_rpb = mvn.bnormpdf(g_mean_diff, g_clk):cmul(g_w)
    g_sigma_inv = torch.Tensor(g_sigma:size()):type(g_sigma:type())
    g_sigma_inv:fill(1):cdiv(g_sigma)
  else
    local g_clk_inv = mvn.btmi(g_clk)
    local g_clk_T_inv = g_clk_inv:transpose(2,3)
    g_sigma_inv = torch.bmm(g_clk_T_inv, g_clk_inv)
    g_rpb = mvn.b3normpdf(g_mean_diff, g_clk_inv):cmul(g_w)
  end
  g_rpb = g_rpb:view(D, N, nm)
  local pdf = torch.cmax(torch.sum(g_rpb, 3), 1e-40)
  --local pdf = torch.sum(g_rpb, 3)

  -- do the loss the gradients
  local loss = - torch.sum(torch.cmul(torch.log(pdf), (self.LW[{{},{},1}]))) -- loss of pixels, Mixture of Gaussians
  local loss_rgb = - torch.squeeze(torch.mean(torch.log(pdf),2))
  print(loss_rgb[1], loss_rgb[2], loss_rgb[3])

  g_rpb = g_rpb:cmul(self.LW)
  local grad_g_w = - torch.cdiv(g_rpb, torch.repeatTensor(pdf,1,1,nm))

  local mean_left = g_mean_diff:view(-1, ps, 1)
  local mean_right = g_mean_diff:view(-1, 1, ps)
  -- though reusing the same name is bad, but it is actually the same thing.
  g_rpb = torch.repeatTensor(grad_g_w:view(-1,1),1,ps)
  -- gradient for mean
  local grad_g_mean = torch.cmul(g_rpb, torch.bmm(g_sigma_inv, mean_left))
  -- gradient for sigma
  local g_temp = torch.bmm(torch.bmm(g_sigma_inv, torch.bmm(mean_left, mean_right)), g_sigma_inv) - g_sigma_inv
  g_rpb = torch.repeatTensor(g_rpb:view(-1,ps,1),1,1,ps)
  local grad_g_sigma = torch.cmul(g_rpb, g_temp):mul(0.5)

  -- back prop encodings
  -- mean undertake no changes
  grad_g_mean = grad_g_mean:view(D, N, -1)
  -- gradient of weight is tricky, making it efficient together with softmax
  grad_g_w:add(g_w:cmul(self.LW))
  grad_g_w = grad_g_w:view(D, N, -1)
  -- gradient of the var, and cov
  local grad_g_var
  local grad_g_cov
  if ps == 3 then
    grad_g_sigma = grad_g_sigma:view(-1, 3, 3)
    local grad_g_clk = self.var_mm:backward({g_clk, g_clk_T}, grad_g_sigma)
    grad_g_clk = grad_g_clk[1]:mul(2)
    grad_g_var = torch.Tensor(D*N*nm, ps):type(grad_g_clk:type())
    grad_g_cov = torch.Tensor(D*N*nm, ps):type(grad_g_clk:type())
    grad_g_var[{{}, 1}] = grad_g_clk[{{},1,1}]
    grad_g_var[{{}, 2}] = grad_g_clk[{{},2,2}]
    grad_g_var[{{}, 3}] = grad_g_clk[{{},3,3}]
    grad_g_var = self.var_exp:backward(g_var_input, grad_g_var)
    grad_g_var = grad_g_var:view(D, N, -1)
    grad_g_cov[{{}, 1}] = grad_g_clk[{{},2,1}]
    grad_g_cov[{{}, 2}] = grad_g_clk[{{},3,1}]
    grad_g_cov[{{}, 3}] = grad_g_clk[{{},3,2}]
    grad_g_cov = grad_g_cov:mul(0.000)
    grad_g_cov = grad_g_cov:view(D, N, -1)
  else
    grad_g_var = torch.cmul(g_var, grad_g_sigma):mul(2)
    grad_g_var = self.var_exp:backward(g_var_input, grad_g_var)
    grad_g_var = grad_g_var:view(D, N, -1)
  end

  local Z = torch.sum(self.LW[{{},{},1}])
  grad_g_mean:div(Z)
  grad_g_var:div(Z)
  if ps == 3 then grad_g_cov:div(Z) end
  grad_g_w:div(Z)

  -- concat to gradInput
  if self.pixel_size == 3 then
    -- torch does not allow us to concat more than 2 tensors for FloatTensors
    self.gradInput = torch.cat(torch.cat(grad_g_mean, grad_g_var), torch.cat(grad_g_cov, grad_g_w))
  else
    self.gradInput = torch.cat(torch.cat(grad_g_mean, grad_g_var), grad_g_w)
  end

  -- return the loss
  self.output = (loss) / (Z)
  return self.output
end

function crit:updateGradInput(input, target)
  -- just return it
  return self.gradInput
end

local gmms = {}

function crit:sample(input, temperature, gt_pixels)
  local N, G = input:size(1), input:size(2)
  local ps = self.pixel_size
  local nm = self.num_mixtures
  if ps == 3 then
    assert(G == nm*(3+3+3+1), 'input dimensions of pixel do not match')
  else
    assert(G == nm*(1+1+0+1), 'input dimensions of pixel do not match')
  end

  -- decode the gmms first
  -- mean undertake no changes
  local g_mean_input = input:narrow(2,1,nm*ps):clone()
  local g_mean = g_mean_input:view(N, nm, ps)
  -- we use 6 numbers to denote the cholesky depositions
  local g_var_input = input:narrow(2, nm*ps+1, nm*ps):clone()
  g_var_input = g_var_input:view(-1, ps)
  local g_var = torch.exp(g_var_input)

  local g_cov_input
  local g_clk
  p = 2
  if ps == 3 then
    g_cov_input = input:narrow(2, p*nm*ps+1, nm*ps):clone()
    g_cov_input = g_cov_input:view(-1, 3)
    p = p + 1
    g_clk = torch.Tensor(N*nm, 3, 3):fill(0):type(g_var_input:type())
    g_clk_T = torch.Tensor(N*nm, 3, 3):fill(0):type(g_var_input:type())
    g_clk[{{}, 1, 1}] = g_var[{{}, 1}]
    g_clk[{{}, 2, 2}] = g_var[{{}, 2}]
    g_clk[{{}, 3, 3}] = g_var[{{}, 3}]
    --g_clk[{{}, 2, 1}] = g_cov_input[{{}, 1}]
    --g_clk[{{}, 3, 1}] = g_cov_input[{{}, 2}]
    --g_clk[{{}, 3, 2}] = g_cov_input[{{}, 3}]
    g_clk = g_clk:view(N, nm, ps, ps)
  else
    g_clk = g_var
    g_clk = g_clk:view(N, nm, ps, ps)
  end
  -- weights coeffs is taken care of at final loss, for computation efficiency and stability
  local g_w_input = input:narrow(2,p*nm*ps+1, nm):clone()
  local g_w = torch.exp(g_w_input:view(-1, nm))
  g_w = g_w:cdiv(torch.repeatTensor(torch.sum(g_w,2),1,nm))
  local g_ws = torch.exp(g_w_input:div(temperature))
  g_ws = g_ws:cdiv(torch.repeatTensor(torch.sum(g_ws,2),1,nm))
  --print(g_w)

  local pixels = torch.Tensor(N, ps):type(input:type())
  local train_pixels = gt_pixels:float()

  -- sampling process
  local mix_idx = torch.multinomial(g_ws, 1)
  --local ignore, mix_idx = torch.max(g_ws, 2)

  for b=1,N do
    local p = mvn.rnd(g_mean[{b, mix_idx[{b,1}], {}}], g_clk[{b, mix_idx[{b,1}], {},{}}])
    pixels[b] = p
  end
  --pixels:clamp(-0.5, 0.5)

  -- evaluate the loss
  local losses
  local train_losses
  if ps == 1 then
    -- for synthesis pixels
    local g_mean_diff = torch.repeatTensor(pixels:view(N, 1, ps),1,nm,1):add(-1, g_mean)
    local g_rpb = mvn.bnormpdf(g_mean_diff, g_clk):cmul(g_w)
    local pdf = torch.sum(g_rpb, 2)
    losses = - torch.sum(torch.log(pdf))
    -- for training pixels
    g_mean_diff = torch.repeatTensor(gt_pixels:view(N, 1, ps),1,nm,1):add(-1, g_mean)
    g_rpb = mvn.bnormpdf(g_mean_diff, g_clk)
    g_rpb = g_rpb:cmul(g_w)
    pdf = torch.sum(g_rpb, 2)
    train_losses = - torch.sum(torch.log(pdf))
  else
    g_clk = g_clk:view(-1, 3, 3)
    local g_clk_inv = mvn.btmi(g_clk)
    -- for synthesis pixels
    local g_mean_diff = torch.repeatTensor(pixels:view(N, 1, ps),1,nm,1):add(-1, g_mean)
    g_mean_diff = g_mean_diff:view(-1, 3, 1)
    local g_rpb = mvn.b3normpdf(g_mean_diff, g_clk_inv):cmul(g_w)
    g_rpb = g_rpb:view(N, nm, 1)
    local pdf = torch.sum(g_rpb, 2)
    losses = - torch.sum(torch.log(pdf))
    -- for training pixels
    g_mean_diff = torch.repeatTensor(gt_pixels:view(N, 1, ps),1,nm,1):add(-1, g_mean)
    g_mean_diff = g_mean_diff:view(-1, 3, 1)
    g_rpb = mvn.b3normpdf(g_mean_diff, g_clk_inv):cmul(g_w)
    g_rpb = g_rpb:view(N, nm, 1)
    pdf = torch.sum(g_rpb, 2)
    train_losses = - torch.sum(torch.log(pdf))
  end

  losses = losses / N
  train_losses = train_losses / N
  return pixels, losses, train_losses
end
