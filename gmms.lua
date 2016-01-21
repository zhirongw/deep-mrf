require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'lstm'
local mvn = require 'misc.mvn'

-------------------------------------------------------------------------------
-- Pixel Model Mixture of Gaussian Density Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.PixelModelCriterion', 'nn.Criterion')
function crit:__init(pixel_size, num_mixtures)
  parent.__init(self)
  self.pixel_size = pixel_size
  self.num_mixtures = num_mixtures
    if self.pixel_size == 3 then
    self.output_size = self.num_mixtures * (3+3+3+1)
  else
    self.output_size = self.num_mixtures * (1+1+0+1)
  end
  if pixel_size == 3 then self.var_mm = nn.MM() end
  self.w_softmax = nn.SoftMax()
  self.var_exp = nn.Exp()
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

  local input_x, target_x
  if ps == 3 and input:type() == 'torch.CudaTensor' then
    input_x = input:float()
    target_x = target:float()
  else
    input_x = input
    target_x = target
  end
  -- decode the gmms first
  -- mean undertake no changes
  local g_mean_input = input_x:narrow(3,1,nm*ps):clone()
  local g_mean = g_mean_input:view(D, N, nm, ps)
  local g_mean_diff = torch.repeatTensor(target_x:view(D, N, 1, ps), 1,1,nm,1):add(-1, g_mean)
  -- we use 6 numbers to denote the cholesky depositions
  local g_var_input = input_x:narrow(3, nm*ps+1, nm*ps):clone()
  g_var_input = g_var_input:view(-1, ps)
  local g_var = self.var_exp:forward(g_var_input)

  local g_cov_input
  local g_clk
  local g_clk_T
  local g_sigma
  p = 2
  if ps == 3 then
    g_cov_input = input_x:narrow(3, p*nm*ps+1, nm*ps):clone()
    g_cov_input = g_cov_input:view(-1, 3)
    p = p + 1
    g_clk = torch.Tensor(D*N*nm, 3, 3):fill(0):type(g_var_input:type())
    g_clk_T = torch.Tensor(D*N*nm, 3, 3):fill(0):type(g_var_input:type())
    g_clk[{{}, 1, 1}] = g_var[{{}, 1}]
    g_clk[{{}, 2, 2}] = g_var[{{}, 2}]
    g_clk[{{}, 3, 3}] = g_var[{{}, 3}]
    g_clk[{{}, 2, 1}] = g_cov_input[{{}, 1}]
    g_clk[{{}, 3, 1}] = g_cov_input[{{}, 2}]
    g_clk[{{}, 3, 2}] = g_cov_input[{{}, 3}]
    g_clk_T[{{}, 1, 1}] = g_var[{{}, 1}]
    g_clk_T[{{}, 2, 2}] = g_var[{{}, 2}]
    g_clk_T[{{}, 3, 3}] = g_var[{{}, 3}]
    g_clk_T[{{}, 1, 2}] = g_cov_input[{{}, 1}]
    g_clk_T[{{}, 1, 3}] = g_cov_input[{{}, 2}]
    g_clk_T[{{}, 2, 3}] = g_cov_input[{{}, 3}]
    g_sigma = self.var_mm:forward({g_clk, g_clk_T})
    g_clk = g_clk:view(D, N, nm, ps, ps)
    g_clk_T = g_clk_T:view(D, N, nm, ps, ps)
  else
    g_clk = g_var
    g_clk = g_clk:view(D, N, nm, ps, ps)
    g_clk_T = g_var
    g_clk_T = g_clk_T:view(D, N, nm, ps, ps)
    g_sigma = torch.cmul(g_clk, g_clk_T)
  end
  g_sigma = g_sigma:view(D, N, nm, ps, ps)
  local g_sigma_inv = torch.Tensor(g_sigma:size()):type(g_sigma:type())
  -- weights coeffs is taken care of at final loss, for computation efficiency and stability
  local g_w_input = input_x:narrow(3,p*nm*ps+1, nm):clone()
  local g_w = self.w_softmax:forward(g_w_input:view(-1, nm))
  g_w = g_w:view(D, N, nm)

  -- do the loss the gradients
  local loss1 = 0 -- loss of pixels, Mixture of Gaussians
  local grad_g_mean = torch.Tensor(g_mean:size()):type(g_mean:type())
  local grad_g_sigma = torch.Tensor(g_sigma:size()):type(g_sigma:type())
  local grad_g_w = torch.Tensor(g_w:size()):type(g_w:type())

  if ps == 1 then
    local g_rpb = mvn.bnormpdf(g_mean_diff, g_clk)
    g_rpb = g_rpb:cmul(g_w)
    local pdf = torch.sum(g_rpb, 3)
    loss1 = - torch.sum(torch.log(pdf))
    g_sigma_inv:fill(1):cdiv(g_sigma)
    grad_g_w = - torch.cdiv(g_rpb, torch.repeatTensor(pdf,1,1,nm,1))
  else
    -- for color pixels, we have to calculate the inverse one at a time. FIX THIS?
    for t=1,D do -- iterate over timestep
      for b=1,N do -- iterate over batches
        -- can we vectorize this? Now constrains by the MVN.PDF
        local g_rpb_ = torch.Tensor(nm):zero()
        for g=1,nm do -- iterate over mixtures
          g_rpb_[g] = mvn.pdf(g_mean_diff[{t,b,g,{}}], g_clk[{t,b,g,{},{}}]) * g_w[{t,b,g}]
          g_sigma_inv[{t,b,g,{},{}}] = torch.inverse(g_sigma[{t,b,g,{},{}}])
        end
        local pdf = torch.sum(g_rpb_)
        loss1 = loss1 - torch.log(pdf)

        -- normalize the responsibilities for backprop
        g_rpb_:div(pdf)

        -- gradient of weight is tricky, making it efficient together with softmax
        grad_g_w[{t,b, {}}] = - g_rpb_
      end
    end
  end

  local mean_left = g_mean_diff:view(-1, ps, 1)
  local mean_right = g_mean_diff:view(-1, 1, ps)
  g_sigma_inv = g_sigma_inv:view(-1, ps, ps)
  local g_rpb = torch.repeatTensor(grad_g_w:view(-1,1),1,ps)
  -- gradient for mean
  grad_g_mean = torch.cmul(g_rpb, torch.bmm(g_sigma_inv, mean_left))
  -- gradient for sigma
  local g_temp = torch.bmm(torch.bmm(g_sigma_inv, torch.bmm(mean_left, mean_right)), g_sigma_inv) - g_sigma_inv
  g_rpb = torch.repeatTensor(g_rpb:view(-1,ps,1),1,1,ps)
  grad_g_sigma = torch.cmul(g_rpb, g_temp):mul(0.5)

  -- back prop encodings
  -- mean undertake no changes
  grad_g_mean = grad_g_mean:view(D, N, -1)
  -- gradient of weight is tricky, making it efficient together with softmax
  grad_g_w:add(g_w)
  grad_g_w = grad_g_w:view(D, N, -1)
  -- gradient of the var, and cov
  local grad_g_var
  local grad_g_cov
  g_clk = g_clk:view(-1, ps, ps)
  g_clk_T = g_clk_T:view(-1, ps, ps)
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
    grad_g_cov = grad_g_cov:view(D, N, -1)
  else
    grad_g_var = torch.cmul(g_var, grad_g_sigma):mul(2)
    grad_g_var = self.var_exp:backward(g_var_input, grad_g_var)
    grad_g_var = grad_g_var:view(D, N, -1)
  end

  grad_g_mean:div(D*N)
  grad_g_var:div(D*N)
  if ps == 3 then grad_g_cov:div(D*N) end
  grad_g_w:div(D*N)

  -- concat to gradInput
  if self.pixel_size == 3 then
    -- torch does not allow us to concat more than 2 tensors for FloatTensors
    self.gradInput = torch.cat(torch.cat(grad_g_mean, grad_g_var), torch.cat(grad_g_cov, grad_g_w))
  else
    self.gradInput = torch.cat(torch.cat(grad_g_mean, grad_g_var), grad_g_w)
  end
  if input:type() == 'torch.CudaTensor' and ps == 3 then
    self.gradInput = self.gradInput:cuda()
  end
  -- return the loss
  self.output = (loss1) / (D*N)
  return self.output
end

function crit:updateGradInput(input, target)
  -- just return it
  return self.gradInput
end

local gmms = {}

function crit:sample(input, gt_pixels)
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
    g_clk[{{}, 2, 1}] = g_cov_input[{{}, 1}]
    g_clk[{{}, 3, 1}] = g_cov_input[{{}, 2}]
    g_clk[{{}, 3, 2}] = g_cov_input[{{}, 3}]
    g_clk = g_clk:view(N, nm, ps, ps)
  else
    g_clk = g_var
    g_clk = g_clk:view(N, nm, ps, ps)
  end
  -- weights coeffs is taken care of at final loss, for computation efficiency and stability
  local g_w_input = input:narrow(2,p*nm*ps+1, nm):clone()
  local g_w = torch.exp(g_w_input:view(-1, nm))
  --print(g_w)
  g_w = g_w:cdiv(torch.repeatTensor(torch.sum(g_w,2),1,nm))
  --print(g_w)

  local pixels = torch.Tensor(N, ps):type(input:type())
  local train_pixels = gt_pixels:float()
  local losses = 0
  local train_losses = 0
  local g_clk_x, g_w_x, g_mean_x
  if input:type() == 'torch.CudaTensor' then
    g_clk_x = g_clk:float()
    g_w_x = g_w:float()
    g_mean_x = g_mean:float()
  else
    g_clk_x = g_clk
    g_w_x = g_w
    g_mean_x = g_mean
  end
  -- sampling process
  local mix_idx
  mix_idx = torch.multinomial(g_w, 1)
  --ignore, mix_idx = torch.max(g_w, 2)
  --mix_idx = mix_idx:resize(N,1)
  for b=1,N do
    -- print('------------------------------------------')
    -- sample from the multinomial
    --print(mix_idx)
    --local max_prob, mix_idx
    --max_prob, mix_idx = torch.max(g_w[b], 1)
    --mix_idx = mix_idx[1]
    --print(mix_idx)
    -- sample from the mvn gaussians
    -- print(g_mean[{b, mix_idx, {}}])
    -- print(g_clk[{b, mix_idx, {}, {}}])
    local p = mvn.rnd(g_mean[{b, mix_idx[{b,1}], {}}], g_clk[{b, mix_idx[{b,1}], {},{}}])
    --p = g_mean[{b, mix_idx[{b,1}], {}}]
    pixels[b] = p
    if ps == 3 then
      -- evaluate the loss
      local g_rpb_ = torch.Tensor(nm):zero()
      local pf = p:float()
      --print(pf)
      for g=1,nm do -- iterate over mixtures
        g_rpb_[g] = mvn.pdf(pf, g_mean_x[{b,g,{}}], g_clk_x[{b,g,{},{}}]) * g_w_x[{b,g}]
      end
      local pdf = torch.sum(g_rpb_)
      losses = losses - torch.log(pdf)
      -- VALIDATION
      local train_g_rpb_ = torch.Tensor(nm):zero()
      local train_pf = train_pixels[b]
      --print(val_pf)
      for g=1,nm do -- iterate over mixtures
        train_g_rpb_[g] = mvn.pdf(train_pf, g_mean_x[{b,g,{}}], g_clk_x[{b,g,{},{}}]) * g_w_x[{b,g}]
      end
      local train_pdf = torch.sum(train_g_rpb_)
      train_losses = train_losses - torch.log(train_pdf)
    end
  end

  -- evaluate the loss
  if ps == 1 then
    -- for synthesis pixels
    local g_mean_diff = torch.repeatTensor(pixels:view(N, 1, ps),1,nm,1):add(-1, g_mean)
    local g_rpb = mvn.bnormpdf(g_mean_diff, g_clk)
    g_rpb = g_rpb:cmul(g_w)
    local pdf = torch.sum(g_rpb, 2)
    losses = - torch.sum(torch.log(pdf))
    -- for training pixels
    g_mean_diff = torch.repeatTensor(gt_pixels:view(N, 1, ps),1,nm,1):add(-1, g_mean)
    g_rpb = mvn.bnormpdf(g_mean_diff, g_clk)
    g_rpb = g_rpb:cmul(g_w)
    pdf = torch.sum(g_rpb, 2)
    train_losses = - torch.sum(torch.log(pdf))
  end

  losses = losses / N
  train_losses = train_losses / N
  return pixels, losses, train_losses
end
