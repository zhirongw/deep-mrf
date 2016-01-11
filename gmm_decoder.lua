-- decode the gaussian mixture models to meaningful mean, cov, weights
--------------------------------------------------------------------------------

require 'nn'
require 'nngraph'

local layer, parent = torch.class('nn.GMMDecoder', 'nn.Module')

function layer:__init(pixel_size, num_mixtures)
  parent.__init(self)
  self.pixel_size = pixel_size
  self.num_mixtures = num_mixtures
  if self.pixel_size == 3 then
    self.output_size = self.num_mixtures * (3+3+3+1) + 1
  else
    self.output_size = self.num_mixures * (1+1+0+1) + 1
  end

  self.var_exp = nn.Exp()
  self.var_mm = nn.MM() -- supports batch mode operations
  if pixel_size == 3 then self.cov_tanh = nn.Tanh() end
  self.w_softmax = nn.SoftMax()
end

--[[
input: NxG Mixture of Gaussian Encodings plus 1 end-token
returns a table of (mean, cov matrix, weights, end)
--]]
function layer:updateOutput(input)
  assert(input:size(2) == self.output_size, 'The length of input encodings do not match with pixel size')

  self.output = {}
  batch_size = input:size(1)
  -- mean undertake no changes
  local g_mean = input:narrow(2,1,self.num_mixtures*self.pixel_size):clone()
  g_mean = g_mean:view(batch_size, self.num_mixtures, self.pixel_size)
  table.insert(self.output, g_mean)
  -- variance should be positive, exponentials
  local g_var = self.var_exp:forward(input:narrow(2,self.num_mixtures*self.pixel_size+1,self.num_mixtures*self.pixel_size))
  local g_var_mat = self.var_mm:forward({g_var:view(batch_size*self.num_mixtures, -1, 1), g_var:view(batch_size*self.num_mixtures, 1, -1)})
  g_var_mat = g_var_mat:view(batch_size, self.num_mixtures, self.pixel_size, self.pixel_size)
  --self.g_var = g_var
  --self.g_var_mat = g_var_mat
  -- covariance coeffs should be (-1,1), tanhs.
  if self.pixel_size == 3 then
    local g_cov = self.cov_tanh:forward(input:narrow(2,2*self.num_mixtures*self.pixel_size+1,self.num_mixtures*self.pixel_size))
    g_cov = g_cov:view(batch_size, self.num_mixtures, 3)
    local cov_mat = torch.Tensor(batch_size, self.num_mixtures, 3, 3):fill(1):type(g_cov:type())
    cov_mat[{{}, {}, 1, 2}] = g_cov[{{}, {}, 1}]
    cov_mat[{{}, {}, 2, 1}] = g_cov[{{}, {}, 1}]
    cov_mat[{{}, {}, 1, 3}] = g_cov[{{}, {}, 2}]
    cov_mat[{{}, {}, 3, 1}] = g_cov[{{}, {}, 2}]
    cov_mat[{{}, {}, 2, 3}] = g_cov[{{}, {}, 3}]
    cov_mat[{{}, {}, 3, 2}] = g_cov[{{}, {}, 3}]
    --self.cov_mat = cov_mat
    g_var_mat:cmul(cov_mat)
  end
  table.insert(self.output, g_var_mat)
  -- weights coeffs is taken care of at final loss, for computation efficiency and stability
  local g_w = self.w_softmax:forward(input:narrow(2,3*self.num_mixtures*self.pixel_size+1,self.num_mixtures))
  table.insert(self.output, g_w)
  -- border is a single scalar indicator
  table.insert(self.output, input[{{},-1}])

  return self.output
end

--[[
input: NxG Mixture of Gaussian Encodings plus 1 end-token
gradOuput:  a table of (mean, cov matrix, weights, end)
--]]
function layer:updateGradInput(input, gradOutput)
  -- mean undertake no changes
  local dg_mean = gradOutput[1]:view(batch_size, -1)
  -- for variance and covariance matrix
  -- g_var, g_var_mat, cov_mat should be recalculated
  -- variance should be positive, exponentials
  local g_var = self.var_exp:forward(input:narrow(2,self.num_mixtures*self.pixel_size+1,self.num_mixtures*self.pixel_size))
  local g_var_mat = self.var_mm:forward({g_var:view(batch_size*self.num_mixtures, -1, 1), g_var:view(batch_size*self.num_mixtures, 1, -1)})
  g_var_mat = g_var_mat:view(batch_size, self.num_mixtures, self.pixel_size, self.pixel_size)

  local dg_cov
  local dvar_mat
  if self.pixel_size == 3 then
    local dcov_mat = torch.cmul(gradOutput[2], g_var_mat)
    dg_cov = torch.Tensor(batch_size, self.num_mixtures, 3):type(dcov_mat:type())
    dg_cov[{{}, {}, 1}] = dcov_mat[{{},{},1,2}] + dcov_mat[{{},{},2,1}]
    dg_cov[{{}, {}, 2}] = dcov_mat[{{},{},1,3}] + dcov_mat[{{},{},3,1}]
    dg_cov[{{}, {}, 3}] = dcov_mat[{{},{},2,3}] + dcov_mat[{{},{},3,2}]
    dg_cov = dg_cov:view(batch_size, -1)
    dg_cov = self.cov_tanh:backward(input:narrow(2,2*self.num_mixtures*self.pixel_size+1,self.num_mixtures*self.pixel_size), dg_cov)

    local g_cov = self.cov_tanh:forward(input:narrow(2,2*self.num_mixtures*self.pixel_size+1,self.num_mixtures*self.pixel_size))
    g_cov = g_cov:view(batch_size, self.num_mixtures, 3)
    local cov_mat = torch.Tensor(batch_size, self.num_mixtures, 3, 3):fill(1):type(g_cov:type())
    cov_mat[{{}, {}, 1, 2}] = g_cov[{{}, {}, 1}]
    cov_mat[{{}, {}, 2, 1}] = g_cov[{{}, {}, 1}]
    cov_mat[{{}, {}, 1, 3}] = g_cov[{{}, {}, 2}]
    cov_mat[{{}, {}, 3, 1}] = g_cov[{{}, {}, 2}]
    cov_mat[{{}, {}, 2, 3}] = g_cov[{{}, {}, 3}]
    cov_mat[{{}, {}, 3, 2}] = g_cov[{{}, {}, 3}]
    dvar_mat = torch.cmul(gradOutput[2], cov_mat)
    dvar_mat = dvar_mat:view(batch_size*self.num_mixtures, self.pixel_size, self.pixel_size)
  else
    dvar_mat = gradOutput[2]
  end

  local dg_var = self.var_mm:backward({g_var:view(batch_size*self.num_mixtures, -1, 1), g_var:view(batch_size*self.num_mixtures, 1, -1)}, dvar_mat)
  dg_var = torch.add(dg_var[1], 1, dg_var[2])
  -- dg_var = dg_var[1] * 2
  dg_var = dg_var:view(batch_size, -1)
  dg_var = self.var_exp:backward(input:narrow(2,self.num_mixtures*self.pixel_size+1,self.num_mixtures*self.pixel_size), dg_var)
  -- weights coeffs should be normalized to one
  local dg_w = self.w_softmax:backward(input:narrow(2,3*self.num_mixtures*self.pixel_size+1, self.num_mixtures), gradOutput[3])
  -- border is a single scalar indicator
  local d_border = gradOutput[4]

  -- concat to gradInput
  if self.pixel_size == 3 then
    self.gradInput = torch.cat(torch.cat(torch.cat(dg_mean, dg_var), torch.cat(dg_cov, dg_w)), d_border)
  else
    self.gradInput = torch.cat(torch.cat(dg_mean, dg_var), torch.cat(dg_w, d_border))
  end
  return self.gradInput
end
