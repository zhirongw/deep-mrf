require('math')

-- all of the functions only process one input.
local mvn = {}

-- cholesky is lower down decomposition of Sigma
-- assumes x = x - mu is already computed
function mvn.logpdf(x, cholesky)
  if x:dim() == 1 then
    x:resize(1, x:nElement())
  end
  local transformed = torch.gesv(x:t(), cholesky)
  local logdet = cholesky:diag():log():sum()
  transformed:apply(function(a) return mvn.logunit(a) end)
  local result = transformed:sum() - logdet
  return result
end

function mvn.pdf(x, cholesky)
  local r = mvn.logpdf(x, cholesky)
  return math.exp(r)
end

-- batch triangle matrix inverse, assumes downside triangle in 3 dimensions
function mvn.btmi(cholesky)
  local inv = cholesky:clone()
  inv[{{}, 1, 1}]:fill(1):cdiv(cholesky[{{}, 1, 1}])
  inv[{{}, 2, 2}]:fill(1):cdiv(cholesky[{{}, 2, 2}])
  inv[{{}, 3, 3}]:fill(1):cdiv(cholesky[{{}, 3, 3}])
  inv[{{}, 2, 1}]:cmul(inv[{{}, 1, 1}])
  inv[{{}, 3, 1}]:cmul(inv[{{}, 1, 1}])
  inv[{{}, 3, 2}]:cmul(inv[{{}, 2, 2}])
  inv[{{}, 3, 1}]:add(-1, torch.cmul(inv[{{}, 2, 1}], inv[{{}, 3, 2}]))
  inv[{{}, 2, 1}]:cmul(inv[{{}, 2, 2}]):mul(-1)
  inv[{{}, 3, 2}]:cmul(inv[{{}, 3, 3}]):mul(-1)
  inv[{{}, 3, 1}]:cmul(inv[{{}, 3, 3}]):mul(-1)
  return inv
end

-- batch gaussian densify for 3 dimensional input
function mvn.b3normpdf(x, cholesky_inv)
  local r = mvn.b3normlogpdf(x, cholesky_inv)
  return torch.exp(r)
end

function mvn.b3normlogpdf(x, cholesky_inv)
  assert(x:size(1) == cholesky_inv:size(1))
  local transformed = torch.bmm(cholesky_inv, x)
  local logdet = torch.log(torch.cmul(torch.cmul(cholesky_inv[{{},1,1}], cholesky_inv[{{},2,2}]), cholesky_inv[{{},3,3}]))
  --transformed:apply(function(a) return mvn.logunit(a) end)
  transformed:cmul(transformed):mul(-0.5):add(- 0.5 * math.log(2*math.pi))
  local result = torch.add(torch.sum(transformed, 2), logdet)
  return result
end

function mvn.logunit(x)
  return -.5 * x*x - 0.5 * math.log(2*math.pi)
end

-- batch gaussian densify for a scalar distribution
function mvn.bnormpdf(x, sigma)
  local r = mvn.bnormlogpdf(x, sigma)
  return torch.exp(r)
end

function mvn.bnormlogpdf(x, sigma)
  assert(x:nElement() == sigma:nElement())
  local r = torch.cmul(x,x):mul(-0.5):cdiv(sigma):cdiv(sigma)
  r:add(-0.5 * math.log(2*math.pi))
  r:add(-1, torch.log(sigma))
  return r
end

function mvn.rnd(mu, cholesky)
  -- L*X + mu
  local n = mu:nElement()
  local x = torch.Tensor(n):type(mu:type())
  for i=1,n do
    x[i] = torch.normal(0,1)
  end
  local y = mu:clone()
  y:addmv(cholesky, x)
  return y
end

return mvn
