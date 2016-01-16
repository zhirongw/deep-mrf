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

function mvn.logunit(x)
  return -.5 * x*x - 0.5 * math.log(2*math.pi)
end

function mvn.bnormpdf(x, sigma)
  local r = mvn.bnormlogpdf(x, sigma)
  return torch.exp(r)
end

function mvn.bnormlogpdf(x, sigma)
  assert(x:nElement() == sigma:nElement())
  local r = torch.cmul(x,x):mul(-0.5):cdiv(sigma):cdiv(sigma)
  r:add(-0.5*math.pi)
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
