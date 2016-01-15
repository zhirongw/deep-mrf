require('math')

-- all of the functions only process one input.
local mvn = {}

-- cholesky is lower down decomposition of Sigma
function mvn.logpdf(x, mu, cholesky)
  local c = x:clone():add(-1, mu)
  if c:dim() == 1 then
    c:resize(1, c:nElement())
  end
  local transformed = torch.gesv(c:t(), cholesky)
  local logdet = cholesky:abs():diag():log():sum()
  transformed:apply(function(a) return mvn.logunit(a) end)
  local result = transformed:sum() - logdet
  return result
end

function mvn.pdf(x, mu, cholesky)
  local r = mvn.logpdf(x, mu, cholesky)
  return math.exp(r)
end

function mvn.logunit(x)
  return -.5 * x*x - 0.5 * math.log(2*math.pi)
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
