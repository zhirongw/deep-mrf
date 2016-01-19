-- from Karpathy's neuraltalk2
-- with my extension of 2d lstm from the paper: http://arxiv.org/abs/0705.2011
-- since my focus is on 2d, I will not implement the general nd case.
--------------------------------------------------------------------------------

require 'nn'
require 'nngraph'

--------------------------------------------------------------------------------
-- First some exercises over single layer lstm module
--------------------------------------------------------------------------------
local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)

  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

local function lstm2d(x, prev_c, prev_h)
  -- Notes about the 2d case:
  ---- the computation of input x is still the same as in 1d.
  ---- prev_c, and prev_h is concat of two states from two dimensions.
  ---- therefore, prev_c, prev_h is twice larger than 1d case.
  local sliced_c = nn.SplitTable(2)(prev_c)
  local prev_c_1 = nn.SelectTable(1)(sliced_c)
  local prev_c_2 = nn.SelectTable(2)(sliced_c)

  -- Calculate all four gates in one go
  ---- for in_gate, in_transform, out_gate, computation is the same excepte
  ---- we have more input dimensions.
  ---- for forget_gate, since it is included in the recurrent computations,
  ---- we will have 2 such gate, each for one recurrent connection.
  local i2h = nn.Linear(params.rnn_size, (3+2*1)*params.rnn_size)(x)
  local h2h = nn.Linear(d*params.rnn_size, (3+2*1)*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape((3+2*1),params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)

  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  -- two forget_gate
  local forget_gate_1      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local forget_gate_2      = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(-1)(sliced_gates))

  -- we will have only one next state
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate_1, prev_c_1}),
      nn.CMulTable()({forget_gate_2, prev_c_2}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

--------------------------------------------------------------------------------
-- Now, begin a stacked lstm network
-- Building a computation graph, and returns a module of the inputs and outputs
--------------------------------------------------------------------------------
local LSTM = {}
function LSTM.lstm(input_size, output_size, rnn_size, n, dropout, mult_in)
  -- based on the original 1d implementation from Zaremba and Karpathy
  -- slightly modified network with more connections from input to deeper layers
  -- and from middle layers to final outputs, paper: http://arxiv.org/abs/1308.0850
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from the lower layer
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = input_size
    else
      if mult_in then
        x = nn.JoinTable(2)({inputs[1], outputs[(L-1)*2]})
        input_size_L = input_size + rnn_size
      else
        x = outputs[(L-1)*2]
        input_size_L = rnn_size
      end
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  local top_size
  local top_h = outputs[#outputs]
  if mult_in then
    top_size = n * rnn_size
    for L = n-1,1,-1 do
      top_h = nn.JoinTable(2)({outputs[2*L], top_h})
    end
  else
    top_size = rnn_size
  end
  -- set up the decoder
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(top_size, output_size)(top_h):annotate{name='encoder'}
  -- local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, proj)

  return nn.gModule(inputs, outputs)
end

function LSTM.lstm2d(input_size, output_size, rnn_size, n, dropout, mult_in)
  -- NOTES about 2d case:
  -- We only extent the lstm2d along the depth dimension, we haven't touched any
  -- core problem of connecting frames in a 2d grid along time.
  dropout = dropout or 0

  -- there will be 4*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_layer_1_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_1_h[L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_2_c[n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_2_h[n+L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from the previous 2 layers
    local prev_c_1 = inputs[L*2]
    local prev_h_1 = inputs[L*2+1]
    local prev_c_2 = inputs[(n+L)*2]
    local prev_h_2 = inputs[(n+L)*2+1]
    local prev_h = nn.JoinTable(2)({prev_h_1, prev_h_2})
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = input_size
    else
      if mult_in then
        x = nn.JoinTable(2)({inputs[1], outputs[(L-1)*2]})
        input_size_L = input_size + rnn_size
      else
        x = outputs[(L-1)*2]
        input_size_L = rnn_size
      end
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, (3+2*1) * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(2*rnn_size, (3+2*1) * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(5, rnn_size)(all_input_sums)
    local n1, n2, n3, n4, n5 = nn.SplitTable(2)(reshaped):split(5)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate_1 = nn.Sigmoid()(n2)
    local forget_gate_2 = nn.Sigmoid()(n3)
    local out_gate = nn.Sigmoid()(n4)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n5)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate_1, prev_c_1}),
        nn.CMulTable()({forget_gate_2, prev_c_2}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  local top_size
  local top_h = outputs[#outputs]
  if mult_in then
    top_size = n * rnn_size
    for L = n-1,1,-1 do
      top_h = nn.JoinTable(2)({outputs[2*L], top_h})
    end
  else
    top_size = rnn_size
  end
  -- set up the encoder
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(top_size, output_size)(top_h):annotate{name='encoder'}
  -- local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, proj)

  return nn.gModule(inputs, outputs)
end

function LSTM.lstm3d(input_size, output_size, rnn_size, n, dropout, mult_in)
  -- extension of 2d case with more than 2 pixel neighbors:
  dropout = dropout or 0

  -- there will be 6*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_layer_1_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_1_h[L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_2_c[n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_2_h[n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_3_c[2n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_3_h[2n+L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from the previous 2 layers
    local prev_c_1 = inputs[L*2]
    local prev_h_1 = inputs[L*2+1]
    local prev_c_2 = inputs[(n+L)*2]
    local prev_h_2 = inputs[(n+L)*2+1]
    local prev_c_3 = inputs[(2*n+L)*2]
    local prev_h_3 = inputs[(2*n+L)*2+1]
    local prev_h = nn.JoinTable(2)({prev_h_1, prev_h_2, prev_h_3})
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = input_size
    else
      if mult_in then
        x = nn.JoinTable(2)({inputs[1], outputs[(L-1)*2]})
        input_size_L = input_size + rnn_size
      else
        x = outputs[(L-1)*2]
        input_size_L = rnn_size
      end
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
    end
    -- evaluate the input sums at once for efficiency, here we have 3 forget gates
    local i2h = nn.Linear(input_size_L, (3+3*1) * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(3*rnn_size, (3+3*1) * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(6, rnn_size)(all_input_sums)
    local n1, n2, n3, n4, n5, n6 = nn.SplitTable(2)(reshaped):split(6)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate_1 = nn.Sigmoid()(n2)
    local forget_gate_2 = nn.Sigmoid()(n3)
    local forget_gate_3 = nn.Sigmoid()(n4)
    local out_gate = nn.Sigmoid()(n5)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n6)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate_1, prev_c_1}),
        nn.CMulTable()({forget_gate_2, prev_c_2}),
        nn.CMulTable()({forget_gate_3, prev_c_3}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  local top_size
  local top_h = outputs[#outputs]
  if mult_in then
    top_size = n * rnn_size
    for L = n-1,1,-1 do
      top_h = nn.JoinTable(2)({outputs[2*L], top_h})
    end
  else
    top_size = rnn_size
  end
  -- set up the encoder
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(top_size, output_size)(top_h):annotate{name='encoder'}
  -- local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, proj)

  return nn.gModule(inputs, outputs)
end

function LSTM.lstm4d(input_size, output_size, rnn_size, n, dropout, mult_in)
  -- extension of 2d case with more than 2 pixel neighbors:
  dropout = dropout or 0

  -- there will be 8*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_layer_1_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_1_h[L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_2_c[n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_2_h[n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_3_c[2n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_3_h[2n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_4_c[3n+L]
    table.insert(inputs, nn.Identity()()) -- prev_layer_4_h[3n+L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from the previous 2 layers
    local prev_c_1 = inputs[L*2]
    local prev_h_1 = inputs[L*2+1]
    local prev_c_2 = inputs[(n+L)*2]
    local prev_h_2 = inputs[(n+L)*2+1]
    local prev_c_3 = inputs[(2*n+L)*2]
    local prev_h_3 = inputs[(2*n+L)*2+1]
    local prev_c_4 = inputs[(3*n+L)*2]
    local prev_h_4 = inputs[(3*n+L)*2+1]
    local prev_h = nn.JoinTable(2)({prev_h_1, prev_h_2, prev_h_3, prev_h_4})
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = input_size
    else
      if mult_in then
        x = nn.JoinTable(2)({inputs[1], outputs[(L-1)*2]})
        input_size_L = input_size + rnn_size
      else
        x = outputs[(L-1)*2]
        input_size_L = rnn_size
      end
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
    end
    -- evaluate the input sums at once for efficiency, here we have 3 forget gates
    local i2h = nn.Linear(input_size_L, (3+4*1) * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(4*rnn_size, (3+4*1) * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(7, rnn_size)(all_input_sums)
    local n1, n2, n3, n4, n5, n6, n7 = nn.SplitTable(2)(reshaped):split(7)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate_1 = nn.Sigmoid()(n2)
    local forget_gate_2 = nn.Sigmoid()(n3)
    local forget_gate_3 = nn.Sigmoid()(n4)
    local forget_gate_4 = nn.Sigmoid()(n5)
    local out_gate = nn.Sigmoid()(n6)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n7)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate_1, prev_c_1}),
        nn.CMulTable()({forget_gate_2, prev_c_2}),
        nn.CMulTable()({forget_gate_3, prev_c_3}),
        nn.CMulTable()({forget_gate_4, prev_c_4}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  local top_size
  local top_h = outputs[#outputs]
  if mult_in then
    top_size = n * rnn_size
    for L = n-1,1,-1 do
      top_h = nn.JoinTable(2)({outputs[2*L], top_h})
    end
  else
    top_size = rnn_size
  end
  -- set up the encoder
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(top_size, output_size)(top_h):annotate{name='encoder'}
  -- local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, proj)

  return nn.gModule(inputs, outputs)
end

--net = LSTM.lstm(4,5,10,2,0,true)
--graph.dot(net.fg, 'LSTM', 'test_net_mult2')

return LSTM
