-- create closure to evaluate f(X) and df/dX
function f_eval(x)
  -- get new parameters
  if x ~= parameters then
    parameters:copy(x)
  end

  -- reset gradients
  grad_parameters:zero()

  -- f is the average of all criterions
  local f = 0

  -- evaluate function for complete mini batch
  for i = 1,#inputs do
    -- estimate f
    output = model:forward(inputs[i])
    err = criterion:forward(output, targets[i])
    f = f + err

    -- estimate df/dW
    df_do = criterion:backward(output, targets[i])
    model:backward(inputs[i], df_do)

    -- update confusion
    confusion:add(output, targets[i])
  end

  -- normalize gradients and f(X)
  grad_parameters:div(#inputs)
  f = f/#inputs

  -- return f and df/dX
  return f,grad_parameters
end
