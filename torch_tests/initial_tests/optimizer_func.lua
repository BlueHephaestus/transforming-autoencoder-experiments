-- Define the function for gradient optimization i.e. for SGD later
-- to evaluate f(X) and df/dX
-- This optimizer originally made by Rudra Poudel - https://github.com/rudrapoudel/hello_ml/blob/master/dnn/dnn.lua
-- I changed it a lot from that one, however.
function f_eval(x)
  -- get new parameters if x is not our parameters
  -- just in case.
  if x ~= params then
    params:copy(x)
  end
  
  -- reset gradients
  grad_params:zero()

  -- f is the average of all criterions, 
  -- to be returned at the end after averaging our error over mini batches.
  f = 0;
  
  -- evaluate function for complete mini batch  
  local outputs = model:forward(inputs)

  --handle cuda again for outputs
  if gpu then
    outputs = outputs:cuda()
  else
    outputs = outputs:float()
  end

  --initialize our df/dX to be the sizes of our first two dims of output
  local df_do = torch.Tensor(outputs:size(1), outputs:size(2)):float()

  for i=1,batch_size do
    --loop through batch, and get error according to criterion chosen.
    local err = 0
    if train_criterion == 'MSE' then
      -- get error
      err = criterion:forward(outputs[i], targets_matrix[i])
      -- estimate df/dW and store in array
      df_do[i]:copy(criterion:backward(outputs[i], targets_matrix[i]))

    elseif train_criterion == 'NLL' then
      -- get error
      err = criterion:forward(outputs[i], targets[i])
      -- estimate df/dW and store in array
      df_do[i]:copy(criterion:backward(outputs[i], targets[i]))

    end    
    --increment average error sum
    f = f + err
  end

  --handle cuda again for df/dX
  if gpu then
    df_do = df_do:cuda()
  end

  -- back prop error
  model:backward(inputs, df_do)

  -- normalize gradients and f(X)
  grad_params:div(batch_size)
  f = f/batch_size

  -- return f and df/dX
  if verbose then
    print("\t"..f)
  end
  return f, grad_params
end 
