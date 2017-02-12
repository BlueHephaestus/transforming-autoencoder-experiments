function train()
  for epoch = 1,epochs do

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- shuffle at each epoch
    shuffle = torch.randperm(training_n)

    -- do one epoch
    print("==> Epoch # " .. epoch)
    for t = 1,training_dataset:size(),batch_size do
      -- disp progress
      xlua.progress(t, training_dataset:size())

      -- create mini batch
      inputs = {}
      targets = {}
      for i = t,math.min(t+batch_size-1,training_dataset:size()) do
        -- load new sample
        input = training_dataset.data[shuffle[i]]
        target = input
        --target = training_dataset.label[shuffle[i]]
        if gpu then
          input = input:cuda()
        else 
          input = input:double()
        end
        --[[
        if gpu then
          input = input:cuda();
        end
        --]]
        
        table.insert(inputs, input)
        table.insert(targets, target)
      end

      -- optimize on current mini-batch
      optim.sgd(f_eval, parameters, optim_state)
    end

    --for flush print formatting
    print("")

    --update confusion matrix
    confusion:updateValids()

    -- update logger/plot
    if log_results then
      train_logger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
      if plot_results then
        train_logger:style{['% mean class accuracy (train set)'] = '-'}
        train_logger:plot()
      end
    end

    --confusion:zero()

  end
end

