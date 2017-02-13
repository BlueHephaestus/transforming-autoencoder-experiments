function init_devices(gpu, gpuid, threads, seed)
  --given the corresponding configuration parameters, 
  --we require and initialize the corresponding hardware/threads/seeds/etc
  torch.setdefaulttensortype('torch.FloatTensor')
  if gpu then
     print 'Using GPU and CUDA'
     require 'cutorch'
     require 'cunn'
     cutorch.setDevice(gpuid)
  end
  torch.setnumthreads(threads)
  torch.manualSeed(seed)
end

function generate_one_hot(label_m, label_v)
  --given a matrix of zeros labels_m, and a vector of indices labels,
  --return the original matrix with zeros in the corresponding index locations for each sample.
  for i = 1, label_v:size(1) do
    label_m[{i,label_v[i]}] = 1
  end
  return label_m
end
