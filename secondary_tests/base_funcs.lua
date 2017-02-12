function init_devices(gpu, gpuid, threads, seed)
  --given the corresponding configuration parameters, 
  --we require and initialize the corresponding hardware/threads/seeds/etc
  torch.setdefaulttensortype('torch.DoubleTensor')
  if gpu then
     print 'Using GPU and CUDA'
     require 'cutorch'
     require 'cunn'
     cutorch.setDevice(gpuid)
  end
  torch.setnumthreads(threads)
  torch.manualSeed(seed)
end

