--[[
-- Example minimal CDNN and DNN with MNIST
--
-- Blake Edwards / Dark Element
--]]
require 'torch'
require 'nn'
require 'optim'
require 'optimizer_func'
require 'base_funcs'
mnist = require 'mnist'

-- Config parameters
verbose = true
gpu = false
gpuid = 1
seed = 1
threads = 1

--init gpu / cpu accordingly
init_devices(gpu, gpuid, threads, seed)

-- NN parameters
class_n = 10
batch_size = 10
iterations = 100
train_criterion = 'NLL' -- MSE | NLL
dropout_p = 0.0

-- Optimizer parameters
optim_state = {
  learningRate = 0.05,
  weightDecay = 1e-5,
  momentum = 0.0,
  learningRateDecay = 5e-7
}

model = nn.Sequential()
model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, class_n))
model:add(nn.Dropout(dropout_p))

-- Add output activation accordingly
if train_criterion == 'MSE' then
  model:add(nn.SoftMax())
  criterion = nn.MSECriterion()
elseif train_criterion == 'NLL' then
  model:add(nn.LogSoftMax())
  criterion = nn.ClassNLLCriterion() 
end

-- Apply cuda
if gpu then
  model = model:cuda()
  criterion = criterion:cuda()
end

--Note on training:
--Data format must be able to be reshaped to (n_samples, data) e.g. (n, 28, 28) == (n, 784) == (n, 1, 28, 28)
--Label format must be (n_samples,)
--Get training dataset
trainset = mnist.traindataset()

--Convert data to a FloatTensor
inputs = (trainset.data):float() 

--Add 1 to the labels section for lua indices
targets = trainset.label+1

--then get our one hot matrix
targets_matrix = torch.Tensor(targets:size(1), class_n):zero():float()
targets_matrix = generate_one_hot(targets_matrix, targets)

--set x and y to cuda versions if we are using cuda
if gpu then
  inputs = inputs:cuda()
  targets = targets:cuda()
  targets_matrix = targets_matrix:cuda()
end

--normalize training data with its mean and std 
--(we want to do this after we convert to cuda to save time in the scalar multiplication)
mean = inputs:mean()
std = inputs:std()
inputs = (inputs - mean)/std

-- The tensor variables for model params and gradient params 
params, grad_params = model:getParameters()

--for printing progress in f_eval
iteration = 0
for i=1,iterations do
  --then loop through number of iterations / epochs and optimize each time
  iteration = iteration + 1
  optim.sgd(f_eval, params, optim_state)
  a = 0
  local o = model:forward(inputs)
  for j=1,batch_size do
    a = a + criterion:forward(o[j], targets[j])
  end
  a = a/batch_size
  print(a)

  --[[
  local y = model:forward(inputs)
  y = y:float()
  e = criterion:forward(y, targets)
  print(e)
  --]]
end

--OI FUTURE SELF
--Talk to mom first - ezpz
--Work on implementing our own printing of results
--And also make sure that we are training on all data, not the exact same mini batch over and over again. - yea it's totally doing the latter fuck
--Good luck, have fun. Get some foods if you want
