----------------------------------------------------------------------
-- Modified from https://github.com/torch/tutorials/tree/master/2_supervised by Clement Farabet
--
-- Example minimal CDNN and DNN with MNIST
--
-- Blake Edwards / Dark Element
----------------------------------------------------------------------

require 'torch'
require 'xlua'    -- for progress bars
require 'optim'   
require 'nn'
require 'closure' -- external file for our closure function
require 'base_funcs' -- external file for some other helpful functions
require 'trainer' -- external file for training model
mnist = require 'mnist'

-- Config parameters
verbose = true
save_dir = "results"
log_results = true
save_net = true
plot_results = true
gpu = false
gpuid = 1
seed = 1
threads = 16

-- NN parameters
class_n = 10
epochs = 100
batch_size = 100
dropout_p = 0.0

-- Optimizer parameters
optim_state = {
  learningRate = 0.05,
  weightDecay = 0,
  momentum = 0.0,
  learningRateDecay = 1e-7
}

--init gpu / cpu accordingly
init_devices(gpu, gpuid, threads, seed)

--Our model
model = nn.Sequential()
model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 10))
model:add(nn.Dropout(dropout_p))
model:add(nn.LogSoftMax())

--Our loss/criterion
criterion = nn.ClassNLLCriterion() 

--initialize datasets
training_dataset = mnist.traindataset()
test_dataset = mnist.testdataset()

--Convert data to a DoubleTensor 
training_dataset.data = training_dataset.data:double()

--convert labels to work with lua
training_dataset.label = training_dataset.label+1

--add appropriate method for dataset
function training_dataset:size() 
  return self.data:size(1) 
end

training_n = training_dataset:size()

-- Apply cuda
if gpu then
  model = model:cuda()
  criterion = criterion:cuda()
end

-- classes, for confusion matrix
classes = {}
for i=1,class_n do classes[i] = i end

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
if log_results then
  train_logger = optim.Logger(paths.concat(save_dir, 'train.log'))
  test_logger = optim.Logger(paths.concat(save_dir, 'test.log'))
end

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,grad_parameters = model:getParameters()
end

--start timer when we start training
time = sys.clock()

--train model
train()

-- print confusion matrix
print(confusion)

-- time taken
time = sys.clock() - time
print("\n==> time to train " .. (time) .. 'ms')

-- save/log current net
if save_net then
  local filename = paths.concat(save_dir, 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> saving model to '..filename)
  torch.save(filename, model)
end
