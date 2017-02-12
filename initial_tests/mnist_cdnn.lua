--[[
--Example minimal CDNN and DNN with MNIST
--
-- Blake Edwards / Dark Element
--]]

require 'cutorch'
require 'cudnn'
require 'cunn'
require 'nn'
--require 'nn'
mnist = require 'mnist'

--get datasets
trainset = mnist.traindataset()
testset = mnist.testdataset()

--get sizes
--APPARENTLY WE CAN'T EASILY MAKE A VALIDATION DATA SET BECAUSE AGH TABLES
n = trainset.size
test_n = testset.size

-- convert the data from a ByteTensor to a DoubleTensor in order to do operations we need.
trainset.data = trainset.data:double() 

--add a size method for optimizer
function trainset:size()
  return self.data:size(1)
end

--add index support for optimizer by referencing the correct way each one:
--we have to return a table with normal indices, where the label is always >= 1 and <= class_n
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]+1} 
                end}
);

--[[
for i=1,10 do
  print(trainset[i][2])
end
--]]

--get mean and std
mean = trainset.data:mean()
std = trainset.data:std()
--print(mean, std)

--normalize training (and test data?) accordingly
trainset.data = (trainset.data - mean)/std

--init layers
model = nn.Sequential()
model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 10))
--[[
model:add(nn.ReLU())
model:add(nn.Linear(1000, 100))
model:add(nn.ReLU())
model:add(nn.Linear(100, 10))
--]]
model:add(nn.LogSoftMax())

--init loss function
criterion = nn.ClassNLLCriterion()

--[[
model2 = model:cuda()
model = nn.Sequential()
model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
model:add(model2)
model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
model = model:cuda()
cudamodel = nn.Sequential()
cudamodel:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
cudamodel:add(model:cuda())
cudamodel:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
--]]

--add cuda support
model = model:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

--init optimizer
optimizer = nn.StochasticGradient(model, criterion)

--train model
optimizer.learningRate = 0.5
optimizer.maxIteration = 100
optimizer:train(trainset)
