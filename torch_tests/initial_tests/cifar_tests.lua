--Example with CIFAR

require 'nn'
require 'paths'
require 'cunn'
if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

--image.save("test.jpg", trainset.data[100])
print(#trainset.data[100])
print(classes[trainset.label[100]])

--now we have to prepare dataset for sgd according to its documentation
--we need a size function and an index operator

-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
print(trainset[1])
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

--then we set the size method to the size of the first dimension
function trainset:size() 
  return self.data:size(1) 
end

--print(trainset:size())
--print(trainset[33])

--now to mean normalize our data
--we can use special tensor indexing to help us with this.
--trainset.data[{ {}, {1}, {}, {} }] gets all images, 1st channel, all vertical pixels, all horizontal pixels.
--we can also use something like {1,3} to get channels 1, 2, and 3
--or [{ {150,300}, {}, {}, {} }] to get 150th to 300th data samples

mean = {}
std = {}
for i = 1,3 do
  mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean()
  std[i] = trainset.data[{ {}, {i}, {}, {} }]:std()

  --then do (x-mean)/std
  trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i]):div(std[i])
end

--define network as before but with 3 channels instead of 1
net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,5,5)) -- 3 input channel, 6 output channels, 5x5 kernel
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2)) -- 2x2 subsample size, 2x2 stride
net:add(nn.SpatialConvolution(6,16,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2)) 

net:add(nn.View(16*5*5)) -- converts 3d tensor of 16x5x5 into 1d tensor of 16*5*5
net:add(nn.Linear(16*5*5,120)) -- fully connected layer
net:add(nn.ReLU())
net:add(nn.Linear(120,84))
net:add(nn.ReLU())
net:add(nn.Linear(84,10))
net:add(nn.LogSoftMax())

--add loss function
--log likelihood again

criterion = nn.ClassNLLCriterion()

--convert everything to gpu versions
net = net:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

--train network
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- epochs
trainer:train(trainset)
