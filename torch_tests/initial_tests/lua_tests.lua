require 'image';
require 'nn'

a = "hello"
b = {}
b[1] = a
b[2] = 30
--i = start, stop; #b is length of b
--[[
for i=1,#b do
  print(b[i])
end
--]]

a = torch.Tensor(5,3)
a = torch.rand(5,3)

b = torch.rand(3,4)

--three ways of doing our dot product
c = a * b
d = torch.mm(a,b)

e = torch.Tensor(5,4)
e:mm(a,b)

--example elementwise multiplication
a = (torch.eye(3) + image.vflip(torch.eye(3)))*2
b = torch.Tensor(3,3):fill(1)
b[1] = torch.Tensor(1,3):fill(2)
b[2] = torch.Tensor(1,3):fill(3)
b[3] = torch.Tensor(1,3):fill(4)
--print(torch.cmul(a,b))

--cuda tensors
--we don't have cuda on our machine so we can't
--a = a:cuda()

--example function
function add_tensors(a,b)
  return a + b
end
a = torch.ones(5,2)
b = torch.Tensor(2,5):fill(4)
--print(add_tensors(a,b))

--example feedforward neural network
net = nn.Sequential()
net:add(nn.SpatialConvolution(1,6,5,5)) -- 1 input channel, 6 output channels, 5x5 kernel
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

--we can print our network
--print ('Lenet5\n' .. net:__tostring());
--print (net:__tostring());

--we can also print individual weights and biases
m = nn.SpatialConvolution(1,3,2,2)
--print(m.weight)
--print(m.bias)

--example test of input
input = torch.rand(1,32,32)
output = net:forward(input)
--print(output)

net:zeroGradParameters() -- apparently this zeros the internal gradient buffers? I am not sure what this does.
gradInput = net:backward(input, torch.rand(10))
--print(#gradInput)


--Now for loss functions / criterion functions
criterion = nn.ClassNLLCriterion() --negative log-likelihood

a = criterion:forward(output, 3) -- assuming the target was 3, we compute the `a` this way
gradients = criterion:backward(output, 3) -- compute gradients for our cost
gradInput = net:backward(input, gradients) -- and now we compute the changes to our parameters by computing the gradients via backpropagation from our cost gradients

--[[
--Now we have computed gradWeight and gradBias parameters for each layer in the network at this point,
--  so now all we need to do if we are doing SGD is go through each layer and do
--    w = w + eta*gradWeight
--  But since this is a high level library we have a built in SGD optimizer for this:
--    nn.StochasticGradient:train(dataset)
--]]

