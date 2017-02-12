local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

print(trainset.size)
print(testset.size)

local ex = trainset[1]

local x = ex.x
local y = ex.y

print(y)
print(x)
