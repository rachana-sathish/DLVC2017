require 'nn'
--require 'cudnn'
require 'cunn'--
require 'gnuplot'
--local matio = require 'matio' 
------------Load data---------------------------------------------
print('Loading data..')
--Loading the data; Normalizing image data
--train_data = torch.div(matio.load('train_data.mat','train_data'):float(),255) --80% of total data; dim => [Nx3X32x32]
train_data=torch.load('train_data.t7')
--local train_label1 = matio.load('train_label.mat','train_label') --dim => [Nx1]
train_label1=torch.load('train_label.t7')
--val_data = torch.div(matio.load('val_data.mat','val_data'):float(),255)  --10% of total data
val_data=torch.load('val_data.t7')
--local val_label1 = matio.load('val_label.mat','val_label')
val_label1=torch.load('val_label.t7')
--test_data = torch.div(matio.load('test_data.mat','test_data'):float(),255)  --10% of total data
test_data=torch.load('test_data.t7')
--test_label = matio.load('test_label.mat','test_label')
test_label=torch.load('test_label.t7')
---Converting labels to 1-D tensor as ClassNLLCriterion expects the targets to be a 1-D tensor
train_label = torch.ByteTensor(train_label1:size(1))
val_label = torch.ByteTensor(val_label1:size(1))


for n1 = 1,train_label1:size(1) do
	train_label[n1] = train_label1[{{n1}}]
end
for n2 = 1,val_label1:size(1) do
	val_label[n2] = val_label1[{{n2}}]
end
train_label1 = nil
val_label1 = nil

collectgarbage()
------------Load model and initialize parameters---------------------
model = require 'LeNet.lua'
--model = cudnn.convert(model, cudnn)
--model:cuda()
criterion = nn.ClassNLLCriterion()
--criterion = criterion:cuda()
LR = 0.01 --Learning rate
epochs = 15  --Number of epochs
------------Train model-------------------------------------------------
data_sz = train_data:size(1)
local train_err = torch.zeros(1,epochs)
local val_err = torch.zeros(1,epochs)
for n = 1,epochs do	
	print('Epoch ' .. n)
	err = 0
	for i = 1,data_sz do	
	  -- feed data to the neural network and the criterion 
	  data = train_data[{{i},{},{},{}}]	  
	  target = train_label[i]  
	  output = model:forward(data)
	  criterion:forward(output,target)	  
	  -- train over this example in 3 steps
	  -- (1) zero the accumulation of the gradients
	  model:zeroGradParameters()
	  -- (2) accumulate gradients
	  model:backward(data, criterion:backward(model.output, target))
	  -- (3) update parameters with a 0.01 learning rate
	  model:updateParameters(LR)	  
	  err_train = criterion:forward(output,target)	 
	  err = err + err_train
	 -- print(err)
	end	
	train_err[{{1},{n}}] = err/data_sz
	----------------Finding validation error--------------------------
	val_output = model:forward(val_data)	
    val_err[{{1},{n}}] = criterion:forward(val_output,val_label)
end
------------Plot performance------------------------------------------
gnuplot.pngfigure('ErrorvsEpochs.png')

gnuplot.figure()
gnuplot.plot({'Training error',train_err[1]},{'Validation error',val_err[1]})
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Training error')
gnuplot.grid(true)
gnuplot.title('Plot of error vs. epochs')
gnuplot.plotflush()
-------------Test model------------------------------------------------
sample_test = test_data[{{5,15},{},{},{}}]
sample_label = test_label[{{5,15}}]
pred = torch.exp(model:forward(sample_test))
pred_val,pred_class = torch.max(pred,2)
-------------Printing results---------------------------------------
print('Predicted class...')
print(pred_class:transpose(1,2)) -- predicted class
print('Ground truth label...')
print(sample_label:transpose(1,2)) -- goundtruth class