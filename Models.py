from torchvision import models
import torch.nn as nn
import pdb


class pronet(nn.Module):
    def __init__(self, nClasses = 200):
        super(pronet,self).__init__();
        
        self.conv_1 = nn.Conv2d(3,32,kernel_size=3,stride=1, padding = 1)
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(32);
	self.conv_6 = nn.Conv2d(32,32,kernel_size=1,stride=2)
        #self.pool_1 = nn.MaxPool2d(kernel_size = 2, stride =2)
        
        self.conv_2 = nn.Conv2d(32,64,kernel_size=5,stride=1, padding = 2)
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(64);
	
        self.conv_3 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding = 1)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(64);
	
        self.conv_4 = nn.Conv2d(64,128, kernel_size=5, stride=1, padding = 2)
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(128);
	self.pool_3 = nn.MaxPool2d(kernel_size = 4, stride =4)

        self.conv_5 = nn.Conv2d(128,256, kernel_size=5, stride=1, padding = 2)
        self.relu_5 = nn.ReLU(True);
        self.batch_norm_5 = nn.BatchNorm2d(256);
	#self.conv_7 = nn.Conv2d(256,256,kernel_size=1,stride=2)
	self.pool_4 = nn.AvgPool2d(kernel_size = 2, stride =2)

        self.fc_1 = nn.Linear(4*4*256, 1024);
        #self.fc_1 = nn.Linear(8192, 200);
        self.relu_6 = nn.ReLU(True);
        self.batch_norm_6 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p = 0.5);
        self.fc_2 = nn.Linear(1024, nClasses);
        
    def forward(self,x):
        #pdb.set_trace();
        y = self.conv_1(x)
	#print y.size();
        y = self.relu_1(y)
	
        #y = self.batch_norm_1(y)
	#print y.size();
        #y = self.pool_1(y)
	y = self.conv_6(y)
        
        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        #y = self.pool_2(y)
        
        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
	
        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)

	
	y = self.pool_3(y)
	#print y.size()
	
        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)
	y = self.pool_4(y)
	#y = self.conv_7(y)
	#print y.size()


        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return(y)

class Demo_Model(nn.Module):
	
    def __init__(self, nClasses = 200):
        super(Demo_Model2,self).__init__();
        
        self.conv_1 = nn.Conv2d(3,32,kernel_size=3,stride=1, padding = 1)
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(32);
        self.pool_1 = nn.MaxPool2d(kernel_size = 2, stride =2)
        
        self.conv_2 = nn.Conv2d(32,64,kernel_size=5,stride=1, padding = 2)
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(64);
        self.pool_2 = nn.MaxPool2d(kernel_size = 4, stride =4)
	
        self.conv_3 = nn.Conv2d(64,64, kernel_size=5, stride=1, padding = 2)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(64);
	
        self.conv_4 = nn.Conv2d(64,128, kernel_size=5, stride=1, padding = 2)
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(128);

        self.fc_1 = nn.Linear(8*8*256, 1024);
        #self.fc_1 = nn.Linear(8192, 200);
        self.relu_5 = nn.ReLU(True);
        self.batch_norm_5 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p = 0.5);
        self.fc_2 = nn.Linear(1024, nClasses);
        
    def forward(self,x):
        #pdb.set_trace();
        y = self.conv_1(x)
	#print y.size();
        y = self.relu_1(y)
	
        y = self.batch_norm_1(y)
	#print y.size();
        y = self.pool_1(y)
        
        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)
        
        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
	
        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)


        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return(y)

class Demo_Model2(nn.Module):
	
    def __init__(self, nClasses = 200):
        super(Demo_Model2,self).__init__();
        
        self.conv_1 = nn.Conv2d(3,32,kernel_size=3,stride=1, padding = 1)
        self.relu_1 = nn.ReLU(True);
        #self.batch_norm_1 = nn.BatchNorm2d(32);
        self.conv_2 = nn.Conv2d(32,32,kernel_size=1,stride=2)
        #self.pool_1 = nn.MaxPool2d(kernel_size = 2, stride =2)
        
        self.conv_3 = nn.Conv2d(32,64,kernel_size=5,stride=1, padding = 2)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(64);
        self.conv_4 = nn.Conv2d(64,64,kernel_size=1,stride=2)
        #self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)
	
        self.conv_5 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding = 1)
        self.relu_5 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(64);
	
        self.conv_6 = nn.Conv2d(64,128, kernel_size=5, stride=1, padding = 2)
        self.relu_6 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(128);
        self.conv_7 = nn.Conv2d(128,128,kernel_size=1,stride=2)
	#self.pool_3 = nn.MaxPool2d(kernel_size = 2, stride =2)

        self.conv_8 = nn.Conv2d(128,256, kernel_size=5, stride=1, padding = 2)
        self.relu_8 = nn.ReLU(True);
        self.batch_norm_5 = nn.BatchNorm2d(256);
	self.pool_3 = nn.AvgPool2d(kernel_size = 2, stride =2)

        self.fc_1 = nn.Linear(4*4*256, 1024);
        #self.fc_1 = nn.Linear(8192, 200);
        self.relu_9 = nn.ReLU(True);
        self.batch_norm_6 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p = 0.5);
        self.fc_2 = nn.Linear(1024, nClasses);
        
    def forward(self,x):
        #pdb.set_trace();
        y = self.conv_1(x)
	#print y.size();
        y = self.relu_1(y)
       # a = self.pool_1(y)
	
	#print (a.size())
	y = self.conv_2(y)
	#print (y.size())
        #y = self.batch_norm_1(y)
        
        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_2(y)
        #y = self.pool_2(y)
	y = self.conv_4(y)        

        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_3(y)
	#print y.size();
	
        y = self.conv_6(y)
        y = self.relu_6(y)
        y = self.batch_norm_4(y)
	y = self.conv_7(y)
	
	#print y.size()
	
        y = self.conv_8(y)
        y = self.relu_8(y)
        y = self.batch_norm_5(y)
	y = self.pool_3(y)
	#print y.size();
	#print y.size()


        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_9(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return(y)
        
        
def resnet18(pretrained = True):
    return models.resnet18(pretrained)
    
#def pronet():
 #   return pronet();

def demo_model2():
    return Demo_Model2();

def alex_net(pretrained = True):
	return models.alexnet(pretrained)
