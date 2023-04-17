# For running the code, these packages are required:
# pip install torch
# pip install prettytable
#
# To run the code issue this command:
# python synthetic_v1.py
#
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import random
import sys, time

# Model parameters
from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params	
		
# non-Linear model		
class nonlinRegression1(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size):
        super().__init__()
        
        self.main = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size_1),  # input -> hidden layer
            nn.Sigmoid(),                        
            nn.Linear(hidden_size_1, hidden_size_2),	
            nn.Sigmoid(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.Sigmoid(),
			nn.Linear(hidden_size_3, hidden_size_4),
            nn.Sigmoid(),
			nn.Linear(hidden_size_4, output_size), # hidden -> output layer
            nn.Sigmoid()	
        )

    def forward(self, x):
        x = self.main(x)
        return x

# Custom dataset
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(MyDataSet, self).__init__()
        # store the raw tensors
        self._x = x
        self._y = y

    def __len__(self):
        # a DataSet must know it size
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]
        return x, y

# Training loop
def trainer(model, criterion, optimizer, dataloader, epochs=5, verbose=True):
    """Simple training wrapper for PyTorch network."""
   
    for epoch in range(epochs):
        losses = 0
		# Batch loop
        batchNo = 0
        for x1, y in dataloader:
         
            x = x1.unsqueeze(dim=1)      
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
			
            # Forward pass to get output
            x = x.cuda()
            y_hat = model(x).flatten()
            
            # Calculate loss
            y_hat = y_hat.cuda()
            y = y.cuda()
            loss = criterion(y_hat, y)  # Calculate loss
			
            # Getting gradients w.r.t. parameters
            loss.backward()  
			# Update parameters
            optimizer.step()
			
			# Add loss for this batch to running total
            losses = loss + loss.item()       
        if verbose: print(f"epoch: {epoch + 1}, loss: {losses / len(dataloader):.4f}")
   
# main
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 784
num_rows = 1024000
hidden_size_1=30000
hidden_size_2=30000
hidden_size_3=30000
hidden_size_4=30000
epochs = 1


# By default, the tensors are generated on the CPU.
# When to move tensors to device

X = torch.randn(num_rows, input_size)
y = X ** 2 + 15 * np.sin(X) **3
y_t = torch.sum(y, dim=1)

print(f'X.shape: {X.shape}, y_t.shape: {y_t.shape}')

#Non-linear model
model = nonlinRegression1(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, hidden_size_3=hidden_size_3, hidden_size_4=hidden_size_4, output_size=1)

count_parameters(model)

#loss
criterion = nn.MSELoss()  

#optimizer
LEARNING_RATE = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Batching data
BATCH_SIZE = 128
dataset = MyDataSet(X, y_t)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Number of samples: {num_rows}")
print(f"Total number of batches: {len(dataloader)}, {num_rows/BATCH_SIZE}")

# Look at batches
XX, yy = next(iter(dataloader))

# Move all tensors and the model to GPU
#X.cuda()
#y.cuda()
#y_t.cuda()
model.cuda()

# Training
trainer(model, criterion, optimizer, dataloader, epochs=epochs, verbose=True)

elapsed_time = time.time() - start_time
print(f'Elapsed time:{elapsed_time}')
