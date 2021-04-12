import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

class DMSNet(nn.Module):
    """ Define a CNN """

    def __init__(self, device='cpu'):
        """ Initialize network with some hyperparameters """
        super(DMSNet, self).__init__()

        self.device = device
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(87584, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.drop = nn.Dropout(p=0.5)

        self.output_dim = 1

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x)))).to(self.device)
        x = self.bn2(self.pool(F.relu(self.conv2(x)))).to(self.device)
        x = x.view(x.size(0), -1).to(self.device)
        x = self.drop(F.relu(self.fc1(x))).to(self.device)
        x = self.drop(F.relu(self.fc2(x))).to(self.device)
        x = F.softmax(self.fc3(x), dim=1).to(self.device)
        return x

# training the neural network
def train(X_train, Y_train, batch_size, model, criterion, optimizer):
        model.train() 
        train_err = 0 
        train_acc = []
        for k in range(batch_size, X_train.shape[0], batch_size):
            preds = model(X_train[k-batch_size:k])
            loss = criterion(preds, Y_train[k-batch_size:k])
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            train_err+=loss.item()
            thresholded_results = np.where(preds.detach().cpu().numpy() > 0.5, 1, 0)
            train_acc.append(accuracy_score(thresholded_results, 
                                    Y_train[k-batch_size:k].detach().cpu().numpy()))
        train_err /= (X_train.shape[0]/(batch_size))
        return train_err, np.mean(np.array(train_acc))

# Function for validation and test
def validate(X_valid, Y_valid, batch_size, model, criterion):
        valid_acc = []
        model.eval() 
        with torch.no_grad():
            valid_err = 0 
            for k in range(batch_size,X_valid.shape[0],batch_size):
                preds = model(X_valid[k-batch_size:k])          
                valid_err += criterion(preds, Y_valid[k-batch_size:k]).item()
                thresholded_results = np.where(preds.detach().cpu().numpy() 
                                               > 0.5, 1, 0)
                valid_acc.append(accuracy_score(thresholded_results, 
                                 Y_valid[k-batch_size:k].detach().cpu().numpy()))
        valid_err /= (X_valid.shape[0]/(batch_size))
        return valid_err, np.mean(np.array(valid_acc))

# The training loop
def learn(model, X_train, Y_train, X_valid, Y_valid, epochs=20, lrate=0.1, 
          batch_size=8, patience=30, model_filename='my_model.pt', 
          best_validation_model = 'valid_best_model.pt',
          training_history = 'training_history.npy', device='cpu',
          smlb_out=None):

        if smlb_out:
            console = smlb_out.log.console
        else:
            console.message('No output console defined, aborting job')
            return 

        best_valid_err = np.inf
        best_train_err = np.inf
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)
        errors = []
        for i in range(epochs):
            train_err, train_acc   = train(X_train, Y_train, batch_size, model,
                                           criterion, optimizer)
            valid_err, valid_acc   = validate(X_valid, Y_valid, batch_size, 
                                              model, criterion)
            if i == 0: patience_counter = 0
            # early stopping        
            if valid_err < best_valid_err:
                console.message('**saving valid current {:5.3f} best {:5.3f}**'.format(
                                                  valid_err, best_valid_err))
                torch.save(model.state_dict(), best_validation_model)
                best_valid_err = valid_err
                patience_counter = 0
            else:
                patience_counter+= 1
            if train_err < best_train_err:
                torch.save(model, model_filename)
                best_train_err = train_err
                console.message('**saving**')

            if patience_counter > patience:
                console.message("Validation error did not get better in the last {} epochs, \
                       early stopping".format(patience))
                break
            console.message("%d epoch, train_err: %.4f, train_acc: %.4f, valid_err: %.4f, valid_acc: %.4f"
                  % (i, train_err, train_acc, valid_err, valid_acc))
            errors.append([train_err, valid_err])
        console.message("BEST, train_err: %.4f, valid_err: %.4f" % (best_train_err, 
                                                          best_valid_err))
        np.save(training_history, np.asarray(errors))
