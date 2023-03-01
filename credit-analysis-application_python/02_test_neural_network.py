import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# load the data set
data = pd.read_csv('german.data-numeric', delim_whitespace=True, header=None, on_bad_lines='skip')

# define the neural network
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

X = data.iloc[:, 0:20]#df.iloc[:, :-1]#df.iloc[:, 0:6]#df.iloc[:, :-1]
y = data.iloc[:, -1] - 1

# split training and testing data
_, x_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# open the pickle file to load the train mean and std
with open('mean_std.pkl', 'rb') as f:
    [x_train_mean, x_train_std] = pickle.load(f)

x_test = (x_test - x_train_mean) / x_train_std

# load the model
model = MLP(20, 10, 2)
model.load_state_dict(torch.load('model.pt'))

# test the model

x_test = torch.tensor(x_test.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)
test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

from sklearn.metrics import roc_auc_score
with torch.no_grad():
    running_predicted_tensor = torch.tensor([])
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        running_predicted_tensor = torch.cat((running_predicted_tensor, predicted), 0)
    auc = roc_auc_score(y_test, running_predicted_tensor)
    print('AUC: {}'.format(auc))