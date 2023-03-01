import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import math

# load the data set from the disk - you can download it here: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
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

for key in model.state_dict().keys():
    print(key)
    print(model.state_dict()[key])

str_list_inputs = []

str_inputs = ""

str_list_inputs.append("[main]\n")

for i, key in enumerate(model.state_dict().keys()):
    if(i==0):
        first_layer = model.state_dict()[key]
    layer_name = ""
    if('weight' in key):
        layer_name = "w"
    if('bias' in key):
        layer_name = "b"
        
    value = model.state_dict()[key]
    for j, val in enumerate(value):
        # get dimension of the value
        if(len(val.shape) == 1):
            for k, val2 in enumerate(val):
                val_fixed_point = int(val2 * 2**7)
                #variable_line = layer_name + str(math.floor(i/2)+1) + str(k) + str(j) + ": " + "u32 = " + str(val_fixed_point) + ";\n"
                variable_line = layer_name + str(math.floor(i/2)+1) + str(k) + str(j) + ": " + "u32 = " + str(0) + ";\n"
                str_list_inputs.append(variable_line)
        else:
            val_fixed_point = int(val * 2**7)
            #variable_line = layer_name + str(math.floor(i/2)+1) + str(j) + ": " + "u32 = " + str(val_fixed_point) + ";\n"
            variable_line = layer_name + str(math.floor(i/2)+1) + str(j) + ": " + "u32 = " + str(0) + ";\n"
            str_list_inputs.append(variable_line)

str_list_inputs.append("\n")

# load the data set
data = pd.read_csv('german.data-numeric', delim_whitespace=True, header=None, on_bad_lines='skip')

X = data.iloc[:, 0:20]#df.iloc[:, :-1]#df.iloc[:, 0:6]#df.iloc[:, :-1]
y = data.iloc[:, -1] - 1

# split training and testing data
_, x_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# open the pickle file to load the train mean and std
with open('mean_std.pkl', 'rb') as f:
    [x_train_mean, x_train_std] = pickle.load(f)

x_test = (x_test - x_train_mean) / x_train_std

x_test = torch.tensor(x_test.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)
test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

for i in range(len((first_layer[0]))):
    value = test_data[0][0][i]
    val_fixed_point = int(value * 2**7)
    str_list_inputs.append("input" + str(i) + ": u32 = " + str(val_fixed_point) + ";\n")

str_list_inputs.append("\n")
str_list_inputs.append("[registers]")
str_list_inputs.append("\n")
str_list_inputs.append("r0: [u32; 2] = [0, 0];")

with open("project.in", "w+") as file:
   file.writelines(str_list_inputs)