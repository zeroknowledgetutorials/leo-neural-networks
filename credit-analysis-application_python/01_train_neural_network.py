import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# load the german.data-numeric data set
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
x_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

# normalize the data
x_train_mean = x_train.mean()
x_train_std = x_train.std()

x_train = (x_train - x_train_mean) / x_train_std

# convert pandas dataframes to tensors
x_train = torch.tensor(x_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)

# combine the data into a dataset and dataloader
dataset = TensorDataset(x_train, y_train)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MLP(20, 10, 2)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# train the model
for epoch in range(100):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# save model
torch.save(model.state_dict(), 'model.pt')

# save the mean and standard deviation using pickle
with open('mean_std.pkl', 'wb') as f:
    pickle.dump((x_train_mean, x_train_std), f)