# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn


df = pd.read_csv('./data/userdata.csv')

df.head()

print("Dataset contains %s actions:", len(df))

"""### Let's plot the actions for one user

From the output can be seen that user completed an offer `0b1e...` and viewed `ae26...`. Offer `2906..` had been ignored twice.
"""

df[df.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6' ]

"""### Preparing data for training
Let's create user-offer matrix by encoding each id into categorical value.
"""

def to_categorical(df, columns):
    for col in columns:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes
    return df

df.dtypes

"""Recommendation matrix is very similar to embeddings. So we will leverage this and will train embedding along the model."""

# Set embedding sizes
N = len(df['id'].unique())
M = len(df['offer_id'].unique())

# Set embedding dimension
D = 100

# Create a neural network
class Model(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, output_dim, layers=[1024], p=0.4):
        super(Model, self).__init__()
        self.N = n_users
        self.M = n_items
        self.D = embed_dim

        self.u_emb = nn.Embedding(self.N, self.D)
        self.m_emb = nn.Embedding(self.M, self.D)

        layerlist = []
        n_in = 2 * self.D
        for i in layers:
            layerlist.append(nn.Linear(n_in,i))
            layerlist.append(nn.ReLU())
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],output_dim))
        self.layers = nn.Sequential(*layerlist)

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, u, m):
        u = self.u_emb(u) # output is (num_samples, D)
        m = self.m_emb(m) # output is (num_samples, D)

        # merge
        out = torch.cat((u, m), 1) # output is (num_samples, 2D)

        x = self.layers(out)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Model(N, M, D, output_dim=df['event'].nunique(), layers=[512, 256])
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)

# offer_specs = ['offer_id', 'difficulty', 'duration', 'reward', 'web',
#        'email', 'mobile', 'social', 'bogo', 'discount', 'informational']
# user_specs = ['age', 'became_member_on', 'gender', 'id', 'income', 'memberdays']

df['offer_id']

# Convert to tensors

# We name events as rating for give better insight on the target value
# and ease of comparison to other similar models

# user_ids_t = torch.from_numpy(df['id'].values).long()
# offer_ids_t = torch.from_numpy(df['offer_id'].values).long()
# ratings_t = torch.from_numpy(df['event'].values).long()

user_ids_t = torch.tensor(df['id'].astype('category').cat.codes.values, dtype=torch.long)
offer_ids_t = torch.tensor(df['offer_id'].astype('category').cat.codes.values, dtype=torch.long)
ratings_t = torch.from_numpy(df['event'].values).long()

# Make datasets

N_train = int(0.8 * len(df['event'].values))
N_test = 1000
train_dataset = torch.utils.data.TensorDataset(
    user_ids_t[:N_train],
    offer_ids_t[:N_train],
    ratings_t[:N_train],
)

val_dataset = torch.utils.data.TensorDataset(
    user_ids_t[N_train:-N_test],
    offer_ids_t[N_train:-N_test],
    ratings_t[N_train:-N_test],
)
test_df = df[-N_test:]

# Data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_iter, test_iter, epochs):

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    acc_list = []

    for it in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for users, offer, targets in train_loader:


            # move data to GPU
            users, offer, targets = users.to(device), offer.to(device), targets.to(device)
            #targets = targets.view(-1, 1).long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(users, offer)

            loss = criterion(outputs, targets.squeeze())

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            # Track the accuracy
            total = targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            acc = correct / total
            acc_list.append(acc)

        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading

        val_loss = []

        # validation
        with torch.no_grad():
            model.eval()

            for users, offer, targets in validation_loader:
                users, offer, targets = users.to(device), offer.to(device), targets.to(device)
                #targets = targets.view(-1, 1).long()
                outputs = model(users, offer)
                loss = criterion(outputs, targets.squeeze())
                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        # Save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Validation Loss: {train_loss:.4f}, '
              f'Test Loss: {val_loss:.4f}, Accuracy: {acc}, Duration: {dt}')

    return train_losses, val_losses

train_losses, val_losses = batch_gd(model, criterion, optimizer, train_loader, validation_loader, 5)

# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='test loss')
plt.legend()
plt.show()

start_ix = 10
end_ix = 20


test_X = torch.tensor(test_df['id'][start_ix:end_ix].astype('category').cat.codes.values, dtype=torch.long)
test_y = torch.from_numpy(test_df.iloc[start_ix:end_ix]['event'].values).long()

with torch.no_grad():

    model.eval()
    pred = model(test_X, test_y)
    print(pred)
_, predicted = torch.max(pred.data, 1)
print(predicted)

from sklearn.metrics import confusion_matrix,  accuracy_score
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

"""#### Plot confusion matrix and baseline accuracy"""

cm = confusion_matrix(test_y, predicted)
classes = [0,1,2]
plot_confusion_matrix(cm, classes)

print("Accuracy so far: " + str(100*accuracy_score(test_y, predicted))+ "%" )

"""Results are decent so far and almost twice better than random quessing.

#### Show some misclassified examples
"""

data = test_df.iloc[start_ix:end_ix][['age', 'became_member_on', 'gender', 'id', 'income', 'memberdays', 'event']]#['offer_id'].values
pred_values = pd.DataFrame(predicted, columns=['predicted'], index=data.index)
pd.concat([data, pred_values], axis=1)

"""Now let's save the model for future reference"""

#!mkdir model
import os

def save_model(model, model_name, model_info):
    # Save the parameters used to construct the model
    with open(model_name, 'wb') as f:
        torch.save(model_info, f)

    # Save the model parameters

    with open(model_name, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

model_info = {
         'n_users': M,
          'n_items': N,
          'embed_dim': D,
          'output_dim': df['event'].nunique(),
          'layers': [512, 256],
          'p': 0.4
    }
save_model(model, './model/BaselineModel2.pth', model_info)

"""During the next step we improve the model to take additional paramenters that describe each user and each offer, which should hopefully, give the model insigths on why a particular customer may like or not like given offer."""