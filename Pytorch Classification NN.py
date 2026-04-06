from sklearn.datasets import make_circles
import torch
from torch import nn 
''' 
# Make 1000 samples 
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values


X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(len(X_train), len(X_test), len(y_train), len(y_test))


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.layer_1 = nn.Linear(2, 25)
        self.layer_2 = nn.Linear(25, 25)
        self.layer_3 = nn.Linear(25, 25)
        self.layer_4 = nn.Linear(25, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))


model_0 = CircleModelV0()

# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)


def eval_pred(y_true, y_pred): 
    list_equal = [y_pred for i in range(len(y_pred)) if y_pred[i] == y_true[i]]
    return 100*(len(list_equal)/len(y_true)) # Get percentage of accurate pieces. 


epochs = 2001
for i in range(epochs): 
    y_logits = model_0(X_train).squeeze()
    y_pred_labls = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    
    acc = eval_pred(y_train, y_pred_labls)
    if i % 50 == 0:
        print(f"Loss: {loss}, Accuracy: {acc}, Epoch: {i}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''


#-----------------------------------------------------------------------------------------------------------
# Multi-Class classification
# Import dependencies


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
    n_features=NUM_FEATURES, # X features
    centers=NUM_CLASSES, # y labels 
    cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
    random_state=RANDOM_SEED
)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# 3. Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED
)


class Blob(nn.Module): 
    def __init__(self, inp_features, out, hidden):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(nn.Linear(inp_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x): 
        return self.linear_layer_stack(x)


b = Blob(NUM_FEATURES, NUM_CLASSES, 8)

epochs = 2001
loss_CE = nn.CrossEntropyLoss()
optim = torch.optim.SGD(b.parameters(), lr = 0.01)
for i in range(epochs): 
    optim.zero_grad()

    output = b.forward(X_blob_train) # Logits
    blob_predictions = torch.softmax(output, dim=1)

    loss = loss_CE(output, y_blob_train,) # Cross Entropy Loss by default applies softmax onto the logits and then compares. 
    loss.backward()
    optim.step()
    if i % 50 == 0: 
        print(f"Epoch: {i}, Loss: {loss}")



