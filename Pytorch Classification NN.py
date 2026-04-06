from sklearn.datasets import make_circles
import torch
from torch import nn 


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


















































