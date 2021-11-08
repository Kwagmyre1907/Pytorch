# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
'''
Prediction: Pytorch Model
Gradients Computation: Autograd
Loss Computation: Pytorch Loss
Parameter Updates: Pytoch Optimizer

1) Design Model (input, output, forward pass)
2) Construct loss and optimizer
3) Training loop
    - Forward pass: compute prediction
    - Backward pass: gradients
    - Update weights
'''
import torch
import torch.nn as nn # torch neural network

# f = w * x
# f = 2 * x (w = 2)
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_sample, n_features = X.shape
print(n_sample, n_features)

input_size = n_features
output_size = n_features


# %%
# prediction model
model = nn.Linear(input_size, output_size)

learning_rate = 0.1
n = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# %%
# training loop
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

for epoch in range(n):
    # prediction
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    #  dl/dw
    l.backward() 

    # update weights
    optimizer.step()
    
    # have to zero gradients with every iteration
    optimizer.zero_grad()

    if epoch % 2 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')


