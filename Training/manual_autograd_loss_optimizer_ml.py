# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
'''
Prediction: Manual
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
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# %%
# model prediction
def forward(x):
    return w * x

learning_rate = 0.01
n = 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)


# %%
# training loop
print(f'Prediction before training: f(10) = {forward(10):.3f}')

for epoch in range(n):
    # prediction
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    #  dl/dw
    l.backward() 

    # update weights
    optimizer.step()
    
    # have to zero gradients with every iteration
    optimizer.zero_grad()

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(10) = {forward(10):.3f}')


