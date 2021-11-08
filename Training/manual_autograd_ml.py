# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch

# Prediction: Manual
# Gradients Computation: Autograd
# Loss Computation: Manual
# Parameter Updates: Manual
# f = w * x
# f = 2 * x (w = 2)
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# %%
# model prediction
def forward(x):
    return w * x

# calculate loss (loss = MeanSqueareError)
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# calculate gradient
# torch automatically calculates the gradient

print(f'Prediction before training: f(5) = {forward(5):.3f}')


# %%
# training
learning_rate = 0.01
n = 100

for epoch in range(n):
    # prediction
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    #  dl/dw
    l.backward() 

    # update weights
    # NB: should not be part of gradient
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # have to zero gradients with every iteration
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')


