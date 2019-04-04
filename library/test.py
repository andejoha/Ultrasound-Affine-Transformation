# -*- coding: utf-8 -*-
import math
import torch
import network
import affine_transformation as at

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
moving_image = torch.randn(1, 1, 15, 15, 15).cuda()
target_image = torch.randn(1, 1, 15, 15, 15).cuda()
x = torch.cat((moving_image, target_image), 1)
true_image = torch.randn(15, 15, 15).cuda()
y = torch.ones(7).cuda()
# Create random Tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = network.Net(32)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for t in range(200):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Applying output constraints
    translation_vector = y_pred[:, :3].detach()
    scaling_vector = y_pred[:, 3:6].detach()
    rotation = y_pred[:, 6:].detach()

    print translation_vector
    print scaling_vector
    print rotation

    scaling_vector = torch.clamp(scaling_vector, 0.5, 1.5)
    rotation = torch.clamp(rotation, -math.pi / 6, math.pi / 6)

    print torch.cat((translation_vector, scaling_vector, rotation), 1)


    print y_pred

    transformed_image = at.translation(moving_image[0, 0].cpu(), translation_vector)
    transformed_image = at.scaling(transformed_image, scaling_vector)
    transformed_image = at.rotation(transformed_image, rotation)
    predicted_image = torch.from_numpy(transformed_image).cuda().requires_grad_()

    # print predicted_image
    # print z

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    #print 'net.conv_net.conv1_m.bias.grad before zero_grad', model.conv_net.conv1_m.bias.grad
    optimizer.zero_grad()

    #print 'net.conv_net.conv1_m.bias.grad before backward', model.conv_net.conv1_m.bias.grad
    loss.backward()

    #print 'net.conv_net.conv1_m.bias.grad after backward', model.conv_net.conv1_m.bias.grad
    #print
    optimizer.step()

print y_pred
print y
