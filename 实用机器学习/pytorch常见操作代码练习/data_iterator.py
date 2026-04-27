# import torch
# import random
#
# from autogluon import features
#
#
# def data_iter(batch_size,features,labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0,num_examples,batch_size):
#         batch_indices = indices[i:min(i+batch_size,num_examples)]
#         yield features[batch_indices],labels[batch_indices]
#
# w = torch.normal(0,0.01,size=(10,1),requires_grad=True)
# b = torch.zeros(1,requires_grad=True)
# num_epochs = 10
#
# for epoch in range(num_epochs):
#     for X,y in data_iter(batch_size,features,labels):
#         y_hat = X@w + b
#         loss = ((y_hat-y)**2/2).mean()
#         loss.backward()
#         for param in [w,b]
#             param -= learning_rate*param.grad
#             param.grad.zero_()
