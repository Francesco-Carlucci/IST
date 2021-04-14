import os
import time

import torch
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2
from AE import AE
from Dataset import myDataset,loadDataset,loadModel

PoolOnly= True
justModel=False
imgdir = './trainsetInpainted'
maskdir= './imgMaskOnly'
mdlFile= './oldModel20.pt'
trndMdl= './oldModel10.pt'
trainRounds=10

if __name__ == '__main__':

    #never tested on GPU, i don't have an nvidia one where to run CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #load the dataset directory, only once at the start
    images,masks=loadDataset(imgdir,maskdir,PoolOnly)
    ih=images[0].shape[0]
    iw=images[0].shape[1]

    model = AE(input_shape=ih * iw)

    # optimizer object with learning rate 10^-3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model,optimizer=loadModel(model,optimizer,trndMdl)

    model = model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # training loss function mean squared error
    criterion = torch.nn.MSELoss()

    train_set= myDataset(images, masks)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=10, shuffle=True, num_workers=4, pin_memory=True
    )
    losses = np.empty(trainRounds, dtype=np.float)
    for epoch in range(trainRounds):
        loss = 0
        d=0
        start = time.time()
        for batch, mask in train_loader:
            #print("batch ",d,":", batch.shape, mask.shape) #check on the data dimensions
            #batch = batch.view(-1, ih*iw).to(device) #flattening, needed only for linear layers
            optimizer.zero_grad() # reset the gradients at each batch

            outputs = model(batch) # give images to the autoencoder

            if(imgdir=='imgPoolOnly' or PoolOnly):   #substitute everything out of the border with the pixel
                for i in range(len(batch)) :      #of the input image, so loss would be 0. It use a mask for each image
                    batch[i].data.copy_((batch[i]*(mask[i]//255)))
                    inputShw=batch[i].permute(1,2,0).detach().numpy()
                    """
                    ax = plt.subplot(1, 1, 1)
                    ax.set_title('Input')
                    plt.imshow(inputShw)
                    plt.show()
                    """
                    outputShw=outputs[i].permute(1,2,0).detach().numpy()
                    """
                    ax = plt.subplot(1, 1, 1)
                    ax.set_title('Reconstructed')
                    plt.imshow(outputShw)
                    plt.show()
                    """
                    maskShw = mask[i].detach().numpy()
                    """
                    cv2.imshow("mask", maskShw)
                    cv2.waitKey()
                    """
                    # tried to reset pixel out of borders with opencv function,
                    # but they need numpy array, to convert from torch tensor to numpy it needs detach()
                    # on the outputs, but then is impossible to calculate the gradient, i finally bypassed that
                    # modifying the data field of the tensor as below

                    #imgres = cv2.bitwise_and(inputShw, inputShw, mask=cv2.bitwise_not(maskShw))
                    #outputShw = cv2.bitwise_and(outputShw, outputShw, mask=maskShw)
                    #outputShw = cv2.add(imgres, outputShw)
                    #outputs[i].data.copy_(torch.Tensor(outputShw).permute(2,0,1)) #using openCV about [100-140]s per epoch
                    outputs[i].data.copy_(outputs[i]*(mask[i]//255)) #paint out of border region in black
                    #to be used with the line after for that paint in black the input too [95-190]s per epoch
                    """
                    criterion = torch.nn.MSELoss(reduction='none')
                    loss=criterion(outputs, batch)
                    ax= plt.subplot(1,1,1)
                    ax.set_title('loss after modification')
                    ax.imshow(loss[i].permute(1, 2, 0).detach().numpy() * 8)
                    plt.show()
                    ax = plt.subplot(1, 1, 1)
                    ax.set_title('modified output')
                    plt.imshow(outputs[i].permute( 1, 2, 0).detach().numpy())
                    plt.show()
                    """
            d+=1

            train_loss = criterion(outputs, batch) # compute training loss

            train_loss.backward() # compute accumulated gradients

            optimizer.step()

            loss += train_loss.item() # add the batch loss to epoch loss
        end = time.time()
        print('duration', end - start)
        losses[epoch] = loss
        loss = loss / len(train_loader)

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, trainRounds, loss))

    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title('Loss')
    ax.plot(losses)
    if(not justModel):
        #save trained model and optimizer parameters for test or continue training
        state = {'epoch': trainRounds, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict() }
        torch.save(state, mdlFile)
    else:
        torch.save(model.state_dict(), mdlFile) #save trained model in a .pt file

    """
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor])
    """
    # visualize 5 input image and the relative reconstructed image
    input,_ = iter(train_loader).next()

    output = model(input)

    criterion = torch.nn.MSELoss(reduction='none')
    trained_loss = criterion(output, input) # compute training loss
    # trying to see regions with an high loss function
    for k in np.arange(input.shape[0]):
        _, ax = plt.subplots(nrows=1, ncols=2, figsize=(iw * 5, ih * 5), sharex=True, sharey=True)
        ax[1].set_title('loss mapping')
        ax[0].imshow(input[k].permute(1, 2, 0).detach().numpy())
        ax[1].imshow(trained_loss[k].permute(1, 2, 0).detach().numpy() * 10)
        plt.show()

    input = input.permute(0, 2, 3, 1).detach().numpy()  # change from (C,H,W) to (H,W,C)
    output = output.permute(0, 2, 3, 1).detach().numpy()  # with H:Height,W:Width,C:Channels

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(iw * 5, ih * 5), sharex=True, sharey=True)
    for i in np.arange(5):
        # ax[i].append(fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[]))
        image = cv2.resize(input[i], (iw * 5, ih * 5), cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[0, i].imshow(image)
    for i in range(5):
        # ax[i+6].append(fig.add_subplot(2, 5, i + 6, xticks=[], yticks=[]))
        image = cv2.resize(output[i], (iw * 5, ih * 5), cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[1, i].imshow(image)
    plt.imshow(image)
    plt.show()



