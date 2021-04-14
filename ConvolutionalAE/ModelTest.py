from matplotlib import pyplot as plt
import cv2
from oldAE import AE
from Dataset import myDataset,loadDataset
import torch
import numpy as np

PoolOnly= False
imgdir = './test_set'
maskdir= './imgMaskOnly'
mdlfile= './oldModel20.pt'
batch_size=5

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the dataset directory, only once at the start
    images, masks = loadDataset(imgdir, maskdir, PoolOnly)
    ih = images[0].shape[0]
    iw = images[0].shape[1]

    model = AE(input_shape=ih * iw).to(device)

    mdl=torch.load(mdlfile)       #allows interoperability between file with only the model
    if ('state_dict' in mdl):     #and file with the optimizer too, to continue the training
        model.load_state_dict(mdl['state_dict'])
    else :
        model.load_state_dict(torch.load(mdlfile))

    # training loss function mean squared error
    criterion = torch.nn.MSELoss( reduction= 'none')  #evaluating region loss
    criterion2 = torch.nn.MSELoss(reduction='sum')

    test_set = myDataset(images, masks)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    loss = 0

    for inputs, _ in test_loader:

        outputs = model(inputs)

        test_loss= criterion(outputs, inputs)

        #n, bins, patches = plt.hist(x=test_loss[0][0][0].view(-1, iw).detach().numpy()*2550, bins='auto')
        #plt.show()

            #visualize reconstructed images
        input = inputs.permute(0, 2, 3, 1).detach().numpy()  # change from (C,H,W) to (H,W,C)
        output = outputs.permute(0, 2, 3, 1).detach().numpy()  # with H:Height,W:Width,C:Channels
        for i in np.arange(input.shape[0]):
            print("mean reconstruction error:", criterion2(outputs[i],inputs[i]).detach().numpy())
            _,ax = plt.subplots(nrows=1, ncols=3, figsize=(iw * 5, ih * 5), sharex=True, sharey=True)
            ax[0].set_title('Input')
            ax[0].imshow(input[i])
            #imgLoss = np.sum(test_loss[i].permute(1, 2, 0).detach().numpy()*8,axis=2)
            #cv2.imshow('Loss', cv2.resize(imgLoss, (iw*5,ih*5)))
            #cv2.waitKey(0)
            ax[1].set_title('Loss')
            ax[1].imshow((test_loss[i].permute(1, 2, 0).detach().numpy()*60)**4)
            ax[2].set_title('Output')
            ax[2].imshow(output[i])
            plt.show()
        fig, axs = plt.subplots(nrows=2, ncols=batch_size, figsize=(iw * 5, ih * 5), sharex=True, sharey=True)
        for i in np.arange(input.shape[0]):
            #fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=(iw * 5, ih * 5), sharex=True, sharey=True)
            #ax[i].append(fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[]))
            image = cv2.resize(input[i], (iw * 5, ih * 5), cv2.INTER_AREA)
            image = cv2.cvtColor(input[i], cv2.COLOR_BGR2RGB)
            axs[0, i].imshow(image)
        for i in range(input.shape[0]):
            # ax[i+6].append(fig.add_subplot(2, 5, i + 6, xticks=[], yticks=[]))
            image = cv2.resize(output[i], (iw * 5, ih * 5), cv2.INTER_AREA)
            image = cv2.cvtColor(output[i], cv2.COLOR_BGR2RGB)
            axs[1, i].imshow(image)
        plt.show()


