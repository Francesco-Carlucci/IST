import os
import numpy as np
import torch
import torchvision
import cv2

class myDataset(torch.utils.data.Dataset) :

    def __init__(self, images,masks):
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        mask=self.masks[index]
        image= self.images[index]
        image = self.transform(image)

        return image, mask
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def loadDataset(imgdir, maskdir, PoolOnly) :
    iw = int(1600 / 5)  # initialized now, but they will be set to the actual image.shape[]/5 later
    ih = int(900 / 5)
    images = []
    masks = []
    for filename in os.listdir(imgdir):

        mask = np.empty((ih,iw,3)) #in case of Dataset without border removal, we fill it with empty masks
        path = os.path.join(imgdir, filename)
        print(path)
        if (imgdir == 'imgPoolOnly' or PoolOnly):
            maskname = filename.replace("Inpainted", "")
            maskpath = os.path.join(maskdir, maskname)
            try:
                mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            except:
                maskpath=maskpath.replace('centre', 'centrÃƒÂ©')
                maskpath = maskpath.replace('centrÃ©', 'centrÃƒÂ©')
                #maskpath = maskpath.replace('centrÃƒÂ©', 'centre')   # 1
                #maskpath = maskpath.replace('centrÃƒÂ©', 'centrÃ©')  # 2
                mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (iw, ih), cv2.INTER_AREA)
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        ih = int(image.shape[0]/5)      # resize image to 1/5 size for processing
        iw = int(image.shape[1]/5)      # so the dimension of first layer is 320*180=57600
        #image = cv2.resize(image, (iw, ih), cv2.INTER_AREA)

        images.append(image)
        masks.append(mask)
        """
        ax= plt.subplot(1, 1, 1)
        ax.set_title('Mask')
        plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        plt.show()
        """
    images = np.array(images)
    masks = np.array(masks)
    return images, masks
def loadModel(model, optimizer, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        zipp = torch.load(filename)
        #start = zipp['epoch']
        model.load_state_dict(zipp['state_dict'])
        optimizer.load_state_dict(zipp['optimizer'])
        print("Loaded model from ",filename,'trained on ',zipp['epoch'],' epochs')
    else:
        print("no trained model found at ",filename)

    return model, optimizer  #,start