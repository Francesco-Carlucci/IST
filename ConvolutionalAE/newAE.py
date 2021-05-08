import torch

class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_layerIn = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3) # ,padding=1
        self.encoder_layerHid1 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5,stride=2)  #, padding=1
        self.encoder_layerHid2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=7, stride=2)   #kernel a 7, senza output padding
        self.encoder_layerOut = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=9, stride=(2,1))  # kernel a 7, senza output padding
        self.decoder_layerIn = torch.nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=9, stride=(2,1))   #, output_padding=1
        self.decoder_layerHid1 = torch.nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=7, stride=2)  # , output_padding=1
        self.decoder_layerHid2 = torch.nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=5, stride=2, output_padding=1)  #, padding=1, output_padding=1
        self.decoder_layerOut = torch.nn.ConvTranspose2d(
            in_channels=16, out_channels=3, kernel_size=3)   # padding=1
        self.pool = torch.nn.MaxPool2d(2,2,return_indices=True) #not used yet
        self.unpool=torch.nn.MaxUnpool2d(2,2)
        self.dropout= torch.nn.Dropout(0.5)

    def forward(self, data):
        print("Autoencoder: ",data.shape) #= [900,1600]
        activation = self.encoder_layerIn(data)
        activation = torch.relu(activation)
        #activation=self.pool(activation)
        print(activation.shape)  #=  [898,1598]
        x = self.encoder_layerHid1(activation)
        x = torch.relu(x)
        x = self.dropout(x)
        print(x.shape) #= (36-3+2)/2+1,(320-3+2)/2+1 => [447,797]
        x = self.encoder_layerHid2(x)
        x = torch.relu(x)
        x,ind1 = self.pool(x)
        x = self.dropout(x)
        print(x.shape) #=[221,396]
        code = self.encoder_layerOut(x)
        code = torch.relu(code)
        code, ind2 = self.pool(code)
        code = self.dropout(code)
        print(code.shape) #= [107,388] output code of the encoder
        x = self.unpool(code, ind2)
        x = self.decoder_layerIn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        print(x.shape) #= [221,396]
        x = self.unpool(x, ind1)
        x = self.decoder_layerHid1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        print(x.shape) #= [447,797]
        x = self.decoder_layerHid2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        print(x.shape)  # = [898,1598]
        activation = self.decoder_layerOut(x)
        #activation = self.dropout(activation)
        reconstructed = torch.sigmoid(activation)
        print(reconstructed.shape) #= [900,1600]
        return reconstructed