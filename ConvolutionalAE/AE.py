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
        #self.pool = torch.nn.MaxPool2d(2,2) #not used yet

    def forward(self, data):
        print("Autoencoder: ",data.shape) #= [180,320,3]
        activation = self.encoder_layerIn(data)
        activation = torch.relu(activation)
        print(activation.shape)  #=  [178,318]
        x = self.encoder_layerHid1(activation)
        x = torch.relu(x)
        print(x.shape) #= (36-3+2)/2+1,(320-3+2)/2+1 => [87,157]
        x = self.encoder_layerHid2(x)
        x = torch.relu(x)
        print(x.shape) #=[41,76]
        code = self.encoder_layerOut(x)
        code = torch.relu(code)
        print(code.shape) #= [17,68] output code of the encoder
        x = self.decoder_layerIn(code)
        x = torch.relu(x)
        print(x.shape) #= [41,76]
        x = self.decoder_layerHid1(x)
        x = torch.relu(x)
        print(x.shape) #= [87,157]
        x = self.decoder_layerHid2(x)
        x = torch.relu(x)
        print(x.shape)  # = [178,318]
        activation = self.decoder_layerOut(x)
        reconstructed = torch.sigmoid(activation)
        print(reconstructed.shape) #= [180,320]
        return reconstructed