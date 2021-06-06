XMLProcessing: script that reads labels in the xml files and inpaint the swimmers

LineDetExp: script that detect the pools, remove outer border and generate a mask for each image

ConvolutionalAE: scripts for training and testing the autoencoder

                 - AE and oldAE : scripts with the autoencoder architectures
                 
                 - Dataset : function to load the dataset and the trained models from file
                 
                 - main : script for autoencoder training
                 
                 - ModelTest : script that load a trained model and show what it does on the testset 
                 
                              (make heatmaps, from loss detect areas of anomalies and their bounding boxes)
                             
                 - newModel [10 50 100] trained models with new arcitecture
                 
                 - oldModel [10 20] trained models with old architecture (less layers)
                 
                 - Models and Results : first batch of reconstructed images and graphs
                 
Exercices : little programs made to understand the basics

Articles : Articles ,collected on internet, similar to this project or that could be useful (in particular 1.Creusot, second one, and Sagi Eppel)
