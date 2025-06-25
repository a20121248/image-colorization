"""
Configuration file for the Colorization GAN
"""

class Config:
    # Data parameters
    external_data_size = 21837
    train_size = 20000
    image_size_1 = 256
    image_size_2 = 256
    batch_size = 32
    
    # Model parameters
    LeakyReLU_slope = 0.2
    dropout = 0.5
    kernel_size = 4
    stride = 2
    padding = 1
    layers_to_cut = -2
    
    # Training parameters
    gen_lr = 2e-4
    disc_lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    lambda_l1 = 100
    gan_mode = 'vanilla'
    epochs = 30
    pretrain_lr = 1e-4
    
    # Paths
    model_save_dir = "models"
    pretrained_generator_path = "models/res18-unet.pt"
    final_model_path = "models/final-model.pt"
    
    # Display parameters
    display_interval = 100
    save_interval = 1  # Save every N epochs