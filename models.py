import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from config import Config

class GANLoss(nn.Module):
    """GAN loss implementation"""
    
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


class UnetBlock(nn.Module):
    """U-Net block for generator architecture"""
    
    def __init__(self, nf, ni, submodule=None, input_channels=None, 
                 dropout=False, innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        
        if input_channels is None: 
            input_channels = nf
            
        downconv = nn.Conv2d(input_channels, ni, kernel_size=Config.kernel_size, 
                           stride=Config.stride, padding=Config.padding, bias=False)
        downrelu = nn.LeakyReLU(Config.LeakyReLU_slope, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=Config.kernel_size, 
                                      stride=Config.stride, padding=Config.padding)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=Config.kernel_size, 
                                      stride=Config.stride, padding=Config.padding, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=Config.kernel_size, 
                                      stride=Config.stride, padding=Config.padding, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: 
                up += [nn.Dropout(Config.dropout)]
            model = down + [submodule] + up
            
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    """U-Net Generator"""
    
    def __init__(self, input_channels=1, output_channels=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, 
                                 submodule=unet_block, dropout=True)
        
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
            
        self.model = UnetBlock(output_channels, out_filters, 
                             input_channels=input_channels, 
                             submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    
    def __init__(self, input_channels, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_channels, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), 
                                stride=1 if i == (n_down-1) else 2) 
                 for i in range(n_down)]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, stride=1, 
                                norm=False, activation=False)]
        self.model = nn.Sequential(*model)
    
    def get_layers(self, ni, nf, kernel_size=Config.kernel_size, 
                  stride=Config.stride, padding=Config.padding, 
                  norm=True, activation=True):
        layers = [nn.Conv2d(ni, nf, kernel_size, stride, padding, bias=not norm)]
        if norm: 
            layers += [nn.BatchNorm2d(nf)]
        if activation:
            layers += [nn.LeakyReLU(Config.LeakyReLU_slope, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ColorizationGAN(nn.Module):
    """Main GAN model combining generator and discriminator"""
    
    def __init__(self, generator=None, gen_lr=Config.gen_lr, disc_lr=Config.disc_lr, 
                 beta1=Config.beta1, beta2=Config.beta2, lambda_l1=Config.lambda_l1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_l1 = lambda_l1
        
        if generator is None:
            self.generator = self._init_model(
                Unet(input_channels=1, output_channels=2, n_down=8, num_filters=64)
            )
        else:
            self.generator = generator.to(self.device)
            
        self.discriminator = self._init_model(
            Discriminator(input_channels=3, num_filters=64, n_down=3)
        )
        
        self.GANloss = GANLoss(gan_mode=Config.gan_mode).to(self.device)
        self.L1loss = nn.L1Loss()
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), 
                                        lr=gen_lr, betas=(beta1, beta2))
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), 
                                         lr=disc_lr, betas=(beta1, beta2))
        
    def _init_model(self, model):
        """Initialize model weights"""
        model = model.to(self.device)
        self._init_weights(model)
        return model
    
    def _init_weights(self, net, init='norm', gain=0.02):
        """Initialize network weights"""
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                if init == 'norm':
                    nn.init.normal_(m.weight.data, mean=0.0, std=gain)
                elif init == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1., gain)
                nn.init.constant_(m.bias.data, 0.)
                
        net.apply(init_func)
        
    def set_requires_grad(self, model, requires_grad=True):
        """Set requires_grad for model parameters"""
        for p in model.parameters():
            p.requires_grad = requires_grad
            
    def prepare_input(self, data):
        """Prepare input data"""
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        """Forward pass through generator"""
        self.gen_output = self.generator(self.L)
        
    def disc_backward(self):
        """Discriminator backward pass"""
        # Fake images
        gen_image = torch.cat([self.L, self.gen_output], dim=1)
        gen_image_preds = self.discriminator(gen_image.detach())
        self.disc_loss_gen = self.GANloss(gen_image_preds, False)
        
        # Real images
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.discriminator(real_image)
        self.disc_loss_real = self.GANloss(real_preds, True)
        
        self.disc_loss = (self.disc_loss_gen + self.disc_loss_real) * 0.5
        self.disc_loss.backward()
    
    def gen_backward(self):
        """Generator backward pass"""
        gen_image = torch.cat([self.L, self.gen_output], dim=1)
        gen_image_preds = self.discriminator(gen_image)
        self.loss_G_GAN = self.GANloss(gen_image_preds, True)
        self.loss_G_L1 = self.L1loss(self.gen_output, self.ab) * self.lambda_l1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize_parameters(self):
        """Optimize both generator and discriminator"""
        self.forward()
        
        # Update discriminator
        self.discriminator.train()
        self.set_requires_grad(self.discriminator, True)
        self.disc_optim.zero_grad()
        self.disc_backward()
        self.disc_optim.step()
        
        # Update generator
        self.generator.train()
        self.set_requires_grad(self.discriminator, False)
        self.gen_optim.zero_grad()
        self.gen_backward()
        self.gen_optim.step()


def build_backbone_generator(input_channels=1, output_channels=2, size=Config.image_size_1):
    """Build generator with ResNet18 backbone"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet18(pretrained=True)
    body = create_body(resnet_model, pretrained=False, n_in=input_channels, cut=Config.layers_to_cut)
    generator = DynamicUnet(body, output_channels, (size, size)).to(device)
    return generator