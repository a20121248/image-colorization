import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
import requests
import os

# Import your modules
from config import Config
from models import ColorizationGAN, build_backbone_generator
from utils import load_model

# ------------------- Model URLs -------------------
MODEL1_PATH = 'models/generator_colorization.pth'
MODEL2_PATH = 'models/modelo_2.pt'
URL1 = 'https://www.dropbox.com/scl/fi/vkme2h1kdoqj74a6k8mrn/modelo_1.pth?rlkey=er10fsftgfy4qk3qjm2muw54o&st=5ybwb3xs&dl=1'
URL2 = 'https://www.dropbox.com/scl/fi/mssjgkyxbuebujgesv54q/modelo_2.pth?rlkey=eljb89ga16ymbxxlxgx4w6ioi&st=kqnwjtqy&dl=1'

def descargar_modelo(url, destino):
    """Download model from URL"""
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    response = requests.get(url, stream=True)
    with open(destino, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# Download models if they don't exist
if not os.path.exists(MODEL1_PATH):
    st.info("Descargando Modelo 1 desde Dropbox...")
    descargar_modelo(URL1, MODEL1_PATH)

if not os.path.exists(MODEL2_PATH):
    st.info("Descargando Modelo 2 desde Dropbox...")
    descargar_modelo(URL2, MODEL2_PATH)

# ------------------- Model 1 (U-Net) -------------------
class UnetBlock(torch.nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False, innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = torch.nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = torch.nn.BatchNorm2d(ni)
        uprelu = torch.nn.ReLU(True)
        upnorm = torch.nn.BatchNorm2d(nf)

        if outermost:
            upconv = torch.nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            model = [downconv, submodule, uprelu, upconv, torch.nn.Tanh()]
        elif innermost:
            upconv = torch.nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = torch.nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if dropout: model += [torch.nn.Dropout(0.5)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], 1)

class Unet(torch.nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)

# ------------------- Load Models -------------------
@st.cache_resource
def load_model1(path=MODEL1_PATH):
    """Load U-Net model"""
    model = Unet(input_c=1, output_c=2)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_model2(path=MODEL2_PATH):
    """Load ColorizationGAN model"""
    device = torch.device("cpu")  # Force CPU for Streamlit
    generator = build_backbone_generator(input_channels=1, output_channels=2, size=Config.image_size_1)
    model = ColorizationGAN(generator=generator)
    model = load_model(model, path, device)
    model.eval()
    return model

# Load both models
try:
    model1 = load_model1()
    st.success("Modelo 1 (U-Net) cargado exitosamente")
except Exception as e:
    st.error(f"Error cargando Modelo 1: {e}")
    model1 = None

try:
    model2 = load_model2()
    st.success("Modelo 2 (ColorizationGAN) cargado exitosamente")
except Exception as e:
    st.error(f"Error cargando Modelo 2: {e}")
    model2 = None

# ------------------- Inference Functions -------------------
def colorize_image_model1(uploaded_file, model):
    """Colorize using U-Net model (Model 1)"""
    SIZE = 256
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((SIZE, SIZE), Image.BICUBIC)
    img_np = np.array(img_resized)
    lab = rgb2lab(img_np).astype("float32")

    L = lab[..., 0]
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0) / 50. - 1.

    with torch.no_grad():
        ab = model(L_tensor)

    L_restored = (L_tensor + 1.) * 50.
    ab_restored = ab * 110.
    Lab = torch.cat([L_restored, ab_restored], dim=1).permute(0, 2, 3, 1).numpy()[0]
    img_rgb = lab2rgb(Lab)
    img_rgb = np.clip(img_rgb, 0, 1)

    return Image.fromarray((img_rgb * 255).astype(np.uint8)), img_resized

def preprocess_image_model2(uploaded_file):
    """Preprocess image for Model Pix2Pix (ResNet-18+DynamicUnet)"""
    img = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((Config.image_size_1, Config.image_size_2)),
        transforms.ToTensor()
    ])
    img = transform(img).permute(1, 2, 0).numpy()  # [H, W, C]
    lab = rgb2lab(img).astype("float32")
    L = lab[:, :, 0] / 50.0 - 1.0  # Normalize to [-1, 1]
    L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return L

def postprocess_ab_model2(L, ab):
    """Reconstruct RGB image from L and ab normalized"""
    L = (L + 1.0) * 50.0
    ab = ab * 128.0
    Lab = torch.cat([L, ab], dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = lab2rgb(Lab)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb)

def colorize_image_model2(uploaded_file, model):
    """Colorize using ColorizationGAN model (Model 2)"""
    # Preprocess image
    L = preprocess_image_model2(uploaded_file)
    
    # Create grayscale version for display
    img_original = Image.open(uploaded_file).convert("RGB")
    img_resized = img_original.resize((Config.image_size_1, Config.image_size_2), Image.BICUBIC)
    grayscale_resized = img_resized.convert("L").convert("RGB")
    
    # Inference
    with torch.no_grad():
        ab = model.generator(L)
    
    # Postprocess
    result = postprocess_ab_model2(L, ab)
    
    return result, grayscale_resized

# ------------------- UI Streamlit -------------------
st.set_page_config(page_title="Colorizador de Imágenes TAFOS", layout="wide")

st.title("🎨 Colorización de Imágenes con GAN")
st.markdown("""
**Tipo de modelo:** GAN Condicional  
**Versión del modelo:** 1.0 y 2.0  
**Curso:** INF658 - COMPUTACIÓN GRÁFICA  
**Alumnos:** Edward Rosales (19910608) / Javier Monzón (20121248)  
""")

tab1, tab2 = st.tabs(["🧠 Modelo 1 (U-Net)", "🧪 Modelo 2 (ColorizationGAN)"])

# Model 1 Tab
with tab1:
    st.markdown("<h2 style='text-align: center;'>Colorización con Modelo 1 (U-Net)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Arquitectura: U-Net Generator con PatchGAN Discriminator</p>", unsafe_allow_html=True)

    if model1 is not None:
        uploaded_file1 = st.file_uploader("📤 Carga tu imagen (.jpg o .png)", type=["jpg", "jpeg", "png"], key="upload1")

        if uploaded_file1:
            st.markdown("<h3>Resultado</h3>", unsafe_allow_html=True)

            with st.spinner("Colorizando imagen con Modelo 1..."):
                colorized_img, grayscale_resized = colorize_image_model1(uploaded_file1, model1)

            # Imágenes lado a lado centradas
            col_empty1, col1, col2, col_empty2 = st.columns([1, 3, 3, 1])
            with col1:
                st.image(grayscale_resized, caption="Escala de grises (256x256)", width=256)
            with col2:
                st.image(colorized_img, caption="Colorizado (256x256)", width=256)
    else:
        st.error("Modelo 1 no está disponible. Verifica la configuración.")

# Model 2 Tab
with tab2:
    st.markdown("<h2 style='text-align: center;'>Colorización con Modelo 2 (ColorizationGAN)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Arquitectura: ColorizationGAN con Backbone Generator</p>", unsafe_allow_html=True)

    if model2 is not None:
        uploaded_file2 = st.file_uploader("📤 Carga tu imagen (.jpg o .png)", type=["jpg", "jpeg", "png"], key="upload2")

        if uploaded_file2:
            st.markdown("<h3>Resultado</h3>", unsafe_allow_html=True)

            with st.spinner("Colorizando imagen con Modelo 2..."):
                try:
                    colorized_img, grayscale_resized = colorize_image_model2(uploaded_file2, model2)

                    # Imágenes lado a lado centradas
                    col_empty1, col1, col2, col_empty2 = st.columns([1, 3, 3, 1])
                    with col1:
                        st.image(grayscale_resized, caption=f"Escala de grises ({Config.image_size_1}x{Config.image_size_2})", width=256)
                    with col2:
                        st.image(colorized_img, caption=f"Colorizado ({Config.image_size_1}x{Config.image_size_2})", width=256)

                except Exception as e:
                    st.error(f"Error durante la colorización: {e}")
    else:
        st.error("Modelo 2 no está disponible. Verifica la configuración.")

# Footer
st.markdown("---")
st.markdown("**Nota:** Los modelos han sido entrenados con diferentes arquitecturas, por lo que pueden producir resultados diferentes para la misma imagen.")