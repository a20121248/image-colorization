import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb
import requests
import os

# ------------------- Descarga del modelo desde Dropbox -------------------
MODEL_PATH = 'models/generator_colorization.pth'
DROPBOX_URL = 'https://www.dropbox.com/scl/fi/vkme2h1kdoqj74a6k8mrn/generator_colorization.pth?rlkey=er10fsftgfy4qk3qjm2muw54o&st=dwxzbeh0&dl=1'

def descargar_modelo(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    response = requests.get(url, stream=True)
    with open(destino, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

if not os.path.exists(MODEL_PATH):
    st.info("Descargando modelo desde Dropbox...")
    descargar_modelo(DROPBOX_URL, MODEL_PATH)

# ------------------- Modelo -------------------
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

# ------------------- Cargar modelo -------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    model = Unet(input_c=1, output_c=2)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

model_G = load_model()

# ------------------- Inferencia -------------------
def colorize_image(uploaded_file):
    SIZE = 256
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((SIZE, SIZE), Image.BICUBIC)
    img_np = np.array(img_resized)
    lab = rgb2lab(img_np).astype("float32")

    L = lab[..., 0]
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0) / 50. - 1.

    with torch.no_grad():
        ab = model_G(L_tensor)

    L_restored = (L_tensor + 1.) * 50.
    ab_restored = ab * 110.
    Lab = torch.cat([L_restored, ab_restored], dim=1).permute(0, 2, 3, 1).numpy()[0]
    img_rgb = lab2rgb(Lab)
    img_rgb = np.clip(img_rgb, 0, 1)

    return Image.fromarray((img_rgb * 255).astype(np.uint8)), img_resized

# ------------------- UI Streamlit -------------------
st.set_page_config(page_title="Colorizador de Imágenes TAFOS", layout="wide")

st.title("🎨 Colorización de Imágenes con GAN")
st.markdown("""
**Tipo de modelo:** GAN Condicional (U-Net + PatchGAN)  
**Versión del modelo:** 1.0  
**Curso:** INF658 - Deep Learning  
**Alumnos:**  
- Javier Monzón - 2020123456  
- Ana Ruiz - 2020123457
""")

st.write("Sube una imagen en blanco y negro para verla coloreada automáticamente.")

uploaded_file = st.file_uploader("📤 Carga tu imagen (.jpg o .png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.subheader("Resultado")

    colorized_img, grayscale_resized = colorize_image(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(grayscale_resized, caption="Escala de grises", use_container_width=True)
    with col2:
        st.image(colorized_img, caption="Colorizado", use_container_width=True)
