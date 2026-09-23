import cv2
import matplotlib.pyplot as plt

# carrega imagem original
img_bgr = cv2.imread("imagem_teste.png")
if img_bgr is None:
    raise FileNotFoundError("Imagem não encontrada. Verifique o caminho especificado.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
altura_orig, largura_orig = img_rgb.shape[:2]

# downscale da imagem
fator_escala = 0.25
largura_baixa = int(largura_orig * fator_escala)
altura_baixa = int(altura_orig * fator_escala)

img_lowres = cv2.resize(
    img_rgb, 
    (largura_baixa, altura_baixa), 
    interpolation=cv2.INTER_AREA
)

# upscale para a dimensão original
dimensoes_originais = (largura_orig, altura_orig)
res_nearest = cv2.resize(img_lowres, dimensoes_originais, interpolation=cv2.INTER_NEAREST)
res_linear  = cv2.resize(img_lowres, dimensoes_originais, interpolation=cv2.INTER_LINEAR)
res_cubic   = cv2.resize(img_lowres, dimensoes_originais, interpolation=cv2.INTER_CUBIC)
res_lanczos = cv2.resize(img_lowres, dimensoes_originais, interpolation=cv2.INTER_LANCZOS4)

# dimensão do corte
roi_y1, roi_y2 = int(altura_orig * 0.35), int(altura_orig * 0.65)
roi_x1, roi_x2 = int(largura_orig * 0.35), int(largura_orig * 0.65)

# janela comparativa
metodos = [
    ("Original", img_rgb),
    ("INTER_NEAREST (Pixelado)", res_nearest),
    ("INTER_LINEAR (Bilinear)", res_linear),
    ("INTER_CUBIC (Bicúbica)", res_cubic),
    ("INTER_LANCZOS4 (Lanczos 8x8)", res_lanczos),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, (titulo, matriz) in zip(axes, metodos):
    # aplica o corte
    detalhe_zoom = matriz[roi_y1:roi_y2, roi_x1:roi_x2]
    
    ax.imshow(detalhe_zoom)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.axis("off")

plt.suptitle("Comparação de Interpolação:", fontsize=14, y=0.98)
plt.tight_layout()
plt.show()
