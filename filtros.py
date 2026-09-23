import cv2
import numpy as np
import matplotlib.pyplot as plt

# carrega imagem de exemplo
img_bgr = cv2.imread("imagem_teste.png")
if img_bgr is None:
    raise FileNotFoundError("Imagem não encontrada. Verifique o caminho.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# adiciona ruido artificial
img_ruidosa = img_rgb.copy()
probabilidade = 0.03
aleatorio = np.random.rand(*img_rgb.shape[:2])
img_ruidosa[aleatorio < (probabilidade / 2)] = 0      # Pimenta (preto)
img_ruidosa[aleatorio > (1 - probabilidade / 2)] = 255  # Sal (branco)

# aplicacao dos filtros
# filtro de media (5x5)
blur_media = cv2.blur(img_ruidosa, (5, 5))

# filtro gaussiano (5x5)
blur_gauss = cv2.GaussianBlur(img_ruidosa, (5, 5), sigmaX=0)

# filtro mediana
blur_mediana = cv2.medianBlur(img_ruidosa, 5)

# filtro bilateral
blur_bilateral = cv2.bilateralFilter(img_ruidosa, d=9, sigmaColor=75, sigmaSpace=75)

# plotagem
filtros = [
    ("Original com Ruído", img_ruidosa),
    ("Média (cv2.blur)", blur_media),
    ("Gaussiano (cv2.GaussianBlur)", blur_gauss),
    ("Mediana (cv2.medianBlur)", blur_mediana),
    ("Bilateral (cv2.bilateralFilter)", blur_bilateral),
]

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, (titulo, matriz) in zip(axes, filtros):
    ax.imshow(matriz)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.axis("off")

plt.suptitle("Comparação de Filtros", fontsize=15, y=0.98)
plt.tight_layout()
plt.show()
