import cv2
import numpy as np
import matplotlib.pyplot as plt

def criar_imagem_teste(largura=600, altura=600):
    img = np.full((altura, largura, 3), 128, dtype=np.uint8)

    # gradiente no fundo
    gradiente = np.linspace(50, 200, largura, dtype=np.uint8)
    img[:, :, :] = np.tile(gradiente, (altura, 1))[:, :, np.newaxis]

    # retângulo preto e círculo branco
    cv2.rectangle(img, (40, 40), (220, 220), (20, 20, 20), -1)
    cv2.circle(img, (130, 130), 60, (240, 240, 240), -1)

    # triângulo colorido
    pontos_triangulo = np.array([[380, 50], [550, 180], [320, 220]], np.int32)
    cv2.fillPoly(img, [pontos_triangulo], (0, 200, 255))

    # linhas radiais
    centro = (300, 360)
    for angulo in range(0, 360, 15):
        rad = np.deg2rad(angulo)
        x_fim = int(centro[0] + 90 * np.cos(rad))
        y_fim = int(centro[1] + 90 * np.sin(rad))
        cv2.line(img, centro, (x_fim, y_fim), (20, 20, 20), 2, lineType=cv2.LINE_AA)

    # texto com fontes de diferentes espessuras
    cv2.putText(img, "IMAGEM TESTE", (40, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Lorem Ipsum", (40, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 1, cv2.LINE_AA)

    # adicionar ruido
    ruido_sp = np.random.rand(altura, largura // 2)
    img[:, :largura // 2][ruido_sp < 0.02] = 0
    img[:, :largura // 2][ruido_sp > 0.98] = 255

    # ruido gaussiano
    ruido_gauss = np.random.normal(0, 18, (altura, largura // 2, 3))
    metade_dir = img[:, largura // 2:].astype(np.float32) + ruido_gauss
    img[:, largura // 2:] = np.clip(metade_dir, 0, 255).astype(np.uint8)

    return img

img_teste = criar_imagem_teste()
cv2.imwrite("imagem_teste.png", img_teste)

plt.figure(figsize=(7, 7))
plt.imshow(cv2.cvtColor(img_teste, cv2.COLOR_BGR2RGB))
plt.title("Imagem para Testes")
plt.axis("off")
plt.show()
