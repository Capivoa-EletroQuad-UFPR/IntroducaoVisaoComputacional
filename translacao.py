import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('imagem.jpg')
altura, largura = imagem.shape[:2]

# Definir os deslocamentos: 100 px para direita, 50 px para baixo
tx, ty = 100, 50

# Construir a matriz de translação float32
M_translacao = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

# Aplicar a transformação afim
imagem_transladada = cv2.warpAffine(imagem, M_translacao, (largura, altura))

cv2.imshow('Original', imagem)
cv2.imshow('Translacao', imagem_transladada)
cv2.waitKey(0)
cv2.destroyAllWindows()
