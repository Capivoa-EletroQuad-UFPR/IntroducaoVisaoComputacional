import cv2
import numpy as np

imagem = cv2.imread('imagem_teste.png')

# coordenada dos 4 cantos do objeto de interesse
pts_origem = np.float32([
    [50, 320],  # canto superior esquerdo distorcido
    [50, 420],  # canto superior direito distorcido
    [560, 430],  # canto inferior direito distorcido
    [50, 430]    # canto inferior esquerdo distorcido
])

# dimensões retangulares desejadas para a visualização corrigida
largura_saida, altura_saida = 400, 300

# coordenadas de destino no plano frontal ortogonal
pts_destino = np.float32([
    [0, 0],
    [largura_saida, 0],
    [largura_saida, altura_saida],
    [0, altura_saida]
])

# calcular a matriz de perspectiva 3x3
M_perspectiva = cv2.getPerspectiveTransform(pts_origem, pts_destino)

imagem_corrigida = cv2.warpPerspective(imagem, M_perspectiva, (largura_saida, altura_saida))

cv2.imshow('Original Inclinada', imagem)
cv2.imshow('Vista Corrigida', imagem_corrigida)
cv2.waitKey(0)
cv2.destroyAllWindows()
