import cv2

imagem = cv2.imread('imagem_teste.png')
altura, largura = imagem.shape[:2]

# definir o pivô (centro da imagem), o ângulo de rotação e o fator de escala
centro = (largura // 2, altura // 2)
angulo = 45.0  # 45 graus anti-horário
escala = 1.0  # Mantém as dimensões 1:1

# gerar a matriz de rotação
M_rotacao = cv2.getRotationMatrix2D(centro, angulo, escala)

# aplicar a rotação
imagem_rotacionada = cv2.warpAffine(imagem, M_rotacao, (largura, altura))

cv2.imshow('Original', imagem)
cv2.imshow('Rotacao', imagem_rotacionada)
cv2.waitKey(0)
cv2.destroyAllWindows()
