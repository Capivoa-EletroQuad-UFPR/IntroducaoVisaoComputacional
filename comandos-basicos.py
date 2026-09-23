import cv2

img = cv2.imread("imagem_teste.png")

altura, largura, canais = img.shape
print("altura:", altura, "| largura:", largura, "| canais:", canais)

cv2.imshow("Minha Janela", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
