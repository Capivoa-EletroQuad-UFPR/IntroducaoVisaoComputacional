"""
CAPIVOA UFPR.

Gera uma imagem de teste ('imagem_teste.png') utilizada pelos scripts do workshop
de Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

A imagem foi projetada para demonstrar e testar os seguintes conceitos:
1. Estrutura da Imagem & Quantização: gradientes de intensidade uint8.
2. Canais de Cor BGR vs RGB: blocos puros Azul, Verde e Vermelho.
3. Transformação de Perspectiva: cartão projetado em perspectiva.
4. Métodos de Interpolação: estrela de linhas radiais.
5. Filtros e Convolução Espacial: ruído Sal & Pimenta e ruído Gaussiano.

Como executar:
  python criar_imagem.py      # Gera a imagem e abre visualização no Matplotlib
  python criar_imagem.py --no-show # Apenas gera e salva o arquivo 'imagem_teste.png'
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# coordenadas fixas da placa em perspectiva para uso em trans_perspectiva.py
# (x, y) na imagem 650x650: [supesq, supdir, infdir, infesq]
PTS_PLACA_ORIGEM = np.float32([
  [370, 70],  # canto superior esquerdo
  [590, 130], # canto superior direito
  [540, 280], # canto inferior direito
  [340, 220]  # canto inferior esquerdo
])


def criar_cartao_ortogonal(largura: int = 240, altura: int = 150) -> np.ndarray:
  """
  Desenha um cartão plano e ortogonal com texto e símbolos nítidos.

  Esse cartão será projetado em perspectiva sobre a cena.

  Parametros
  ----------
  largura : int, opcional
    Largura em pixels do cartão. Padrão é 240.
  altura : int, opcional
    Altura em pixels do cartão. Padrão é 150.

  Returns
  -------
  np.ndarray
    Matriz NumPy (uint8) contendo a imagem do cartão gerado.
  """
  cartao = np.full((altura, largura, 3), 245, dtype=np.uint8)

  # moldura externa azul escuro
  cv2.rectangle(cartao, (0, 0), (largura - 1, altura - 1), (140, 40, 20), 4)

  # faixa superior azul marinho
  cv2.rectangle(cartao, (6, 6), (largura - 6, 42), (180, 90, 20), -1)
  cv2.putText(
    cartao, "UFPR - CAPIVOA", (15, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
  )

  # símbolo circular
  cv2.circle(cartao, (50, 95), 28, (0, 140, 255), -1)
  cv2.circle(cartao, (50, 95), 16, (255, 255, 255), -1)

  # textos da placa
  cv2.putText(
    cartao, "PLACA-01", (95, 85),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA
  )
  cv2.putText(
    cartao, "PERSPECTIVA", (95, 115),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA
  )

  return cartao


def criar_imagem_teste(largura: int = 650, altura: int = 650) -> np.ndarray:
  """
  Constrói a imagem de teste completa.

  Gera uma imagem sintética contendo gradientes de quantização, blocos BGR puros,
  estrela de linhas radiais para teste de interpolação, cartão em perspectiva e
  faixas de ruído Sal & Pimenta e Gaussiano.

  Parametros
  ----------
  largura : int, opcional
    Largura em pixels da imagem. Padrão é 650.
  altura : int, opcional
    Altura em pixels da imagem. Padrão é 650.

  Returns
  -------
  np.ndarray
    Matriz NumPy (uint8) representando a imagem de teste sintetizada.
  """
  # 1. fundo com gradiente horizontal (demonstra amostragem e quantizacao 0 a 255)
  img = np.zeros((altura, largura, 3), dtype=np.uint8)
  gradiente = np.linspace(45, 205, largura, dtype=np.uint8)
  img[:, :] = gradiente[None, :, None]

  # 2. cores primarias puras para canais bgr
  # azul puro (255, 0, 0), verde puro (0, 255, 0), vermelho puro (0, 0, 255)
  tamanho_bloco = 60
  y_topo = 40
  # azul
  cv2.rectangle(img, (40, y_topo), (40 + tamanho_bloco, y_topo + tamanho_bloco), (255, 0, 0), -1)
  cv2.putText(img, "B", (60, y_topo + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
  # verde
  cv2.rectangle(img, (115, y_topo), (115 + tamanho_bloco, y_topo + tamanho_bloco), (0, 255, 0), -1)
  cv2.putText(img, "G", (133, y_topo + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
  # vermelho
  cv2.rectangle(img, (190, y_topo), (190 + tamanho_bloco, y_topo + tamanho_bloco), (0, 0, 255), -1)
  cv2.putText(img, "R", (208, y_topo + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

  # 3. estrela de linhas radiais (interpolacao, corte e aliasing)
  centro_radial = (145, 235)
  raio = 65
  for angulo in range(0, 360, 15):
    rad = np.deg2rad(angulo)
    xf = int(centro_radial[0] + raio * np.cos(rad))
    yf = int(centro_radial[1] + raio * np.sin(rad))
    cv2.line(img, centro_radial, (xf, yf), (30, 30, 30), 2, cv2.LINE_AA)
  cv2.circle(img, centro_radial, 6, (0, 0, 255), -1)

  # 4. placa inclinada em perspectiva (transformacao de perspectiva)
  w_c, h_c = 240, 150
  cartao_reto = criar_cartao_ortogonal(w_c, h_c)
  pts_cartao = np.float32([[0, 0], [w_c, 0], [w_c, h_c], [0, h_c]])

  M_warp = cv2.getPerspectiveTransform(pts_cartao, PTS_PLACA_ORIGEM)
  cartao_deformado = cv2.warpPerspective(cartao_reto, M_warp, (largura, altura))
  mascara_cartao = cv2.warpPerspective(
    np.ones((h_c, w_c), dtype=np.uint8) * 255, M_warp, (largura, altura)
  )

  # sobrepõe o cartão deformado sobre a imagem de fundo
  indices_cartao = mascara_cartao > 0
  img[indices_cartao] = cartao_deformado[indices_cartao]

  # 5. textos informativos de identificação
  cv2.putText(
    img, "CAPIVOA - MODULO 1", (40, 345),
    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA
  )
  cv2.putText(
    img, "Estrutura, Geometria e Filtros", (40, 375),
    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA
  )

  # 6. regiao com ruidos controlados (filtros)
  y_faixa_topo, y_faixa_base = 410, 620
  x_meio = largura // 2

  # metade esquerda: ruído sal & pimenta (ruído impulsivo)
  altura_faixa = y_faixa_base - y_faixa_topo
  ruido_sp = np.random.rand(altura_faixa, x_meio)
  regiao_esq = img[y_faixa_topo:y_faixa_base, :x_meio]
  regiao_esq[ruido_sp < 0.03] = 0   # pimenta (preto)
  regiao_esq[ruido_sp > 0.97] = 255  # sal (branco)
  img[y_faixa_topo:y_faixa_base, :x_meio] = regiao_esq

  cv2.putText(
    img, "Ruido Sal e Pimenta", (30, 595),
    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
  )

  # metade direita: ruído gaussiano (distribuição normal)
  largura_dir = largura - x_meio
  ruido_gauss = np.random.normal(0, 22, (altura_faixa, largura_dir, 3))
  regiao_dir = img[y_faixa_topo:y_faixa_base, x_meio:].astype(np.float32) + ruido_gauss
  img[y_faixa_topo:y_faixa_base, x_meio:] = np.clip(regiao_dir, 0, 255).astype(np.uint8)

  cv2.putText(
    img, "Ruido Gaussiano", (x_meio + 30, 595),
    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA
  )

  # linha divisória da faixa de ruído
  cv2.line(img, (x_meio, y_faixa_topo), (x_meio, y_faixa_base), (60, 60, 60), 2)

  return img


def main():
  """
  Executa a geração e salvamento da imagem de teste didática.

  Gera o arquivo 'imagem_teste.png' no diretório do script e opcionalmente
  exibe a renderização gráfica via Matplotlib.
  """
  parser = argparse.ArgumentParser(description="Gera a imagem de teste didática para o Módulo 1.")
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Salva a imagem sem exibir janela gráfica (útil em modo não-interativo)."
  )
  args = parser.parse_args()

  caminho_saida = Path(__file__).resolve().parent / "imagem_teste.png"
  img = criar_imagem_teste()
  cv2.imwrite(str(caminho_saida), img)
  print(f"[OK] Imagem didática salva com sucesso: '{caminho_saida}' ({img.shape[1]}x{img.shape[0]} px)")

  if not args.no_show:
    plt.figure(figsize=(7, 7))
    # opencv usa bgr; matplotlib espera rgb
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("imagem_teste.png - Elementos Didáticos do Módulo 1", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
  main()
