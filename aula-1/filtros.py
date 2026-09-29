"""
CAPIVOA UFPR.

Demonstração didática de Convolução Espacial e Filtros de Suavização.
Módulo 1 (Capivoa - EletroQuad-UFPR).

Conceitos Teóricos:
1. Convolução Espacial :
  - Deslocamento de uma máscara (kernel) de dimensões ímpares (3x3, 5x5, 7x7)
   sobre cada pixel da imagem original.
  - O pixel central resultante é a soma ponderada dos pixels vizinhos.

2. Comparação dos Filtros :
  - Média (cv2.blur):
   Kernel homogêneo onde todos os pesos são iguais a 1/(k*k).
   Muito rápido, mas borra fortemente bordas finas.
  - Gaussiano (cv2.GaussianBlur):
   Pesos decaem exponencialmente a partir do centro seguindo uma distribuição normal.
   Excelente para atenuar ruído gaussiano (eletrônico/sensor) preservando a forma geral.
  - Mediana (cv2.medianBlur):
   Filtro não-linear que ordena os valores da vizinhança e seleciona o elemento mediano.
   Extremamente eficaz contra ruído impulsivo (Sal & Pimenta) sem criar tons intermediários.
  - Bilateral (cv2.bilateralFilter):
   Combina proximidade espacial e diferença de intensidade/cor.
   Suaviza ruídos mantendo bordas perfeitamente nítidas (edge-preserving), mas é mais custoso.

Como executar:
  python filtros.py
  python filtros.py --midia imagem_teste.png
  python filtros.py --kernel 5
  python filtros.py --no-show
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import carregar_imagem, criar_parser_midia


def aplicar_filtros(img_rgb: np.ndarray, ksize: int = 5) -> dict:
  """
  Aplica os quatro filtros espaciais de suavização sobre a imagem.

  Executa as operações de filtragem da Média, Gaussiano, Mediana e Bilateral
  conforme os tópicos de filtragem.

  Parametros
  ----------
  img_rgb : np.ndarray
    Matriz da imagem em formato RGB ou escala de cinza.
  ksize : int, opcional
    Tamanho da dimensão do kernel ímpar (3, 5, 7, etc.). Padrão é 5.

  Returns
  -------
  dict
    Dicionário contendo os nomes dos filtros e suas respectivas matrizes processadas.
  """
  if ksize % 2 == 0:
    ksize += 1
    print(f"[Aviso] Dimensões do kernel devem ser ímpares. Ajustado para ksize={ksize}")

  # 1. filtro da média (box filter) - item 01
  blur_media = cv2.blur(img_rgb, (ksize, ksize))

  # 2. filtro gaussiano - item 02
  blur_gauss = cv2.GaussianBlur(img_rgb, (ksize, ksize), sigmaX=0)

  # 3. filtro da mediana - item 03
  blur_mediana = cv2.medianBlur(img_rgb, ksize)

  # 4. filtro bilateral - item 04
  # d: diâmetro da vizinhança, sigmacolor: tolerância de cor, sigmaspace: alcance espacial
  blur_bilateral = cv2.bilateralFilter(img_rgb, d=ksize * 2, sigmaColor=75, sigmaSpace=75)

  return {
    "Original": img_rgb,
    f"Média ({ksize}x{ksize})\ncv2.blur": blur_media,
    f"Gaussiano ({ksize}x{ksize})\ncv2.GaussianBlur": blur_gauss,
    f"Mediana ({ksize}x{ksize})\ncv2.medianBlur": blur_mediana,
    f"Bilateral (d={ksize*2})\ncv2.bilateralFilter": blur_bilateral
  }


def plotar_comparacao_completa(resultados: dict, no_show: bool = False):
  """
  Plota a visão geral comparativa dos filtros em múltiplas colunas.

  Parametros
  ----------
  resultados : dict
    Dicionário contendo os títulos e as respectivas matrizes de imagens.
  no_show : bool, opcional
    Se True, não exibe a janela gráfica interativa. Padrão é False.
  """
  fig, axes = plt.subplots(1, 5, figsize=(22, 5))
  for ax, (titulo, matriz) in zip(axes, resultados.items()):
    ax.imshow(matriz)
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.axis("off")

  plt.suptitle(
    "FILTROS ESPACIAIS E SUAVIZAÇÃO ",
    fontsize=14,
    fontweight="bold",
    y=0.98
  )
  plt.tight_layout()

  if not no_show:
    plt.show()
  else:
    plt.close()


def main():
  """
  Executa a comparação dos filtros espaciais de suavização.

  Recebe os argumentos de linha de comando, carrega a imagem de entrada,
  aplica os filtros e exibe o gráfico comparativo com Matplotlib.
  """
  parser = criar_parser_midia(
    descricao="Comparação Didática de Filtros Espaciais de Suavização",
    nome="imagem_teste.png"
  )
  parser.add_argument(
    "-k", "--kernel",
    type=int,
    default=5,
    help="Tamanho do kernel ímpar (ex: 3, 5, 7). Padrão: 5"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janela gráfica."
  )
  args = parser.parse_args()

  print("=" * 60)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 1)")
  print("FILTROS E CONVOLUÇÃO ESPACIAL")
  print("=" * 60)
  print(f"Mídia de entrada: '{args.midia}'")
  print(f"Kernel de filtro: {args.kernel}x{args.kernel} px")

  # carrega imagem original em formato rgb para renderização fiel no matplotlib
  img_rgb = carregar_imagem(args.midia, to_rgb=True)

  print("\nAplicando filtros...")
  resultados = aplicar_filtros(img_rgb, ksize=args.kernel)

  print("\nResumo do Comportamento dos Filtros:")
  print("1. Média   : borra uniformemente todos os pixels vizinhos (kernel constante).")
  print("2. Gaussiano : pondera pelo sino de Gauss; atenua ruído contínuo/gaussiano.")
  print("3. Mediana  : elimina ruído pontual (Sal & Pimenta) perfeitamente.")
  print("4. Bilateral : preserva bordas nítidas enquanto suaviza áreas planas.")

  plotar_comparacao_completa(resultados, no_show=args.no_show)
  print("\n[Concluído] Processamento de filtros finalizado.")


if __name__ == "__main__":
  main()
