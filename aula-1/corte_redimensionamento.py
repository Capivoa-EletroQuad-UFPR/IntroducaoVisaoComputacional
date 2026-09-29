"""
CAPIVOA UFPR.

Demonstração didática de Corte (ROI), Redimensionamento e Métodos de Interpolação.
Módulo 1 (Capivoa - EletroQuad-UFPR).

Conceitos abordados:
1. Corte (ROI - Região de Interesse):
  - Extração matricial através de fatiamento (slicing) direto: img[y_min:y_max, x_min:x_max].
  - Como isolar um objeto e manipulá-lo individualmente.
2. Redimensionamento e Amostragem :
  - Subamostragem (Downsampling): redução de resolução (pode gerar serrilhamento/aliasing).
  - Sobreamostragem (Upsampling): ampliação com interpolação de novos pixels.
3. Métodos de Interpolação :
  - cv2.INTER_NEAREST : Vizinho mais próximo (1 pixel vizinho). Mais rápido, mas pixelado.
  - cv2.INTER_LINEAR  : Bilinear (4 pixels 2x2). Padrão do OpenCV, boa relação custo/benefício.
  - cv2.INTER_CUBIC  : Bicúbica (16 pixels 4x4). Gera curvas mais suaves, ideal para ampliação.
  - cv2.INTER_LANCZOS4 : Lanczos (64 pixels 8x8). Alta fidelidade com preservação de detalhes finos.

Como executar:
  python corte_redimensionamento.py
  python corte_redimensionamento.py --midia imagem_teste.png
  python corte_redimensionamento.py --no-show
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


def demonstrar_corte(img_rgb: np.ndarray) -> np.ndarray:
  """
  Demonstra a extração de Região de Interesse (ROI) via fatiamento NumPy.

  Isola matricialmente a sub-região contendo as linhas radiais da imagem.

  Parametros
  ----------
  img_rgb : np.ndarray
    Matriz da imagem de entrada em formato RGB.

  Returns
  -------
  np.ndarray
    Matriz recortada correspondente à ROI extraída.
  """
  altura, largura = img_rgb.shape[:2]

  # delimita uma região de interesse (roi) ao redor das linhas radiais
  y_min, y_max = int(altura * 0.22), int(altura * 0.52)
  x_min, x_max = int(largura * 0.08), int(largura * 0.38)

  roi = img_rgb[y_min:y_max, x_min:x_max]
  print(f"\n[Corte/ROI] Dimensão original: {img_rgb.shape[:2]} -> ROI recortada: {roi.shape[:2]}")
  return roi


def comparar_interpolacoes(img_rgb: np.ndarray, no_show: bool = False):
  """
  Compara os quatro métodos de interpolação do OpenCV em subamostragem e ampliação.

  Subamostra a imagem para 25% da resolução original e reconstrói utilizando
  os algoritmos INTER_NEAREST, INTER_LINEAR, INTER_CUBIC e INTER_LANCZOS4 ,
  gerando um painel de zoom para avaliação visual de perdas e artefatos.

  Parametros
  ----------
  img_rgb : np.ndarray
    Matriz da imagem em formato RGB.
  no_show : bool, opcional
    Se True, não exibe a janela gráfica interativa. Padrão é False.
  """
  altura_orig, largura_orig = img_rgb.shape[:2]

  # 1. subamostragem (redução severa para 25% da resolução original)
  fator_escala = 0.25
  largura_reduzida = max(1, int(largura_orig * fator_escala))
  altura_reduzida = max(1, int(altura_orig * fator_escala))

  # inter_area é o método recomendado pelo opencv para redução (decimação)
  img_lowres = cv2.resize(
    img_rgb,
    (largura_reduzida, altura_reduzida),
    interpolation=cv2.INTER_AREA
  )

  print(f"[Redimensionamento] Imagem reduzida de {largura_orig}x{altura_orig} para {largura_reduzida}x{altura_reduzida} (25%)")

  # 2. sobreamostragem (ampliação de volta às dimensões originais pelos 4 métodos)
  dimensoes_orig = (largura_orig, altura_orig)
  res_nearest = cv2.resize(img_lowres, dimensoes_orig, interpolation=cv2.INTER_NEAREST)
  res_linear = cv2.resize(img_lowres, dimensoes_orig, interpolation=cv2.INTER_LINEAR)
  res_cubic  = cv2.resize(img_lowres, dimensoes_orig, interpolation=cv2.INTER_CUBIC)
  res_lanczos = cv2.resize(img_lowres, dimensoes_orig, interpolation=cv2.INTER_LANCZOS4)

  # 3. região de zoom para avaliar detalhes finos (linhas radiais centrais)
  # se a imagem for a imagem_teste padrão, foca na estrela radial; senão, no centro
  y_centro, x_centro = int(altura_orig * 0.36), int(largura_orig * 0.23)
  raio_box = int(min(altura_orig, largura_orig) * 0.12)
  y1, y2 = max(0, y_centro - raio_box), min(altura_orig, y_centro + raio_box)
  x1, x2 = max(0, x_centro - raio_box), min(largura_orig, x_centro + raio_box)

  metodos = [
    ("Original (Referência)", img_rgb[y1:y2, x1:x2], "Sem perdas"),
    ("INTER_NEAREST\n(Vizinho: 1 px)", res_nearest[y1:y2, x1:x2], "Efeito de blocos / pixelado"),
    ("INTER_LINEAR\n(Bilinear: 4 px)", res_linear[y1:y2, x1:x2], "Padrão rápido / borrado suave"),
    ("INTER_CUBIC\n(Bicúbica: 16 px)", res_cubic[y1:y2, x1:x2], "Suavização e curvas finas"),
    ("INTER_LANCZOS4\n(Lanczos: 64 px)", res_lanczos[y1:y2, x1:x2], "Máxima nitidez / bordas vivas"),
  ]

  fig, axes = plt.subplots(1, 5, figsize=(20, 5))
  for ax, (titulo, crop, subtitulo) in zip(axes, metodos):
    ax.imshow(crop)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel(subtitulo, fontsize=9, color="#444444")
    ax.set_xticks([])
    ax.set_yticks([])

  plt.suptitle(
    "MÉTODOS DE INTERPOLAÇÃO : Zoom de Detalhes após Redução (25%) e Ampliação",
    fontsize=13,
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
  Executa a demonstração de corte (ROI) e comparação de interpolações.

  Carrega a imagem de teste, realiza o fatiamento matricial e exibe
  a comparação dos quatro métodos de interpolação de interpolação.
  """
  parser = criar_parser_midia(
    descricao="Corte (ROI), Redimensionamento e Métodos de Interpolação ",
    nome="imagem_teste.png"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  print("=" * 60)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 1)")
  print("CORTE, REDIMENSIONAMENTO E INTERPOLAÇÃO")
  print("=" * 60)

  img_rgb = carregar_imagem(args.midia, to_rgb=True)
  demonstrar_corte(img_rgb)
  comparar_interpolacoes(img_rgb, no_show=args.no_show)


if __name__ == "__main__":
  main()
