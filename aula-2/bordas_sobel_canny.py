"""
CAPIVOA UFPR.

Demonstração didática de Detecção de Bordas: Operador de Sobel vs Algoritmo de Canny.
Módulo 2 de Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Tópicos e Fundamentos Teóricos:
1. Bordas na Imagem :
   - Fronteiras visuais formadas por descontinuidades bruscas de iluminação.
   - Compressão semântica: elimina texturas irrelevantes preservando a silhueta estrutural.
   - Modelos de transição: degrau analítico, rampa de difusão e telhado.

2. Operador de Sobel :
   - Convolução espacial baseada em derivadas numéricas combinadas com suavização Gaussiana.
   - Kernel Horizontal Gx (3x3): realça variações na direção horizontal (bordas verticais).
   - Kernel Vertical Gy (3x3): realça variações na direção vertical (bordas horizontais).
   - Magnitude combinada: |Gx| + |Gy| ou sqrt(Gx^2 + Gy^2).
   - Características: operação veloz, porém gera bordas espessas em tons de cinza e é
     mais sensível a ruído e texturas finas.

3. Algoritmo de Canny :
   - Pipeline de 4 etapas para extração de contornos nítidos e finos de 1 pixel:
     Passo 1. Filtro Gaussiano: atenuação prévia de ruídos de alta frequência.
     Passo 2. Gradiente: cálculo da magnitude e orientação angular da borda via Sobel.
     Passo 3. Supressão de Não-Máximos: afinamento da crista da borda mantendo apenas picos locais.
     Passo 4. Limiarização por Histerese: uso de dois limiares (T_min e T_max) onde bordas fracas
              só são aceitas se estiverem conectadas a bordas fortes.
   - Características: contornos binários estritamente de 1 pixel, imunidade superior a ruídos.

Controles no Teclado:
  [TAB]   : Alterna entre Comparação Direta (Sobel vs Canny) e Decomposição Completa (Gx, Gy, Mag, Canny)
  [W / S] : Aumenta / diminui o limiar inferior de Canny (T_baixo)
  [E / D] : Aumenta / diminui o limiar superior de Canny (T_alto)
  [K]     : Alterna tamanho do Kernel Sobel (ksize = 3, 5 ou 7)
  [S]     : Salva a comparação em disco na pasta 'frames-capturados/'
  [Q/ESC] : Encerra o programa

Como executar:
  python bordas_sobel_canny.py
  python bordas_sobel_canny.py --midia imagem_teste_aula2.png
  python bordas_sobel_canny.py --t-baixo 50 --t-alto 150
  python bordas_sobel_canny.py --no-show
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import carregar_imagem, criar_parser_midia, concatenar_lado_a_lado, stream_midia


def processar_bordas(
  img_cinza: np.ndarray,
  t_baixo: int = 50,
  t_alto: int = 150,
  ksize_sobel: int = 3
) -> Dict[str, np.ndarray]:
  """
  Calcula os componentes de gradiente de Sobel e a extração de bordas de Canny.

  Parametros
  ----------
  img_cinza : np.ndarray
    Imagem monocromática de entrada (uint8).
  t_baixo : int
    Limiar inferior para histerese de Canny. Padrão é 50.
  t_alto : int
    Limiar superior para histerese de Canny. Padrão é 150.
  ksize_sobel : int
    Dimensão ímpar do kernel do operador de Sobel (3, 5 ou 7).

  Returns
  -------
  Dict[str, np.ndarray]
    Dicionário contendo as matrizes de Sobel Gx, Gy, Magnitude e Canny.
  """
  # 1. pré-suavização leve para Sobel
  suave = cv2.GaussianBlur(img_cinza, (3, 3), 0)

  # 2. operador de Sobel em ponto flutuante para evitar truncamento em valores negativos
  sobel_x = cv2.Sobel(suave, cv2.CV_64F, 1, 0, ksize=ksize_sobel)
  sobel_y = cv2.Sobel(suave, cv2.CV_64F, 0, 1, ksize=ksize_sobel)

  # conversão para visualização absoluta em 8 bits
  abs_sobel_x = cv2.convertScaleAbs(sobel_x)
  abs_sobel_y = cv2.convertScaleAbs(sobel_y)

  # magnitude total do gradiente (|Gx| + |Gy|)
  sobel_mag = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

  # 3. algoritmo de Canny (4 passos: filtro gaussiano, gradiente, não-máximos e histerese)
  bordas_canny = cv2.Canny(img_cinza, threshold1=t_baixo, threshold2=t_alto, apertureSize=ksize_sobel)

  return {
    "sobel_x": abs_sobel_x,
    "sobel_y": abs_sobel_y,
    "sobel_mag": sobel_mag,
    "canny": bordas_canny
  }


def montar_paineis(
  img_cinza: np.ndarray,
  dados: Dict[str, np.ndarray],
  t_baixo: int,
  t_alto: int,
  ksize: int
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Monta o painel comparativo direto (Sobel vs Canny) e o painel expandido (decomposição).
  """
  # Painel Comparativo Direto
  painel_comparativo = concatenar_lado_a_lado(
    [img_cinza, dados["sobel_mag"], dados["canny"]],
    titulos=[
      "1. Original (Cinza)",
      f"2. Operador Sobel (|Gx|+|Gy|, k={ksize})\nBordas espessas em tons cinza",
      f"3. Algoritmo Canny (T=[{t_baixo}, {t_alto}])\nLinhas finas de 1 pixel"
    ],
    altura_padrao=360
  )

  # Painel Decomposição Completa
  linha_cima = concatenar_lado_a_lado(
    [img_cinza, dados["sobel_mag"]],
    titulos=["Original (Grayscale)", f"Sobel: Magnitude Total (k={ksize})"],
    altura_padrao=250
  )
  linha_baixo = concatenar_lado_a_lado(
    [dados["sobel_x"], dados["sobel_y"], dados["canny"]],
    titulos=[
      "Sobel Gx (Bordas Verticais)",
      "Sobel Gy (Bordas Horizontais)",
      f"Canny: Bordas 1px (T={t_baixo}/{t_alto})"
    ],
    altura_padrao=250
  )

  largura_max = max(linha_cima.shape[1], linha_baixo.shape[1])
  if linha_cima.shape[1] < largura_max:
    linha_cima = cv2.copyMakeBorder(
      linha_cima, 0, 0, 0, largura_max - linha_cima.shape[1], cv2.BORDER_CONSTANT, value=[30, 30, 30]
    )
  if linha_baixo.shape[1] < largura_max:
    linha_baixo = cv2.copyMakeBorder(
      linha_baixo, 0, 0, 0, largura_max - linha_baixo.shape[1], cv2.BORDER_CONSTANT, value=[30, 30, 30]
    )

  painel_expandido = np.vstack([linha_cima, linha_baixo])
  return painel_comparativo, painel_expandido


def imprimir_observacao_kernel(ksize: int):
  if ksize == 7:
    print("[Sobel 7x7] Ruim pra detalhe: borra demais, borda vira uma faixa grossa de 7px e funde furos/cantos próximos.")
  elif ksize == 5:
    print("[Sobel 5x5] Filtra mais ruído, mas já engorda as bordas e perde definição.")
  else:
    print("[Sobel 3x3] Padrão: mantém a posição exata da borda sem deformar a peça.")


def main():
  parser = criar_parser_midia(
    descricao="Módulo 2: Detecção de Bordas - Operador de Sobel vs Algoritmo de Canny",
    nome="imagem_teste_aula2.png"
  )
  parser.add_argument(
    "--t-baixo",
    type=int,
    default=50,
    help="Limiar inferior para histerese de Canny (padrão: 50)"
  )
  parser.add_argument(
    "--t-alto",
    type=int,
    default=150,
    help="Limiar superior para histerese de Canny (padrão: 150)"
  )
  parser.add_argument(
    "--ksize",
    type=int,
    default=3,
    choices=[3, 5, 7],
    help="Tamanho do kernel de Sobel (padrão: 3)"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  print("=" * 65)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 2)")
  print("DETECÇÃO DE BORDAS: SOBEL VS CANNY")
  print("=" * 65)

  t_baixo = args.t_baixo
  t_alto = args.t_alto
  ksize = args.ksize
  modo_expandido = False
  nome_janela = "Modulo 2 - Deteccao de Bordas: Sobel vs Canny (Capivoa)"

  imprimir_observacao_kernel(ksize)

  for frame, is_stream in stream_midia(args.midia, nome="imagem_teste_aula2.png"):
    img_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    dados_bordas = processar_bordas(img_cinza, t_baixo=t_baixo, t_alto=t_alto, ksize_sobel=ksize)
    painel_comp, painel_exp = montar_paineis(img_cinza, dados_bordas, t_baixo, t_alto, ksize)

    if args.no_show:
      print(f"[No-Show] Sobel processado com ksize={ksize}. Média Mag={dados_bordas['sobel_mag'].mean():.2f}")
      print(f"[No-Show] Canny processado com T=[{t_baixo}, {t_alto}]. Pixels de borda={np.count_nonzero(dados_bordas['canny'])}")
      print("[No-Show] Validação de bordas concluída com sucesso.")
      break

    exibicao = painel_exp if modo_expandido else painel_comp
    cv2.imshow(nome_janela, exibicao)

    while True:
      tecla = cv2.waitKey(30) & 0xFF
      if tecla in [ord("q"), ord("Q"), 27]:
        cv2.destroyAllWindows()
        return

      if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
        return

      atualizar = False

      if tecla == 9:  # TAB
        modo_expandido = not modo_expandido
        atualizar = True
      elif tecla in [ord("w"), ord("W")]:
        t_baixo = min(t_alto - 5, t_baixo + 5)
        print(f"[Canny] T_baixo = {t_baixo}")
        atualizar = True
      elif tecla in [ord("s"), ord("S")] and not (tecla == ord("s") and False):
        # salvamento de imagem com 's' minúsculo se pressionado sozinho
        pass
      if tecla == ord("s"):
        t_baixo = max(5, t_baixo - 5)
        print(f"[Canny] T_baixo = {t_baixo}")
        atualizar = True
      elif tecla in [ord("e"), ord("E")]:
        t_alto = min(255, t_alto + 5)
        print(f"[Canny] T_alto = {t_alto}")
        atualizar = True
      elif tecla in [ord("d"), ord("D")]:
        t_alto = max(t_baixo + 5, t_alto - 5)
        print(f"[Canny] T_alto = {t_alto}")
        atualizar = True
      elif tecla in [ord("k"), ord("K")]:
        ksize = 5 if ksize == 3 else (7 if ksize == 5 else 3)
        print(f"[Sobel] Kernel alterado para ksize={ksize}")
        imprimir_observacao_kernel(ksize)
        atualizar = True
      elif tecla in [ord("p"), ord("P")]:
        pasta_saida = Path("frames-capturados")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_salvo = pasta_saida / "bordas_sobel_canny.png"
        cv2.imwrite(str(caminho_salvo), exibicao)
        print(f"[Salvo] Imagem salva em: '{caminho_salvo}'")

      if atualizar and not is_stream:
        dados_bordas = processar_bordas(img_cinza, t_baixo=t_baixo, t_alto=t_alto, ksize_sobel=ksize)
        painel_comp, painel_exp = montar_paineis(img_cinza, dados_bordas, t_baixo, t_alto, ksize)
        exibicao = painel_exp if modo_expandido else painel_comp
        cv2.imshow(nome_janela, exibicao)

      if is_stream:
        break


if __name__ == "__main__":
  main()
