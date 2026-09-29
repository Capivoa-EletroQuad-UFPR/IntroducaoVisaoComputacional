"""
CAPIVOA UFPR.

Demonstração prática de Rotação de Imagens e Vídeo.
Módulo 1 - Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Conceito teórico:
- A rotação gira a imagem em torno de um ponto pivô (geralmente o centro da imagem).
- É uma transformação euclidiana (preserva comprimentos, ângulos internos e formas).
- No OpenCV, a matriz afim 2x3 de rotação e escala combinadas é obtida com:
   M = cv2.getRotationMatrix2D(centro, angulo, escala)
 Onde:
 - centro: tupla (cx, cy) do ponto pivô em pixels.
 - angulo: valor em graus (positivo = anti-horário).
 - escala: fator isotrópico de escala (1.0 mantém tamanho original).

Controles interativos pelo teclado:
- 'a' / 'd' : gira a imagem (altera o ângulo em -5° / +5°)
- 'w' / 's' : ajusta o zoom (altera a escala em +0.1 / -0.1)
- 'r'    : reseta ângulo e escala para o padrão inicial
- 'q' / ESC : encerra a demonstração

Como executar:
  python rotacao.py
  python rotacao.py --midia video.mp4
  python rotacao.py --midia 0 # Webcam
  python rotacao.py --angulo 90 --escala 0.8
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import stream_midia, criar_parser_midia, concatenar_lado_a_lado


def aplicar_transformacao_rotacao(
  frame: np.ndarray,
  angulo: float,
  escala: float
) -> np.ndarray:
  """
  Aplica rotacao e escala sobre a imagem e monta painel comparativo lado a lado.

  Parametros
  ----------
  frame : np.ndarray
    Matriz da imagem original (BGR).
  angulo : float
    Angulo de rotacao em graus.
  escala : float
    Fator isotropico de escala.

  Returns
  -------
  np.ndarray
    Painel lado a lado contendo imagem original com pivo e imagem rotacionada.
  """
  altura, largura = frame.shape[:2]

  # 1. definir o centro pivo de rotacao
  centro = (largura // 2, altura // 2)

  # 2. gerar a matriz de rotacao 2x3 via opencv
  m_rotacao = cv2.getRotationMatrix2D(centro, angulo, escala)

  # 3. aplicar a transformacao afim
  frame_rotacionado = cv2.warpAffine(
    frame,
    m_rotacao,
    (largura, altura),
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(30, 30, 30)
  )

  # 4. desenhar o pivo de rotacao na imagem original de referencia
  frame_com_pivo = frame.copy()
  cv2.circle(frame_com_pivo, centro, 7, (0, 0, 255), -1)
  cv2.circle(frame_com_pivo, centro, 14, (0, 255, 255), 2)
  cv2.putText(
    frame_com_pivo,
    f"Pivo: {centro}",
    (centro[0] + 15, centro[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 255, 255),
    1,
    cv2.LINE_AA
  )

  titulo_rot = f"Rotacionada ({angulo:.1f} deg, escala={escala:.2f}x)"
  painel = concatenar_lado_a_lado(
    [frame_com_pivo, frame_rotacionado],
    titulos=["Original (Pivo Marcado)", titulo_rot],
    altura_padrao=400
  )
  return painel


def main():
  """
  Executa a demonstracao interativa de rotacao geometrica e escala afim.

  Aplica a matriz afim 2x3 com cv2.getRotationMatrix2D e cv2.warpAffine
  sobre o fluxo de midia, permitindo ajuste dinamico de angulo e escala via teclado.
  """
  parser = criar_parser_midia(
    descricao="Rotacao Geometrica e Escala com cv2.getRotationMatrix2D",
    nome="imagem_teste.png"
  )
  parser.add_argument(
    "--angulo",
    type=float,
    default=35.0,
    help="Ângulo inicial de rotação em graus (positivo = anti-horário). Padrão: 35.0"
  )
  parser.add_argument(
    "--escala",
    type=float,
    default=1.0,
    help="Fator de escala inicial. Padrão: 1.0"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  angulo = args.angulo
  escala = args.escala
  angulo_inicial, escala_inicial = angulo, escala
  nome_janela = "Rotacao (Use A/D para girar, W/S para escala, Q para sair)"

  print("=" * 60)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 1)")
  print("ROTAÇÃO GEOMÉTRICA")
  print("=" * 60)
  print(f"Mídia de entrada : '{args.midia}'")
  print(f"Ângulo inicial  : {angulo}° | Escala inicial: {escala}x")
  print("Controles interativos:")
  print(" [A / D] : Gira anti-horário / horário")
  print(" [W / S] : Aumenta / reduz escala (zoom)")
  print(" [R]   : Reseta parâmetros")
  print(" [Q/ESC] : Encerra o programa")
  print("-" * 60)

  for i, (frame, is_stream) in enumerate(stream_midia(args.midia)):
    if args.no_show:
      _ = aplicar_transformacao_rotacao(frame, angulo, escala)
      if i >= 5 or not is_stream:
        print(f"[Modo no-show] {i+1} frames processados com sucesso.")
        break
      continue

    # tratamento especifico para imagem estatica: interacao continua no mesmo quadro
    if not is_stream:
      painel = aplicar_transformacao_rotacao(frame, angulo, escala)
      cv2.imshow(nome_janela, painel)

      while True:
        tecla = cv2.waitKey(50) & 0xFF
        if tecla in [ord("q"), ord("Q"), 27]:
          break
        if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
          break

        mudou = False
        if tecla in [ord("d"), ord("D")]:
          angulo = (angulo - 5.0) % 360
          mudou = True
        elif tecla in [ord("a"), ord("A")]:
          angulo = (angulo + 5.0) % 360
          mudou = True
        elif tecla in [ord("w"), ord("W")]:
          escala = min(2.5, escala + 0.05)
          mudou = True
        elif tecla in [ord("s"), ord("S")]:
          escala = max(0.2, escala - 0.05)
          mudou = True
        elif tecla in [ord("r"), ord("R")]:
          angulo, escala = angulo_inicial, escala_inicial
          mudou = True

        if mudou:
          painel = aplicar_transformacao_rotacao(frame, angulo, escala)
          cv2.imshow(nome_janela, painel)
      break

    # tratamento para stream de video ou webcam
    painel = aplicar_transformacao_rotacao(frame, angulo, escala)
    cv2.imshow(nome_janela, painel)

    tecla = cv2.waitKey(30) & 0xFF
    if tecla in [ord("q"), ord("Q"), 27]:
      break
    if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
      break

    if tecla in [ord("d"), ord("D")]:
      angulo = (angulo - 5.0) % 360
    elif tecla in [ord("a"), ord("A")]:
      angulo = (angulo + 5.0) % 360
    elif tecla in [ord("w"), ord("W")]:
      escala = min(2.5, escala + 0.05)
    elif tecla in [ord("s"), ord("S")]:
      escala = max(0.2, escala - 0.05)
    elif tecla in [ord("r"), ord("R")]:
      angulo, escala = angulo_inicial, escala_inicial

  cv2.destroyAllWindows()
  print("[Fim] Demonstração de rotação encerrada.")


if __name__ == "__main__":
  main()
