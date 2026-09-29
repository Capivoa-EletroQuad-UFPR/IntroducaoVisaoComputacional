"""
CAPIVOA UFPR.

Demonstração prática de Translação de Imagens e Vídeo.
Módulo 1 - Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Conceito teórico:
- A translação desloca todos os pixels da imagem por meio de uma matriz afim 2x3.
- É uma transformação euclidiana: preserva distâncias, ângulos e formas.
- Equação matricial da translação para cada ponto (x, y):
   [ x' ]  [ 1 0 tx ]  [ x ]
   [ y' ] = [ 0 1 ty ] * [ y ]
               [ 1 ]
 Onde tx é o deslocamento horizontal e ty é o deslocamento vertical.

Controles interativos pelo teclado:
- 'a' / 'd' : desloca para a esquerda / direita (altera tx)
- 'w' / 's' : desloca para cima / baixo (altera ty)
- 'r'    : reseta para o deslocamento inicial
- 'q' / ESC : encerra a demonstração

Como executar:
  python translacao.py
  python translacao.py --midia video.mp4
  python translacao.py --midia 0 # Webcam
  python translacao.py --tx 120 --ty 60
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import stream_midia, criar_parser_midia, concatenar_lado_a_lado


def aplicar_transformacao_translacao(
  frame: np.ndarray,
  tx: int,
  ty: int
) -> np.ndarray:
  """
  Aplica translacao afim sobre a imagem e monta painel comparativo lado a lado.

  Parametros
  ----------
  frame : np.ndarray
    Matriz da imagem original (BGR).
  tx : int
    Deslocamento horizontal em pixels (+direita, -esquerda).
  ty : int
    Deslocamento vertical em pixels (+baixo, -cima).

  Returns
  -------
  np.ndarray
    Painel lado a lado contendo imagem original e imagem transladada.
  """
  altura, largura = frame.shape[:2]

  # 1. construcao da matriz afim 2x3 de translacao
  m_translacao = np.float32([
    [1.0, 0.0, float(tx)],
    [0.0, 1.0, float(ty)]
  ])

  # 2. aplicacao da transformacao afim via cv2.warpaffine
  frame_transladado = cv2.warpAffine(
    frame,
    m_translacao,
    (largura, altura),
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(40, 40, 40)
  )

  titulo_trans = f"Transladada (tx={tx:+d}px, ty={ty:+d}px)"
  painel = concatenar_lado_a_lado(
    [frame, frame_transladado],
    titulos=["Original", titulo_trans],
    altura_padrao=400
  )
  return painel


def main():
  """
  Executa a demonstracao interativa de translacao geometrica com matriz afim 2x3.

  Calcula a transformacao de translacao via cv2.warpAffine sobre o fluxo de midia,
  permitindo ajuste dinamico do deslocamento horizontal e vertical via teclado.
  """
  parser = criar_parser_midia(
    descricao="Translacao Geometrica com Matriz Afim 2x3",
    nome="imagem_teste.png"
  )
  parser.add_argument(
    "--tx",
    type=int,
    default=80,
    help="Deslocamento horizontal inicial em pixels (+direita, -esquerda). Padrão: 80"
  )
  parser.add_argument(
    "--ty",
    type=int,
    default=40,
    help="Deslocamento vertical inicial em pixels (+baixo, -cima). Padrão: 40"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  tx = args.tx
  ty = args.ty
  tx_inicial, ty_inicial = tx, ty
  nome_janela = "Translacao (Use W/A/S/D para mover, Q para sair)"

  print("=" * 60)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 1)")
  print("TRANSLAÇÃO GEOMÉTRICA")
  print("=" * 60)
  print(f"Mídia de entrada : '{args.midia}'")
  print(f"Deslocamento   : tx={tx} px | ty={ty} px")
  print("Controles interativos:")
  print(" [A / D] : Move horizontalmente (-/+ tx)")
  print(" [W / S] : Move verticalmente  (-/+ ty)")
  print(" [R]   : Reseta deslocamento")
  print(" [Q/ESC] : Encerra o programa")
  print("-" * 60)

  for i, (frame, is_stream) in enumerate(stream_midia(args.midia)):
    if args.no_show:
      _ = aplicar_transformacao_translacao(frame, tx, ty)
      if i >= 5 or not is_stream:
        print(f"[Modo no-show] {i+1} frames processados com sucesso.")
        break
      continue

    # tratamento especifico para imagem estatica: interacao continua no mesmo quadro
    if not is_stream:
      painel = aplicar_transformacao_translacao(frame, tx, ty)
      cv2.imshow(nome_janela, painel)

      while True:
        tecla = cv2.waitKey(50) & 0xFF
        if tecla in [ord("q"), ord("Q"), 27]:
          break
        if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
          break

        mudou = False
        if tecla in [ord("d"), ord("D")]:
          tx += 15
          mudou = True
        elif tecla in [ord("a"), ord("A")]:
          tx -= 15
          mudou = True
        elif tecla in [ord("s"), ord("S")]:
          ty += 15
          mudou = True
        elif tecla in [ord("w"), ord("W")]:
          ty -= 15
          mudou = True
        elif tecla in [ord("r"), ord("R")]:
          tx, ty = tx_inicial, ty_inicial
          mudou = True

        if mudou:
          painel = aplicar_transformacao_translacao(frame, tx, ty)
          cv2.imshow(nome_janela, painel)
      break

    # tratamento para stream de video ou webcam
    painel = aplicar_transformacao_translacao(frame, tx, ty)
    cv2.imshow(nome_janela, painel)

    tecla = cv2.waitKey(30) & 0xFF
    if tecla in [ord("q"), ord("Q"), 27]:
      break
    if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
      break

    if tecla in [ord("d"), ord("D")]:
      tx += 15
    elif tecla in [ord("a"), ord("A")]:
      tx -= 15
    elif tecla in [ord("s"), ord("S")]:
      ty += 15
    elif tecla in [ord("w"), ord("W")]:
      ty -= 15
    elif tecla in [ord("r"), ord("R")]:
      tx, ty = tx_inicial, ty_inicial

  cv2.destroyAllWindows()
  print("[Fim] Demonstração de translação encerrada.")


if __name__ == "__main__":
  main()
