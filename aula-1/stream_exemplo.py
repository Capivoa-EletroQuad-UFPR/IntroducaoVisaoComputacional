"""
CAPIVOA UFPR.

Exemplo Didático de Processamento Polimórfico com OpenCV.
Módulo 1 - Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Conceito:
O gerador 'stream_midia' em 'media_io.py' unifica a leitura de:
1. Imagens estáticas : python stream_exemplo.py --midia imagem_teste.png
2. Arquivos de vídeo : python stream_exemplo.py --midia video.mp4
3. Webcams ao vivo  : python stream_exemplo.py --midia 0

Qualquer algoritmo implementado dentro do loop funcionará de forma transparente
para qualquer uma dessas fontes sem modificar uma única linha de código.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import stream_midia, criar_parser_midia, concatenar_lado_a_lado


def processar_frame(frame: np.ndarray) -> np.ndarray:
  """
  Aplica pipeline de exemplo com conversão para escala de cinza e filtro gaussiano.

  Parametros
  ----------
  frame : np.ndarray
    Matriz da imagem em formato BGR.

  Returns
  -------
  np.ndarray
    Matriz da imagem convertida em escala de cinza e suavizada.
  """
  cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  suavizado = cv2.GaussianBlur(cinza, (7, 7), sigmaX=1.5)
  return suavizado


def main():
  """
  Executa a demonstração de processamento polimórfico de mídias com stream_midia.

  Processa de forma unificada imagens estáticas, arquivos de vídeo ou webcam
  utilizando a mesma estrutura de iteração.
  """
  parser = criar_parser_midia(
    descricao="Processamento Unificado para Imagem, Vídeo e Webcam (Módulo 1)",
    nome="imagem_teste.png"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  nome_janela = "Processamento Polimorfico (stream_midia) - 'q' para sair"

  print("=" * 60)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 1)")
  print("EXEMPLO DE PROCESSAMENTO POLIMÓRFICO (stream_midia)")
  print("=" * 60)
  print(f"Mídia de entrada: '{args.midia}'")
  print("Pressione 'q' ou ESC na janela para encerrar.")
  print("-" * 60)

  for i, (frame, is_stream) in enumerate(stream_midia(args.midia)):
    resultado = processar_frame(frame)

    if args.no_show:
      if i >= 5 or not is_stream:
        print(f"[Modo no-show] {i+1} frames processados com sucesso.")
        break
      continue

    painel = concatenar_lado_a_lado(
      [frame, resultado],
      titulos=["Entrada Original", "Grayscale + Filtro Gaussiano"],
      altura_padrao=400
    )

    cv2.imshow(nome_janela, painel)

    # se for imagem estatica, aguarda em loop interativo ate 'q', esc ou fechar a janela
    if not is_stream:
      while True:
        tecla = cv2.waitKey(50) & 0xFF
        if tecla in [ord("q"), ord("Q"), 27]:
          break
        if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
          break
      break

    # se for video/webcam (is_stream=true), aguarda 30ms para taxa aproximada de 30 fps
    tecla = cv2.waitKey(30) & 0xFF
    if tecla in [ord("q"), ord("Q"), 27]:
      print("Execução interrompida pelo usuário.")
      break
    if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
      break

  cv2.destroyAllWindows()
  print("[Fim] Exemplo concluído.")


if __name__ == "__main__":
  main()
