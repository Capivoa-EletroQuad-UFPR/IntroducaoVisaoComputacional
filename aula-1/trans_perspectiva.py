"""
CAPIVOA UFPR.

Demonstração didática da Transformação de Perspectiva.
Módulo 1 - Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Conceito teórico:
- A transformação de perspectiva (homografia projetiva) corrige distorções
 causadas pelo ângulo oblíquo da câmera em relação ao plano de interesse.
- Enquanto em transformações afins (translação/rotação) o paralelismo é
 estritamente mantido, na perspectiva linhas paralelas no mundo real convergem
 para pontos de fuga.
- Diferença matemática:
 - Transformação Afim: precisa de 3 pontos de correspondência (matriz 2x3).
 - Transformação de Perspectiva: precisa de 4 pontos (matriz 3x3 com 8 graus de liberdade).
- Funções OpenCV:
 - cv2.getPerspectiveTransform(pts_origem, pts_destino): calcula a matriz 3x3.
 - cv2.warpPerspective(img, M, (largura, altura)): aplica a deformação matricial.

Como executar:
  python trans_perspectiva.py
  python trans_perspectiva.py --midia video.mp4
  python trans_perspectiva.py --midia 0 # Webcam
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import stream_midia, criar_parser_midia, concatenar_lado_a_lado


def obter_pontos_origem_padrao(frame: np.ndarray, nome_midia: str) -> np.ndarray:
  """
  Retorna os quatro pontos de origem no formato [SupEsq, SupDir, InfDir, InfEsq].

  Para 'imagem_teste.png', utiliza as coordenadas exatas da placa didática.
  Caso contrário, calcula um trapézio central proporcional à resolução do frame.

  Parametros
  ----------
  frame : np.ndarray
    Matriz do quadro de imagem ou vídeo.
  nome_midia : str
    Nome ou caminho da mídia fornecida.

  Returns
  -------
  np.ndarray
    Matriz NumPy (float32) contendo as 4 coordenadas (x, y) de origem.
  """
  h, w = frame.shape[:2]
  nome_base = Path(nome_midia).name.lower()

  if "imagem_teste" in nome_base:
    # coordenadas exatas da placa desenhada por criar_imagem.py
    return np.float32([
      [370, 70],  # p1: superior esquerdo
      [590, 130], # p2: superior direito
      [540, 280], # p3: inferior direito
      [340, 220]  # p4: inferior esquerdo
    ])

  # para qualquer outra imagem, vídeo ou webcam, define um trapézio de interesse didático
  margem_x = int(w * 0.20)
  topo_x = int(w * 0.32)
  topo_y = int(h * 0.25)
  base_y = int(h * 0.85)

  return np.float32([
    [topo_x, topo_y],       # p1: superior esquerdo
    [w - topo_x, topo_y],     # p2: superior direito
    [w - margem_x, base_y],    # p3: inferior direito
    [margem_x, base_y]      # p4: inferior esquerdo
  ])


def main():
  """
  Executa a demonstração de retificação por transformação de perspectiva.

  Calcula a homografia projetiva 3x3 com cv2.getPerspectiveTransform e
  aplica cv2.warpPerspective para corrigir distorções de ângulo da câmera.
  """
  parser = criar_parser_midia(
    descricao="Transformação de Perspectiva e Retificação",
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
  print("TRANSFORMAÇÃO DE PERSPECTIVA")
  print("=" * 60)
  print(f"Mídia de entrada: '{args.midia}'")

  # dimensões de saída para a vista frontal retificada
  largura_saida, altura_saida = 360, 220
  pts_destino = np.float32([
    [0, 0],              # p1' -> topo esquerdo
    [largura_saida, 0],        # p2' -> topo direito
    [largura_saida, altura_saida],  # p3' -> base direita
    [0, altura_saida]         # p4' -> base esquerda
  ])

  # cores didáticas para cada um dos 4 vértices (bgr)
  cores_pontos = [
    (0, 0, 255),  # p1: vermelho
    (0, 255, 0),  # p2: verde
    (255, 0, 0),  # p3: azul
    (0, 255, 255)  # p4: amarelo
  ]

  for i, (frame, is_stream) in enumerate(stream_midia(args.midia)):
    pts_origem = obter_pontos_origem_padrao(frame, args.midia)

    # 1. calcular a matriz 3x3 de perspectiva 
    M_perspectiva = cv2.getPerspectiveTransform(pts_origem, pts_destino)

    # 2. aplicar cv2.warpperspective para projetar o plano de volta à vista regular
    frame_retificado = cv2.warpPerspective(
      frame,
      M_perspectiva,
      (largura_saida, altura_saida)
    )

    if args.no_show:
      if i >= 5 or not is_stream:
        print(f"[Modo no-show] {i+1} frames processados com sucesso.")
        break
      continue

    # 3. desenhar o polígono e vértices no frame original para visualização
    frame_anotado = frame.copy()
    pts_int = pts_origem.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame_anotado, [pts_int], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for i, (pt, cor) in enumerate(zip(pts_origem, cores_pontos)):
      pt_xy = (int(pt[0]), int(pt[1]))
      cv2.circle(frame_anotado, pt_xy, 6, cor, -1, lineType=cv2.LINE_AA)
      cv2.putText(
        frame_anotado, f"P{i+1}",
        (pt_xy[0] + 8, pt_xy[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2, cv2.LINE_AA
      )

    # 4. montagem lado a lado comparativa
    painel = concatenar_lado_a_lado(
      [frame_anotado, frame_retificado],
      titulos=["Origem (Perspectiva Inclinada)", "Vista Retificada (Ortogonal)"],
      altura_padrao=420
    )

    nome_janela = "Transformacao de Perspectiva - Pressione Q para sair"
    cv2.imshow(nome_janela, painel)

    # se for imagem estatica, mantem a janela aberta ate o usuario encerrar
    if not is_stream:
      while True:
        tecla = cv2.waitKey(50) & 0xFF
        if tecla in [ord("q"), ord("Q"), 27]:
          break
        if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
          break
      break

    # se for stream de video ou webcam
    tecla = cv2.waitKey(30) & 0xFF
    if tecla in [ord("q"), ord("Q"), 27]:
      break
    if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
      break

  cv2.destroyAllWindows()
  print("[Fim] Demonstração de perspectiva encerrada.")


if __name__ == "__main__":
  main()
