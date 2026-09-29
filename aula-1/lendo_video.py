"""
CAPIVOA UFPR.

Demonstração didática de Captura e Processamento de Vídeo / Câmera .
Módulo 1 - Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Conceito teórico :
- Um vídeo digital é uma sequência temporal contínua de matrizes (frames).
- No OpenCV, cv2.VideoCapture gerencia streams de arquivos de vídeo e webcams.
- A leitura ocorre frame a frame através do método cap.read(), que retorna:
 1. ret: booleano indicando se o frame foi lido com sucesso.
 2. frame: matriz NumPy com as dimensões (altura, largura, 3 canais BGR).
- O controle de taxa de quadros (FPS) é sincronizado pelo tempo de espera em cv2.waitKey(ms).

Controles interativos pelo teclado:
- ESPAÇO : Pausa / Retoma a reprodução do vídeo
- 's'   : Salva o frame atual em disco na pasta 'frames-capturados/'
- 'q'/ESC : Encerra a reprodução

Como executar:
  python lendo_video.py           # Executa com 'video.mp4' padrão
  python lendo_video.py --midia 0      # Captura ao vivo da Webcam
  python lendo_video.py --midia video.mp4 -t "Minha Legenda"
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import obter_captura_video, criar_parser_midia


def main():
  """
  Executa a leitura, reprodução e salvamento de quadros de vídeo ou webcam.

  Inicializa o leitor cv2.VideoCapture, calcula taxa de quadros (FPS) em tempo real,
  sobrepõe telemetria nos quadros e gerencia pausas e capturas de telas.
  """
  parser = criar_parser_midia(
    descricao="Leitura e Exibição de Vídeo/Câmera frame a frame com OpenCV",
    nome="video.mp4"
  )
  parser.add_argument(
    "-t", "--texto",
    type=str,
    default="Visao Computacional - Capivoa",
    help="Texto descritivo sobreposto no vídeo"
  )
  parser.add_argument(
    "-d", "--output-dir",
    type=str,
    default="frames-capturados",
    help="Diretório de destino para frames salvos com tecla 's'"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa a leitura sem abrir janela gráfica."
  )
  args = parser.parse_args()

  diretorio_saida = Path(args.output_dir)
  diretorio_saida.mkdir(parents=True, exist_ok=True)
  nome_janela = "Video Player (Capivoa)"

  print("=" * 60)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 1)")
  print("LEITURA DE VÍDEO E WEBCAM COM OpenCV")
  print("=" * 60)
  print(f"Fonte de vídeo : '{args.midia}'")
  print(f"Pasta de capturas: '{diretorio_saida}/'")
  print("Atalhos no teclado:")
  print(" [ESPAÇO] : Pausar / Continuar")
  print(" [S]   : Salvar frame atual")
  print(" [Q / ESC]: Encerrar")
  print("-" * 60)

  # 1. abertura da captura de video (cv2.videocapture)
  cap = obter_captura_video(args.midia, nome="video.mp4")

  # metadados do stream de vídeo
  largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps_video = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  if fps_video <= 0 or np.isnan(fps_video):
    fps_video = 30.0 # padrão razoável para webcams

  # intervalo de espera em milissegundos para sincronizar com o fps original
  atraso_ms = max(1, int(1000.0 / fps_video))

  print(f"Resolução    : {largura}x{altura} px")
  print(f"FPS nominal   : {fps_video:.1f} frames/segundo (espera ~{atraso_ms} ms)")
  if total_frames > 0:
    print(f"Total de frames : {total_frames}")

  indice_frame = 0
  pausado = False
  tempo_anterior = time.time()
  fps_calculado = fps_video

  try:
    while True:
      if not pausado:
        ret, frame = cap.read()
        if not ret or frame is None:
          print("\n[Aviso] Fim do vídeo ou falha na leitura do frame.")
          break

        indice_frame += 1

        # cálculo de fps real medido
        tempo_atual = time.time()
        delta_tempo = tempo_atual - tempo_anterior
        tempo_anterior = tempo_atual
        if delta_tempo > 0:
          fps_instantaneo = 1.0 / delta_tempo
          # média móvel exponencial para suavizar a leitura de fps
          fps_calculado = 0.9 * fps_calculado + 0.1 * fps_instantaneo

        if args.no_show:
          if indice_frame >= 10:
            print(f"[Modo no-show] {indice_frame} frames processados com sucesso.")
            break
          continue

      # cria cópia do frame para desenhar elementos de interface (hud)
      frame_exibicao = frame.copy()

      # faixa superior translúcida para status
      h, w = frame_exibicao.shape[:2]
      overlay = frame_exibicao.copy()
      cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
      cv2.addWeighted(overlay, 0.6, frame_exibicao, 0.4, 0, frame_exibicao)

      # texto do usuario (cv2.puttext)
      cv2.putText(
        frame_exibicao,
        args.texto,
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 230, 255),
        2,
        cv2.LINE_AA
      )

      # informações de telemetria (frame, fps, status)
      texto_telemetria = f"Frame: #{indice_frame:04d} | FPS: {fps_calculado:.1f}"
      if pausado:
        texto_telemetria += " [PAUSADO - Pressione ESPAÇO]"

      cv2.putText(
        frame_exibicao,
        texto_telemetria,
        (20, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA
      )

      # instruções na barra inferior
      cv2.putText(
        frame_exibicao,
        "[ESPACO] Pausar | [S] Salvar Frame | [Q] Sair",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
      )

      cv2.imshow(nome_janela, frame_exibicao)

      if pausado:
        encerrar = False
        while pausado:
          tecla_pausa = cv2.waitKey(50) & 0xFF
          if tecla_pausa in [ord("q"), ord("Q"), 27]:
            encerrar = True
            break
          if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
            encerrar = True
            break
          if tecla_pausa == 32: # barra de espaco
            pausado = False
            print("[Controle] Vídeo retomado.")
          elif tecla_pausa in [ord("s"), ord("S")]:
            nome_foto = diretorio_saida / f"frame_{indice_frame:05d}.jpg"
            cv2.imwrite(str(nome_foto), frame)
            print(f"[Salvo] Frame capturado com sucesso: '{nome_foto}'")
        if encerrar:
          break
      else:
        tecla = cv2.waitKey(atraso_ms) & 0xFF
        if tecla in [ord("q"), ord("Q"), 27]:
          break
        if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
          break
        if tecla == 32: # barra de espaco
          pausado = True
          print("[Controle] Vídeo pausado.")
        elif tecla in [ord("s"), ord("S")]:
          nome_foto = diretorio_saida / f"frame_{indice_frame:05d}.jpg"
          cv2.imwrite(str(nome_foto), frame)
          print(f"[Salvo] Frame capturado com sucesso: '{nome_foto}'")

  finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[Fim] Reprodução concluída.")


if __name__ == "__main__":
  main()
