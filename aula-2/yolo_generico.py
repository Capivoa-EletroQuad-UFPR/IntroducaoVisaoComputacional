"""
CAPIVOA UFPR.

Detector e classificador genérico de objetos em tempo real utilizando YOLO (Ultralytics).
Módulo 2 de Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Como executar:
  python yolo_generico.py
  python yolo_generico.py --midia video.mp4
  python yolo_generico.py --midia 0                  # Webcam ao vivo
  python yolo_generico.py --modelo yolov8s.pt        # Outro modelo
  python yolo_generico.py --conf 0.60
  python yolo_generico.py --no-show
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import criar_parser_midia, stream_midia


def detectar_objetos(model: YOLO, frame: np.ndarray, conf_min: float = 0.50) -> List[Dict]:
  """
  Executa a inferência da rede YOLO sobre o frame e retorna as detecções filtradas.
  """
  results = model(frame, conf=conf_min, verbose=False)
  deteccoes = []

  for r in results:
    for box in r.boxes:
      cls_id = int(box.cls[0])
      conf = float(box.conf[0])
      nome = model.names[cls_id]
      x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

      deteccoes.append({
        "id": cls_id,
        "classe": nome,
        "conf": conf,
        "bbox": (x1, y1, x2 - x1, y2 - y1),  # (x, y, w, h)
        "coords": (x1, y1, x2, y2)
      })

  return deteccoes


def desenhar_deteccoes(frame: np.ndarray, deteccoes: List[Dict]) -> np.ndarray:
  """
  Desenha bounding boxes, classes, percentual de confiança e centróide nos objetos.
  """
  anotado = frame.copy()

  for d in deteccoes:
    x1, y1, x2, y2 = d["coords"]
    cls_nome = d["classe"]
    conf = d["conf"]

    # gera cor única consistente por classe usando hash
    np.random.seed(d["id"] * 37 + 11)
    cor = tuple(int(c) for c in np.random.randint(60, 255, size=3))

    # Bounding Box
    cv2.rectangle(anotado, (x1, y1), (x2, y2), cor, 2, lineType=cv2.LINE_AA)

    # Centróide
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.circle(anotado, (cx, cy), 4, (0, 0, 255), -1)

    # Rótulo de classe e confiança
    rotulo = f"{cls_nome}: {int(conf * 100)}%"
    (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.rectangle(anotado, (x1, y1 - th - 8), (x1 + tw + 6, y1), cor, -1)
    cv2.putText(
      anotado,
      rotulo,
      (x1 + 3, y1 - 4),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 0, 0),
      1,
      lineType=cv2.LINE_AA
    )

  return anotado


def main():
  parser = criar_parser_midia(
    descricao="Detector e classificador genérico de objetos com YOLO",
    nome="video.mp4"
  )
  parser.add_argument(
    "--modelo",
    type=str,
    default="yolov8n.pt",
    help="Modelo YOLO pré-treinado ou customizado (padrão: yolov8n.pt)"
  )
  parser.add_argument(
    "--conf",
    type=float,
    default=0.50,
    help="Limiar de confiança mínima (padrão: 0.50)"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa sem abrir janela gráfica."
  )
  args = parser.parse_args()

  print("=" * 65)
  print("CAPIVOA - DETECTOR GENÉRICO YOLO")
  print("=" * 65)
  print(f"Modelo     : {args.modelo}")
  print(f"Mídia      : {args.midia}")
  print(f"Confiança  : {args.conf:.2f}")
  print("=" * 65)

  print("[1] Carregando pesos da rede neural...")
  model = YOLO(args.modelo)
  print("[OK] Modelo carregado com sucesso.")

  conf_atual = args.conf
  pausado = False
  num_frame = 0
  nome_janela = "CAPIVOA - Deteccao de Objetos YOLO"

  for frame, is_stream in stream_midia(args.midia, nome="video.mp4"):
    num_frame += 1
    deteccoes = detectar_objetos(model, frame, conf_min=conf_atual)

    # print direto e conciso no terminal
    classes_contagem = {}
    for d in deteccoes:
      classes_contagem[d["classe"]] = classes_contagem.get(d["classe"], 0) + 1

    resumo = ", ".join([f"{k}: {v}" for k, v in classes_contagem.items()]) if classes_contagem else "Nenhum"
    print(f"[Frame {num_frame:04d}] {len(deteccoes)} objetos detectados | {resumo}")

    if args.no_show:
      if not is_stream or num_frame >= 5:
        print("[No-Show] Validação de inferência YOLO concluída.")
        break
      continue

    anotado = desenhar_deteccoes(frame, deteccoes)

    # HUD superior
    cv2.putText(
      anotado,
      f"YOLO ({args.modelo}) | Objetos: {len(deteccoes)} | Conf: {conf_atual:.2f}",
      (15, 25),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (0, 255, 255),
      2,
      lineType=cv2.LINE_AA
    )

    cv2.imshow(nome_janela, anotado)

    while True:
      tecla = cv2.waitKey(30 if not pausado else 0) & 0xFF

      if tecla in [ord("q"), ord("Q"), 27]:
        cv2.destroyAllWindows()
        return

      if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
        return

      if tecla == 32:  # ESPAÇO
        pausado = not pausado
        print(f"[Status] {'Pausado' if pausado else 'Reproduzindo'}")

      elif tecla in [ord("+"), ord("=")]:
        conf_atual = min(0.95, round(conf_atual + 0.05, 2))
        print(f"[Confiança] {conf_atual:.2f}")

      elif tecla in [ord("-"), ord("_")]:
        conf_atual = max(0.10, round(conf_atual - 0.05, 2))
        print(f"[Confiança] {conf_atual:.2f}")

      elif tecla in [ord("s"), ord("S")]:
        pasta_saida = Path("frames-capturados")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_salvo = pasta_saida / f"yolo_frame_{num_frame}.png"
        cv2.imwrite(str(caminho_salvo), anotado)
        print(f"[Salvo] Frame salvo em: '{caminho_salvo}'")

      if not pausado or not is_stream:
        break


if __name__ == "__main__":
  main()
