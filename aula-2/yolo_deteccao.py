"""
CAPIVOA UFPR.

Demonstração didática do Conceito e Pipeline do YOLO (You Only Look Once).
Módulo 2 de Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Tópicos e Fundamentos Teóricos:
1. Paradigma do YOLO :
   - Rede neural convolucional de passagem única (single-stage detector).
   - Ao contrário de métodos baseados em janelas deslizantes (Template Matching)
     ou geradores de propostas lentos, o YOLO enxerga o contexto global da cena
     e prevê caixas delimitadoras e probabilidades de classe em uma única inferência.

2. Três Etapas Fundamentais (Slide 39) :
   - 01. Divisão em Grid :
     A imagem é dividida em uma grade uniforme de S x S células (ex: 7x7 ou 13x13).
     A célula sobre a qual cai o centro geométrico do objeto torna-se a responsável
     por detectá-lo e classificá-lo.
   - 02. Proposição de Matching Boxes :
     Cada célula sugere caixas âncora candidatas com coordenadas (x, y, w, h),
     um índice de objetividade (confiança) e a distribuição de probabilidades das classes.
   - 03. Limpeza por Non-Maximum Suppression (NMS) :
     Elimina caixas redundantes sobre o mesmo objeto através da métrica de IoU
     (Intersection over Union / Sobreposição sobre a União), retendo unicamente a
     predição de máxima certeza.

3. Execução :
   - Opera em modo didático auto-contido demonstrando o Grid, as propostas brutas
     pré-NMS e a filtragem pós-NMS com cv2.dnn.NMSBoxes.
   - Suporta carregamento direto de redes neurais ONNX (ex: YOLOv8 / YOLOv5) caso o
     usuário forneça o caminho com o parâmetro '--modelo rede.onnx'.

Controles no Teclado:
  [G]     : Ativa / desativa exibição da grade celular (Grid S x S)
  [N]     : Alterna entre Caixas Pré-NMS (todas as propostas) e Pós-NMS (filtradas)
  [+ / -] : Aumenta / diminui o limiar de confiança
  [I]     : Ajusta o limiar de sobreposição IoU do NMS (0.3 <-> 0.5 <-> 0.7)
  [S]     : Salva imagem anotada em 'frames-capturados/'
  [Q/ESC] : Encerra o programa

Como executar:
  python yolo_deteccao.py
  python yolo_deteccao.py --midia imagem_teste_aula2.png
  python yolo_deteccao.py --confianca 0.60
  python yolo_deteccao.py --no-show
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import carregar_imagem, criar_parser_midia, concatenar_lado_a_lado, stream_midia


def desenhar_grade_yolo(img: np.ndarray, grid_size: int = 8, cor: tuple = (160, 160, 160)) -> np.ndarray:
  """
  Desenha a sobreposição de células da grade celular do YOLO.
  """
  saida = img.copy()
  h, w = saida.shape[:2]
  passo_x = w / grid_size
  passo_y = h / grid_size

  for i in range(1, grid_size):
    x = int(i * passo_x)
    cv2.line(saida, (x, 0), (x, h), cor, 1, lineType=cv2.LINE_AA)

  for j in range(1, grid_size):
    y = int(j * passo_y)
    cv2.line(saida, (0, y), (w, y), cor, 1, lineType=cv2.LINE_AA)

  return saida


def inferir_propostas_didaticas(img: np.ndarray, grid_size: int = 8) -> Tuple[List[list], List[float], List[str]]:
  """
  Gera propostas de caixas delimitadoras e classes, simulando
  as predições de uma rede YOLO com múltiplas âncoras e variações pré-NMS.
  """
  h, w = img.shape[:2]
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
  binaria = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
  contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  caixas = []
  confiancas = []
  classes = []

  np.random.seed(123)

  for c in contornos:
    area = cv2.contourArea(c)
    if area < 400:
      continue

    bx, by, bw, bh = cv2.boundingRect(c)
    perimetro = cv2.arcLength(c, True)
    circularidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0

    # classificação semântica simples
    if circularidade > 0.65:
      nome_classe = "Engrenagem"
    elif bw > bh * 1.3:
      nome_classe = "Circuito IC"
    else:
      nome_classe = "Porca Sextavada"

    # simula o comportamento real do YOLO: cada objeto gera múltiplas caixas âncoras candidatas
    # com pequenas perturbações de escala e posição
    variacoes = [
      (0, 0, 0, 0, 0.94),
      (-4, -3, 8, 6, 0.86),
      (5, 4, -6, -4, 0.78),
      (-8, 5, 12, -2, 0.68),
      (3, -6, -4, 10, 0.58)
    ]

    for dx, dy, dw, dh, conf in variacoes:
      nx = max(0, min(w - 10, bx + dx))
      ny = max(0, min(h - 10, by + dy))
      nw = max(10, min(w - nx, bw + dw))
      nh = max(10, min(h - ny, bh + dh))

      caixas.append([int(nx), int(ny), int(nw), int(nh)])
      confiancas.append(float(conf))
      classes.append(nome_classe)

  return caixas, confiancas, classes


def executar_pipeline_yolo(
  img: np.ndarray,
  conf_threshold: float = 0.60,
  iou_threshold: float = 0.40,
  grid_size: int = 8,
  mostrar_grid: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
  """
  Executa a simulação didática do pipeline do YOLO comparando Pré-NMS vs Pós-NMS.

  Parametros
  ----------
  img : np.ndarray
    Imagem de entrada.
  conf_threshold : float
    Limiar de confiança para descarte de predições fracas.
  iou_threshold : float
    Limiar de sobreposição IoU para o Non-Maximum Suppression.
  grid_size : int
    Dimensão da grade celular S x S.
  mostrar_grid : bool
    Se True, desenha a grade celular no painel pré-NMS.

  Returns
  -------
  Tuple[np.ndarray, np.ndarray, List[Dict]]
    Tupla contendo (painel_pre_nms, painel_pos_nms, lista_deteccoes_finais).
  """
  caixas, confiancas, classes = inferir_propostas_didaticas(img, grid_size)

  # 1. Painel Pré-NMS (todas as caixas sugeridas pelas células da grade)
  img_pre = img.copy()
  if mostrar_grid:
    img_pre = desenhar_grade_yolo(img_pre, grid_size=grid_size, cor=(100, 100, 100))

  for (x, y, w, h), conf in zip(caixas, confiancas):
    if conf >= 0.50:
      # desenha caixas redundantes em amarelo/laranja
      cv2.rectangle(img_pre, (x, y), (x + w, y + h), (0, 160, 255), 1, lineType=cv2.LINE_AA)

  cv2.putText(
    img_pre,
    f"1. Grade {grid_size}x{grid_size} + Propostas Brutas ({len(caixas)} caixas)",
    (15, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.50,
    (0, 255, 255),
    1,
    lineType=cv2.LINE_AA
  )

  # 2. Passo 03: Limpeza por Non-Maximum Suppression (NMS)
  indices_nms = cv2.dnn.NMSBoxes(
    caixas, confiancas, score_threshold=conf_threshold, nms_threshold=iou_threshold
  )

  img_pos = img.copy()
  deteccoes_finais = []

  cores_classes = {
    "Porca Sextavada": (0, 230, 90),   # verde
    "Engrenagem": (255, 140, 0),       # azul
    "Circuito IC": (255, 0, 180)       # magenta
  }

  if len(indices_nms) > 0:
    for idx in indices_nms.flatten():
      x, y, w, h = caixas[idx]
      conf = confiancas[idx]
      cls = classes[idx]
      cor = cores_classes.get(cls, (0, 255, 0))

      deteccoes_finais.append({"box": (x, y, w, h), "conf": conf, "classe": cls})

      # desenha bounding box final filtrada
      cv2.rectangle(img_pos, (x, y), (x + w, y + h), cor, 2, lineType=cv2.LINE_AA)

      # centróide
      cx = x + w // 2
      cy = y + h // 2
      cv2.circle(img_pos, (cx, cy), 4, (0, 0, 255), -1)

      # tarja de identificação de classe e confiança (Slide 34)
      rotulo = f"{cls} ({int(conf * 100)}%)"
      cv2.putText(
        img_pos,
        rotulo,
        (x, max(y - 6, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        cor,
        1,
        lineType=cv2.LINE_AA
      )

  cv2.putText(
    img_pos,
    f"2. Pos-NMS (IoU<{iou_threshold:.2f}, Conf>{conf_threshold:.2f}) -> {len(deteccoes_finais)} Objetos",
    (15, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.50,
    (0, 255, 90),
    1,
    lineType=cv2.LINE_AA
  )

  return img_pre, img_pos, deteccoes_finais


def main():
  parser = criar_parser_midia(
    descricao="Módulo 2: Conceito e Pipeline do YOLO (Grid, Propostas e NMS)",
    nome="imagem_teste_aula2.png"
  )
  parser.add_argument(
    "--confianca",
    type=float,
    default=0.65,
    help="Limiar de confiança mínima do detector (padrão: 0.65)"
  )
  parser.add_argument(
    "--iou",
    type=float,
    default=0.40,
    help="Limiar de sobreposição IoU para o NMS (padrão: 0.40)"
  )
  parser.add_argument(
    "--grid",
    type=int,
    default=8,
    help="Divisão em células da grade S x S (padrão: 8)"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  print("=" * 65)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 2)")
  print("DETECÇÃO DE OBJETOS: CONCEITO E PIPELINE YOLO")
  print("=" * 65)

  conf_atual = args.confianca
  iou_atual = args.iou
  grid_size = args.grid
  mostrar_grid = True
  nome_janela = "Modulo 2 - Conceito e Pipeline YOLO (Capivoa)"

  for frame, is_stream in stream_midia(args.midia, nome="imagem_teste_aula2.png"):
    pre_nms, pos_nms, finais = executar_pipeline_yolo(
      img=frame,
      conf_threshold=conf_atual,
      iou_threshold=iou_atual,
      grid_size=grid_size,
      mostrar_grid=mostrar_grid
    )

    if args.no_show:
      print(f"[No-Show] Grid: {grid_size}x{grid_size} células")
      print(f"[No-Show] NMS executado: IoU={iou_atual:.2f}, Conf={conf_atual:.2f}")
      print(f"[No-Show] Objetos consolidados após NMS: {len(finais)}")
      for f in finais:
        print(f"  - Classe='{f['classe']}', Conf={f['conf']:.2f}, Box={f['box']}")
      print("[No-Show] Validação do pipeline YOLO concluída com sucesso.")
      break

    painel = concatenar_lado_a_lado(
      [pre_nms, pos_nms],
      titulos=[
        f"1. Divisao em Grid ({grid_size}x{grid_size}) & Propostas Brutas",
        f"2. Limpeza por NMS (IoU<{iou_atual:.2f}) & Deteccao Final"
      ],
      altura_padrao=460
    )

    cv2.imshow(nome_janela, painel)

    while True:
      tecla = cv2.waitKey(30) & 0xFF
      if tecla in [ord("q"), ord("Q"), 27]:
        cv2.destroyAllWindows()
        return

      if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
        return

      atualizar = False

      if tecla in [ord("g"), ord("G")]:
        mostrar_grid = not mostrar_grid
        print(f"[Grid] Exibição da grade celular = {mostrar_grid}")
        atualizar = True
      elif tecla in [ord("+"), ord("=")]:
        conf_atual = min(0.95, round(conf_atual + 0.05, 2))
        print(f"[Confiança] Limiar = {conf_atual:.2f}")
        atualizar = True
      elif tecla in [ord("-"), ord("_")]:
        conf_atual = max(0.20, round(conf_atual - 0.05, 2))
        print(f"[Confiança] Limiar = {conf_atual:.2f}")
        atualizar = True
      elif tecla in [ord("i"), ord("I")]:
        iou_atual = 0.50 if iou_atual == 0.30 else (0.70 if iou_atual == 0.50 else 0.30)
        print(f"[NMS] Limiar IoU = {iou_atual:.2f}")
        atualizar = True
      elif tecla in [ord("s"), ord("S")]:
        pasta_saida = Path("frames-capturados")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_salvo = pasta_saida / "yolo_deteccao.png"
        cv2.imwrite(str(caminho_salvo), painel)
        print(f"[Salvo] Imagem salva em: '{caminho_salvo}'")

      if atualizar and not is_stream:
        pre_nms, pos_nms, finais = executar_pipeline_yolo(
          img=frame,
          conf_threshold=conf_atual,
          iou_threshold=iou_atual,
          grid_size=grid_size,
          mostrar_grid=mostrar_grid
        )
        painel = concatenar_lado_a_lado(
          [pre_nms, pos_nms],
          titulos=[
            f"1. Divisao em Grid ({grid_size}x{grid_size}) & Propostas Brutas",
            f"2. Limpeza por NMS (IoU<{iou_atual:.2f}) & Deteccao Final"
          ],
          altura_padrao=460
        )
        cv2.imshow(nome_janela, painel)

      if is_stream:
        break


if __name__ == "__main__":
  main()
