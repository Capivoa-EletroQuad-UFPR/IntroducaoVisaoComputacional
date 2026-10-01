"""
CAPIVOA UFPR.

Demonstração prática de Extração de Contornos, Hierarquia e Momentos Espaciais.
Módulo 2 de Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Tópicos e Fundamentos Teóricos:
1. Contornos vs Bordas :
   - Uma borda é uma imagem binária com pixels soltos de alta derivada.
   - Um contorno é uma estrutura de dados vetorial (lista ordenada de coordenadas (x, y))
     que modela um polígono fechado contínuo.
   - Vetorização: comprime informação matricial em vértices matemáticos.

2. Hierarquia de Contornos :
   - Relação de parentesco entre contornos aninhados: [Next, Previous, First_Child, Parent].
   - cv2.RETR_EXTERNAL: extrai estritamente os contornos mais externos, ignorando furos.
   - cv2.RETR_TREE: reconstrói a árvore genealógica completa (diferencia o objeto externo
     dos furos internos da peça).

3. Momentos Espaciais e Centróide :
   - Calculados via cv2.moments(contorno):
     * Área: momento de ordem zero m00 (ou cv2.contourArea).
     * Centróide (Ponto Central de Gravidade):
         cx = m10 / m00
         cy = m01 / m00
       Mais robusto e estável do que o centro geométrico simples da bounding box.
     * Perímetro: comprimento do arco fechado (cv2.arcLength).
     * Caixa Delimitadora (Bounding Box): cv2.boundingRect -> (x, y, w, h).

Controles no Teclado:
  [H]     : Alterna modo de hierarquia (RETR_EXTERNAL <-> RETR_TREE com furos)
  [B]     : Ativa / desativa caixas delimitadoras (Bounding Boxes)
  [+ / -] : Aumenta / diminui o limiar de área mínima para descarte de ruído
  [S]     : Salva imagem anotada em 'frames-capturados/'
  [Q/ESC] : Encerra a execução

Como executar:
  python contornos_momentos.py
  python contornos_momentos.py --midia imagem_teste_aula2.png
  python contornos_momentos.py --area-min 150
  python contornos_momentos.py --no-show
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


def analisar_contornos(
  img_binaria: np.ndarray,
  img_colorida: np.ndarray,
  modo_hierarquia: int = cv2.RETR_TREE,
  area_minima: float = 100.0,
  desenhar_bbox: bool = True
) -> Tuple[np.ndarray, List[Dict]]:
  """
  Extrai os contornos da imagem binária, filtra ruídos e calcula momentos espaciais.

  Parametros
  ----------
  img_binaria : np.ndarray
    Máscara monocromática binária (0 e 255).
  img_colorida : np.ndarray
    Quadro colorido original (BGR) onde serão desenhadas as anotações visuais.
  modo_hierarquia : int
    Flag do OpenCV para hierarquia (cv2.RETR_EXTERNAL ou cv2.RETR_TREE).
  area_minima : float
    Limite de corte em pixels para descartar ruídos espúrios.
  desenhar_bbox : bool
    Se True, desenha as caixas delimitadoras retangulares.

  Returns
  -------
  Tuple[np.ndarray, List[Dict]]
    Tupla contendo a imagem anotada em cores e a lista de dados métricos dos contornos válidos.
  """
  contornos, hierarquia = cv2.findContours(img_binaria, modo_hierarquia, cv2.CHAIN_APPROX_SIMPLE)
  saida = img_colorida.copy()

  objetos_validos = []
  total_ruidos = 0

  if hierarquia is None or len(contornos) == 0:
    return saida, objetos_validos

  hierarquia = hierarquia[0]  # remove dimensão extra do OpenCV

  for i, c in enumerate(contornos):
    area = cv2.contourArea(c)

    # 1. descarte de ruído por área mínima
    if area < area_minima:
      total_ruidos += 1
      # desenha ruído descartado em vermelho translúcido sutil
      cv2.drawContours(saida, [c], -1, (40, 40, 180), 1)
      continue

    # 2. análise de parentesco da hierarquia [Next, Prev, First_Child, Parent]
    parent_idx = hierarquia[i][3]
    e_furo_interno = (parent_idx != -1)

    # se for furo interno, usamos cor amarela/laranja; se for externo, verde/ciano
    cor_contorno = (0, 180, 255) if e_furo_interno else (0, 230, 90)

    # 3. momentos espaciais
    m = cv2.moments(c)
    if m["m00"] > 0:
      cx = int(m["m10"] / m["m00"])
      cy = int(m["m01"] / m["m00"])
    else:
      cx, cy = 0, 0

    perimetro = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)

    # circularidade: 4*pi*area / (perimetro^2) (1.0 = círculo perfeito)
    circularidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0.0

    objetos_validos.append({
      "indice": i,
      "area": area,
      "perimetro": perimetro,
      "centro": (cx, cy),
      "bbox": (x, y, w, h),
      "furo": e_furo_interno,
      "circularidade": circularidade
    })

    # 4. renderização gráfica
    # desenha contorno poligonal
    cv2.drawContours(saida, [c], -1, cor_contorno, 2, lineType=cv2.LINE_AA)

    # marca centróide com cruz e ponto
    if cx > 0 and cy > 0:
      cv2.circle(saida, (cx, cy), 4, (0, 0, 255), -1)
      cv2.drawMarker(saida, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

    # caixa delimitadora
    if desenhar_bbox and not e_furo_interno:
      cv2.rectangle(saida, (x, y), (x + w, y + h), (255, 120, 0), 2, lineType=cv2.LINE_AA)

    # texto de telemetria sobre o objeto
    rotulo_tipo = "Furo" if e_furo_interno else f"Obj #{len(objetos_validos)}"
    pos_texto_y = max(y - 8, 18)
    cv2.putText(
      saida,
      f"{rotulo_tipo} (Area:{int(area)})",
      (x, pos_texto_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.45,
      cor_contorno,
      1,
      lineType=cv2.LINE_AA
    )

    if cx > 0 and cy > 0 and not e_furo_interno:
      cv2.putText(
        saida,
        f"C:({cx},{cy}) {w}x{h}",
        (cx + 10, cy + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA
      )

  # HUD informativo superior
  modo_texto = "RETR_TREE (Hierarquia Completa com Furos)" if modo_hierarquia == cv2.RETR_TREE else "RETR_EXTERNAL (Apenas Externos)"
  hud_info = f"Modo: {modo_texto} | Validos: {len(objetos_validos)} | Ruidos (<{int(area_minima)}px): {total_ruidos}"
  cv2.putText(saida, hud_info, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, lineType=cv2.LINE_AA)

  return saida, objetos_validos


def main():
  parser = criar_parser_midia(
    descricao="Módulo 2: Contornos, Hierarquias e Momentos Espaciais",
    nome="imagem_teste_aula2.png"
  )
  parser.add_argument(
    "--area-min",
    type=float,
    default=80.0,
    help="Área mínima de corte para eliminação de ruídos (padrão: 80.0)"
  )
  parser.add_argument(
    "--apenas-externos",
    action="store_true",
    help="Utiliza cv2.RETR_EXTERNAL em vez de cv2.RETR_TREE."
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  print("=" * 65)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 2)")
  print("EXTRAÇÃO DE CONTORNOS, HIERARQUIA E MOMENTOS ESPACIAIS")
  print("=" * 65)

  area_minima = args.area_min
  modo_hierarquia = cv2.RETR_EXTERNAL if args.apenas_externos else cv2.RETR_TREE
  desenhar_bbox = True
  nome_janela = "Modulo 2 - Contornos e Momentos Espaciais (Capivoa)"

  for frame, is_stream in stream_midia(args.midia, nome="imagem_teste_aula2.png"):
    img_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    # segmentação prévia adaptativa para isolar os objetos
    binaria = cv2.adaptiveThreshold(
      img_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    anotada, objetos = analisar_contornos(
      binaria, frame, modo_hierarquia=modo_hierarquia, area_minima=area_minima, desenhar_bbox=desenhar_bbox
    )

    if args.no_show:
      print(f"[No-Show] Contornos válidos detectados: {len(objetos)}")
      for obj in objetos[:5]:
        print(f"  - Objeto centro={obj['centro']}, area={obj['area']:.1f}, bbox={obj['bbox']}, furo={obj['furo']}")
      print("[No-Show] Validação de contornos e momentos concluída com sucesso.")
      break

    painel = concatenar_lado_a_lado(
      [binaria, anotada],
      titulos=["1. Segmentacao Binaria Adaptativa", "2. Contornos Vetorizados, Bounding Box e Centroide"],
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

      if tecla in [ord("h"), ord("H")]:
        modo_hierarquia = cv2.RETR_EXTERNAL if modo_hierarquia == cv2.RETR_TREE else cv2.RETR_TREE
        nome_m = "RETR_EXTERNAL" if modo_hierarquia == cv2.RETR_EXTERNAL else "RETR_TREE"
        print(f"[Hierarquia] Alternado para: {nome_m}")
        atualizar = True
      elif tecla in [ord("b"), ord("B")]:
        desenhar_bbox = not desenhar_bbox
        print(f"[Bounding Box] Exibição = {desenhar_bbox}")
        atualizar = True
      elif tecla in [ord("+"), ord("=")]:
        area_minima += 25
        print(f"[Área Mínima] Corte de ruído = {area_minima} px")
        atualizar = True
      elif tecla in [ord("-"), ord("_")]:
        area_minima = max(5.0, area_minima - 25)
        print(f"[Área Mínima] Corte de ruído = {area_minima} px")
        atualizar = True
      elif tecla in [ord("s"), ord("S")]:
        pasta_saida = Path("frames-capturados")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_salvo = pasta_saida / "contornos_momentos.png"
        cv2.imwrite(str(caminho_salvo), anotada)
        print(f"[Salvo] Imagem salva em: '{caminho_salvo}'")

      if atualizar and not is_stream:
        anotada, objetos = analisar_contornos(
          binaria, frame, modo_hierarquia=modo_hierarquia, area_minima=area_minima, desenhar_bbox=desenhar_bbox
        )
        painel = concatenar_lado_a_lado(
          [binaria, anotada],
          titulos=["1. Segmentacao Binaria Adaptativa", "2. Contornos Vetorizados, Bounding Box e Centroide"],
          altura_padrao=460
        )
        cv2.imshow(nome_janela, painel)

      if is_stream:
        break


if __name__ == "__main__":
  main()
