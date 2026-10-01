"""
CAPIVOA UFPR.

Demonstração prática de Detecção de Objetos via Template Matching.
Módulo 2 de Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Tópicos e Fundamentos Teóricos:
1. Template Matching :
   - Técnica clássica de detecção por correlação espacial.
   - Desliza uma matriz menor de referência (molde/template) pixel a pixel sobre a cena,
     gerando uma matriz 2D de similaridade (mapa de calor).
   - Não requer treinamento prévio e possui alta velocidade de inferência.

2. Métodos de Correlação no OpenCV :
   - Diferença Quadrática Normalizada (cv2.TM_SQDIFF_NORMED) :
     Calcula a soma das diferenças quadráticas entre os pixels.
     O melhor casamento corresponde ao ponto de MÍNIMO global (valor mais próximo de 0).
   - Correlação Cruzada Normalizada (cv2.TM_CCORR_NORMED) :
     Multiplica os valores dos pixels correspondentes.
     O melhor casamento é o ponto de MÁXIMO global (valor mais alto).
   - Coeficiente de Correlação Normalizado (cv2.TM_CCOEFF_NORMED) :
     Subtrai a intensidade média local antes do produto escalar.
     Imune a variações lineares de brilho/iluminação. Gera valores padronizados de -1 a +1
     (sendo +1 a correspondência exata). É o método mais confiável e utilizado.

3. Limitações Críticas :
   - Rotação: o molde falha drasticamente se o objeto girar no espaço 2D.
   - Escala / Tamanho: variações de distância alteram as dimensões do objeto, anulando o match.
   - Ângulo de Visão / Perspectiva: deformações afins e projetivas quebram a matriz de pesos.
   - Oclusão Parcial: sobreposição ou cortes parciais reduzem a pontuação abaixo do limiar.

Controles no Teclado:
  [M]     : Alterna método (TM_CCOEFF_NORMED <-> TM_SQDIFF_NORMED <-> TM_CCORR_NORMED)
  [T]     : Alterna template (Porca <-> Engrenagem <-> Chip)
  [R]     : Rotaciona o template em +15 graus (demonstra limitação de rotação)
  [W / S] : Aumenta / diminui a escala do template (demonstra limitação de tamanho)
  [0]     : Reseta rotação e escala do template para o padrão
  [P]     : Salva a cena anotada em 'frames-capturados/'
  [Q/ESC] : Encerra a execução

Como executar:
  python template_matching.py
  python template_matching.py --midia imagem_teste_aula2.png
  python template_matching.py --template template_engrenagem.png
  python template_matching.py --no-show
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import (
  carregar_imagem,
  criar_parser_midia,
  concatenar_lado_a_lado,
  stream_midia,
  resolver_caminho
)


METODOS_OPENCV = {
  "ccoeff": (cv2.TM_CCOEFF_NORMED, "TM_CCOEFF_NORMED (Coeficiente Normalizado - Maximo)"),
  "sqdiff": (cv2.TM_SQDIFF_NORMED, "TM_SQDIFF_NORMED (Diferenca Quadratica - Minimo)"),
  "ccorr": (cv2.TM_CCORR_NORMED, "TM_CCORR_NORMED (Correlacao Cruzada - Maximo)")
}


def transformar_template(template_base: np.ndarray, angulo: float, escala: float) -> np.ndarray:
  """
  Aplica rotação e escala arbitrárias ao template para simular condições adversas.
  """
  if angulo == 0 and escala == 1.0:
    return template_base

  h, w = template_base.shape[:2]
  centro = (w / 2.0, h / 2.0)
  matriz_afim = cv2.getRotationMatrix2D(centro, angulo, escala)

  # calcula nova bounding box para evitar corte das pontas
  cos = np.abs(matriz_afim[0, 0])
  sin = np.abs(matriz_afim[0, 1])
  novo_w = int((h * sin) + (w * cos))
  novo_h = int((h * cos) + (w * sin))

  matriz_afim[0, 2] += (novo_w / 2.0) - centro[0]
  matriz_afim[1, 2] += (novo_h / 2.0) - centro[1]

  template_transformado = cv2.warpAffine(
    template_base, matriz_afim, (novo_w, novo_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
  )
  return template_transformado


def executar_matching(
  img_cena: np.ndarray,
  template: np.ndarray,
  chave_metodo: str = "ccoeff",
  limiar_deteccao: float = 0.70
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
  """
  Executa a correlação entre a imagem de cena e o template.

  Parametros
  ----------
  img_cena : np.ndarray
    Imagem da cena completa (BGR ou Grayscale).
  template : np.ndarray
    Molde de referência a ser localizado.
  chave_metodo : str
    Chave do método de correlação ('ccoeff', 'sqdiff' ou 'ccorr').
  limiar_deteccao : float
    Pontuação mínima para aceitar múltiplos casamentos.

  Returns
  -------
  Tuple[np.ndarray, np.ndarray, List[Dict]]
    Tupla contendo (cena_anotada, mapa_calor_visual, lista_deteccoes).
  """
  flag_metodo, nome_metodo = METODOS_OPENCV[chave_metodo]

  # conversão para escala de cinza para correlacionar intensidade luminosa
  cena_cinza = cv2.cvtColor(img_cena, cv2.COLOR_BGR2GRAY) if img_cena.ndim == 3 else img_cena
  tpl_cinza = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template

  th, tw = tpl_cinza.shape[:2]
  ch, cw = cena_cinza.shape[:2]

  if th >= ch or tw >= cw:
    # template maior que a imagem
    return img_cena.copy(), np.zeros_like(cena_cinza), []

  # 1. correlação espacial 2D (cv2.matchTemplate)
  mapa_res = cv2.matchTemplate(cena_cinza, tpl_cinza, flag_metodo)

  # 2. busca dos extremos globais (cv2.minMaxLoc)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mapa_res)

  if flag_metodo == cv2.TM_SQDIFF_NORMED:
    melhor_loc = min_loc
    melhor_score = 1.0 - min_val  # inverte para que 1.0 seja pontuação máxima
    pontuacao_exibicao = min_val
  else:
    melhor_loc = max_loc
    melhor_score = max_val
    pontuacao_exibicao = max_val

  # 3. normalização do mapa de calor para renderização colorida em JET
  mapa_norm = cv2.normalize(mapa_res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  mapa_calor = cv2.applyColorMap(mapa_norm, cv2.COLORMAP_JET)

  # redimensiona mapa de calor para bater exatamente com a cena
  mapa_calor_cena = cv2.resize(mapa_calor, (img_cena.shape[1], img_cena.shape[0]), interpolation=cv2.INTER_LINEAR)

  cena_anotada = img_cena.copy()
  deteccoes = []

  # 4. identifica ocorrências que ultrapassam o limiar (Non-Maximum Suppression simplificado)
  if flag_metodo == cv2.TM_SQDIFF_NORMED:
    locs = np.where(mapa_res <= (1.0 - limiar_deteccao))
  else:
    locs = np.where(mapa_res >= limiar_deteccao)

  caixas_brutas = []
  scores_brutos = []
  for pt in zip(*locs[::-1]):
    score = (1.0 - mapa_res[pt[1], pt[0]]) if flag_metodo == cv2.TM_SQDIFF_NORMED else mapa_res[pt[1], pt[0]]
    caixas_brutas.append([pt[0], pt[1], tw, th])
    scores_brutos.append(float(score))

  indices_nms = cv2.dnn.NMSBoxes(caixas_brutas, scores_brutos, score_threshold=limiar_deteccao, nms_threshold=0.3)

  if len(indices_nms) > 0:
    for idx in indices_nms.flatten():
      bx, by, bw, bh = caixas_brutas[idx]
      sc = scores_brutos[idx]
      deteccoes.append({"box": (bx, by, bw, bh), "score": sc})
      # desenha bounding box verde
      cv2.rectangle(cena_anotada, (bx, by), (bx + bw, by + bh), (0, 230, 90), 2, lineType=cv2.LINE_AA)
      cv2.circle(cena_anotada, (bx + bw // 2, by + bh // 2), 4, (0, 0, 255), -1)
      cv2.putText(
        cena_anotada,
        f"Match: {sc:.2f}",
        (bx, max(by - 6, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 230, 90),
        1,
        lineType=cv2.LINE_AA
      )
  else:
    # se nenhuma passou o limiar estrito, destaca ao menos o melhor extremo com aviso
    bx, by = melhor_loc
    cv2.rectangle(cena_anotada, (bx, by), (bx + tw, by + th), (0, 140, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(
      cena_anotada,
      f"Melhor Pico: {melhor_score:.2f} (< Limiar)",
      (bx, max(by - 6, 15)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.45,
      (0, 140, 255),
      1,
      lineType=cv2.LINE_AA
    )

  return cena_anotada, mapa_calor_cena, deteccoes


def main():
  parser = criar_parser_midia(
    descricao="Módulo 2: Detecção de Objetos com Template Matching",
    nome="imagem_teste_aula2.png"
  )
  parser.add_argument(
    "-t", "--template",
    type=str,
    default="template_porca.png",
    help="Nome do arquivo do template de referência (padrão: template_porca.png)"
  )
  parser.add_argument(
    "--metodo",
    choices=["ccoeff", "sqdiff", "ccorr"],
    default="ccoeff",
    help="Método de correlação do OpenCV (padrão: ccoeff)"
  )
  parser.add_argument(
    "--limiar",
    type=float,
    default=0.75,
    help="Limiar de confiança para detecção múltipla (padrão: 0.75)"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  print("=" * 65)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 2)")
  print("DETECÇÃO DE OBJETOS: TEMPLATE MATCHING & MAPA DE SIMILARIDADE")
  print("=" * 65)

  # carrega imagem de entrada e template
  caminho_tpl = resolver_caminho(args.template, nome="template_porca.png")
  template_base = carregar_imagem(caminho_tpl)

  lista_templates = ["template_porca.png", "template_engrenagem.png", "template_chip.png"]
  idx_tpl_atual = 0

  chaves_metodo = ["ccoeff", "sqdiff", "ccorr"]
  idx_metodo = chaves_metodo.index(args.metodo)

  angulo_atual = 0.0
  escala_atual = 1.0
  limiar_atual = args.limiar

  nome_janela = "Modulo 2 - Template Matching (Capivoa)"

  for frame, is_stream in stream_midia(args.midia, nome="imagem_teste_aula2.png"):
    tpl_corrente = transformar_template(template_base, angulo_atual, escala_atual)
    chave_m = chaves_metodo[idx_metodo]

    anotada, mapa_calor, detec = executar_matching(
      img_cena=frame,
      template=tpl_corrente,
      chave_metodo=chave_m,
      limiar_deteccao=limiar_atual
    )

    if args.no_show:
      print(f"[No-Show] Template: {Path(caminho_tpl).name} (Dim: {template_base.shape[:2]})")
      print(f"[No-Show] Método: {METODOS_OPENCV[chave_m][1]}")
      print(f"[No-Show] Detecções validadas: {len(detec)}")
      for d in detec:
        print(f"  - Box={d['box']}, Score={d['score']:.3f}")
      print("[No-Show] Validação de Template Matching concluída com sucesso.")
      break

    # desenha moldura do template ativo no canto do mapa de calor
    th, tw = tpl_corrente.shape[:2]
    cv2.putText(
      anotada,
      f"Template: {Path(caminho_tpl).stem} | Metodo: {chave_m.upper()} | Rot:{int(angulo_atual)}deg Esc:{escala_atual:.2f}",
      (15, 25),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.50,
      (0, 255, 255),
      1,
      lineType=cv2.LINE_AA
    )

    painel = concatenar_lado_a_lado(
      [anotada, mapa_calor],
      titulos=[
        f"1. Deteccoes Validadas (N={len(detec)})",
        "2. Mapa de Calor de Semelhanca (cv2.matchTemplate)"
      ],
      altura_padrao=440
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

      if tecla in [ord("m"), ord("M")]:
        idx_metodo = (idx_metodo + 1) % len(chaves_metodo)
        print(f"[Método] Alternado para: {chaves_metodo[idx_metodo].upper()}")
        atualizar = True
      elif tecla in [ord("t"), ord("T")]:
        idx_tpl_atual = (idx_tpl_atual + 1) % len(lista_templates)
        caminho_tpl = resolver_caminho(lista_templates[idx_tpl_atual])
        template_base = carregar_imagem(caminho_tpl)
        print(f"[Template] Alternado para: {lista_templates[idx_tpl_atual]}")
        atualizar = True
      elif tecla in [ord("r"), ord("R")]:
        angulo_atual = (angulo_atual + 15) % 360
        print(f"[Perturbação] Rotação do Template = {angulo_atual}° (Observe a queda do score!)")
        atualizar = True
      elif tecla in [ord("w"), ord("W")]:
        escala_atual = round(min(2.0, escala_atual + 0.1), 2)
        print(f"[Perturbação] Escala do Template = {escala_atual}x")
        atualizar = True
      elif tecla in [ord("s"), ord("S")] and not (tecla == ord("s") and False):
        escala_atual = round(max(0.5, escala_atual - 0.1), 2)
        print(f"[Perturbação] Escala do Template = {escala_atual}x")
        atualizar = True
      elif tecla == ord("0"):
        angulo_atual = 0.0
        escala_atual = 1.0
        print("[Reset] Template restaurado para rotação=0° e escala=1.0x.")
        atualizar = True
      elif tecla in [ord("p"), ord("P")]:
        pasta_saida = Path("frames-capturados")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_salvo = pasta_saida / "template_matching.png"
        cv2.imwrite(str(caminho_salvo), painel)
        print(f"[Salvo] Imagem salva em: '{caminho_salvo}'")

      if atualizar and not is_stream:
        tpl_corrente = transformar_template(template_base, angulo_atual, escala_atual)
        chave_m = chaves_metodo[idx_metodo]
        anotada, mapa_calor, detec = executar_matching(
          img_cena=frame,
          template=tpl_corrente,
          chave_metodo=chave_m,
          limiar_deteccao=limiar_atual
        )
        painel = concatenar_lado_a_lado(
          [anotada, mapa_calor],
          titulos=[
            f"1. Deteccoes Validadas (N={len(detec)})",
            "2. Mapa de Calor de Semelhanca (cv2.matchTemplate)"
          ],
          altura_padrao=440
        )
        cv2.imshow(nome_janela, painel)

      if is_stream:
        break


if __name__ == "__main__":
  main()
