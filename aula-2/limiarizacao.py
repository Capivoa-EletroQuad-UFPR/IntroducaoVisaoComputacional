"""
CAPIVOA UFPR.

Demonstração prática dos Métodos de Limiarização (Thresholding).
Módulo 2 de Introdução à Visão Computacional (Capivoa - EletroQuad-UFPR).

Tópicos e Fundamentos Teóricos:
1. Limiarização :
   - Conversão de matrizes monocromáticas (8 bits, 0-255) em máscaras binárias (0 ou 255).
   - Separação de planos: isola os objetos de interesse do plano de fundo.
   - Base fundamental para extração de contornos e momentos geométricos.

2. Métodos abordados :
   - Limiar Fixo (Global) :
     Aplica um único escalar T para todos os pixels: g(x, y) = 255 se f(x, y) > T, senão 0.
     Muito veloz, porém vulnerável a gradientes de luz e sombras graduais.
   - Limiarização Adaptativa :
     Calcula limiares locais por janelas de vizinhança (bloco ímpar) menos uma constante C.
     * Média Aritmética (cv2.ADAPTIVE_THRESH_MEAN_C).
     * Média Gaussiana (cv2.ADAPTIVE_THRESH_GAUSSIAN_C).
   - Método de Otsu :
     Busca exaustiva do limiar que maximiza a variância inter-classes:
       sigma_B^2(t) = w0(t) * w1(t) * [mu0(t) - mu1(t)]^2
     Totalmente autônomo para histogramas bimodais.
   - Algoritmo de Kittler-Illingworth (Minimum Error Thresholding) :
     Modela o histograma como uma mistura de duas distribuições normais (Gaussianas)
     e minimiza a função de critério de erro de Bayes.
     Superior em situações onde o objeto ocupa uma área muito menor do que o fundo.

Controles no Teclado:
  [1]     : Visualizar Limiar Fixo
  [2]     : Visualizar Adaptativo (Média Aritmética)
  [3]     : Visualizar Adaptativo (Média Gaussiana)
  [4]     : Visualizar Método de Otsu
  [5]     : Visualizar Kittler-Illingworth
  [TAB]   : Alternar entre Modo Único e Painel Comparativo Geral
  [+ / -] : Ajustar o valor do limiar fixo T
  [I]     : Inverter polaridade (Normal <-> Invertido)
  [S]     : Salvar imagem binária em 'frames-capturados/'
  [Q/ESC] : Encerrar

Como executar:
  python limiarizacao.py
  python limiarizacao.py --midia imagem_teste_aula2.png
  python limiarizacao.py --metodo otsu
  python limiarizacao.py --no-show
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import carregar_imagem, criar_parser_midia, concatenar_lado_a_lado, stream_midia


def calcular_limiar_kittler_illingworth(img_cinza: np.ndarray) -> int:
  """
  Calcula o limiar ótimo segundo o critério de Kittler-Illingworth (1986).

  Modela as intensidades do objeto e do fundo como uma mistura de duas
  distribuições Gaussianas e busca o limiar T que minimiza a probabilidade
  de erro de classificação bayesiana.

  Parametros
  ----------
  img_cinza : np.ndarray
    Matriz 2D da imagem em escala de cinza (uint8).

  Returns
  -------
  int
    Valor inteiro do limiar ótimo T na faixa [1, 254].
  """
  # histograma normalizado (distribuição de probabilidade)
  hist = cv2.calcHist([img_cinza], [0], None, [256], [0, 256]).flatten()
  hist = hist / (hist.sum() + 1e-12)

  # probabilidades acumuladas (w0 e w1)
  omega = np.cumsum(hist)

  # médias acumuladas
  indices = np.arange(256)
  mu = np.cumsum(indices * hist)
  mu_total = mu[-1]

  melhor_custo = float("inf")
  melhor_t = 128

  # varre todos os limiares candidatos com suporte numérico
  for t in range(1, 255):
    w0 = omega[t]
    w1 = 1.0 - w0

    if w0 <= 1e-6 or w1 <= 1e-6:
      continue

    mu0 = mu[t] / w0
    mu1 = (mu_total - mu[t]) / w1

    # cálculo das variâncias das duas classes
    var0 = np.sum(((indices[:t + 1] - mu0) ** 2) * hist[:t + 1]) / w0
    var1 = np.sum(((indices[t + 1:] - mu1) ** 2) * hist[t + 1:]) / w1

    # evita variância zero para estabilidade logarítmica
    var0 = max(var0, 1e-6)
    var1 = max(var1, 1e-6)

    # função de custo J(t) de Kittler-Illingworth
    custo_j = (
      1.0
      + 2.0 * (w0 * np.log(np.sqrt(var0)) + w1 * np.log(np.sqrt(var1)))
      - 2.0 * (w0 * np.log(w0) + w1 * np.log(w1))
    )

    if custo_j < melhor_custo:
      melhor_custo = custo_j
      melhor_t = t

  return int(melhor_t)


def processar_limiarizacao(
  img_cinza: np.ndarray,
  limiar_fixo: int = 127,
  tamanho_bloco: int = 31,
  constante_c: int = 10,
  invertido: bool = True
) -> Dict[str, Tuple[np.ndarray, str]]:
  """
  Aplica os métodos de limiarização apresentados nos slides e retorna os resultados.

  Parametros
  ----------
  img_cinza : np.ndarray
    Imagem monocromática de entrada.
  limiar_fixo : int
    Valor escalar T para limiar fixo global. Padrão é 127.
  tamanho_bloco : int
    Tamanho da vizinhança ímpar para métodos adaptativos. Padrão é 31.
  constante_c : int
    Constante subtraída nos métodos adaptativos. Padrão é 10.
  invertido : bool
    Se True, objetos escuros tornam-se brancos (255) e fundo torna-se preto (0).

  Returns
  -------
  Dict[str, Tuple[np.ndarray, str]]
    Dicionário mapeando a chave do método para a tupla (matriz_binaria, descricao_detalhada).
  """
  if tamanho_bloco % 2 == 0:
    tamanho_bloco += 1

  tipo_thresh = cv2.THRESH_BINARY_INV if invertido else cv2.THRESH_BINARY

  # 1. Limiar Fixo
  _, bin_fixo = cv2.threshold(img_cinza, limiar_fixo, 255, tipo_thresh)
  desc_fixo = f"Fixo (T={limiar_fixo})"

  # 2. Adaptativo por Média Aritmética
  adapt_tipo = cv2.THRESH_BINARY_INV if invertido else cv2.THRESH_BINARY
  bin_adapt_media = cv2.adaptiveThreshold(
    img_cinza, 255, cv2.ADAPTIVE_THRESH_MEAN_C, adapt_tipo, tamanho_bloco, constante_c
  )
  desc_adapt_media = f"Adapt. Media (B={tamanho_bloco}, C={constante_c})"

  # 3. Adaptativo por Média Gaussiana
  bin_adapt_gauss = cv2.adaptiveThreshold(
    img_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, adapt_tipo, tamanho_bloco, constante_c
  )
  desc_adapt_gauss = f"Adapt. Gauss (B={tamanho_bloco}, C={constante_c})"

  # 4. Método de Otsu
  flag_otsu = tipo_thresh | cv2.THRESH_OTSU
  val_otsu, bin_otsu = cv2.threshold(img_cinza, 0, 255, flag_otsu)
  desc_otsu = f"Otsu (T_auto={int(val_otsu)})"

  # 5. Kittler-Illingworth
  t_kittler = calcular_limiar_kittler_illingworth(img_cinza)
  _, bin_kittler = cv2.threshold(img_cinza, t_kittler, 255, tipo_thresh)
  desc_kittler = f"Kittler-Illingworth (T_opt={t_kittler})"

  return {
    "fixo": (bin_fixo, desc_fixo),
    "adapt_media": (bin_adapt_media, desc_adapt_media),
    "adapt_gauss": (bin_adapt_gauss, desc_adapt_gauss),
    "otsu": (bin_otsu, desc_otsu),
    "kittler": (bin_kittler, desc_kittler),
  }


def montar_painel_comparativo(img_cinza: np.ndarray, resultados: dict, altura_padrao: int = 280) -> np.ndarray:
  """
  Monta uma matriz visual em grade comparando todos os métodos com a imagem original.
  """
  lista_imagens = [
    img_cinza,
    resultados["fixo"][0],
    resultados["adapt_media"][0],
    resultados["adapt_gauss"][0],
    resultados["otsu"][0],
    resultados["kittler"][0]
  ]
  titulos = [
    "Original (Grayscale)",
    f"1. {resultados['fixo'][1]}",
    f"2. {resultados['adapt_media'][1]}",
    f"3. {resultados['adapt_gauss'][1]}",
    f"4. {resultados['otsu'][1]}",
    f"5. {resultados['kittler'][1]}"
  ]

  linha_superior = concatenar_lado_a_lado(lista_imagens[:3], titulos[:3], altura_padrao=altura_padrao)
  linha_inferior = concatenar_lado_a_lado(lista_imagens[3:], titulos[3:], altura_padrao=altura_padrao)

  largura_max = max(linha_superior.shape[1], linha_inferior.shape[1])
  if linha_superior.shape[1] < largura_max:
    linha_superior = cv2.copyMakeBorder(
      linha_superior, 0, 0, 0, largura_max - linha_superior.shape[1], cv2.BORDER_CONSTANT, value=[30, 30, 30]
    )
  if linha_inferior.shape[1] < largura_max:
    linha_inferior = cv2.copyMakeBorder(
      linha_inferior, 0, 0, 0, largura_max - linha_inferior.shape[1], cv2.BORDER_CONSTANT, value=[30, 30, 30]
    )

  return np.vstack([linha_superior, linha_inferior])


def main():
  parser = criar_parser_midia(
    descricao="Módulo 2: Métodos de Limiarização (Fixo, Adaptativo, Otsu e Kittler-Illingworth)",
    nome="imagem_teste_aula2.png"
  )
  parser.add_argument(
    "--metodo",
    choices=["fixo", "adapt_media", "adapt_gauss", "otsu", "kittler", "painel"],
    default="painel",
    help="Método inicial exibido (padrão: painel com todos lado a lado)"
  )
  parser.add_argument(
    "--limiar",
    type=int,
    default=120,
    help="Limiar escalar inicial para o método fixo (padrão: 120)"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  print("=" * 65)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 2)")
  print("MÉTODOS DE LIMIARIZAÇÃO E SEGMENTAÇÃO DE PLANOS")
  print("=" * 65)

  limiar_atual = args.limiar
  metodo_ativo = args.metodo
  modo_painel = (args.metodo == "painel")
  invertido = True  # objetos como foreground branco

  nome_janela = "Modulo 2 - Limiarizacao e Segmentacao (Capivoa)"

  for frame, is_stream in stream_midia(args.midia, nome="imagem_teste_aula2.png"):
    img_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    resultados = processar_limiarizacao(
      img_cinza=img_cinza,
      limiar_fixo=limiar_atual,
      tamanho_bloco=31,
      constante_c=10,
      invertido=invertido
    )

    if args.no_show:
      print(f"[No-Show] Limiar Fixo: T={limiar_atual}")
      print(f"[No-Show] Otsu calculado: {resultados['otsu'][1]}")
      print(f"[No-Show] Kittler-Illingworth calculado: {resultados['kittler'][1]}")
      print("[No-Show] Validação concluída com sucesso.")
      break

    # renderização visual
    if modo_painel:
      exibicao = montar_painel_comparativo(img_cinza, resultados, altura_padrao=250)
    else:
      matriz_bin, rotulo = resultados[metodo_ativo]
      exibicao = concatenar_lado_a_lado(
        [img_cinza, matriz_bin],
        titulos=["Entrada (Grayscale)", f"Segmentada: {rotulo}"],
        altura_padrao=400
      )

    cv2.imshow(nome_janela, exibicao)

    # loop de interação se for imagem estática
    while True:
      tecla = cv2.waitKey(30) & 0xFF

      if tecla in [ord("q"), ord("Q"), 27]:
        cv2.destroyAllWindows()
        return

      if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
        return

      atualizar = False

      if tecla == ord("1"):
        metodo_ativo = "fixo"
        modo_painel = False
        atualizar = True
      elif tecla == ord("2"):
        metodo_ativo = "adapt_media"
        modo_painel = False
        atualizar = True
      elif tecla == ord("3"):
        metodo_ativo = "adapt_gauss"
        modo_painel = False
        atualizar = True
      elif tecla == ord("4"):
        metodo_ativo = "otsu"
        modo_painel = False
        atualizar = True
      elif tecla == ord("5"):
        metodo_ativo = "kittler"
        modo_painel = False
        atualizar = True
      elif tecla == 9:  # TAB
        modo_painel = not modo_painel
        atualizar = True
      elif tecla in [ord("+"), ord("=")]:
        limiar_atual = min(254, limiar_atual + 5)
        print(f"[Ajuste] Limiar fixo T = {limiar_atual}")
        atualizar = True
      elif tecla in [ord("-"), ord("_")]:
        limiar_atual = max(1, limiar_atual - 5)
        print(f"[Ajuste] Limiar fixo T = {limiar_atual}")
        atualizar = True
      elif tecla in [ord("i"), ord("I")]:
        invertido = not invertido
        print(f"[Polaridade] Inversão = {invertido}")
        atualizar = True
      elif tecla in [ord("s"), ord("S")]:
        pasta_saida = Path("frames-capturados")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_salvo = pasta_saida / f"limiarizacao_{metodo_ativo}.png"
        cv2.imwrite(str(caminho_salvo), exibicao)
        print(f"[Salvo] Imagem salva em: '{caminho_salvo}'")

      if atualizar and not is_stream:
        resultados = processar_limiarizacao(
          img_cinza=img_cinza,
          limiar_fixo=limiar_atual,
          tamanho_bloco=31,
          constante_c=10,
          invertido=invertido
        )
        if modo_painel:
          exibicao = montar_painel_comparativo(img_cinza, resultados, altura_padrao=250)
        else:
          matriz_bin, rotulo = resultados[metodo_ativo]
          exibicao = concatenar_lado_a_lado(
            [img_cinza, matriz_bin],
            titulos=["Entrada (Grayscale)", f"Segmentada: {rotulo}"],
            altura_padrao=400
          )
        cv2.imshow(nome_janela, exibicao)

      if is_stream:
        break


if __name__ == "__main__":
  main()
