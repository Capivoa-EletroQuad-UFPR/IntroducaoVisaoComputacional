"""
CAPIVOA UFPR.

Módulo utilitário para entrada/saída padronizada de imagens, vídeos e webcam.
Desenvolvido como ferramenta de suporte didático para o curso de Introdução
à Visão Computacional (Capivoa - EletroQuad-UFPR).

Principais funcionalidades:
- Carregamento seguro de imagens com validação de caminhos e decodificação.
- Captura unificada de vídeo (arquivo local ou índice numérico de webcam).
- Gerador `stream_midia` polimórfico (trata imagem, vídeo e webcam no mesmo loop).
- Parser de argumentos padronizado (`--midia` / `-m` ou `--origem` / `-o`).
- Utilitário visual `concatenar_lado_a_lado` para comparar etapas antes/depois.
"""

from pathlib import Path
import argparse
from typing import Generator, Tuple, Union, Optional, List
import cv2
import numpy as np

# diretório base do repositório (raiz)
BASE_DIR = Path(__file__).resolve().parent.parent

# extensões comuns de imagem estática reconhecidas
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def resolver_caminho(caminho: Optional[Union[str, Path]], nome: Optional[str] = None) -> Path:
  """
  Resolve um caminho de arquivo garantindo que caminhos relativos sejam encontrados.

  Busca o arquivo na ordem: diretório atual de trabalho, subpasta 'aula-1',
  raiz do repositório ou pasta 'shared'.

  Parametros
  ----------
  caminho : str ou Path, opcional
    Caminho relativo ou absoluto do arquivo de mídia.
  nome : str, opcional
    Nome padrão de fallback caso 'caminho' não seja informado.

  Returns
  -------
  Path
    Instância de Path apontando para o arquivo localizado.

  Raises
  ------
  ValueError
    Se nem 'caminho' nem 'nome' forem fornecidos.
  """
  alvo = caminho if caminho is not None else nome
  if alvo is None:
    raise ValueError("Nenhum caminho ou nome padrão foi especificado.")

  caminho_path = Path(alvo)
  if caminho_path.is_absolute():
    return caminho_path

  # ordem de verificação para encontrar o arquivo
  candidatos = [
    Path.cwd() / caminho_path,
    BASE_DIR / "aula-2" / "desafio" / caminho_path,
    BASE_DIR / "aula-2" / caminho_path,
    BASE_DIR / "aula-1" / "desafio" / caminho_path,
    BASE_DIR / "aula-1" / caminho_path,
    BASE_DIR / caminho_path,
    BASE_DIR / "shared" / caminho_path,
  ]

  for candidato in candidatos:
    if candidato.exists():
      return candidato

  # se ainda não existe fisicamente (ex: arquivo novo para salvar), usa pasta de trabalho, aula-2 ou aula-1
  return Path.cwd() / caminho_path


def carregar_imagem(
  caminho: Optional[Union[str, Path]] = None,
  nome: str = "imagem_teste.png",
  to_rgb: bool = False
) -> np.ndarray:
  """
  Carrega uma imagem do disco com verificação detalhada de erros.

  Garante a existência do arquivo em disco e a decodificação válida
  por meio do OpenCV, com opção de conversão do espaço de cor BGR para RGB.

  Parametros
  ----------
  caminho : str ou Path, opcional
    Caminho relativo ou absoluto da imagem.
  nome : str, opcional
    Arquivo padrão caso caminho seja None. Padrão é "imagem_teste.png".
  to_rgb : bool, opcional
    Se True, converte a imagem de BGR para RGB. Padrão é False.

  Returns
  -------
  np.ndarray
    Matriz NumPy (uint8) contendo a imagem carregada.

  Raises
  ------
  FileNotFoundError
    Se o arquivo não existir fisicamente em nenhum dos caminhos de busca.
  ValueError
    Se o OpenCV não conseguir decodificar o arquivo de imagem.
  """
  arquivo = resolver_caminho(caminho, nome)

  if not arquivo.exists():
    raise FileNotFoundError(
      f"[ERRO media_io] Imagem não encontrada: '{arquivo}'. "
      f"Verifique se o nome do arquivo foi digitado corretamente."
    )

  img = cv2.imread(str(arquivo))
  if img is None:
    raise ValueError(
      f"[ERRO media_io] Não foi possível decodificar o arquivo como imagem: '{arquivo}'. "
      f"O formato pode ser inválido ou o arquivo estar corrompido."
    )

  if to_rgb:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img


def obter_captura_video(
  origem: Optional[Union[int, str, Path]] = None,
  nome: str = "video.mp4"
) -> cv2.VideoCapture:
  """
  Abre uma fonte de captura de vídeo (webcam pelo índice ou arquivo de vídeo).

  Inicializa e valida a abertura de um objeto cv2.VideoCapture.

  Parametros
  ----------
  origem : int, str ou Path, opcional
    Índice numérico para webcam ou caminho para arquivo de vídeo.
  nome : str, opcional
    Arquivo de vídeo padrão caso origem seja None. Padrão é "video.mp4".

  Returns
  -------
  cv2.VideoCapture
    Instância aberta de cv2.VideoCapture.

  Raises
  ------
  FileNotFoundError
    Se o arquivo de vídeo não existir no sistema de arquivos.
  RuntimeError
    Se houver falha ao inicializar o dispositivo ou ler os codecs do arquivo.
  """
  alvo = origem if origem is not None else nome

  eh_webcam = isinstance(alvo, int) or (isinstance(alvo, str) and alvo.strip().isdigit())

  if eh_webcam:
    indice = int(alvo)
    cap = cv2.VideoCapture(indice)
    descricao = f"Webcam (índice {indice})"
  else:
    arquivo = resolver_caminho(alvo, nome)
    if not arquivo.exists():
      raise FileNotFoundError(
        f"[ERRO media_io] Arquivo de vídeo não encontrado: '{arquivo}'"
      )
    cap = cv2.VideoCapture(str(arquivo))
    descricao = f"Arquivo de vídeo '{arquivo.name}'"

  if not cap.isOpened():
    raise RuntimeError(
      f"[ERRO media_io] Falha ao abrir fonte de captura: {descricao}. "
      f"Certifique-se de que a câmera está conectada ou o codec é suportado."
    )

  return cap


def stream_midia(
  origem: Optional[Union[int, str, Path]] = None,
  nome: str = "imagem_teste.png",
  to_rgb: bool = False
) -> Generator[Tuple[np.ndarray, bool], None, None]:
  """
  Gerador polimórfico unificado para processamento de imagem, vídeo ou webcam.

  Entrega quadros sucessivos para iteração homogênea em loops de processamento.
  Caso a entrada seja uma imagem estática, entrega um único frame com a flag
  is_stream=False. Caso seja vídeo ou câmera, entrega cada frame com is_stream=True.

  Parametros
  ----------
  origem : int, str ou Path, opcional
    Caminho do arquivo ou índice numérico da câmera.
  nome : str, opcional
    Arquivo padrão de fallback. Padrão é "imagem_teste.png".
  to_rgb : bool, opcional
    Se True, converte cada frame de BGR para RGB. Padrão é False.

  Yields
  ------
  frame : np.ndarray
    Matriz NumPy representando a imagem ou quadro atual.
  is_stream : bool
    Booleano indicando se o fluxo é contínuo (vídeo/webcam) ou pontual (imagem).
  """
  alvo = origem if origem is not None else nome
  eh_webcam = isinstance(alvo, int) or (isinstance(alvo, str) and alvo.strip().isdigit())

  if not eh_webcam:
    caminho_arquivo = resolver_caminho(alvo, nome)
    extensao = caminho_arquivo.suffix.lower()

    # se for imagem estática, carrega e entrega imediatamente
    if extensao in EXTENSOES_IMAGEM:
      frame = carregar_imagem(caminho_arquivo, to_rgb=to_rgb)
      yield frame, False
      return

  # caso seja vídeo ou webcam
  padrao_video = "video.mp4"
  cap = obter_captura_video(alvo, nome=nome if not eh_webcam else padrao_video)
  try:
    while True:
      ret, frame = cap.read()
      if not ret or frame is None:
        break

      if to_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      yield frame, True
  finally:
    cap.release()


def concatenar_lado_a_lado(
  imagens: List[np.ndarray],
  titulos: Optional[List[str]] = None,
  altura_padrao: int = 400
) -> np.ndarray:
  """
  Concatena horizontalmente duas ou mais imagens para visualização lado a lado.

  Padroniza dimensões e número de canais entre imagens (convertendo grayscale
  em BGR e redimensionando proporcionalmente para a mesma altura) e adiciona
  faixas superiores opcionais com títulos explicativos.

  Parametros
  ----------
  imagens : list de np.ndarray
    Lista contendo as matrizes das imagens a serem concatenadas.
  titulos : list de str, opcional
    Lista com os títulos textuais correspondentes a cada imagem.
  altura_padrao : int, opcional
    Altura vertical em pixels para redimensionar as imagens. Padrão é 400.

  Returns
  -------
  np.ndarray
    Matriz NumPy combinada contendo todas as imagens dispostas horizontalmente.

  Raises
  ------
  ValueError
    Se a lista de imagens fornecida for vazia.
  """
  if not imagens:
    raise ValueError("A lista de imagens não pode ser vazia.")

  processadas = []
  for i, img in enumerate(imagens):
    item = img.copy()

    # converte 1 canal (grayscale/máscara) para 3 canais bgr para poder concatenar
    if len(item.shape) == 2:
      item = cv2.cvtColor(item, cv2.COLOR_GRAY2BGR)

    # redimensiona proporcionalmente para a altura_padrao
    h, w = item.shape[:2]
    if h != altura_padrao:
      fator = altura_padrao / float(h)
      largura_nova = int(w * fator)
      item = cv2.resize(item, (largura_nova, altura_padrao), interpolation=cv2.INTER_LINEAR)

    # se houver título correspondente, adiciona barra superior com o texto
    if titulos and i < len(titulos) and titulos[i]:
      barra = np.zeros((32, item.shape[1], 3), dtype=np.uint8)
      cv2.putText(
        barra,
        titulos[i],
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
      )
      item = np.vstack([barra, item])

    processadas.append(item)

  # concatena horizontalmente com uma linha divisória cinza entre elas
  divisor = np.full((processadas[0].shape[0], 4, 3), 60, dtype=np.uint8)
  resultado = processadas[0]
  for p in processadas[1:]:
    resultado = np.hstack([resultado, divisor, p])

  return resultado


class MediaArgumentParser(argparse.ArgumentParser):
  """
  ArgumentParser personalizado que aceita tanto '--midia'/'-m' quanto '--origem'/'-o'.
  """

  def parse_args(self, args=None, namespace=None):
    ns = super().parse_args(args=args, namespace=namespace)
    if hasattr(ns, "origem"):
      ns.midia = ns.origem
    elif hasattr(ns, "midia"):
      ns.origem = ns.midia
    return ns


def criar_parser_midia(
  descricao: str = "Processamento de Imagem/Vídeo com OpenCV (Módulo 1)",
  nome: str = "imagem_teste.png"
) -> MediaArgumentParser:
  """
  Cria um parser padronizado de argumentos de linha de comando.

  Configura os argumentos comuns como '--midia'/'-m' e '--origem'/'-o'.

  Parametros
  ----------
  descricao : str, opcional
    Texto descritivo do parser. Padrão é "Processamento de Imagem/Vídeo com OpenCV (Módulo 1)".
  nome : str, opcional
    Arquivo padrão sugerido como entrada. Padrão é "imagem_teste.png".

  Returns
  -------
  MediaArgumentParser
    Instância configurada de MediaArgumentParser.
  """
  parser = MediaArgumentParser(
    description=descricao,
    formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "-m", "--midia", "-o", "--origem",
    dest="origem",
    type=str,
    default=nome,
    help=f"Arquivo de imagem/vídeo ou índice da webcam (ex: 0). Padrão: {nome}"
  )
  return parser
