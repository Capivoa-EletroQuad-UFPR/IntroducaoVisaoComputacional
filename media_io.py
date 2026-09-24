"""
Módulo para padronização de entrada de imagens e vídeos.
"""

from pathlib import Path
import argparse
from typing import Generator, Tuple, Union, Optional
import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def resolver_caminho(caminho: Optional[Union[str, Path]], padrao: Optional[str] = None) -> Path:
    """
    Resolve um caminho de arquivo em relação ao diretório base (BASE_DIR).
    Se o caminho for None, utiliza o valor padrão especificado.
    """
    alvo = caminho if caminho is not None else padrao
    if alvo is None:
        raise ValueError("Nenhum caminho ou padrão foi especificado.")

    caminho_path = Path(alvo)
    if not caminho_path.is_absolute():
        caminho_path = BASE_DIR / caminho_path

    return caminho_path


def carregar_imagem(
    caminho: Optional[Union[str, Path]] = None,
    nome: str = "imagem_teste.png",
    to_rgb: bool = False
) -> np.ndarray:
    """
    Carrega uma imagem

    Args:
        caminho: Caminho relativo ou absoluto da imagem (opcional).
        nome: Nome do arquivo padrão no diretório base caso caminho seja None.
        to_rgb: Se True, converte de BGR (OpenCV) para RGB (Matplotlib).

    Returns:
        Matriz numpy contendo a imagem.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o OpenCV não conseguir decodificar a imagem.
    """
    arquivo = resolver_caminho(caminho, nome)

    if not arquivo.exists():
        raise FileNotFoundError(f"Imagem não encontrada: '{arquivo}'")

    img = cv2.imread(str(arquivo))
    if img is None:
        raise ValueError(f"Falha ao decodificar a imagem (arquivo corrompido ou formato inválido): '{arquivo}'")

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def obter_captura_video(
    origem: Optional[Union[int, str, Path]] = None,
    nome: str = "video.mp4"
) -> cv2.VideoCapture:
    """
    Abre uma fonte de captura de vídeo (webcam ou arquivo de vídeo).

    Args:
        origem: Índice numérico da webcam (int ou str ex: '0') ou caminho do vídeo.
        nome: Arquivo de vídeo padrão caso origem seja None.

    Returns:
        Instância aberta de cv2.VideoCapture.

    Raises:
        FileNotFoundError: Se for arquivo de vídeo e não existir.
        RuntimeError: Se a captura não puder ser aberta.
    """
    alvo = origem if origem is not None else nome

    if isinstance(alvo, int) or (isinstance(alvo, str) and alvo.strip().isdigit()):
        indice_camera = int(alvo)
        cap = cv2.VideoCapture(indice_camera)
        origem_desc = f"Webcam (índice {indice_camera})"
    else:
        arquivo = resolver_caminho(alvo, nome)
        if not arquivo.exists():
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: '{arquivo}'")
        cap = cv2.VideoCapture(str(arquivo))
        origem_desc = f"Arquivo de vídeo '{arquivo}'"

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a captura de vídeo: {origem_desc}")

    return cap


def stream_midia(
    origem: Optional[Union[int, str, Path]] = None,
    nome: str = "imagem_teste.png",
    to_rgb: bool = False
) -> Generator[Tuple[np.ndarray, bool], None, None]:
    """
    Leitor genérico para arquivos de:
    - Imagem: entrega o frame único e encerra (`is_stream = False`).
    - Vídeo ou Webcam: entrega frame a frame sucessivamente (`is_stream = True`).

    Args:
        origem: Caminho da imagem/vídeo ou índice numérico da câmera.
        nome: Recurso padrão caso origem seja None.
        to_rgb: Se True, converte os frames para RGB.

    Yields:
        Tupla `(frame, is_stream)` onde `is_stream` indica se a entrada possui continuação.
    """
    alvo = origem if origem is not None else nome

    # verifica se eh webcam pelo indice
    eh_webcam = isinstance(alvo, int) or (isinstance(alvo, str) and alvo.strip().isdigit())

    if not eh_webcam:
        caminho_arquivo = resolver_caminho(alvo, nome)
        extensao = caminho_arquivo.suffix.lower()

        # caso seja uma imagem estatica
        if extensao in EXTENSOES_IMAGEM:
            frame = carregar_imagem(caminho_arquivo, to_rgb=to_rgb)
            yield frame, False
            return

    # caso seja video ou webcam
    cap = obter_captura_video(alvo, nome=nome if not eh_webcam else "video.mp4")
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


def criar_parser_midia(
    descricao: str = "Processamento de Imagem/Vídeo com OpenCV",
    nome: str = "imagem_teste.png"
) -> argparse.ArgumentParser:
    """
    Lê o parâmetro `--origem` / `-o` e retorna os argumentos.
    """
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument(
        "-o", "--origem",
        type=str,
        default=nome,
        help=f"Caminho para imagem/vídeo ou índice de webcam (padrão: {nome})"
    )
    return parser
