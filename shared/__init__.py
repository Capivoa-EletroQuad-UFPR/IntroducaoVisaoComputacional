"""
CAPIVOA UFPR.

Pacote de utilitários compartilhados para o curso de Introdução à Visão Computacional.
"""

from .media_io import (
  resolver_caminho,
  carregar_imagem,
  obter_captura_video,
  stream_midia,
  concatenar_lado_a_lado,
  criar_parser_midia,
  MediaArgumentParser,
)

__all__ = [
  "resolver_caminho",
  "carregar_imagem",
  "obter_captura_video",
  "stream_midia",
  "concatenar_lado_a_lado",
  "criar_parser_midia",
  "MediaArgumentParser",
]
