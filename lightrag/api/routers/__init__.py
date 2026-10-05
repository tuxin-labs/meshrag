"""
This module contains all the routers for the LightRAG API.

Routers are created per app instance through their factory functions
(``create_document_routes``, ``create_query_routes``, ``create_graph_routes``,
``OllamaAPI``); no module-level router singletons are exported so that multiple
server instances never share route tables.
"""

from .ollama_api import OllamaAPI

__all__ = ["OllamaAPI"]
