# tools/external_search_tool.py

from typing import List, Tuple
from ddgs import DDGS
from utils.logger import get_logger

logger = get_logger(__name__, "external_search.log")

def search_web(query: str, max_results: int = 5) -> List[Tuple[str, str]]:
    """
    Cari informasi dari internet menggunakan DuckDuckGo (bebas API Key).

    Args:
        query (str): Pertanyaan pengguna.
        max_results (int): Jumlah hasil yang diambil.

    Returns:
        List[Tuple[str, str]]: List hasil berupa (judul, deskripsi/ringkasan).
    """
    results = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                title = result.get("title", "").strip()
                snippet = result.get("body", "").strip()

                if title or snippet:
                    results.append((title, snippet))

        if results:
            logger.info(f"[EXTERNAL SEARCH] {len(results)} hasil ditemukan.")
        else:
            logger.warning(f"[EXTERNAL SEARCH] Tidak ditemukan hasil untuk: {query}")

    except Exception as e:
        logger.error(f"[EXTERNAL SEARCH] Gagal melakukan pencarian web: {e}")

    return results
