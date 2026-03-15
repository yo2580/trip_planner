"""Unsplash图片服务"""

import time
import requests
from typing import Dict, List, Optional, Tuple
from ..config import get_settings


class UnsplashService:
    """Unsplash图片服务类"""
    
    def __init__(self):
        """初始化服务"""
        settings = get_settings()
        self.access_key = settings.unsplash_access_key
        self.base_url = "https://api.unsplash.com"
        self._success_cache: Dict[str, Tuple[Optional[str], float]] = {}
        self._fail_cache: Dict[str, float] = {}
        self._last_error_log_ts: float = 0.0
    
    def search_photos(self, query: str, per_page: int = 5) -> List[dict]:
        """
        搜索图片
        
        Args:
            query: 搜索关键词
            per_page: 每页数量
            
        Returns:
            图片列表
        """
        try:
            if not self.access_key:
                return []

            now = time.time()
            fail_until = self._fail_cache.get(query)
            if fail_until and fail_until > now:
                return []

            url = f"{self.base_url}/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "client_id": self.access_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # 提取图片URL
            photos = []
            for photo in results:
                photos.append({
                    "id": photo.get("id"),
                    "url": photo.get("urls", {}).get("regular"),
                    "thumb": photo.get("urls", {}).get("thumb"),
                    "description": photo.get("description") or photo.get("alt_description"),
                    "photographer": photo.get("user", {}).get("name")
                })
            
            return photos
            
        except Exception as e:
            now = time.time()
            self._fail_cache[query] = now + 300

            if now - self._last_error_log_ts > 30:
                print(f"⚠️ Unsplash请求失败（将暂时跳过一段时间）: {str(e)}")
                self._last_error_log_ts = now
            return []
    
    def get_photo_url(self, query: str) -> Optional[str]:
        """
        获取单张图片URL

        Args:
            query: 搜索关键词

        Returns:
            图片URL
        """
        if not query:
            return None

        now = time.time()
        cached = self._success_cache.get(query)
        if cached:
            value, expire_at = cached
            if expire_at > now:
                return value

        photos = self.search_photos(query, per_page=1)
        url = photos[0].get("url") if photos else None
        self._success_cache[query] = (url, now + 86400)
        return url


# 全局服务实例
_unsplash_service = None


def get_unsplash_service() -> UnsplashService:
    """获取Unsplash服务实例(单例模式)"""
    global _unsplash_service
    
    if _unsplash_service is None:
        _unsplash_service = UnsplashService()
    
    return _unsplash_service
