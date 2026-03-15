"""LLM服务模块"""

import os
from langchain_openai import ChatOpenAI
from ..config import get_settings

# 全局LLM实例
_llm_instance = None


def get_llm() -> ChatOpenAI:
    """
    获取LLM实例(单例模式)
    
    Returns:
        ChatOpenAI实例
    """
    global _llm_instance
    
    if _llm_instance is None:
        settings = get_settings()
        
        # 优先从 Pydantic settings 获取已加载的配置
        api_key = settings.llm_api_key or os.getenv("LLM_API_KEY") or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        base_url = settings.llm_base_url or os.getenv("LLM_BASE_URL") or settings.openai_base_url or os.getenv("OPENAI_BASE_URL")
        
        # 依次尝试多种模型变量名
        model = (
            settings.llm_model_id or 
            os.getenv("LLM_MODEL_ID") or 
            settings.llm_model or 
            os.getenv("LLM_MODEL") or 
            settings.openai_model or 
            os.getenv("OPENAI_MODEL") or
            "qwen-plus"
        )
        
        timeout = settings.llm_timeout or int(os.getenv("LLM_TIMEOUT", 60))

        # 记录关键配置（隐藏敏感信息）
        print(f"🔧 初始化 LLM 实例...")
        print(f"   模型: {model}")
        print(f"   服务地址: {base_url}")
        print(f"   API Key: {api_key[:8]}...{api_key[-4:] if api_key else ''}")

        _llm_instance = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.7,
            timeout=float(timeout),
            max_retries=2,
        )
        
        print(f"✅ LLM服务初始化成功 (LangChain ChatOpenAI)")
    
    return _llm_instance


def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_instance
    _llm_instance = None

