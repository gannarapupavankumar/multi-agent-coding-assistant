# Backend


FastAPI backend for the Multi-Agent Coding Assistant.

This backend implements:
- API routes for repo search, RAG retrieval, agent execution, workflow run, test execution, and safe apply
- LangGraph workflow orchestration
- Planner, developer, reviewer, and improver agents
- Hybrid BM25 + ChromaDB context retrieval
- Token-aware context pruning and prompt packing
- Groq LLM support with optional Ollama fallback
- Safe apply and test-feedback repair loop