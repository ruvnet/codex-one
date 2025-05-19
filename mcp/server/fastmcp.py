from types import SimpleNamespace

class FastMCP:
    def __init__(self, name: str):
        self.name = name
        self.settings = SimpleNamespace(port=3001)
    def tool(self, name: str, description: str):
        def decorator(func):
            setattr(self, name, func)
            return func
        return decorator
    async def run_stdio_async(self):
        pass
    async def run_sse_async(self):
        pass
