"""Echo tool implementation for MCP server"""

from typing import Optional
from mcp import types

def echo(text: str, transform: Optional[str] = None) -> types.TextContent:
    """
    Echo the input text back to the caller with optional case transformation.
    
    Args:
        text: The text to echo back
        transform: Optional case transformation ('upper' or 'lower')
        
    Returns:
        TextContent: The transformed text as MCP TextContent
    """
    if transform == "upper":
        result = text.upper()
    elif transform == "lower":
        result = text.lower()
    else:
        result = text
        
    return types.TextContent(
        type="text",
        text=result,
        format="text/plain"
    ) 