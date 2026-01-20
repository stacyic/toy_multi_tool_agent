"""LLM utility functions for consistent API call handling."""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from ..logging.trace_logger import TraceLogger


async def invoke_with_logging(
    llm: ChatOpenAI,
    messages: List[Dict[str, str]],
    logger: Optional["TraceLogger"],
    component: str,
    model: str,
) -> Any:
    """
    Invoke an LLM with consistent timing and logging.

    This utility consolidates the common pattern of:
    1. Recording start time
    2. Making the API call
    3. Extracting token usage from response metadata
    4. Logging the API call with timing and token info

    Args:
        llm: The LangChain ChatOpenAI instance
        messages: List of message dicts with 'role' and 'content'
        logger: Optional TraceLogger for API call logging
        component: Component name for logging (e.g., 'routing', 'sql_generation', 'synthesis')
        model: Model name for logging

    Returns:
        The LLM response object

    Raises:
        Exception: Re-raises any exception from the LLM call after logging
    """
    start_time = time.perf_counter()

    try:
        response = await llm.ainvoke(messages)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Extract token usage if available
        input_tokens = None
        output_tokens = None
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            input_tokens = usage.get('prompt_tokens')
            output_tokens = usage.get('completion_tokens')

        if logger:
            logger.log_api_call(
                component=component,
                model=model,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )

        return response

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        if logger:
            logger.log_api_call(
                component=component,
                model=model,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            )
        raise
