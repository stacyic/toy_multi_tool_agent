"""Custom exceptions for the multi-tool agent."""

from typing import List, Optional


class MultiToolAgentError(Exception):
    """Base exception for all multi-tool agent errors."""

    pass


# SQL Accessor Exceptions
class SQLAccessorError(MultiToolAgentError):
    """Base exception for SQL accessor errors."""

    pass


class SQLGenerationError(SQLAccessorError):
    """Error during SQL query generation."""

    pass


class SQLValidationError(SQLAccessorError):
    """Error during SQL query validation."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class SQLExecutionError(SQLAccessorError):
    """Error during SQL query execution."""

    def __init__(self, message: str, sql: Optional[str] = None):
        super().__init__(message)
        self.sql = sql


class SQLRetryExhaustedError(SQLAccessorError):
    """All SQL retry attempts exhausted."""

    def __init__(self, message: str, attempts: int = 0, last_error: Optional[str] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# Policy Accessor Exceptions
class PolicyAccessorError(MultiToolAgentError):
    """Base exception for policy accessor errors."""

    pass


class PolicyNotFoundError(PolicyAccessorError):
    """Requested policy information not found."""

    pass


class PolicyLoadError(PolicyAccessorError):
    """Error loading policy documents."""

    pass


# Agent Exceptions
class AgentError(MultiToolAgentError):
    """Base exception for agent orchestration errors."""

    pass


class RoutingError(AgentError):
    """Error during query routing."""

    pass


class ToolExecutionError(AgentError):
    """Error during tool execution."""

    def __init__(self, message: str, tool_name: Optional[str] = None):
        super().__init__(message)
        self.tool_name = tool_name


class ContextPassingError(AgentError):
    """Error passing context between tools."""

    pass


# Configuration Exceptions
class ConfigurationError(MultiToolAgentError):
    """Configuration-related error."""

    pass


class APIKeyMissingError(ConfigurationError):
    """Required API key is missing."""

    pass


# Vector Store Exceptions
class VectorStoreError(MultiToolAgentError):
    """Base exception for vector store errors."""

    pass


class EmbeddingError(VectorStoreError):
    """Error generating embeddings."""

    pass
