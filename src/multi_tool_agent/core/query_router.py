"""Query router for determining which tools to use."""

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from langchain_openai import ChatOpenAI

from ..utils.llm_utils import invoke_with_logging
from .exceptions import RoutingError

if TYPE_CHECKING:
    from ..logging.trace_logger import TraceLogger

# Tool dependency graph - defines execution order
# If tool B depends on tool A, A runs first and can pass context to B
TOOL_DEPENDENCIES: Dict[str, List[str]] = {
    "policy_accessor": [],  # No dependencies
    "sql_accessor": ["policy_accessor"],  # Can receive context from policy
}


@dataclass
class RoutingDecision:
    """Result of query routing."""

    tools: List[str]
    reasoning: str
    requires_context_passing: bool

    @property
    def is_policy_only(self) -> bool:
        """Check if only policy tool is needed."""
        return self.tools == ["policy_accessor"]

    @property
    def is_sql_only(self) -> bool:
        """Check if only SQL tool is needed."""
        return self.tools == ["sql_accessor"]

    @property
    def needs_both(self) -> bool:
        """Check if both tools are needed."""
        return "policy_accessor" in self.tools and "sql_accessor" in self.tools


class QueryRouter:
    """
    Routes queries to appropriate tools.

    Features:
    - LLM-based intent classification
    - Determines which tools are needed
    - Supports context passing between tools
    - Fallback heuristics for common patterns
    """

    SYSTEM_PROMPT = """You are a query classifier for a business intelligence system. Your job is to determine which tools are needed to answer a user's question.

Available tools:
1. policy_accessor - Searches company policy documents for business rules, definitions, criteria, and procedures
2. sql_accessor - Queries a database with customer, product, and order information

DECISION RULES:
- Use ONLY policy_accessor for questions about:
  - Return policies, shipping policies, warranties
  - Business definitions (e.g., "What is a VIP customer?")
  - Procedures and rules
  - Contact information, hours of operation
  - Discount policies, promotions

- Use ONLY sql_accessor for questions about:
  - Specific customer data without business logic (e.g., "What is Alex's email?")
  - Product listings, prices, inventory
  - Order details, order counts
  - Simple statistics that don't need business context

- Use BOTH policy_accessor THEN sql_accessor when:
  - The query asks if someone/something QUALIFIES, is ELIGIBLE, or MEETS CRITERIA
  - The query uses business-specific terms that need definition (e.g., "VIP", "membership", "tier")
  - The query combines policy context with data needs
  - The query asks about a specific person/entity AND a business concept
  - Examples:
    - "Does Alex qualify for membership?" (needs policy for criteria + SQL for Alex's data)
    - "How many VIP customers?" (needs policy for VIP definition + SQL for count)
    - "Is order #123 eligible for return?" (needs policy for return rules + SQL for order data)
    - "Which customers qualify for free shipping?" (needs policy for criteria + SQL for customer data)

IMPORTANT: When a query mentions a specific person/entity AND asks about qualification, eligibility, membership, tier status, or any business rule - ALWAYS use BOTH tools.

OUTPUT FORMAT (JSON only):
{
    "tools": ["tool1", "tool2"],
    "reasoning": "Brief explanation",
    "requires_context_passing": true/false
}

Only output valid JSON, no other text."""

    # Keyword patterns for quick routing (fallback)
    # Note: patterns with articles/determiners use trailing space to avoid false positives
    # e.g., "what is a " won't match "what is alex"
    POLICY_KEYWORDS = {
        "policy",
        "policies",
        "return policy",
        "refund",
        "warranty",
        "shipping policy",
        "discount",
        "vip definition",
        "what is a ",  # trailing space to avoid matching names like "alex"
        "what is the ",  # trailing space for precision
        "what are the ",  # trailing space for precision
        "how do i",
        "how can i",
        "can i return",
        "can i get",
        "rules",
        "procedure",
        "contact info",
        "hours of operation",
        "business hours",
    }

    SQL_KEYWORDS = {
        "list",
        "show",
        "count",
        "how many",
        "total",
        "sum",
        "average",
        "customers",
        "products",
        "orders",
        "sales",
        "revenue",
        "inventory",
        "stock",
        "price",
        "top",
        "most",
        "least",
        "email",
        "phone",
        "address",
        "name",
        "order date",
        "amount",
    }

    # Keywords that indicate need for BOTH policy context AND data lookup
    CONTEXT_KEYWORDS = {
        "vip",
        "tier",
        "eligible",
        "eligibility",
        "qualify",
        "qualifies",
        "qualified",
        "qualification",
        "membership",
        "member",
        "returnable",
        "meets criteria",
        "criteria",
        "requirement",
        "requirements",
        "benefits",
        "status",
    }

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        use_llm: bool = True,
        logger: Optional["TraceLogger"] = None,
    ):
        """
        Initialize the query router.

        Args:
            model: OpenAI model for classification
            api_key: OpenAI API key
            temperature: LLM temperature
            use_llm: Whether to use LLM routing (False for heuristic-only)
            logger: Optional TraceLogger for API call logging
        """
        self.use_llm = use_llm
        self.model = model
        self.logger = logger
        self._llm: Optional[ChatOpenAI] = None

        if use_llm:
            kwargs = {"model": model, "temperature": temperature}
            if api_key:
                kwargs["api_key"] = api_key
            self._llm = ChatOpenAI(**kwargs)

    async def route(self, query: str) -> RoutingDecision:
        """
        Route a query to appropriate tools.

        Args:
            query: The user's natural language query

        Returns:
            RoutingDecision with selected tools and reasoning
        """
        if self.use_llm and self._llm:
            try:
                return await self._route_with_llm(query)
            except Exception as e:
                # Fall back to heuristics on LLM failure
                return self._route_with_heuristics(query)
        else:
            return self._route_with_heuristics(query)

    async def _route_with_llm(self, query: str) -> RoutingDecision:
        """Route using LLM classification."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ]

        response = await invoke_with_logging(
            llm=self._llm,
            messages=messages,
            logger=self.logger,
            component="routing",
            model=self.model,
        )
        content = response.content.strip()

        # Parse JSON response
        try:
            # Remove markdown if present
            if content.startswith("```"):
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
                if match:
                    content = match.group(1).strip()

            data = json.loads(content)
            tools = data.get("tools", [])
            reasoning = data.get("reasoning", "")
            requires_context = data.get("requires_context_passing", False)

            # Validate tools
            valid_tools = set(TOOL_DEPENDENCIES.keys())
            tools = [t for t in tools if t in valid_tools]

            if not tools:
                # Default to policy if no valid tools
                return RoutingDecision(
                    tools=["policy_accessor"],
                    reasoning="Defaulting to policy search",
                    requires_context_passing=False,
                )

            return RoutingDecision(
                tools=tools,
                reasoning=reasoning,
                requires_context_passing=requires_context,
            )

        except json.JSONDecodeError as e:
            raise RoutingError(f"Failed to parse LLM routing response: {e}")

    def _route_with_heuristics(self, query: str) -> RoutingDecision:
        """Route using keyword-based heuristics."""
        query_lower = query.lower()

        # Check for context-requiring terms (needs both tools)
        # These keywords inherently require understanding business rules + looking up data
        has_context_keywords = any(kw in query_lower for kw in self.CONTEXT_KEYWORDS)

        # Check for data query indicators
        has_sql_keywords = any(kw in query_lower for kw in self.SQL_KEYWORDS)

        # Check for policy indicators
        has_policy_keywords = any(kw in query_lower for kw in self.POLICY_KEYWORDS)

        # Check for entity references (names, specific items) that suggest data lookup
        # Simple heuristic: capitalized words that aren't at sentence start
        words = query.split()
        has_entity_reference = any(
            word[0].isupper() and i > 0 and words[i-1][-1] not in '.?!'
            for i, word in enumerate(words) if word and word[0].isalpha()
        )

        # Decision logic
        # Context keywords ALWAYS require both tools (policy understanding + data lookup)
        if has_context_keywords:
            return RoutingDecision(
                tools=["policy_accessor", "sql_accessor"],
                reasoning="Query requires business context (policy) combined with data lookup",
                requires_context_passing=True,
            )
        elif has_sql_keywords and has_policy_keywords:
            # Both indicators present
            return RoutingDecision(
                tools=["policy_accessor", "sql_accessor"],
                reasoning="Query contains both policy and data indicators",
                requires_context_passing=True,
            )
        elif has_sql_keywords and not has_policy_keywords:
            # Pure data query
            return RoutingDecision(
                tools=["sql_accessor"],
                reasoning="Query appears to be a data query",
                requires_context_passing=False,
            )
        elif has_policy_keywords and not has_sql_keywords:
            # Pure policy query
            return RoutingDecision(
                tools=["policy_accessor"],
                reasoning="Query appears to be a policy question",
                requires_context_passing=False,
            )
        else:
            # Default to policy (safer for unknown queries)
            return RoutingDecision(
                tools=["policy_accessor"],
                reasoning="Defaulting to policy search for unknown query type",
                requires_context_passing=False,
            )

    @staticmethod
    def resolve_execution_order(tools: List[str]) -> List[str]:
        """
        Resolve tool execution order based on dependency graph.

        Uses topological sort to ensure tools run in correct order.

        Args:
            tools: List of tool names to execute

        Returns:
            Ordered list of tools respecting dependencies
        """
        if len(tools) <= 1:
            return tools

        # Build dependency subset for requested tools
        tool_set = set(tools)
        ordered = []
        visited: Set[str] = set()

        def visit(tool: str) -> None:
            if tool in visited or tool not in tool_set:
                return
            visited.add(tool)

            # Visit dependencies first
            for dep in TOOL_DEPENDENCIES.get(tool, []):
                if dep in tool_set:
                    visit(dep)

            ordered.append(tool)

        for tool in tools:
            visit(tool)

        return ordered
