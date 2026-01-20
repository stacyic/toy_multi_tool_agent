"""Unit tests for QueryRouter and routing logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from multi_tool_agent.core.query_router import (
    QueryRouter,
    RoutingDecision,
    TOOL_DEPENDENCIES,
)


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_is_policy_only(self):
        """Test is_policy_only property."""
        decision = RoutingDecision(
            tools=["policy_accessor"],
            reasoning="Policy query",
            requires_context_passing=False,
        )
        assert decision.is_policy_only is True
        assert decision.is_sql_only is False
        assert decision.needs_both is False

    def test_is_sql_only(self):
        """Test is_sql_only property."""
        decision = RoutingDecision(
            tools=["sql_accessor"],
            reasoning="Data query",
            requires_context_passing=False,
        )
        assert decision.is_sql_only is True
        assert decision.is_policy_only is False
        assert decision.needs_both is False

    def test_needs_both(self):
        """Test needs_both property."""
        decision = RoutingDecision(
            tools=["policy_accessor", "sql_accessor"],
            reasoning="Combined query",
            requires_context_passing=True,
        )
        assert decision.needs_both is True
        assert decision.is_policy_only is False
        assert decision.is_sql_only is False


class TestQueryRouterHeuristics:
    """Tests for QueryRouter heuristic-based routing."""

    @pytest.fixture
    def router(self):
        """Create QueryRouter with heuristic-only routing."""
        return QueryRouter(use_llm=False)

    # Policy-only queries
    @pytest.mark.asyncio
    async def test_route_return_policy_query(self, router):
        """Test routing of return policy query."""
        decision = await router.route("What is the return policy?")
        assert decision.is_policy_only
        assert "policy_accessor" in decision.tools

    @pytest.mark.asyncio
    async def test_route_refund_query(self, router):
        """Test routing of refund query."""
        decision = await router.route("How do I get a refund?")
        assert decision.is_policy_only

    @pytest.mark.asyncio
    async def test_route_warranty_query(self, router):
        """Test routing of warranty query."""
        decision = await router.route("What is the warranty period?")
        assert decision.is_policy_only

    @pytest.mark.asyncio
    async def test_route_vip_definition_query(self, router):
        """Test routing 'what is a VIP' uses both tools (VIP is context keyword)."""
        decision = await router.route("What is a VIP customer?")
        # VIP is a context keyword, so it needs both tools
        assert decision.needs_both

    # SQL-only queries
    @pytest.mark.asyncio
    async def test_route_list_products_query(self, router):
        """Test routing of product listing query."""
        decision = await router.route("List all products under $100")
        assert decision.is_sql_only
        assert "sql_accessor" in decision.tools

    @pytest.mark.asyncio
    async def test_route_count_orders_query(self, router):
        """Test routing of order count query."""
        decision = await router.route("How many orders were placed?")
        assert decision.is_sql_only

    @pytest.mark.asyncio
    async def test_route_email_lookup_query(self, router):
        """Test routing of email lookup query - should be SQL only."""
        decision = await router.route("What is Alex email?")
        assert decision.is_sql_only
        assert "sql_accessor" in decision.tools

    @pytest.mark.asyncio
    async def test_route_email_lookup_with_possessive(self, router):
        """Test routing of email lookup with possessive."""
        decision = await router.route("What is Alex's email?")
        assert decision.is_sql_only

    @pytest.mark.asyncio
    async def test_route_phone_lookup_query(self, router):
        """Test routing of phone lookup query - should be SQL only."""
        decision = await router.route("What is John's phone number?")
        assert decision.is_sql_only

    @pytest.mark.asyncio
    async def test_route_address_lookup_query(self, router):
        """Test routing of address lookup query - should be SQL only."""
        decision = await router.route("Show me the address for customer 123")
        assert decision.is_sql_only

    @pytest.mark.asyncio
    async def test_route_customer_name_query(self, router):
        """Test routing of customer name query - should be SQL only."""
        decision = await router.route("Show me customer names")
        assert decision.is_sql_only

    @pytest.mark.asyncio
    async def test_route_top_products_query(self, router):
        """Test routing of top products query."""
        decision = await router.route("Show me top 5 selling products")
        assert decision.is_sql_only

    # Context-requiring queries (both tools needed)
    @pytest.mark.asyncio
    async def test_route_vip_count_query(self, router):
        """Test routing of VIP count query - needs both tools."""
        decision = await router.route("How many VIP customers do we have?")
        assert decision.needs_both
        assert "policy_accessor" in decision.tools
        assert "sql_accessor" in decision.tools
        assert decision.requires_context_passing is True

    @pytest.mark.asyncio
    async def test_route_qualification_query(self, router):
        """Test routing of qualification query - needs both tools."""
        decision = await router.route("Does Alex qualify for membership?")
        assert decision.needs_both
        assert decision.requires_context_passing is True

    @pytest.mark.asyncio
    async def test_route_eligibility_query(self, router):
        """Test routing of eligibility query - needs both tools."""
        decision = await router.route("Is order #123 eligible for return?")
        assert decision.needs_both

    @pytest.mark.asyncio
    async def test_route_tier_query(self, router):
        """Test routing of tier query - needs both tools."""
        decision = await router.route("What tier is customer John?")
        assert decision.needs_both

    @pytest.mark.asyncio
    async def test_route_membership_query(self, router):
        """Test routing of membership query - needs both tools."""
        decision = await router.route("Which customers have membership status?")
        assert decision.needs_both

    @pytest.mark.asyncio
    async def test_route_benefits_query(self, router):
        """Test routing of benefits query - needs both tools."""
        decision = await router.route("What benefits does customer 1 have?")
        assert decision.needs_both

    @pytest.mark.asyncio
    async def test_route_criteria_query(self, router):
        """Test routing of criteria query - needs both tools."""
        decision = await router.route("Which customers meet the criteria?")
        assert decision.needs_both

    @pytest.mark.asyncio
    async def test_route_free_shipping_eligibility(self, router):
        """Test routing of free shipping eligibility query."""
        decision = await router.route("Which customers are eligible for free shipping?")
        assert decision.needs_both

    # Edge cases - avoiding false positive matches
    @pytest.mark.asyncio
    async def test_route_name_starting_with_a_not_policy(self, router):
        """Test that 'what is alex' doesn't match 'what is a ' pattern."""
        decision = await router.route("What is Alex's order total?")
        # Should be SQL query (orders, total are SQL keywords)
        assert decision.is_sql_only

    @pytest.mark.asyncio
    async def test_route_actual_what_is_a_query(self, router):
        """Test that 'what is a product' matches policy pattern."""
        # "what is a " with trailing space should match
        decision = await router.route("What is a returnable item?")
        # 'returnable' is a context keyword, so should use both
        assert decision.needs_both

    # Default fallback
    @pytest.mark.asyncio
    async def test_route_unknown_query_defaults_to_policy(self, router):
        """Test that unknown queries default to policy."""
        decision = await router.route("Tell me something interesting")
        assert decision.is_policy_only
        assert "Defaulting" in decision.reasoning


class TestQueryRouterLLM:
    """Tests for QueryRouter LLM-based routing."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock()
        return mock

    @pytest.fixture
    def router_llm(self, mock_llm):
        """Create QueryRouter with mocked LLM."""
        with patch("multi_tool_agent.core.query_router.ChatOpenAI") as mock_class:
            mock_class.return_value = mock_llm
            router = QueryRouter(model="gpt-4", api_key="test-key", use_llm=True)
            router._llm = mock_llm
            return router

    @pytest.mark.asyncio
    async def test_route_with_llm_sql_response(self, router_llm, mock_llm):
        """Test routing with LLM returning SQL-only decision."""
        mock_llm.ainvoke.return_value = MagicMock(content='''
        {
            "tools": ["sql_accessor"],
            "reasoning": "Simple data query",
            "requires_context_passing": false
        }
        ''')

        decision = await router_llm.route("Count all orders")
        assert "sql_accessor" in decision.tools

    @pytest.mark.asyncio
    async def test_route_with_llm_both_tools(self, router_llm, mock_llm):
        """Test routing with LLM returning both tools."""
        mock_llm.ainvoke.return_value = MagicMock(content='''
        {
            "tools": ["policy_accessor", "sql_accessor"],
            "reasoning": "Needs policy context for VIP definition",
            "requires_context_passing": true
        }
        ''')

        decision = await router_llm.route("How many VIP customers?")
        assert decision.needs_both
        assert decision.requires_context_passing is True

    @pytest.mark.asyncio
    async def test_route_with_llm_markdown_response(self, router_llm, mock_llm):
        """Test routing handles markdown-wrapped JSON."""
        mock_llm.ainvoke.return_value = MagicMock(content='''```json
        {
            "tools": ["policy_accessor"],
            "reasoning": "Policy question",
            "requires_context_passing": false
        }
        ```''')

        decision = await router_llm.route("What is the return policy?")
        assert decision.is_policy_only

    @pytest.mark.asyncio
    async def test_route_llm_fallback_on_error(self, router_llm, mock_llm):
        """Test that router falls back to heuristics on LLM error."""
        mock_llm.ainvoke.side_effect = Exception("API Error")

        decision = await router_llm.route("What is the return policy?")

        # Should fall back to heuristics and still work
        assert decision is not None
        assert len(decision.tools) > 0

    @pytest.mark.asyncio
    async def test_route_llm_invalid_json_fallback(self, router_llm, mock_llm):
        """Test fallback when LLM returns invalid JSON."""
        mock_llm.ainvoke.return_value = MagicMock(content="Not valid JSON at all")

        decision = await router_llm.route("What is the return policy?")

        # Should fall back to heuristics
        assert decision is not None

    @pytest.mark.asyncio
    async def test_route_llm_invalid_tools_defaults(self, router_llm, mock_llm):
        """Test handling of invalid tool names in LLM response."""
        mock_llm.ainvoke.return_value = MagicMock(content='''
        {
            "tools": ["invalid_tool", "another_bad_tool"],
            "reasoning": "Invalid tools",
            "requires_context_passing": false
        }
        ''')

        decision = await router_llm.route("Some query")

        # Should default to policy when no valid tools
        assert decision.is_policy_only


class TestExecutionOrder:
    """Tests for tool execution order resolution."""

    def test_resolve_execution_order_single_tool(self):
        """Test execution order with single tool."""
        tools = ["policy_accessor"]
        ordered = QueryRouter.resolve_execution_order(tools)
        assert ordered == ["policy_accessor"]

    def test_resolve_execution_order_both_tools(self):
        """Test that policy comes before SQL due to dependency."""
        tools = ["sql_accessor", "policy_accessor"]
        ordered = QueryRouter.resolve_execution_order(tools)

        # Policy should come before SQL
        assert ordered.index("policy_accessor") < ordered.index("sql_accessor")

    def test_resolve_execution_order_correct_order_input(self):
        """Test with tools already in correct order."""
        tools = ["policy_accessor", "sql_accessor"]
        ordered = QueryRouter.resolve_execution_order(tools)
        assert ordered == ["policy_accessor", "sql_accessor"]

    def test_resolve_execution_order_empty(self):
        """Test with empty tools list."""
        tools = []
        ordered = QueryRouter.resolve_execution_order(tools)
        assert ordered == []


class TestToolDependencies:
    """Tests for tool dependency configuration."""

    def test_tool_dependencies_structure(self):
        """Test that TOOL_DEPENDENCIES is properly structured."""
        assert "policy_accessor" in TOOL_DEPENDENCIES
        assert "sql_accessor" in TOOL_DEPENDENCIES

    def test_policy_has_no_dependencies(self):
        """Test that policy_accessor has no dependencies."""
        assert TOOL_DEPENDENCIES["policy_accessor"] == []

    def test_sql_depends_on_policy(self):
        """Test that sql_accessor can depend on policy_accessor."""
        assert "policy_accessor" in TOOL_DEPENDENCIES["sql_accessor"]


class TestKeywordSets:
    """Tests for keyword set configuration."""

    @pytest.fixture
    def router(self):
        """Create router to access keyword sets."""
        return QueryRouter(use_llm=False)

    def test_policy_keywords_no_false_positives(self, router):
        """Test policy keywords don't cause false positives."""
        # "what is a " has trailing space to avoid matching "what is alex"
        assert "what is a " in router.POLICY_KEYWORDS
        assert "what is the " in router.POLICY_KEYWORDS

        # Simple "what is" should NOT be in keywords
        assert "what is" not in router.POLICY_KEYWORDS

    def test_sql_keywords_include_data_fields(self, router):
        """Test SQL keywords include common data field names."""
        assert "email" in router.SQL_KEYWORDS
        assert "phone" in router.SQL_KEYWORDS
        assert "address" in router.SQL_KEYWORDS
        assert "name" in router.SQL_KEYWORDS

    def test_context_keywords_include_qualification_terms(self, router):
        """Test context keywords include qualification-related terms."""
        context_terms = ["qualify", "qualifies", "eligible", "eligibility", "membership"]
        for term in context_terms:
            assert term in router.CONTEXT_KEYWORDS, f"Missing context keyword: {term}"

    def test_context_keywords_include_tier_terms(self, router):
        """Test context keywords include tier-related terms."""
        tier_terms = ["vip", "tier", "status", "benefits"]
        for term in tier_terms:
            assert term in router.CONTEXT_KEYWORDS, f"Missing context keyword: {term}"
