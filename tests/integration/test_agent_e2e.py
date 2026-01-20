"""End-to-end integration tests for MultiToolAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from multi_tool_agent.core.agent import MultiToolAgent
from multi_tool_agent.core.query_router import QueryRouter, RoutingDecision
from multi_tool_agent.tools.policy_accessor import PolicyAccessor, PolicyResult
from multi_tool_agent.tools.sql_accessor.sql_tool import SQLAccessor, SQLAccessorResult
from multi_tool_agent.logging.trace_logger import TraceLogger


class TestQueryRouter:
    """Tests for QueryRouter."""

    @pytest.fixture
    def router_heuristic(self):
        """Create QueryRouter with heuristic-only routing."""
        return QueryRouter(use_llm=False)

    @pytest.fixture
    def router_llm(self, mock_llm):
        """Create QueryRouter with mocked LLM."""
        with patch("multi_tool_agent.core.query_router.ChatOpenAI") as mock_class:
            mock_class.return_value = mock_llm
            router = QueryRouter(model="gpt-4", api_key="test-key", use_llm=True)
            router._llm = mock_llm
            return router

    # Heuristic routing tests
    @pytest.mark.asyncio
    async def test_route_policy_only_query(self, router_heuristic):
        """Test routing of policy-only queries."""
        decision = await router_heuristic.route("What is the return policy?")

        assert decision.is_policy_only
        assert "policy_accessor" in decision.tools

    @pytest.mark.asyncio
    async def test_route_sql_only_query(self, router_heuristic):
        """Test routing of SQL-only queries."""
        decision = await router_heuristic.route("List all products under $100")

        assert decision.is_sql_only
        assert "sql_accessor" in decision.tools

    @pytest.mark.asyncio
    async def test_route_combined_query(self, router_heuristic):
        """Test routing of queries requiring both tools."""
        decision = await router_heuristic.route("How many VIP customers do we have?")

        assert decision.needs_both
        assert "policy_accessor" in decision.tools
        assert "sql_accessor" in decision.tools
        assert decision.requires_context_passing is True

    @pytest.mark.asyncio
    async def test_route_qualification_query_uses_both_tools(self, router_heuristic):
        """Test that qualification queries use both policy and SQL tools."""
        decision = await router_heuristic.route("Does Alex qualify for membership?")

        assert decision.needs_both
        assert "policy_accessor" in decision.tools
        assert "sql_accessor" in decision.tools
        assert decision.requires_context_passing is True

    @pytest.mark.asyncio
    async def test_route_email_lookup_is_sql_only(self, router_heuristic):
        """Test that simple email lookup uses SQL only."""
        decision = await router_heuristic.route("What is Alex email?")

        assert decision.is_sql_only
        assert "sql_accessor" in decision.tools

    @pytest.mark.asyncio
    async def test_route_email_lookup_possessive_is_sql_only(self, router_heuristic):
        """Test that possessive email lookup uses SQL only."""
        decision = await router_heuristic.route("What is Alex's email?")

        assert decision.is_sql_only

    @pytest.mark.asyncio
    async def test_route_eligibility_query_uses_both_tools(self, router_heuristic):
        """Test that eligibility queries use both tools."""
        decision = await router_heuristic.route("Is order #123 eligible for return?")

        assert decision.needs_both
        assert decision.requires_context_passing is True

    @pytest.mark.asyncio
    async def test_route_with_llm(self, router_llm, mock_llm):
        """Test routing with LLM classification."""
        mock_llm.ainvoke.return_value.content = '''
        {
            "tools": ["sql_accessor"],
            "reasoning": "Data query",
            "requires_context_passing": false
        }
        '''

        decision = await router_llm.route("Count all orders")

        assert "sql_accessor" in decision.tools

    @pytest.mark.asyncio
    async def test_route_llm_fallback_on_error(self, router_llm, mock_llm):
        """Test that router falls back to heuristics on LLM error."""
        mock_llm.ainvoke.side_effect = Exception("API Error")

        decision = await router_llm.route("What is the return policy?")

        # Should fall back to heuristics
        assert decision is not None
        assert len(decision.tools) > 0

    def test_resolve_execution_order(self):
        """Test tool execution order resolution."""
        tools = ["sql_accessor", "policy_accessor"]
        ordered = QueryRouter.resolve_execution_order(tools)

        # Policy should come before SQL due to dependency
        assert ordered.index("policy_accessor") < ordered.index("sql_accessor")

    def test_resolve_execution_order_single_tool(self):
        """Test execution order with single tool."""
        tools = ["policy_accessor"]
        ordered = QueryRouter.resolve_execution_order(tools)

        assert ordered == ["policy_accessor"]


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


class TestMultiToolAgent:
    """End-to-end tests for MultiToolAgent."""

    @pytest.fixture
    def mock_policy_accessor(self):
        """Create mock PolicyAccessor."""
        accessor = MagicMock(spec=PolicyAccessor)
        accessor.initialize = AsyncMock()
        accessor.search = AsyncMock(return_value=PolicyResult(
            content="Return policy: 30 days return window.",
            sources=["Return Policy"],
            from_cache=False,
        ))
        return accessor

    @pytest.fixture
    def mock_sql_accessor(self):
        """Create mock SQLAccessor."""
        accessor = MagicMock(spec=SQLAccessor)
        accessor.run = AsyncMock(return_value=SQLAccessorResult(
            success=True,
            data="Results (3 rows):\nid | name\n---\n1 | John\n2 | Jane\n3 | Bob",
            sql_executed="SELECT id, name FROM customers",
            attempts=1,
            pii_masked=None,
        ))
        accessor.get_schema_description = MagicMock(return_value="Schema info")
        return accessor

    @pytest.fixture
    def mock_router(self):
        """Create mock QueryRouter."""
        router = MagicMock(spec=QueryRouter)
        router.route = AsyncMock(return_value=RoutingDecision(
            tools=["policy_accessor"],
            reasoning="Policy query",
            requires_context_passing=False,
        ))
        return router

    @pytest.fixture
    def agent(self, temp_db, temp_policies, mock_policy_accessor, mock_sql_accessor, mock_router):
        """Create MultiToolAgent with mocked components."""
        with patch("multi_tool_agent.core.agent.PolicyAccessor") as mock_pa_class, \
             patch("multi_tool_agent.core.agent.SQLAccessor") as mock_sa_class, \
             patch("multi_tool_agent.core.agent.QueryRouter") as mock_router_class, \
             patch("multi_tool_agent.core.agent.ChatOpenAI"):

            mock_pa_class.return_value = mock_policy_accessor
            mock_sa_class.return_value = mock_sql_accessor
            mock_router_class.return_value = mock_router

            agent = MultiToolAgent(
                db_path=temp_db,
                policies_path=temp_policies,
                model="gpt-4",
                api_key="test-key",
            )

            agent.policy_accessor = mock_policy_accessor
            agent.sql_accessor = mock_sql_accessor
            agent.router = mock_router
            agent._tools = {
                "policy_accessor": mock_policy_accessor,
                "sql_accessor": mock_sql_accessor,
            }
            agent._initialized = True

            return agent

    @pytest.mark.asyncio
    async def test_execute_policy_query(self, agent, mock_router, mock_policy_accessor):
        """Test executing a policy-only query."""
        mock_router.route.return_value = RoutingDecision(
            tools=["policy_accessor"],
            reasoning="Policy query",
            requires_context_passing=False,
        )

        response = await agent.execute("What is the return policy?")

        assert "Return policy" in response
        mock_policy_accessor.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_sql_query(self, agent, mock_router, mock_sql_accessor):
        """Test executing a SQL-only query."""
        mock_router.route.return_value = RoutingDecision(
            tools=["sql_accessor"],
            reasoning="Data query",
            requires_context_passing=False,
        )

        response = await agent.execute("List all customers")

        mock_sql_accessor.run.assert_called_once()
        # Response should contain data
        assert response is not None

    @pytest.mark.asyncio
    async def test_execute_combined_query(self, agent, mock_router, mock_policy_accessor, mock_sql_accessor):
        """Test executing a query requiring both tools."""
        mock_router.route.return_value = RoutingDecision(
            tools=["policy_accessor", "sql_accessor"],
            reasoning="Combined query",
            requires_context_passing=True,
        )

        response = await agent.execute("How many VIP customers do we have?")

        # Both tools should be called
        mock_policy_accessor.search.assert_called_once()
        mock_sql_accessor.run.assert_called_once()

        # SQL should receive context from policy
        call_kwargs = mock_sql_accessor.run.call_args[1]
        assert "context" in call_kwargs

    @pytest.mark.asyncio
    async def test_execute_with_tool_error(self, agent, mock_router, mock_sql_accessor):
        """Test graceful degradation when tool fails."""
        mock_router.route.return_value = RoutingDecision(
            tools=["sql_accessor"],
            reasoning="Data query",
            requires_context_passing=False,
        )
        mock_sql_accessor.run.side_effect = Exception("Database error")

        # Should not raise with graceful_degradation=True
        response = await agent.execute("List customers")

        assert "error" in response.lower() or "issue" in response.lower()

    @pytest.mark.asyncio
    async def test_execute_with_logging(self, agent, mock_router):
        """Test that logging is called during execution."""
        mock_logger = MagicMock(spec=TraceLogger)
        mock_logger.start_request = MagicMock()
        mock_logger.end_request = MagicMock()
        mock_logger.log_tool_selection = MagicMock()
        mock_logger.log_policy_search = MagicMock()

        agent.logger = mock_logger

        mock_router.route.return_value = RoutingDecision(
            tools=["policy_accessor"],
            reasoning="Policy query",
            requires_context_passing=False,
        )

        await agent.execute("What is the return policy?")

        mock_logger.start_request.assert_called_once()
        mock_logger.log_tool_selection.assert_called_once()
        mock_logger.end_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_passing_between_tools(self, agent, mock_router, mock_policy_accessor, mock_sql_accessor):
        """Test that context is properly passed between tools."""
        mock_router.route.return_value = RoutingDecision(
            tools=["policy_accessor", "sql_accessor"],
            reasoning="Needs policy context",
            requires_context_passing=True,
        )

        mock_policy_accessor.search.return_value = PolicyResult(
            content="VIP = customers with >$1000 spent",
            sources=["Customer Tiers"],
            from_cache=False,
        )

        await agent.execute("How many VIP customers?")

        # Verify SQL accessor received policy context
        call_args, call_kwargs = mock_sql_accessor.run.call_args
        context = call_kwargs.get("context", {})
        assert "policy_accessor" in context
        assert "VIP" in context["policy_accessor"]

    @pytest.mark.asyncio
    async def test_execute_qualification_query(self, agent, mock_router, mock_policy_accessor, mock_sql_accessor):
        """Test executing a qualification query that requires both tools."""
        mock_router.route.return_value = RoutingDecision(
            tools=["policy_accessor", "sql_accessor"],
            reasoning="Qualification query needs policy for criteria and SQL for data",
            requires_context_passing=True,
        )

        mock_policy_accessor.search.return_value = PolicyResult(
            content="Membership qualification: customers with 10+ orders",
            sources=["Membership Policy"],
            from_cache=False,
        )

        mock_sql_accessor.run.return_value = SQLAccessorResult(
            success=True,
            data="Results (1 row):\nname | order_count\n---\nAlex | 5",
            sql_executed="SELECT name, COUNT(*) as order_count FROM orders WHERE customer_name='Alex'",
            attempts=1,
            pii_masked=None,
        )

        response = await agent.execute("Does Alex qualify for membership?")

        # Both tools should be called
        mock_policy_accessor.search.assert_called_once()
        mock_sql_accessor.run.assert_called_once()

        # SQL should receive context from policy about membership criteria
        call_kwargs = mock_sql_accessor.run.call_args[1]
        assert "context" in call_kwargs
        assert "policy_accessor" in call_kwargs["context"]

    @pytest.mark.asyncio
    async def test_execute_simple_data_lookup(self, agent, mock_router, mock_policy_accessor, mock_sql_accessor):
        """Test executing a simple data lookup that only needs SQL."""
        mock_router.route.return_value = RoutingDecision(
            tools=["sql_accessor"],
            reasoning="Simple data lookup",
            requires_context_passing=False,
        )

        mock_sql_accessor.run.return_value = SQLAccessorResult(
            success=True,
            data="Results (1 row):\nemail\n---\nalex@example.com",
            sql_executed="SELECT email FROM customers WHERE name='Alex'",
            attempts=1,
            pii_masked=None,
        )

        response = await agent.execute("What is Alex's email?")

        # Only SQL should be called, not policy
        mock_policy_accessor.search.assert_not_called()
        mock_sql_accessor.run.assert_called_once()


class TestPolicyAccessor:
    """Tests for PolicyAccessor."""

    @pytest.fixture
    def policy_accessor(self, temp_policies, mock_embeddings):
        """Create PolicyAccessor with mocked embeddings."""
        with patch("multi_tool_agent.stores.faiss_store.OpenAIEmbeddings") as mock_emb_class:
            mock_emb_class.return_value = mock_embeddings

            accessor = PolicyAccessor(
                policies_path=temp_policies,
                api_key="test-key",
            )
            return accessor

    def test_load_policies(self, policy_accessor):
        """Test that policies are loaded and chunked."""
        assert policy_accessor.raw_policies is not None
        assert len(policy_accessor.policy_chunks) > 0

    def test_chunk_policies(self, policy_accessor):
        """Test policy chunking by sections."""
        chunks = policy_accessor.policy_chunks

        # Should have chunks for each h2 section
        titles = [chunk.metadata.get("title") for chunk in chunks]
        assert any("Return Policy" in t for t in titles if t)
        assert any("Customer Tiers" in t for t in titles if t)

    @pytest.mark.asyncio
    async def test_search_returns_relevant_content(self, policy_accessor, mock_embeddings):
        """Test that search returns relevant policy content."""
        # Initialize first
        await policy_accessor.initialize()

        result = await policy_accessor.search("return policy")

        assert result.content is not None
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_fallback_search(self, policy_accessor):
        """Test fallback search when vector store fails."""
        result = policy_accessor._fallback_search("VIP customer")

        assert result.from_cache is True
        assert "VIP" in result.content

    def test_get_context_for_sql(self, policy_accessor):
        """Test formatting policy result for SQL context."""
        policy_result = PolicyResult(
            content="VIP = >$1000 spent",
            sources=["Customer Tiers"],
            from_cache=False,
        )

        context = policy_accessor.get_context_for_sql("query", policy_result)

        assert "Business Policies" in context
        assert "VIP" in context


class TestTraceLogger:
    """Tests for TraceLogger."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create TraceLogger."""
        log_file = tmp_path / "test.log"
        return TraceLogger(
            destination="file",
            log_file=log_file,
            level="DEBUG",
            include_costs=True,
        )

    def test_start_end_request(self, logger):
        """Test starting and ending a request trace."""
        trace = logger.start_request("test-123", "Test query")

        assert trace.request_id == "test-123"
        assert trace.query == "Test query"

        logger.end_request(success=True)

        assert trace.end_time is not None
        assert trace.success is True

    def test_log_tool_selection(self, logger):
        """Test logging tool selection."""
        logger.start_request("test", "query")
        logger.log_tool_selection(
            tools=["policy_accessor", "sql_accessor"],
            reasoning="Combined query",
        )

        assert len(logger.current_trace.events) > 0
        assert any(e.event_type == "tool_selection" for e in logger.current_trace.events)

    def test_log_sql_attempt(self, logger):
        """Test logging SQL attempt."""
        logger.start_request("test", "query")
        logger.log_sql_attempt(
            attempt=1,
            sql="SELECT * FROM test",
            success=True,
        )

        events = logger.current_trace.events
        sql_events = [e for e in events if e.event_type == "sql_attempt"]
        assert len(sql_events) > 0

    def test_log_pii_masked(self, logger):
        """Test logging PII masking."""
        logger.start_request("test", "query")
        logger.log_pii_masked({"email": 5, "phone": 3})

        events = logger.current_trace.events
        pii_events = [e for e in events if e.event_type == "pii_masked"]
        assert len(pii_events) > 0

    def test_log_token_usage(self, logger):
        """Test logging token usage."""
        logger.start_request("test", "query")
        logger.log_token_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
        )

        assert len(logger.current_trace.token_usage) == 1
        usage = logger.current_trace.token_usage[0]
        assert usage.model == "gpt-4"
        assert usage.input_tokens == 100

    def test_cost_calculation(self, logger):
        """Test cost calculation."""
        logger.start_request("test", "query")
        logger.log_token_usage("gpt-4", 1000, 500)

        trace = logger.current_trace
        # GPT-4: $0.03/1K input, $0.06/1K output
        expected_cost = (1000 / 1000 * 0.03) + (500 / 1000 * 0.06)
        assert abs(trace.total_cost - expected_cost) < 0.001


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @pytest.fixture
    def full_agent(self, temp_db, temp_policies, mock_embeddings, mock_llm):
        """Create agent with real components but mocked external services."""
        with patch("multi_tool_agent.stores.faiss_store.OpenAIEmbeddings") as mock_emb, \
             patch("multi_tool_agent.tools.sql_accessor.query_generator.ChatOpenAI") as mock_gen, \
             patch("multi_tool_agent.core.query_router.ChatOpenAI") as mock_router_llm, \
             patch("multi_tool_agent.core.agent.ChatOpenAI") as mock_synth:

            mock_emb.return_value = mock_embeddings
            mock_gen.return_value = mock_llm
            mock_router_llm.return_value = mock_llm
            mock_synth.return_value = mock_llm

            # Set up LLM responses
            mock_llm.ainvoke.return_value.content = '''
            {
                "sql": "SELECT id, name FROM customers LIMIT 10",
                "explanation": "Query customers",
                "columns": ["id", "name"]
            }
            '''

            agent = MultiToolAgent(
                db_path=temp_db,
                policies_path=temp_policies,
                model="gpt-4",
                api_key="test-key",
                graceful_degradation=True,
            )

            return agent

    @pytest.mark.asyncio
    async def test_scenario_policy_question(self, full_agent, mock_llm):
        """Test scenario: User asks about return policy."""
        # Mock router to return policy-only
        mock_llm.ainvoke.return_value.content = '''
        {
            "tools": ["policy_accessor"],
            "reasoning": "Policy question",
            "requires_context_passing": false
        }
        '''

        response = await full_agent.execute("What is the return policy?")

        # Should return policy content
        assert response is not None

    @pytest.mark.asyncio
    async def test_scenario_data_query(self, full_agent, mock_llm):
        """Test scenario: User queries database."""
        mock_llm.ainvoke.side_effect = [
            # Router response
            MagicMock(content='''
            {
                "tools": ["sql_accessor"],
                "reasoning": "Data query",
                "requires_context_passing": false
            }
            '''),
            # SQL generation response
            MagicMock(content='''
            {
                "sql": "SELECT id, name FROM customers",
                "explanation": "List customers",
                "columns": ["id", "name"]
            }
            '''),
        ]

        response = await full_agent.execute("List all customers")

        assert response is not None

    @pytest.mark.asyncio
    async def test_scenario_graceful_degradation(self, full_agent, mock_llm):
        """Test scenario: External service fails."""
        # Make all LLM calls fail
        mock_llm.ainvoke.side_effect = Exception("API unavailable")

        # Should not raise, but return fallback response
        response = await full_agent.execute("What is VIP status?")

        # Graceful degradation should return something useful
        # Either an error message or fallback content from cache
        assert response is not None
        assert len(response) > 0
