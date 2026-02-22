"""Tests for server.py tool functions — critical paths for deep-researcher agent."""

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from notebooklm_mcp.api_client import NotebookLMClient, AuthenticationError


# FastMCP wraps tool functions into FunctionTool objects.
# Access the raw callable via .fn for testing.
def _get_tool_fn(name):
    """Import a server tool and return its raw callable function."""
    import notebooklm_mcp.server as srv
    tool = getattr(srv, name)
    return tool.fn


@pytest.fixture
def mock_client():
    """Create a mock NotebookLMClient."""
    with patch.object(NotebookLMClient, '_refresh_auth_tokens'):
        client = NotebookLMClient(
            cookies={'SID': 'test_sid'},
            csrf_token='test_csrf',
            session_id='test_sid',
        )
        return client


class TestNotebookListTool:
    """Test notebook_list server tool."""

    def test_notebook_list_success(self, mock_client):
        """Test successful notebook listing."""
        fn = _get_tool_fn('notebook_list')

        mock_nb = MagicMock()
        mock_nb.id = 'nb-uuid-1'
        mock_nb.title = 'Test Notebook'
        mock_nb.source_count = 5
        mock_nb.url = 'https://notebooklm.google.com/notebook/nb-uuid-1'
        mock_nb.ownership = 'owned'
        mock_nb.is_owned = True
        mock_nb.is_shared = False
        mock_nb.created_at = '2026-01-01'
        mock_nb.modified_at = '2026-02-01'

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            mock_client.list_notebooks = MagicMock(return_value=[mock_nb])
            result = fn()

        assert result['status'] == 'success'
        assert result['count'] == 1
        assert result['owned_count'] == 1
        assert result['notebooks'][0]['id'] == 'nb-uuid-1'

    def test_notebook_list_auth_error(self):
        """Test notebook_list returns actionable error on auth failure."""
        fn = _get_tool_fn('notebook_list')

        with patch('notebooklm_mcp.server.get_client', side_effect=ValueError('No authentication found')):
            result = fn()

        assert result['status'] == 'error'
        assert 'notebooklm-mcp-auth' in result['error']

    def test_notebook_list_http_error(self, mock_client):
        """Test notebook_list handles HTTP errors."""
        fn = _get_tool_fn('notebook_list')

        req = httpx.Request('POST', 'https://notebooklm.google.com/')
        resp = httpx.Response(500, request=req)

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            mock_client.list_notebooks = MagicMock(
                side_effect=httpx.HTTPStatusError('Server Error', request=req, response=resp)
            )
            result = fn()

        assert result['status'] == 'error'
        assert '500' in result['error']


class TestResearchStartTool:
    """Test research_start server tool."""

    def test_research_start_creates_notebook(self, mock_client):
        """Test research_start creates notebook when none provided."""
        fn = _get_tool_fn('research_start')

        mock_nb = MagicMock()
        mock_nb.id = 'new-nb-uuid'

        mock_client.create_notebook = MagicMock(return_value=mock_nb)
        mock_client.start_research = MagicMock(return_value={
            'task_id': 'task-123',
            'source': 'web',
            'mode': 'deep',
        })

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            result = fn(query='quantum computing advances', mode='deep')

        assert result['status'] == 'success'
        assert result['notebook_id'] == 'new-nb-uuid'
        assert result['created_notebook'] is True
        assert result['task_id'] == 'task-123'

    def test_research_start_deep_drive_rejected(self, mock_client):
        """Test that deep + drive combination is rejected."""
        fn = _get_tool_fn('research_start')

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            result = fn(query='test', source='drive', mode='deep')

        assert result['status'] == 'error'
        assert 'Drive' in result['error']

    def test_research_start_uses_existing_notebook(self, mock_client):
        """Test research_start uses provided notebook_id."""
        fn = _get_tool_fn('research_start')

        mock_client.start_research = MagicMock(return_value={
            'task_id': 'task-456',
            'source': 'web',
            'mode': 'fast',
        })

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            result = fn(
                query='test query',
                notebook_id='existing-nb',
                mode='fast',
            )

        assert result['status'] == 'success'
        assert result['created_notebook'] is False
        assert result['notebook_id'] == 'existing-nb'


class TestNotebookQueryTool:
    """Test notebook_query server tool."""

    def test_notebook_query_success(self, mock_client):
        """Test successful notebook query."""
        fn = _get_tool_fn('notebook_query')

        mock_client.query = MagicMock(return_value={
            'answer': 'The answer is 42.',
            'conversation_id': 'conv-123',
        })

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            result = fn(notebook_id='nb-1', query='What is the answer?')

        assert result['status'] == 'success'
        assert result['answer'] == 'The answer is 42.'
        assert result['conversation_id'] == 'conv-123'

    def test_notebook_query_with_conversation_id(self, mock_client):
        """Test follow-up query uses conversation_id."""
        fn = _get_tool_fn('notebook_query')

        mock_client.query = MagicMock(return_value={
            'answer': 'Follow-up answer.',
            'conversation_id': 'conv-123',
        })

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            result = fn(
                notebook_id='nb-1',
                query='Tell me more',
                conversation_id='conv-123',
            )

        assert result['status'] == 'success'
        mock_client.query.assert_called_once_with(
            'nb-1',
            query_text='Tell me more',
            source_ids=None,
            conversation_id='conv-123',
            timeout=120.0,
        )

    def test_notebook_query_string_source_ids(self, mock_client):
        """Test that string source_ids are parsed as JSON."""
        fn = _get_tool_fn('notebook_query')

        mock_client.query = MagicMock(return_value={
            'answer': 'Answer.',
            'conversation_id': None,
        })

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            result = fn(
                notebook_id='nb-1',
                query='test',
                source_ids='["src-1", "src-2"]',
            )

        assert result['status'] == 'success'
        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs['source_ids'] == ['src-1', 'src-2']


class TestChatConfigureTool:
    """Test chat_configure server tool."""

    def test_chat_configure_custom(self, mock_client):
        """Test configuring chat with custom prompt."""
        fn = _get_tool_fn('chat_configure')

        mock_client.configure_chat = MagicMock(return_value={
            'status': 'success',
            'goal': 'custom',
        })

        with patch('notebooklm_mcp.server.get_client', return_value=mock_client):
            result = fn(
                notebook_id='nb-1',
                goal='custom',
                custom_prompt='Extract implementation details.',
                response_length='longer',
            )

        assert result['status'] == 'success'
        mock_client.configure_chat.assert_called_once_with(
            notebook_id='nb-1',
            goal='custom',
            custom_prompt='Extract implementation details.',
            response_length='longer',
        )


class TestHealthCheck:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_with_valid_tokens(self):
        """Test health check returns auth status."""
        from notebooklm_mcp.server import health_check
        from notebooklm_mcp.auth import AuthTokens

        mock_tokens = AuthTokens(
            cookies={'SID': 'test'},
            extracted_at=time.time() - 3600,  # 1 hour ago
        )

        mock_request = MagicMock()
        with patch('notebooklm_mcp.auth.load_cached_tokens', return_value=mock_tokens):
            response = await health_check(mock_request)

        body = json.loads(response.body)
        assert body['status'] == 'healthy'
        assert body['auth']['status'] == 'valid'
        assert body['auth']['token_age_hours'] is not None

    @pytest.mark.asyncio
    async def test_health_check_no_tokens(self):
        """Test health check when no tokens exist."""
        from notebooklm_mcp.server import health_check

        mock_request = MagicMock()
        with patch('notebooklm_mcp.auth.load_cached_tokens', return_value=None):
            response = await health_check(mock_request)

        body = json.loads(response.body)
        assert body['status'] == 'healthy'
        assert body['auth']['status'] == 'no_tokens'


class TestAuthTokenPermissions:
    """Test auth token file permissions."""

    def test_save_tokens_sets_600_permissions(self, tmp_path):
        """Test that saved auth tokens have 0600 permissions."""
        from notebooklm_mcp.auth import AuthTokens, save_tokens_to_cache

        tokens = AuthTokens(
            cookies={'SID': 'test'},
            extracted_at=time.time(),
        )

        with patch('notebooklm_mcp.auth.get_cache_path', return_value=tmp_path / 'auth.json'):
            save_tokens_to_cache(tokens, silent=True)

        cache_file = tmp_path / 'auth.json'
        assert cache_file.exists()
        # Check permissions (0600 = owner read/write only)
        assert oct(cache_file.stat().st_mode)[-3:] == '600'
