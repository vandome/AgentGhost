
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
from pathlib import Path
from ghost_agent.tools.file_system import tool_download_file

@pytest.mark.asyncio
async def test_tool_download_file_writes_asynchronously():
    # Setup mocks
    mock_url = "http://example.com/file.txt"
    Path("/tmp/sandbox/file.txt").unlink(missing_ok=True)   # §4OG-1 (h): a leftover from an earlier run is an unread existing file
    mock_sandbox = Path("/tmp/sandbox")
    mock_content = b"chunk1chunk2"
    
    # Mock httpx client
    with patch("ghost_agent.tools.file_system.curl_requests", None), \
         patch("ghost_agent.tools.file_system.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        # Mock stream response
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Length": "100"}
        
        # Mock aiter_bytes to yield chunks
        async def mock_aiter_bytes():
            yield b"chunk1"
            yield b"chunk2"
        mock_resp.aiter_bytes = mock_aiter_bytes
        
        # ensure stream is NOT an AsyncMock (which is awaited on call), but returns an async context manager
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_resp
        mock_context.__aexit__.return_value = None
        
        mock_client.stream = MagicMock(return_value=mock_context)

        # §4PE: the body is staged in a real temp file beside the target and moved
        # over it; the flush still goes through asyncio.to_thread (spied here, but run)
        import tempfile, asyncio as _aio
        tmpdir = Path(tempfile.mkdtemp())
        target = tmpdir / "file.txt"
        real_to_thread = _aio.to_thread
        calls = []

        async def _spy(fn, *a, **k):
            calls.append((fn, a))
            return await real_to_thread(fn, *a, **k)
        with patch("ghost_agent.tools.file_system.asyncio.to_thread", side_effect=_spy):
            with patch("ghost_agent.tools.file_system._get_safe_path", return_value=target):
                out = await tool_download_file(mock_url, mock_sandbox, None, "file.txt")
        assert "SUCCESS" in out and target.read_bytes() == b"chunk1chunk2"
        writes = [c for c in calls if getattr(c[0], "__name__", "") == "write"]
        assert len(writes) == 1 and writes[0][1][0] == bytearray(b"chunk1chunk2")
