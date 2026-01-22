import pytest
from httpx import AsyncClient, ASGITransport
from service_api import app

@pytest.mark.asyncio
async def test_home():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
