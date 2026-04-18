"""
tests/test_api.py — Integration tests for FastAPI endpoints.
"""

import pytest
from httpx import AsyncClient, ASITransport

from api.main import app

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
async def async_client():
    transport = ASITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testServer") as client:
        yield client

@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
async def test_auth_register_and_login(async_client: AsyncClient):
    # Register
    reg_res = await async_client.post("/api/v1/auth/register", json={
        "username": "testuser",
        "password": "testpassword"
    })
    
    # Simple check for conflict as fake_db is kept across test cases
    if reg_res.status_code == 400:
        assert reg_res.json()["detail"] == "Username already registered"
    else:
        assert reg_res.status_code == 200
        assert reg_res.json()["username"] == "testuser"

    # Login
    log_res = await async_client.post("/api/v1/auth/login", data={
        "username": "testuser",
        "password": "testpassword"
    })
    assert log_res.status_code == 200
    data = log_res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_protected_routes(async_client: AsyncClient):
    # Login to get token
    log_res = await async_client.post("/api/v1/auth/login", data={
        "username": "testuser",
        "password": "testpassword"
    })
    token = log_res.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}

    # Agents List
    ag_res = await async_client.get("/api/v1/agents", headers=headers)
    assert ag_res.status_code == 200

    # Tasks List
    ts_res = await async_client.get("/api/v1/tasks", headers=headers)
    assert ts_res.status_code == 200

    # Approval List
    ap_res = await async_client.get("/api/v1/approve", headers=headers)
    assert ap_res.status_code == 200

    # Test WS (Just the HTTP upgrade part won't work in pure httpx easily, so skipping WebSocket)
