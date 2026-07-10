import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _get(path: str, params: dict = None):
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return {"error": f"{e} — {detail}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            # Nest the API's own payload under "detail" instead of spreading it —
            # spreading let the API's `status: "healthy"` silently overwrite our
            # `status: "online"` sentinel below, since dict-literal keys are
            # last-write-wins (a real bug caught by testing, not hypothetical).
            return {"status": "online", "detail": response.json()}
        return {"status": "error", "code": response.status_code}
    except requests.exceptions.RequestException:
        return {"status": "offline"}


def list_users(limit: int = 300):
    return _get("/users", params={"limit": limit})


def get_popular(k: int = 10, temperature: float = 0.5):
    return _get("/recommendations/popular", params={"k": k, "temperature": temperature})


def get_personalized(user_id: str, k: int = 10, temperature: float = 0.5):
    return _get("/recommendations/personalized", params={"user_id": user_id, "k": k, "temperature": temperature})


def get_because_you_liked(user_id: str, k: int = 10, temperature: float = 0.3):
    return _get(f"/users/{user_id}/because-you-liked", params={"k": k, "temperature": temperature})


def get_item(item_id: str):
    return _get(f"/items/{item_id}")


def get_similar_items(item_id: str, k: int = 10, temperature: float = 0.3):
    return _get(f"/items/{item_id}/similar", params={"k": k, "temperature": temperature})


def get_similar_users_liked(user_id: str, k: int = 10, temperature: float = 0.5):
    return _get(f"/users/{user_id}/similar-users-liked", params={"k": k, "temperature": temperature})


def get_same_author(item_id: str, k: int = 10):
    return _get(f"/items/{item_id}/same-author", params={"k": k})


def get_similar_authors(item_id: str, k: int = 10, temperature: float = 0.4):
    return _get(f"/items/{item_id}/similar-authors", params={"k": k, "temperature": temperature})


def get_metrics():
    return _get("/metrics")
