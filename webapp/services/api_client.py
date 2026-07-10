import os
import threading

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Shared/pooled connection — the bare `requests.get/post` module functions
# each open a fresh TCP connection, which adds up fast on pages that make
# many calls per render (e.g. one impression per card).
_session = requests.Session()


def _get(path: str, params: dict = None):
    try:
        response = _session.get(f"{API_URL}{path}", params=params, timeout=10)
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


def _post(path: str, json: dict = None, params: dict = None, timeout: float = 10):
    try:
        response = _session.post(f"{API_URL}{path}", json=json, params=params, timeout=timeout)
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


def _delete(path: str, params: dict = None):
    try:
        response = _session.delete(f"{API_URL}{path}", params=params, timeout=10)
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
        response = _session.get(f"{API_URL}/health", timeout=2)
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


# --- Categories & guest discovery ---

def list_categories():
    return _get("/categories")


def get_discover(trending_k: int = 20, random_k: int = 10, temperature: float = 0.5):
    return _get("/recommendations/discover", params={"trending_k": trending_k, "random_k": random_k, "temperature": temperature})


# --- Signup / onboarding ---

def signup(user_id: str):
    return _post("/users/signup", json={"user_id": user_id})


def set_category_preferences(user_id: str, categories: list):
    return _post(f"/users/{user_id}/preferences/categories", json={"categories": categories})


def get_onboarding_suggestions(user_id: str, k: int = 12):
    return _get(f"/users/{user_id}/onboarding-suggestions", params={"k": k})


def get_onboarding_suggestions_by_categories(categories: list, k: int = 12):
    """Step 2 of signup, before the account exists — requests never need a
    persisted user_id, since account creation is deferred to Finish."""
    return _get("/onboarding-suggestions", params=[("categories", c) for c in categories] + [("k", k)])


# --- Implicit events ---

def log_event(session_id: str, event_type: str, user_id: str = None, parent_asin: str = None, position: int = None, value: float = None):
    """Fire-and-forget on a background thread — this fires on every card
    render (up to ~30/page load for the guest homepage alone). Blocking the
    script on each one was the actual cause behind cover clicks
    intermittently "doing nothing": a slow rerun (many sequential blocking
    calls) can get superseded by the next browser interaction before
    Streamlit ever finishes processing the click that triggered it, silently
    discarding that click's navigation. Nothing reads this call's result, so
    there's nothing lost by not waiting for it."""
    def _fire():
        _post("/events", json={
            "session_id": session_id, "user_id": user_id, "parent_asin": parent_asin,
            "event_type": event_type, "position": position, "value": value,
        }, timeout=3)
    threading.Thread(target=_fire, daemon=True).start()


# --- Wishlist ---

def get_wishlist(user_id: str, session_id: str):
    return _get(f"/users/{user_id}/wishlist", params={"session_id": session_id})


def add_to_wishlist(user_id: str, parent_asin: str, session_id: str):
    return _post(f"/users/{user_id}/wishlist", json={"parent_asin": parent_asin, "session_id": session_id})


def remove_from_wishlist(user_id: str, parent_asin: str, session_id: str):
    return _delete(f"/users/{user_id}/wishlist/{parent_asin}", params={"session_id": session_id})


# --- Cart & checkout ---

def get_cart(user_id: str):
    return _get(f"/users/{user_id}/cart")


def add_to_cart(user_id: str, parent_asin: str, session_id: str):
    return _post(f"/users/{user_id}/cart", json={"parent_asin": parent_asin, "session_id": session_id})


def remove_from_cart(user_id: str, parent_asin: str, session_id: str):
    return _delete(f"/users/{user_id}/cart/{parent_asin}", params={"session_id": session_id})


def checkout(user_id: str, session_id: str):
    return _post(f"/users/{user_id}/checkout", json={"session_id": session_id})


# --- Ratings ---

def submit_rating(user_id: str, parent_asin: str, rating: float, session_id: str):
    return _post(f"/users/{user_id}/ratings", json={"parent_asin": parent_asin, "rating": rating, "session_id": session_id})


# --- Activity log ---

def get_user_events(user_id: str, limit: int = 200):
    return _get(f"/users/{user_id}/events", params={"limit": limit})
