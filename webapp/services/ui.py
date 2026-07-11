import html
import time
import uuid

import streamlit as st

from services import api_client

PLACEHOLDER_IMAGE = "https://placehold.co/200x300?text=No+Image"
COVER_HEIGHT_PX = 260
TITLE_LINES = 2

TOP_ROW_MAX_POSITION = 5  # keep in sync with model_service/recommend.py's TOP_ROW_MAX_POSITION
QUICK_BOUNCE_SECONDS = 5

def cover_image_html(image_url: str, height: int = COVER_HEIGHT_PX) -> str:
    """A fixed-height, cropped (`object-fit: cover`) book cover — plain
    `st.image` sizes only by width and preserves each image's own aspect
    ratio, so covers of different proportions come out different heights
    and throw off row alignment. Render with
    `st.markdown(cover_image_html(...), unsafe_allow_html=True)`."""
    src = image_url or PLACEHOLDER_IMAGE
    return f'<img src="{src}" style="width:100%; height:{height}px; object-fit:cover; border-radius:8px;" />'


def clamped_title_html(title: str, lines: int = TITLE_LINES) -> str:
    """Fixed-height title block, clamped to a set number of lines with an
    ellipsis — otherwise a 2-line title makes its card taller than
    neighboring cards with a 1-line title, breaking row alignment the same
    way uneven cover heights did."""
    return (
        f'<div style="height:{lines * 1.3}em; overflow:hidden; margin:0.3em 0;'
        f'display:-webkit-box; -webkit-line-clamp:{lines}; -webkit-box-orient:vertical;">'
        f'<strong>{html.escape(title)}</strong></div>'
    )


def cover_button(image_url: str, key: str, height: int = COVER_HEIGHT_PX, label: str = "View details", selected: bool = None) -> bool:
    """Book cover (plain `st.image`, guaranteed to render) immediately
    followed by a plain, real, full-width `st.button` — no custom CSS, no
    reliance on Streamlit's internal DOM structure, nothing that can fail
    silently in a browser I can't see. This is deliberately *not* a
    pixel-perfect "click the cover pixels" solution (an earlier version
    tried to paint the cover directly onto the button via injected CSS —
    scrapped after it kept failing in ways invisible to every test method
    available here). Returns True on the rerun where the button was just
    clicked, same as `st.button`.

    Pass `selected=True/False` (instead of `label`) to render this as a
    toggle — used by the signup flow's "pick which of these you like" step —
    rather than a plain one-off action button."""
    st.markdown(cover_image_html(image_url, height=height), unsafe_allow_html=True)
    if selected is None:
        return st.button(label, key=key, width="stretch")
    button_label = "✅ Selected" if selected else "Select"
    button_type = "primary" if selected else "secondary"
    return st.button(button_label, key=key, width="stretch", type=button_type)


def ensure_session():
    """Assigns each browser session a stable id, generated once and reused
    for every event it logs — including guest (pre-signup) activity, which
    otherwise couldn't be tied together at all. Call at the top of every page."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "hidden_asins" not in st.session_state:
        st.session_state["hidden_asins"] = set()
    return st.session_state["session_id"]


def log_event(event_type: str, parent_asin: str = None, position: int = None, value: float = None):
    ensure_session()
    api_client.log_event(
        session_id=st.session_state["session_id"], event_type=event_type,
        user_id=st.session_state.get("user_id"), parent_asin=parent_asin, position=position, value=value,
    )


def ensure_wishlist_cart_cache(user_id: str = None):
    """Fetch the logged-in user's current wishlist/cart item sets once and
    cache them in `session_state`, so `render_item_card` can show a filled
    vs. empty heart/cart icon without an API call per card. Refetched
    whenever the logged-in user changes (switching accounts, logging out);
    every add/remove path also mutates these sets directly (see
    `render_item_card`, `pages/2_Item.py`, `pages/3_My_Account.py`) so they
    never need a re-fetch just to reflect an action taken this session."""
    owner_key = "wishlist_cart_cache_owner"
    if user_id is None:
        st.session_state["wishlist_asins"] = set()
        st.session_state["cart_asins"] = set()
        st.session_state[owner_key] = None
        return
    if st.session_state.get(owner_key) == user_id:
        return
    ensure_session()
    wishlist_result = api_client.get_wishlist(user_id, st.session_state["session_id"])
    cart_result = api_client.get_cart(user_id)
    st.session_state["wishlist_asins"] = (
        {row["item"]["parent_asin"] for row in wishlist_result["items"]} if "error" not in wishlist_result else set()
    )
    st.session_state["cart_asins"] = (
        {row["item"]["parent_asin"] for row in cart_result["items"]} if "error" not in cart_result else set()
    )
    st.session_state[owner_key] = user_id


def clear_pending_view():
    """Call right after a positive action (wishlist/cart/rating/like) on the
    item page, so that action isn't overwritten by a bounce/no-action
    negative signal once the user navigates away."""
    st.session_state.pop("pending_view", None)


def sync_pending_view(current_item_id: str = None):
    """Tracks whether the item currently open (if any) ends up with no
    positive action before the user leaves it — logs `quick_bounce` (fast
    exit) or `viewed_no_action` (slower exit, still no action) the moment
    that happens. Call at the top of every page: the Item page passes its
    own `item_id`; every other page passes nothing, meaning "the user just
    left whatever item they had open, if any"."""
    ensure_session()
    pending = st.session_state.get("pending_view")

    if pending and pending["parent_asin"] != current_item_id:
        dwell = time.time() - pending["started_at"]
        event_type = "quick_bounce" if dwell < QUICK_BOUNCE_SECONDS else "viewed_no_action"
        log_event(event_type, parent_asin=pending["parent_asin"])
        del st.session_state["pending_view"]
        pending = None

    if current_item_id and pending is None:
        st.session_state["pending_view"] = {"parent_asin": current_item_id, "started_at": time.time()}
        log_event("view", parent_asin=current_item_id)


def render_item_card(column, rec, source_model: str = None, position: int = None):
    """Renders one recommended item (image, title, category, source-model
    badge) as a clickable card that navigates to the item detail page. Logs
    an `impression` on every render, plus — for logged-in users — wishlist/
    cart/not-interested/hide actions. See `cover_button` for the click."""
    item = rec["item"]
    key_suffix = f"{item['parent_asin']}_{source_model}_{rec['score']:.4f}_{position}"
    user_id = st.session_state.get("user_id")

    log_event("impression", parent_asin=item["parent_asin"], position=position)

    with column:
        # `switch_page` clears query params by default — pass them explicitly
        # or `item_id` is gone by the time the Item page reads it.
        if cover_button(item.get("image_url"), key=f"open_{key_suffix}", label="View details"):
            st.switch_page("pages/2_Item.py", query_params={"item_id": item["parent_asin"]})
        title = (item.get("title") or item["parent_asin"])[:60]
        st.markdown(clamped_title_html(title), unsafe_allow_html=True)
        if item.get("category"):
            st.caption(item["category"])
        if item.get("average_rating"):
            st.caption(f"⭐ {item['average_rating']:.1f}" + (f" · ${item['price']:.2f}" if item.get("price") else ""))

        if user_id:
            wishlist_asins = st.session_state.get("wishlist_asins", set())
            cart_asins = st.session_state.get("cart_asins", set())
            in_wishlist = item["parent_asin"] in wishlist_asins
            in_cart = item["parent_asin"] in cart_asins

            b1, b2, b3, b4 = st.columns(4)
            wishlist_icon = "❤️" if in_wishlist else "🤍"
            if b1.button(wishlist_icon, key=f"wishlist_{key_suffix}", help="Remove from wishlist" if in_wishlist else "Add to wishlist"):
                if in_wishlist:
                    api_client.remove_from_wishlist(user_id, item["parent_asin"], st.session_state["session_id"])
                    wishlist_asins.discard(item["parent_asin"])
                    st.toast("Removed from wishlist")
                else:
                    api_client.add_to_wishlist(user_id, item["parent_asin"], st.session_state["session_id"])
                    wishlist_asins.add(item["parent_asin"])
                    st.toast("Added to wishlist")
                st.rerun()
            cart_icon = "✅" if in_cart else "🛒"
            if b2.button(cart_icon, key=f"cart_{key_suffix}", help="Remove from cart" if in_cart else "Add to cart"):
                if in_cart:
                    api_client.remove_from_cart(user_id, item["parent_asin"], st.session_state["session_id"])
                    cart_asins.discard(item["parent_asin"])
                    st.toast("Removed from cart")
                else:
                    api_client.add_to_cart(user_id, item["parent_asin"], st.session_state["session_id"])
                    cart_asins.add(item["parent_asin"])
                    st.toast("Added to cart")
                st.rerun()
            if b3.button("👎", key=f"notint_{key_suffix}", help="Not interested"):
                log_event("not_interested", parent_asin=item["parent_asin"], position=position)
                st.toast("Thanks — we'll show you less of this")
            if b4.button("✕", key=f"hide_{key_suffix}", help="Hide this recommendation"):
                log_event("dismiss_recommendation", parent_asin=item["parent_asin"], position=position)
                st.session_state["hidden_asins"].add(item["parent_asin"])
                st.rerun()


def render_item_row(items, source_model: str, n_cols: int = 5):
    """Renders a horizontal row of item cards, with the model that produced
    them shown once above the row (project.md: "show which model produced
    each recommendation"). Skips items the user dismissed earlier this session."""
    ensure_session()
    ensure_wishlist_cart_cache(st.session_state.get("user_id"))
    hidden = st.session_state["hidden_asins"]
    visible_items = [rec for rec in items if rec["item"]["parent_asin"] not in hidden]
    st.caption(f"Powered by: **{source_model}**")
    cols = st.columns(n_cols)
    for i, rec in enumerate(visible_items):
        render_item_card(cols[i % n_cols], rec, source_model, position=i)


def show_api_error(result: dict):
    st.error("Something went wrong calling the recommendation API.")
    with st.expander("Show error details"):
        st.write(result.get("error", result))
