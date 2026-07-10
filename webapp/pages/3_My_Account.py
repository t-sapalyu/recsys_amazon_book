import streamlit as st

from services import api_client
from services.ui import PLACEHOLDER_IMAGE, ensure_session, show_api_error, sync_pending_view

st.title("My Account")

ensure_session()
sync_pending_view()

user_id = st.session_state.get("user_id")
if not user_id:
    st.info("Log in or sign up from the sidebar to see your wishlist and cart.")
    st.stop()

tab_wishlist, tab_cart = st.tabs(["🤍 Wishlist", "🛒 Cart"])

with tab_wishlist:
    result = api_client.get_wishlist(user_id, st.session_state["session_id"])
    if "error" in result:
        show_api_error(result)
    elif not result["items"]:
        st.caption("Your wishlist is empty.")
    else:
        for row in result["items"]:
            item = row["item"]
            col_img, col_info, col_action = st.columns([1, 3, 1])
            with col_img:
                st.image(item.get("image_url") or PLACEHOLDER_IMAGE, width="stretch")
            with col_info:
                st.markdown(f"**{item.get('title') or item['parent_asin']}**")
                if item.get("category"):
                    st.caption(item["category"])
                st.caption(f"Added {row['added_at'][:10]} · viewed {row['view_count']}x")
            with col_action:
                if st.button("Remove", key=f"wishlist_remove_{item['parent_asin']}"):
                    api_client.remove_from_wishlist(user_id, item["parent_asin"], st.session_state["session_id"])
                    st.rerun()
            st.divider()

with tab_cart:
    result = api_client.get_cart(user_id)
    if "error" in result:
        show_api_error(result)
    elif not result["items"]:
        st.caption("Your cart is empty.")
    else:
        for row in result["items"]:
            item = row["item"]
            col_img, col_info, col_action = st.columns([1, 3, 1])
            with col_img:
                st.image(item.get("image_url") or PLACEHOLDER_IMAGE, width="stretch")
            with col_info:
                st.markdown(f"**{item.get('title') or item['parent_asin']}**")
                if item.get("price"):
                    st.write(f"${item['price']:.2f}")
            with col_action:
                if st.button("Remove", key=f"cart_remove_{item['parent_asin']}"):
                    api_client.remove_from_cart(user_id, item["parent_asin"], st.session_state["session_id"])
                    st.rerun()
            st.divider()

        st.markdown(f"### Total: ${result['total_price']:.2f}")
        if st.button("✅ Confirm purchase", type="primary"):
            checkout_result = api_client.checkout(user_id, st.session_state["session_id"])
            if "error" in checkout_result:
                show_api_error(checkout_result)
            else:
                st.success("Order placed — thank you! (Demo only — no real payment was made.)")
                st.rerun()
