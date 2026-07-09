import streamlit as st

PLACEHOLDER_IMAGE = "https://placehold.co/200x300?text=No+Image"


def render_item_card(column, rec, source_model: str = None):
    """Renders one recommended item (title, image, category, source-model badge)
    as a clickable card that navigates to the item detail page."""
    item = rec["item"]
    with column:
        st.image(item.get("image_url") or PLACEHOLDER_IMAGE, width="stretch")
        st.markdown(f"**{(item.get('title') or item['parent_asin'])[:60]}**")
        if item.get("category"):
            st.caption(item["category"])
        if item.get("average_rating"):
            st.caption(f"⭐ {item['average_rating']:.1f}" + (f" · ${item['price']:.2f}" if item.get("price") else ""))
        if st.button("View item", key=f"view_{item['parent_asin']}_{source_model}_{rec['score']:.4f}"):
            st.query_params["item_id"] = item["parent_asin"]
            st.switch_page("pages/2_Item.py")


def render_item_row(items, source_model: str, n_cols: int = 5):
    """Renders a horizontal row of item cards, with the model that produced
    them shown once above the row (project.md: "show which model produced
    each recommendation")."""
    st.caption(f"Powered by: **{source_model}**")
    cols = st.columns(n_cols)
    for i, rec in enumerate(items):
        render_item_card(cols[i % n_cols], rec, source_model)


def show_api_error(result: dict):
    st.error("Something went wrong calling the recommendation API.")
    with st.expander("Show error details"):
        st.write(result.get("error", result))
