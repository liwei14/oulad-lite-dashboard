"""Shared Streamlit layout helpers."""

from __future__ import annotations

import streamlit as st

_HOME_HEADER_HTML = (
    '<div style="margin-bottom: 1.75rem;">'
    '<h1 style="font-family: inherit; font-size: clamp(1.85rem, 4vw, 2.75rem); '
    "font-weight: 600; color: #1d1d1f; margin: 0; padding: 0; line-height: 1.12; "
    'letter-spacing: -0.02em;">OULAD-Lite Learning Analytics Dashboard</h1>'
    '<p style="font-family: inherit; font-size: clamp(1.35rem, 2.6vw, 1.85rem); '
    "color: #515154; font-weight: 400; margin: 0.45rem 0 0 0; padding: 0; "
    "line-height: 1.35; letter-spacing: -0.01em;\">"
    "Adaptive for teacher agency and student self-regulated learning</p>"
    "</div>"
)

_PAGE_FOOTER_HTML = (
    '<p style="text-align: center; color: #6e6e73; font-size: 0.875rem; '
    'margin-top: 2.5rem; margin-bottom: 0;">Built by Hugo together with '
    "Cursor (2026)</p>"
)


def render_home_header() -> None:
    """Home page main title and styled subtitle."""
    st.markdown(_HOME_HEADER_HTML, unsafe_allow_html=True)


def render_page_footer() -> None:
    """Centered credit line at the bottom of every app page."""
    st.markdown(_PAGE_FOOTER_HTML, unsafe_allow_html=True)
