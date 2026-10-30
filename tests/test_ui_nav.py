"""
test_ui_nav.py
==============
Tests for UI navigation state management and routing constants.
"""

from unittest.mock import patch
import pytest
from src.ui.nav import PAGES, PAGE_HOME, PAGE_INTAKE, PAGE_RESULTS, PAGE_ABOUT, navigate_to


def test_navigation_pages_definition():
    """Verify all 4 core pages are defined in sequence."""
    assert len(PAGES) == 4
    assert PAGES[0] == PAGE_HOME
    assert PAGES[1] == PAGE_INTAKE
    assert PAGES[2] == PAGE_RESULTS
    assert PAGES[3] == PAGE_ABOUT


def test_navigate_to_updates_session_state():
    """Verify navigate_to sets canonical state and pending navigation queue."""
    mock_session_state = {}
    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.rerun") as mock_rerun:
        navigate_to(PAGE_INTAKE)
        assert mock_session_state["current_page"] == PAGE_INTAKE
        assert mock_session_state["_pending_nav"] == PAGE_INTAKE
        mock_rerun.assert_called_once()


def test_navigate_to_ignores_invalid_page():
    """Verify invalid page names do not trigger rerun or mutate state."""
    mock_session_state = {}
    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.rerun") as mock_rerun:
        navigate_to("NonExistentPage")
        assert "current_page" not in mock_session_state
        assert "nav_radio_selection" not in mock_session_state
        mock_rerun.assert_not_called()
