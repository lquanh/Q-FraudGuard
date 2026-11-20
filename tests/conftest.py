from pathlib import Path

import pytest


@pytest.fixture
def open_dashboard(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from streamlit.util import calc_hash

    monkeypatch.delenv("QFG_DASHBOARD_OPERATOR_PASSWORD", raising=False)
    monkeypatch.delenv("QFG_DASHBOARD_ADMIN_PASSWORD", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

    def open_app():
        app = AppTest.from_file(str(app_path), default_timeout=40).run()
        # AppTest resets the requested page on each run for callable pages.
        app._page_hash = calc_hash("login")
        app.text_input(key="login_username").set_value("chuyenvien1")
        app.text_input(key="login_password").set_value("1")
        next(button for button in app.button if button.label == "Đăng nhập").click().run()
        app._page_hash = calc_hash("dashboard")
        return app

    return open_app
