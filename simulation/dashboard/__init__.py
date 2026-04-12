"""Energie Delen interactive dashboard (Panel)."""

__all__ = ["create_app"]


def create_app():  # noqa: ANN201
    from dashboard.app import create_app as _create_app
    return _create_app()
