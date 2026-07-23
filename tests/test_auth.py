import types

from app.auth import COOKIE_NAME, AuthGuard


def _settings(token="", ttl=3600):
    return types.SimpleNamespace(auth_token=token, auth_session_ttl=ttl)


def _guard(token="secret", ttl=3600):
    return AuthGuard(_settings(token, ttl))


class _Req:
    def __init__(self, cookies=None, headers=None):
        self.cookies = cookies or {}
        self.headers = headers or {}


class _Ws:
    def __init__(self, cookies=None, headers=None, query=None):
        self.cookies = cookies or {}
        self.headers = headers or {}
        self.query_params = query or {}


def test_disabled_when_token_empty():
    guard = _guard(token="")
    assert guard.enabled is False
    assert guard.verify_credential("anything") is False
    assert guard.valid_cookie(guard.issue_cookie()) is False


def test_verify_credential_matches_exact_token_only():
    guard = _guard(token="secret")
    assert guard.verify_credential("secret") is True
    assert guard.verify_credential("wrong") is False
    assert guard.verify_credential(None) is False
    assert guard.verify_credential("") is False


def test_cookie_round_trip():
    guard = _guard()
    cookie = guard.issue_cookie(now=1000.0)
    assert guard.valid_cookie(cookie, now=1000.0) is True


def test_expired_cookie_is_rejected():
    guard = _guard(ttl=100)
    cookie = guard.issue_cookie(now=1000.0)
    assert guard.valid_cookie(cookie, now=1000.0 + 50) is True
    assert guard.valid_cookie(cookie, now=1000.0 + 200) is False


def test_tampered_cookie_is_rejected():
    guard = _guard()
    cookie = guard.issue_cookie(now=1000.0)
    expiry, _, sig = cookie.partition(".")
    # bumped expiry with the old signature must not validate
    forged = f"{int(expiry) + 99999}.{sig}"
    assert guard.valid_cookie(forged, now=1000.0) is False
    assert guard.valid_cookie("garbage", now=1000.0) is False
    assert guard.valid_cookie(f"{expiry}.deadbeef", now=1000.0) is False


def test_rotating_token_invalidates_existing_cookie():
    cookie = _guard(token="old").issue_cookie(now=1000.0)
    assert _guard(token="new").valid_cookie(cookie, now=1000.0) is False


def test_request_ok_via_cookie_or_bearer():
    # Far-future TTL so the freshly issued cookie is live under real time.
    guard = _guard(ttl=10 ** 9)
    live_cookie = guard.issue_cookie()
    assert guard.request_ok(_Req(cookies={COOKIE_NAME: live_cookie})) is True
    assert guard.request_ok(_Req(headers={"authorization": "Bearer secret"})) is True
    assert guard.request_ok(_Req(headers={"authorization": "Bearer nope"})) is False
    assert guard.request_ok(_Req()) is False


def test_websocket_ok_via_cookie_header_or_query():
    guard = _guard(ttl=10 ** 9)
    live_cookie = guard.issue_cookie()
    assert guard.websocket_ok(_Ws(cookies={COOKIE_NAME: live_cookie})) is True
    assert guard.websocket_ok(_Ws(headers={"authorization": "Bearer secret"})) is True
    assert guard.websocket_ok(_Ws(query={"token": "secret"})) is True
    assert guard.websocket_ok(_Ws(query={"token": "wrong"})) is False
    assert guard.websocket_ok(_Ws()) is False
