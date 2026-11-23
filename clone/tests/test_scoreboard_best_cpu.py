"""tools/scoreboard.best_seat_entry: between seats the better measurement (row direction, N >= theirs) wins."""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), "..", "tools", "scoreboard.py")
_spec = importlib.util.spec_from_file_location("scoreboard", _p)
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)


def test_best_seat_entry_by_direction_and_n():
    row = dict(id="x", direction="higher", n=500)
    a, b, small = dict(value=.596, n=500, seat="q"), dict(value=.684, n=500, seat="s"), dict(value=.9, n=100, seat="z")
    assert sb.best_seat_entry(row, [a, b])["seat"] == "s"
    assert sb.best_seat_entry(row, [a, b, small])["seat"] == "s"          # N below theirs only if nothing else
    assert sb.best_seat_entry(dict(row, direction="lower"), [a, b])["seat"] == "q"
    assert sb.best_seat_entry(row, [dict(blocked="no data", seat="r"), a])["seat"] == "q"
    assert sb.best_seat_entry(row, [a, dict(a, seat="later")])["seat"] == "q"  # ties: first seat, as before
