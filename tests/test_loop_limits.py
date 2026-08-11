from src.training.loop_limits import hit_max_steps, parse_epochs, parse_max_steps


def test_parse_max_steps():
    assert parse_max_steps(None) is None
    assert parse_max_steps("") is None
    assert parse_max_steps(0) is None
    assert parse_max_steps("0") is None
    assert parse_max_steps(50) == 50
    assert parse_max_steps("50") == 50


def test_parse_epochs():
    assert parse_epochs(None) == 1
    assert parse_epochs(0) == 1
    assert parse_epochs(2) == 2


def test_hit_max_steps_after_full_accum():
    assert not hit_max_steps(399, 8, 50)
    assert hit_max_steps(400, 8, 50)
    assert not hit_max_steps(400, 8, None)
