"""Tests voor stepper.py — statuslogica van de voortgangsindicator."""

from edu_synth.ui.stepper import STEP_LABELS, step_status


def test_step_status_done_active_future():
    # huidige stap = 3
    assert step_status(1, 3) == "done"
    assert step_status(2, 3) == "done"
    assert step_status(3, 3) == "active"
    assert step_status(4, 3) == "future"
    assert step_status(5, 3) == "future"


def test_step_status_first_step():
    assert step_status(1, 1) == "active"
    assert step_status(2, 1) == "future"


def test_step_status_last_step_all_done():
    statuses = [step_status(i, 5) for i in range(1, 6)]
    assert statuses == ["done", "done", "done", "done", "active"]


def test_five_labels():
    assert len(STEP_LABELS) == 5
