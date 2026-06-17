"""Tests voor stepper.py — statuslogica van de voortgangsindicator."""

from edu_synth.ui.stepper import STEP_LABELS, step_status


def test_step_status_done_active_future():
    # huidige stap = 2 (van 3)
    assert step_status(1, 2) == "done"
    assert step_status(2, 2) == "active"
    assert step_status(3, 2) == "future"


def test_step_status_first_step():
    assert step_status(1, 1) == "active"
    assert step_status(2, 1) == "future"


def test_step_status_last_step_all_done():
    statuses = [step_status(i, 3) for i in range(1, 4)]
    assert statuses == ["done", "done", "active"]


def test_three_labels():
    assert STEP_LABELS == ["Data laden", "Genereren", "Resultaten"]
