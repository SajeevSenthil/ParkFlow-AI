from __future__ import annotations

from parkflow import operations as ops


def test_log_and_history_roundtrip(tmp_path):
    db = tmp_path / "ops.db"
    rid = ops.log_deployment(
        db, team="Team A", zone="KR Market", priority_score=87.0,
        predicted_violations=12.0, operator="op1",
    )
    history = ops.deployment_history(db)
    assert len(history) == 1
    row = history.iloc[0]
    assert int(row["id"]) == rid
    assert row["team"] == "Team A"
    assert row["zone"] == "KR Market"
    assert row["status"] == "deployed"


def test_override_then_mark_complete(tmp_path):
    db = tmp_path / "ops.db"
    ops.log_deployment(db, team="Team B", zone="Z1")
    oid = ops.override_assignment(db, team="Team B", new_zone="Z2", operator="op2")
    ops.mark_complete(db, oid)
    history = ops.deployment_history(db).set_index("id")
    assert history.loc[oid, "status"] == "completed"
    assert history.loc[oid, "zone"] == "Z2"


def test_history_empty_when_db_absent(tmp_path):
    history = ops.deployment_history(tmp_path / "does_not_exist.db")
    assert len(history) == 0


def test_history_is_most_recent_first(tmp_path):
    db = tmp_path / "ops.db"
    first = ops.log_deployment(db, team="Team A", zone="Z1")
    second = ops.log_deployment(db, team="Team B", zone="Z2")
    history = ops.deployment_history(db)
    assert int(history.iloc[0]["id"]) == second
    assert int(history.iloc[1]["id"]) == first
