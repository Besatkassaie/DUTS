"""Tests for the filesystem backup/stage/restore logic in
``scripts.wdc_extract_via_santos_checkpoint`` (WDC scalability study Phase 3).

Deliberately narrow and GPU-free: this only covers the side-effecting parts
(backup/restore/stage), tested against a tmp_path standing in for the real
``data/santos/`` staging directory -- never touches the real one. The
subprocess call to ``extractVectors.py`` itself is integration/smoke-only
(matches ``santoslarge_study.py``'s confirmed zero-pytest-coverage precedent
for GPU/real-data-dependent scripts), not covered here.
"""
import os

from scripts.wdc_extract_via_santos_checkpoint import (
    QUERY_PLACEHOLDER_COUNT, _stage, backup_existing_dir, restore_or_absent,
)


def test_backup_returns_none_when_target_absent(tmp_path):
    target = tmp_path / "santos"
    assert backup_existing_dir(str(target)) is None
    assert not target.exists()


def test_backup_renames_existing_dir_aside(tmp_path):
    target = tmp_path / "santos"
    target.mkdir()
    (target / "marker.txt").write_text("original")

    backup_path = backup_existing_dir(str(target))

    assert backup_path is not None
    assert not target.exists()
    assert os.path.isfile(os.path.join(backup_path, "marker.txt"))


def test_restore_or_absent_with_backup_restores_original_contents(tmp_path):
    target = tmp_path / "santos"
    target.mkdir()
    (target / "marker.txt").write_text("original")
    backup_path = backup_existing_dir(str(target))

    # simulate staging: something new now occupies `target`
    target.mkdir()
    (target / "staged.txt").write_text("staged")

    restore_or_absent(str(target), backup_path)

    assert os.path.isfile(str(target / "marker.txt"))
    assert not os.path.exists(str(target / "staged.txt"))
    assert not os.path.exists(backup_path)  # renamed back, not left behind


def test_restore_or_absent_without_backup_leaves_target_absent(tmp_path):
    target = tmp_path / "santos"
    target.mkdir()
    (target / "staged.txt").write_text("staged")

    restore_or_absent(str(target), None)

    assert not target.exists()


def test_full_backup_stage_restore_cycle_is_idempotent_on_pre_existing_dir(tmp_path):
    """End-to-end: pre-existing dir survives a full stage/restore cycle
    byte-for-byte, which is the property the real script's `finally` block
    depends on to never lose real data in `data/santos/`."""
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    for i in range(5):
        (csv_dir / ("t%d.csv" % i)).write_text("a,b\n1,2\n")

    target = tmp_path / "santos"
    target.mkdir()
    (target / "important_marker.txt").write_text("do not lose me")

    backup_path = backup_existing_dir(str(target))
    n_staged = _stage(str(csv_dir), target_dir=str(target))
    assert n_staged == 5
    assert os.path.isdir(str(target / "datalake"))
    assert len(os.listdir(str(target / "datalake"))) == 5
    assert len(os.listdir(str(target / "query"))) == min(QUERY_PLACEHOLDER_COUNT, 5)
    # extractVectors.py writes here but never creates it itself -- found by
    # running the real script against tier_10k, where its absence crashed
    # the pickle.dump() call with FileNotFoundError.
    assert os.path.isdir(str(target / "vectors"))

    restore_or_absent(str(target), backup_path)

    assert (target / "important_marker.txt").read_text() == "do not lose me"
    assert not (target / "datalake").exists()
