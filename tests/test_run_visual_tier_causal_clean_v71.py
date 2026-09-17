from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_visual_tier_clean_is_metric_fail_closed():
    source = (ROOT / "src/chaoyang/ops/run_visual_tier_causal_clean_v71.py").read_text()
    assert "PASS_CAUSAL_REAL_DONOR_VISUAL_TIER_GRADE_B" in source
    assert '"metric_geometry": False' in source
    assert '"POSE_ONLY_VISUAL_ROBOT_INPUT"' in source
    assert "run_clean_synthetic_propainter_baseline" not in source


def test_shared_prepare_has_explicit_visual_tier_contract():
    source = (ROOT / "src/chaoyang/ops/prepare_exact78_clean_wave_session_v52.py").read_text()
    assert "exact78-visual-clean-wave1-selection-v71-v1" in source
    assert "ABSENT_VISUAL_TIER_CALIBRATION_MISSING" in source


def test_shared_prepare_requires_30fps_contract():
    source = (ROOT / "src/chaoyang/ops/prepare_exact78_clean_wave_session_v52.py").read_text()
    assert "abs(fps - 30.0) > 0.05" in source
    assert "SUPPORTED_SOURCE_FPS" not in source


def test_visual_tier_runner_supports_bounded_immutable_session_repair():
    source = (ROOT / "src/chaoyang/ops/run_visual_tier_causal_clean_v71.py").read_text()
    assert 'parser.add_argument("--session", action="append"' in source
    assert 'parser.add_argument("--artifact-revision"' in source
    assert 'parser.add_argument("--result-path"' in source
