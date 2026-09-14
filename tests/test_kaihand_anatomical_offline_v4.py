import numpy as np
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tools import diagnose_kaihand_anatomical_temporal_v3 as anatomy


class FakeOfficial:
    @staticmethod
    def forward_kinematics(model, values):
        return model


def T(x, y, z):
    out = np.eye(4)
    out[:3, 3] = (x, y, z)
    return out


def contract(prefix, fk):
    return {"model": fk, "names": (), "basis": np.eye(3), "mcp_scale": 2.0, "prefix": prefix}


def test_semantic_target_excludes_fixed_palm_bone():
    target = {"mcp": np.ones(3), "bones": np.arange(12).reshape(4, 3), "tip": np.ones(3) * 2}
    result = anatomy.semantic_target(target)
    assert np.array_equal(result["bones"], target["bones"][1:])
    assert result["bones"].shape == (3, 3)


def test_nonthumb_landmarks_are_mcp_pip_dip_mesh_tip():
    fk = {
        "hand_r_index_link2": T(2, 0, 0),
        "hand_r_index_link3": T(2, 1, 0),
        "hand_r_index_link4": T(2, 1, 1),
    }
    result = anatomy.features(FakeOfficial, contract("hand_r", fk), np.empty(0), "index", np.array((0, 0, 1)))
    assert result["bones"].shape == (3, 3)
    assert np.allclose(result["mcp"], (1, 0, 0))
    assert np.allclose(result["bones"], ((0, 1, 0), (0, 0, 1), (0, 0, 1)))


def test_thumb_is_independent_cmc_mcp_ip_mesh_tip_chain():
    fk = {
        "hand_l_thumb_link3": T(1, 0, 0),
        "hand_l_thumb_link5": T(2, 0, 0),
        "hand_l_thumb_link6": T(3, 0, 0),
    }
    result = anatomy.features(FakeOfficial, contract("hand_l", fk), np.empty(0), "thumb", np.array((1, 0, 0)))
    assert result["bones"].shape == (3, 3)
    assert np.allclose(result["mcp"], (1, 0, 0))
    assert np.allclose(result["bones"], np.tile((1, 0, 0), (3, 1)))
