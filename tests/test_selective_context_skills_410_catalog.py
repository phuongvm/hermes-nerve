"""410-skill ``<available_skills>`` regression: mandatory pins and discretionary matches survive past index 128.

Baseline Hermes catalog is 410 active skills (> 18,000 rendered chars). Before this change the skills
surface capped discretionary scoring at 128 candidates, so every skill rendered after row 128 was
rejected with ``CANDIDATE_CAP`` and lost its ``: <description>`` regardless of relevance.
"""
from __future__ import annotations

import tempfile

import pytest

from hermes_nerve.consumer import _controller, plan_skills_index, select_skills_index, settings_from_config

CATALOG_SIZE = 410
MANDATORY_INDEX = 350
DISCRETIONARY_INDEX = 300
MANDATORY_NAME = "slice3-mandatory-governance-gate"
MANDATORY_DESC = "Govern admission of critical skills before any claim."
DISCRETIONARY_NAME = "kubernetes-rollout-debug"
DISCRETIONARY_DESC = "Debug kubernetes rollout failures and pod crashloops."
GOAL = "debug kubernetes rollout pod crashloop"

_FILLER_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")


def _filler_desc(i: int) -> str:
    # Lexically disjoint from GOAL so only the planted discretionary skill can match.
    return f"Filler capability {_FILLER_WORDS[i % 10]} {_FILLER_WORDS[(i // 10) % 10]} number {i:03d} for archive."


def catalog_names() -> list[str]:
    names = [f"skill-{i:03d}" for i in range(1, CATALOG_SIZE + 1)]
    names[MANDATORY_INDEX - 1] = MANDATORY_NAME
    names[DISCRETIONARY_INDEX - 1] = DISCRETIONARY_NAME
    return names


def build_stock_index() -> str:
    """Stock-shaped index: 4-space rows grouped into categories (the row form Nerve may strip)."""
    rows = []
    for idx, name in enumerate(catalog_names(), start=1):
        if name == MANDATORY_NAME:
            desc = MANDATORY_DESC
        elif name == DISCRETIONARY_NAME:
            desc = DISCRETIONARY_DESC
        else:
            desc = _filler_desc(idx)
        rows.append((idx, name, desc))
    lines = []
    for cat in range(0, CATALOG_SIZE, 41):
        lines.append(f"  category-{cat // 41:02d}:")
        lines.extend(f"    - {n}: {d}" for _, n, d in rows[cat:cat + 41])
    return "<available_skills>\n" + "\n".join(lines) + "\n</available_skills>"


def settings(**extra):
    base = {
        "selective_context_enabled": True,
        "selective_context_skills_mode": "apply",
        "selective_context_mandatory_skills": [MANDATORY_NAME],
    }
    base.update(extra)
    return base


def _decision(plan: dict, name: str) -> dict:
    hits = [d for d in plan["decisions"] if d["name"] == name]
    assert len(hits) == 1, f"expected exactly one decision for {name}, got {hits}"
    return hits[0]


def _row_kept(rendered: str, name: str, desc: str) -> bool:
    return f"    - {name}: {desc}" in rendered


def test_fixture_is_a_realistic_410_skill_catalog():
    stock = build_stock_index()
    names = catalog_names()
    assert len(names) == CATALOG_SIZE
    assert len(stock) >= 18_000
    assert names[MANDATORY_INDEX - 1] == MANDATORY_NAME and MANDATORY_INDEX > 128
    assert names[DISCRETIONARY_INDEX - 1] == DISCRETIONARY_NAME and DISCRETIONARY_INDEX > 128
    assert stock.count("    - ") == CATALOG_SIZE


def test_skills_controller_default_candidate_cap_is_512():
    cfg = _controller({"selective_context_skills_mode": "apply"}, "skills").config.skills
    assert cfg.max_candidates == 512
    assert cfg.max_admitted == 5
    assert cfg.min_score == pytest.approx(0.22)


def test_mandatory_skill_at_index_350_is_pinned_and_keeps_description():
    stock = build_stock_index()
    rendered, plan = plan_skills_index(stock, GOAL, settings())
    assert plan["fail_open"] is False and plan["mode"] == "apply"
    decision = _decision(plan, MANDATORY_NAME)
    assert decision["action"] == "pin"
    assert decision["reason_code"] == "MANDATORY"
    assert _row_kept(rendered, MANDATORY_NAME, MANDATORY_DESC)
    # Backward-compatible string seam returns the identical rendering.
    assert select_skills_index(stock, GOAL, settings()) == rendered


@pytest.mark.parametrize("index", [1, 128, 129, 300, 350, 410])
def test_mandatory_skill_is_retained_regardless_of_catalog_index(index):
    names = catalog_names()
    target = names[index - 1]
    stock = build_stock_index()
    rendered = select_skills_index(stock, GOAL, settings(selective_context_mandatory_skills=[target]))
    kept = [line for line in rendered.splitlines() if line.startswith(f"    - {target}: ")]
    assert len(kept) == 1, f"mandatory {target} at index {index} lost its description"


def test_discretionary_skill_at_index_300_is_admitted_by_direct_task_match():
    stock = build_stock_index()
    rendered, plan = plan_skills_index(stock, GOAL, settings())
    decision = _decision(plan, DISCRETIONARY_NAME)
    assert decision["action"] == "admit"
    assert decision["reason_code"] == "DIRECT_TASK_MATCH"
    assert decision["score"] >= 0.22
    assert _row_kept(rendered, DISCRETIONARY_NAME, DISCRETIONARY_DESC)
    # Everything else past the pin/admit set is demoted to a bare name, never dropped.
    for name in catalog_names():
        assert f"    - {name}" in rendered
    assert rendered.count(": ") == 2  # exactly the pinned + admitted rows keep a description


def test_old_128_cap_reproduces_candidate_cap_regression():
    """Guard against a tautological green: with the legacy cap the same skill is rejected by CANDIDATE_CAP."""
    stock = build_stock_index()
    rendered, plan = plan_skills_index(stock, GOAL, settings(selective_context_skills_max_candidates=128))
    decision = _decision(plan, DISCRETIONARY_NAME)
    assert decision["reason_code"] == "CANDIDATE_CAP"
    assert not _row_kept(rendered, DISCRETIONARY_NAME, DISCRETIONARY_DESC)
    # Mandatory pins are cap-independent even under the legacy bound.
    assert _decision(plan, MANDATORY_NAME)["reason_code"] == "MANDATORY"


def test_mandatory_skills_accept_tuple_and_comma_separated_string():
    stock = build_stock_index()
    for value in ((MANDATORY_NAME,), f" {MANDATORY_NAME} , skill-001 "):
        rendered, plan = plan_skills_index(stock, GOAL, settings(selective_context_mandatory_skills=value))
        assert plan["fail_open"] is False
        assert _decision(plan, MANDATORY_NAME)["reason_code"] == "MANDATORY"
        assert _row_kept(rendered, MANDATORY_NAME, MANDATORY_DESC)


def test_plan_skills_index_fails_open_on_invalid_input():
    stock = build_stock_index()
    rendered, plan = plan_skills_index(stock, "", settings())
    assert rendered == stock and plan["fail_open"] is True
    rendered, plan = plan_skills_index(stock, GOAL, settings(selective_context_skills_mode="off"))
    assert rendered == stock and plan["fail_open"] is True


def _config(**nerve_settings):
    return {"plugins": {"entries": {"nerve": {"settings": {"nerve_profile": "operator", **nerve_settings}}}}}


def test_settings_from_config_forwards_candidate_cap_and_mandatory_skills():
    with tempfile.TemporaryDirectory() as home:
        effective = settings_from_config(
            _config(selective_context_mandatory_skills=f"{MANDATORY_NAME}, tdd"), home=home)
        assert effective["selective_context_enabled"] is True
        assert effective["selective_context_skills_mode"] == "apply"
        assert effective["selective_context_skills_max_candidates"] == 512
        assert effective["selective_context_mandatory_skills"] == [MANDATORY_NAME, "tdd"]

        explicit = settings_from_config(
            _config(selective_context_skills_max_candidates=256,
                    selective_context_mandatory_skills=["a", "b"]), home=home)
        assert explicit["selective_context_skills_max_candidates"] == 256
        assert explicit["selective_context_mandatory_skills"] == ["a", "b"]
        assert _controller(explicit, "skills").config.skills.max_candidates == 256
