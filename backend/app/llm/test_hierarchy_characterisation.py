"""Characterisation of the hierarchy (written BEFORE the Phase 3 rebind).

Records the pre-rebind wiring so the change is visible rather than
discovered. Phase 1 proved this wiring structurally impossible against
production data — 100% of teams span multiple offices, and the level
called "unit head" had 138 members sitting above a "zonal head" level of
59 — so most of what is pinned here is expected to change.

What must NOT change is the CAPABILITY set: every level that could be
filtered, grouped, ranked or broken down before must still be, and every
phrase a user could say must still reach a level. Those are pinned
separately at the bottom and are the real regression guard.
"""

import pytest

from app.database.models import Advisor
from app.llm import hierarchy


# ---------------------------------------------------------------------
# RETIRED (Phase 3 landed).
#
# This file also pinned the pre-rebind wiring: unit_head -> Advisor.bm,
# zonal_head -> Advisor.zm, business_center -> Advisor.office, team as
# the level directly above advisor, and a `unit` level with zero
# production rows. All five were written to fail once the verified chain
# replaced them, and all five did. The post-rebind state is asserted in
# test_hierarchy.py and test_hierarchy_traversal.py.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# CAPABILITIES — these must survive the rebind unchanged
# ---------------------------------------------------------------------

CAPABILITY_LEVELS = ("team", "company", "unit_head", "zonal_head", "advisor")


@pytest.mark.parametrize("level", CAPABILITY_LEVELS)
def test_every_capability_level_stays_addressable(level):
    """Whatever each level BINDS to, it must remain a real level with a
    column, a label and a keyword vocabulary."""
    assert hierarchy.column_for(level) is not None, level
    assert hierarchy.label_for(level), level
    assert hierarchy.LEVEL_KEYWORDS.get(level), level


@pytest.mark.parametrize("phrase", [
    "unit head", "zonal head", "zone head", "team", "company", "advisor",
])
def test_user_vocabulary_still_reaches_some_level(phrase):
    """A phrase that resolved to a level before must still resolve to
    one — the level it names may be rebound, but it must not vanish."""
    hit = [lvl for lvl, words in hierarchy.LEVEL_KEYWORDS.items() if phrase in words]
    assert hit, phrase


def test_reverse_role_lookup_covers_the_manager_words():
    from app.llm import relations

    aliases = dict(relations.role_alias_pairs())
    for word in ("bm", "zm", "rm", "unit head", "zonal head"):
        assert word in aliases, word


def test_group_levels_exclude_advisor():
    assert "advisor" not in hierarchy.GROUP_LEVELS


def test_every_level_is_declared_in_every_table():
    """The invariant that makes a level usable end to end."""
    for level in hierarchy.HIERARCHY_LEVELS:
        assert level in hierarchy.LEVEL_COLUMNS, level
        assert level in hierarchy.LEVEL_LABELS, level
        assert level in hierarchy.LEVEL_KEYWORDS, level
        assert level in hierarchy.PARENT_LEVEL, level
        if level != "advisor":
            assert level in hierarchy.LEVEL_ENTITY_KEYS, level
            assert level in hierarchy.LEVEL_MATCH_KIND, level
