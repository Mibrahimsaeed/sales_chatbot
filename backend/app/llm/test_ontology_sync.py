"""
Enforces the invariant stated at the top of metric_ontology.py: every
(metric, level) pair a MetricDef declares in entity_levels must have a
matching ColumnBinding in that same MetricDef's `bindings` dict.

This used to check metric_ontology.py against a SEPARATE registry in
sql_generator.py — two places that had to be kept in sync, which is
exactly the class of bug this test existed to catch (ontology renamed to
business-friendly keys, resolvers left on the old names). Per the
redesign's Phase 1, bindings now live inside the same MetricDef as
entity_levels, so there is one source of truth and this test is really
just checking the dict is internally consistent — kept anyway since
that's a one-line typo away from silently breaking a metric.
"""

from app.llm.metric_ontology import METRICS


def test_every_declared_entity_level_has_a_binding():
    missing = [
        (metric.key, level)
        for metric in METRICS.values()
        for level in metric.entity_levels
        if level not in metric.bindings
    ]
    assert not missing, f"Ontology declares entity_levels with no matching binding: {missing}"


def test_no_binding_references_a_level_not_in_entity_levels():
    """Catches the opposite drift: a binding left behind after a level was
    removed from entity_levels."""
    orphaned = [
        (metric.key, level)
        for metric in METRICS.values()
        for level in metric.bindings
        if level not in metric.entity_levels
    ]
    assert not orphaned, f"Bindings exist for levels no longer declared in entity_levels: {orphaned}"


def test_primary_level_always_has_a_binding():
    missing = [metric.key for metric in METRICS.values() if metric.primary_level not in metric.bindings]
    assert not missing, f"Metrics whose primary_level has no binding: {missing}"
