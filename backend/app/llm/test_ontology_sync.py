"""
Enforces the invariant stated at the top of metric_ontology.py: every
(metric, level) pair a MetricDef declares in entity_levels must have a
matching resolver in sql_generator.py. This is exactly the class of bug
reported — ontology renamed to business-friendly keys, resolvers left on
the old names — and it should fail CI, not get discovered via a chat
query returning suspiciously flat data.
"""

from app.llm.metric_ontology import METRICS
from app.llm.sql_generator import RESOLVERS


def test_every_declared_metric_level_has_a_resolver():
    missing = [
        (metric.key, level)
        for metric in METRICS.values()
        for level in metric.entity_levels
        if (metric.key, level) not in RESOLVERS
    ]
    assert not missing, f"Ontology declares entity_levels with no matching resolver: {missing}"


def test_no_resolver_references_a_metric_not_in_the_ontology():
    """Catches the opposite drift: a resolver left behind after a metric
    was renamed or removed from the ontology."""
    known_keys = set(METRICS.keys())
    orphaned = [key for (key, _level) in RESOLVERS if key not in known_keys]
    assert not orphaned, f"Resolvers exist for metrics no longer in the ontology: {orphaned}"