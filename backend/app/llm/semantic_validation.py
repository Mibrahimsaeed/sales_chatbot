"""Validation: is this interpretation executable, as stated?

    semantic model -> entity grounding -> hierarchy grounding -> VALIDATION

The rule this module exists to enforce:

    VALIDATION MUST NOT SILENTLY REINTERPRET USER INTENT.

So it has no repair path. Nothing here edits a metric, retypes a subject,
substitutes a level or drops a condition to make a query run. It reads
the interpretation and the two grounding reports and returns a verdict.
An interpretation that cannot be executed as stated is REJECTED or sent
for CLARIFICATION — both of which leave the user's meaning intact — and
never quietly turned into a different query that happens to work.

That distinction is the whole point. A rewritten query returns a
well-formed answer to a question nobody asked, and there is no signal in
the response that it happened. A rejection is visible.

WHICH FAILURES ASK AND WHICH REFUSE:

    reject   nothing to choose between — the metric does not exist, the
             operation is not supported, the relationship is not one this
             data can express. Asking would be theatre.
    clarify  several readings are genuinely open — an ambiguous name, a
             stated type that exists as something else, a required slot
             the query never filled. The user can settle it in a word.

Both are answers. Neither is a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.llm import grounding as entity_grounding
from app.llm import hierarchy_grounding, operations
from app.llm.metric_ontology import METRICS
from app.llm.semantic_model import SemanticModel

VALID = "valid"
INVALID = "invalid"
NEEDS_CLARIFICATION = "needs_clarification"

REJECT = "reject"
CLARIFY = "clarify"

# The eight checks, named so a verdict says which one spoke.
SCHEMA = "schema"
OPERATION = "operation"
METRIC = "metric"
ENTITY = "entity"
ENTITY_TYPE = "entity_type"
HIERARCHY = "hierarchy"
REQUIRED_FIELDS = "required_fields"
AUTHORIZATION = "authorization"


@dataclass(frozen=True)
class Finding:
    """One reason an interpretation is not executable as stated."""
    check: str
    severity: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict:
        return {"check": self.check, "severity": self.severity,
                "message": self.message, "field": self.field}


@dataclass
class Verdict:
    findings: list[Finding] = field(default_factory=list)

    @property
    def rejections(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == REJECT]

    @property
    def clarifications(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == CLARIFY]

    @property
    def status(self) -> str:
        """A rejection outranks a clarification: there is no point asking
        which Yasir Ali was meant when the metric does not exist."""
        if self.rejections:
            return INVALID
        if self.clarifications:
            return NEEDS_CLARIFICATION
        return VALID

    @property
    def is_valid(self) -> bool:
        return self.status == VALID

    @property
    def is_executable(self) -> bool:
        """Only a clean verdict may run. Deliberately not "no rejections":
        a query needing clarification has more than one meaning, and
        picking one is the thing this module forbids."""
        return self.status == VALID

    def to_dict(self) -> dict:
        return {"status": self.status,
                "findings": [f.to_dict() for f in self.findings]}


# Operations whose answer IS a measurement, so a metric is required.
# Read from the registry's dispatch modes rather than listed by hand
# where possible; the metric-bearing set is small and explicit.
_NEEDS_METRIC = frozenset({
    "leaderboard", "group_metric", "comparison", "trend", "breakdown",
})


def _check_schema(model: SemanticModel) -> list[Finding]:
    """Structural wellformedness pydantic cannot express.

    The model type itself is validated on construction, so what is left
    is agreement BETWEEN fields — a comparison with one target is
    schema-valid and meaningless.
    """
    findings: list[Finding] = []
    if model.comparison_subjects and len(model.comparison_subjects) < 2:
        findings.append(Finding(
            SCHEMA, REJECT,
            "a comparison needs at least two subjects; the interpretation names one",
            "comparison_subjects"))
    if model.relationship is not None and model.requested_level is None:
        findings.append(Finding(
            SCHEMA, CLARIFY,
            "a relationship was expressed without a level to enumerate",
            "requested_level"))
    return findings


def _check_operation(model: SemanticModel) -> list[Finding]:
    if model.operation not in operations.OPERATIONS:
        return [Finding(OPERATION, REJECT,
                        f"'{model.operation}' is not a supported operation",
                        "operation")]
    return []


def _check_metrics(model: SemanticModel) -> list[Finding]:
    """Every named measure must exist in the ontology.

    Rejected rather than clarified, and never swapped for the closest
    match: substituting a metric answers a different question with full
    confidence. Fuzzy widening belongs BEFORE this point, in parsing,
    where it is visible as a parse decision.
    """
    findings = [
        Finding(METRIC, REJECT, f"'{m.name}' is not a known metric", "metrics")
        for m in model.metrics if m.name not in METRICS
    ]
    if not model.metrics and model.operation in _NEEDS_METRIC:
        findings.append(Finding(
            REQUIRED_FIELDS, CLARIFY,
            f"'{model.operation}' reports a measure and none was named", "metrics"))
    return findings


def _check_entities(grounded: entity_grounding.Grounding) -> list[Finding]:
    """Entity validity and entity TYPE are separate checks because they
    have different answers: a name that does not exist cannot be
    clarified into existing, while a name that exists as something else
    is a question worth asking."""
    findings: list[Finding] = []

    for entity in grounded.not_found:
        findings.append(Finding(
            ENTITY, REJECT,
            f"'{entity.name}' does not exist"
            + (f" as a {entity.stated_level}" if entity.stated_level else ""),
            entity.role))

    for entity in grounded.ambiguous:
        where = ", ".join(entity.found_at) if entity.found_at else "several records"
        findings.append(Finding(
            ENTITY, CLARIFY,
            f"'{entity.name}' matches {len(entity.candidates)} records ({where})",
            entity.role))

    for entity in grounded.mismatched:
        findings.append(Finding(
            ENTITY_TYPE, CLARIFY,
            f"'{entity.name}' was read as a {entity.stated_level}, but exists as "
            f"{'/'.join(entity.found_at)}",
            entity.role))

    return findings


def _check_read_is_supported(model: SemanticModel,
                             entities: dict | None) -> list[Finding]:
    """A hierarchy read must be something the query actually asked for.

    PHASE 11. `SemanticModel.requested_level` is documented as "the level
    the user EXPLICITLY asked to see — set only when they said it, never
    inferred from a subject having members". A model that enumerates
    advisors for "connects of Blue Area" has broken that contract: the
    sentence names no level and no relationship, so there is nothing to
    enumerate.

    ir_validator used to SILENTLY REWRITE this — nulling target_level,
    subject_of and relation behind the model's back. That produced the
    right answer and no signal that anything had happened, so a parser
    regression looked exactly like a parser working. This reports the
    conflict instead, from the same deterministic facts extraction
    already produced: whether the turn contained a level noun
    (`level_word`) or a relationship phrase (`relation_word`).

    It is a CLARIFY, not a rejection: the query is answerable, the two
    readings are both coherent, and which one was meant is a question a
    person can settle in a word.
    """
    if entities is None or not model.is_hierarchy_query():
        return []
    if entities.get("level_word") or entities.get("relation_word"):
        return []
    return [Finding(
        HIERARCHY, CLARIFY,
        f"the interpretation enumerates {model.requested_level or 'a level'} "
        "beneath the named entity, but the query names neither a level nor a "
        "relationship — it may be asking for that entity's own figure",
        "requested_level")]


def _check_hierarchy(result: hierarchy_grounding.HierarchyGrounding) -> list[Finding]:
    """An UNSUPPORTED relationship is rejected; an EMPTY one is valid.

    "No BCMs work under her" is a true statement about the organisation,
    and turning it into an error — or quietly descending to a level that
    does have members — would both misreport it.
    """
    if result.status == hierarchy_grounding.UNSUPPORTED:
        return [Finding(HIERARCHY, REJECT,
                        result.reason or "the requested relationship is not supported",
                        "relationship")]
    return []


def _check_required_fields(model: SemanticModel,
                           result: hierarchy_grounding.HierarchyGrounding) -> list[Finding]:
    """Slots the operation needs and the query never filled.

    `model.missing` is what the parser itself flagged; this adds what the
    shape requires. Both are clarifications — a missing slot has an
    answer the user can give.
    """
    findings = [
        Finding(REQUIRED_FIELDS, CLARIFY, f"'{slot}' was not specified", slot)
        for slot in model.missing
    ]
    if result.is_hierarchy and result.target_level is None:
        findings.append(Finding(
            REQUIRED_FIELDS, CLARIFY,
            "a hierarchy read needs a level to enumerate", "requested_level"))
    return findings


def _check_authorization(model: SemanticModel, principal) -> list[Finding]:
    """DECLARED, NOT DECIDED.

    There is no authorization policy in this system to enforce: tokens
    carry a `role` claim and nothing consumes it, and the posture — which
    roles may read which levels — was raised as an open decision and
    never settled. Inventing one here would be exactly the silent
    reinterpretation the rest of this module refuses, and it would be
    invisible: queries would start returning less with no stated reason.

    So this check exists, runs, and passes with no principal. When a
    policy is decided, it goes here, and every caller already routes
    through it.
    """
    if principal is None:
        return []
    return []


def validate(model: SemanticModel,
             grounded: entity_grounding.Grounding,
             hierarchy_result: hierarchy_grounding.HierarchyGrounding,
             db: Session | None = None,
             *, principal=None, entities: dict | None = None) -> Verdict:
    """Run every check and report all of them.

    Deliberately NOT short-circuiting on the first failure: a user with
    two problems should be told both, and a trace that stops at the first
    one hides the rest of the picture.
    """
    findings: list[Finding] = []
    findings += _check_schema(model)
    findings += _check_operation(model)
    findings += _check_metrics(model)
    findings += _check_entities(grounded)
    findings += _check_hierarchy(hierarchy_result)
    findings += _check_read_is_supported(model, entities)
    findings += _check_required_fields(model, hierarchy_result)
    findings += _check_authorization(model, principal)
    return Verdict(findings=findings)
