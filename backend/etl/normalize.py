"""
Canonical normalization for org/person name values (Area 3 of the
reliability work).

The problem this solves, straight from the live-database audit: the same
real-world entity was stored under several spellings that differ only in
whitespace or casing, so every GROUP BY / gazetteer / leaderboard silently
split it into two:

    "Agency21- HQ Islamabad"  (111 advisors)
    "Agency21-HQ Islamabad"   ( 61 advisors)   <- same office, undercounted

    "adeel dogar"  vs  "Ali Murtaza"           <- inconsistent person casing

Applied at WRITE time (etl/transform.py's `_clean`, the single choke point
every org field already flows through) rather than at read time, so the
database holds one canonical spelling and every consumer — the compiler's
GROUP BY, entity_extractor's gazetteer, hierarchy rollups — agrees without
each needing its own fuzzy pass.

Deliberately conservative. These functions only fix *mechanical* variance
(whitespace, separator spacing, casing of clearly-lowercase names). They
never merge values that differ in their actual characters — "Graana-RO RWP"
and "Graana RO Rawalpindi" stay distinct, because deciding those are the
same place is a data-ownership call, not a string-processing one. Anything
this module can't safely merge is reported instead by etl/validation.py's
near-duplicate check, for a human to resolve at the source sheet.
"""

from __future__ import annotations

import re

# Runs of any whitespace (including non-breaking spaces pasted from Sheets)
# collapse to exactly one plain space.
_WHITESPACE_RUN = re.compile(r"[\s ]+")

# " - " / "- " / " -" around a hyphen that joins two name parts all collapse
# to a single "-", which is what fixes the Agency21 split above. Applied
# only to hyphens that sit BETWEEN word characters so a legitimately spaced
# dash in prose ("Region - North") isn't silently reflowed into a compound.
_HYPHEN_SPACING = re.compile(r"(?<=\w)\s*-\s*(?=\w)")

# Name particles that stay lowercase when title-casing a person name, and
# prefixes that keep their following letter capitalized.
_LOWERCASE_PARTICLES = {"bin", "binte", "bint", "al", "ul", "ur", "e", "de", "da", "van", "von"}
_ACRONYMS = {"hq", "ro", "moi", "bc", "dha", "kpk", "gcc", "amd", "npr", "ccmc", "gro", "uk", "usa"}


def normalize_org_name(value: str | None) -> str | None:
    """Canonical form for an ORG-unit name (office/business center, team,
    company, region). Whitespace and separator spacing only — casing is
    left exactly as the source wrote it, because org names carry
    intentional capitalization ("Mall of Imarat (MOI)", "Graana-RO RWP")
    that a blanket title-case would damage."""
    if value is None:
        return None
    cleaned = _WHITESPACE_RUN.sub(" ", str(value)).strip()
    if not cleaned:
        return None
    return _HYPHEN_SPACING.sub("-", cleaned)


def normalize_person_name(value: str | None) -> str | None:
    """Canonical form for a PERSON name (advisor, unit head/bm, zonal
    head/zm, rm, portfolio lead, management lead). Same whitespace handling
    as org names, plus casing repair for the all-lowercase and ALL-CAPS
    entries the audit found sitting next to correctly-cased ones.

    A name that is already mixed-case is left ALONE — "McDonald",
    "O'Brien", "bin Rashid" and similar are correct as typed, and
    re-casing them would be a regression, so only the unambiguously
    machine-mangled cases (all-lower / all-upper) get rewritten."""
    if value is None:
        return None
    cleaned = _WHITESPACE_RUN.sub(" ", str(value)).strip()
    if not cleaned:
        return None
    cleaned = _HYPHEN_SPACING.sub("-", cleaned)

    letters = [c for c in cleaned if c.isalpha()]
    if not letters:
        return cleaned
    is_all_lower = all(c.islower() for c in letters)
    is_all_upper = all(c.isupper() for c in letters)
    if not (is_all_lower or is_all_upper):
        return cleaned

    return " ".join(_titlecase_token(tok, i) for i, tok in enumerate(cleaned.split(" ")))


def _titlecase_token(token: str, index: int) -> str:
    if not token:
        return token
    bare = token.strip(".,")
    if bare.lower() in _ACRONYMS:
        return token.upper()
    # a non-leading particle ("bin", "ul") stays lowercase
    if index > 0 and bare.lower() in _LOWERCASE_PARTICLES:
        return token.lower()
    # hyphenated compounds capitalize on both sides ("abdul-rehman")
    return "-".join(part[:1].upper() + part[1:].lower() if part else part for part in token.split("-"))


# Which normalizer each Advisor field uses. transform.py reads this so
# adding a field is one entry here, not a new branch in the ETL loop.
ORG_FIELDS = ("company", "region", "team", "office", "unit")
PERSON_FIELDS = ("name", "portfolio_lead", "management_lead", "bm", "zm", "rm")


def normalize_field(key: str, value):
    """Dispatch by Advisor field name. A field in neither list passes
    through untouched — normalization is opt-in per field, never applied
    blindly to arbitrary values (numbers, statuses, timestamps)."""
    if key in ORG_FIELDS:
        return normalize_org_name(value)
    if key in PERSON_FIELDS:
        return normalize_person_name(value)
    return value


def normalization_key(value: str | None) -> str:
    """An aggressive comparison key used ONLY to FIND near-duplicates
    (etl/validation.py), never to write a value. Strips every non-
    alphanumeric character and lowercases, so "Agency21- HQ Islamabad" and
    "agency21 hq  islamabad" collapse to the same key for reporting
    purposes even though normalize_org_name deliberately won't merge them
    on its own."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())
