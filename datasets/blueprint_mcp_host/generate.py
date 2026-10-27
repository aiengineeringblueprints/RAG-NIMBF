#!/usr/bin/env python3
"""Generate the blueprint_mcp_host corpus and questions deterministically.

Ground truths for MCP questions are computed by the same functions the MCP
fixture server exposes (``fixture_data.TOOLS``), so labels and server cannot
drift apart. Candidate questions are filtered (document answers must occur in
their document; tool answers must not occur anywhere in the corpus) and then
sampled per category and sub-type with a fixed seed.

Run from anywhere: ``python datasets/blueprint_mcp_host/generate.py``.
"""

from __future__ import annotations

import json
import re
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))
import fixture_data as data
import text_bank as TB
from benchmark.answer_match import answer_matches_normalized, normalize_answer

CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "questions.jsonl"

CATEGORY_TARGETS = {
    "doc_single": 300,
    "doc_multi": 100,
    "mcp_single": 250,
    "mcp_chain": 100,
    "mixed": 250,
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def long_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


# --- documents -------------------------------------------------------------

FAMILY_DOC = {prefix: f"KI-{51 + i:03d}" for i, (prefix, *_rest) in enumerate(data.FAMILIES)}
PRODUCT_DOC = {code: f"KI-{i:03d}" for i, code in enumerate(data.PRODUCTS, start=1)}
CUSTOMER_BY_NAME = {c["name"]: c for c in data.CUSTOMERS}
# Products documented as full installation and operating manuals.
MANUALS = set(random.Random("manuals").sample(sorted(data.PRODUCTS), 10))
# Filler dates must not collide with MCP-only dates (restock dates are matched alone).
RESERVED_DATES = {e["restock_date"] for e in data.STOCK.values() if e["restock_date"]}
TIERS = list(data.TIERS)  # Bronze, Silver, Gold, Platinum


def _index(items, key):
    out = defaultdict(list)
    for item in items:
        out[item[key]].append(item)
    return out


PROJECTS_BY_CUSTOMER = _index(data.PROJECTS, "customer")
PROJECTS_BY_PRODUCT = _index(data.PROJECTS, "product")
INCIDENTS_BY_CUSTOMER = _index(data.INCIDENTS, "customer")
INCIDENTS_BY_PRODUCT = _index(data.INCIDENTS, "product")


def project_title(p: dict) -> str:
    return f"Project Brief {p['code']}: {p['name']}"


def incident_title(i: dict) -> str:
    return f"Incident Report {i['id']}"


def datasheet_title(code: str) -> str:
    p = data.PRODUCTS[code]
    return f"{p['name']} {'Installation and Operating Manual' if code in MANUALS else 'Datasheet'}"


def subject_of(title: str) -> str:
    """Short noun phrase for a document title, used in closing paragraphs."""
    short = title.split(" (")[0]
    for pattern, replacement in [
        (r"^Project Brief (PRJ-\d+):.*$", r"project \1"),
        (r"^Incident Report (.+)$", r"incident \1"),
        (r"^(Kestrel \S+) (Datasheet|Installation and Operating Manual|Discontinuation Notice)$", r"the \1"),
        (r"^(\w+) Series Firmware Release Notes$", r"\1 series firmware"),
        (r"^(?:Customer|Supplier) Profile: (.+)$", r"\1"),
        (r"^Training Course: (.+)$", r"the course \1"),
    ]:
        if re.match(pattern, short):
            return re.sub(pattern, replacement, short)
    return f"the {short}"


class Doc:
    """Markdown builder with a per-document random generator."""

    def __init__(self, doc_id: str, doc_kind: str, title: str, /, **names: str) -> None:
        self.id, self.kind = doc_id, doc_kind
        self.rng = random.Random(f"doc-{doc_id}")
        # Documents differ in how thorough they are, like real ones.
        self.verbosity = self.rng.choice(["brief", "normal", "detailed", "detailed"])
        self.names = names
        self.subject = subject_of(title)
        self.parts = [f"# {title}", self.rng.choice([
            "Kestrel Instruments, internal document.",
            "Kestrel Instruments. Internal use only.",
            "Owner: Kestrel Instruments. Classification: internal.",
        ])]

    def fill(self, template: str) -> str:
        return template.format(**self.names)

    def h2(self, title: str) -> None:
        self.parts.append(f"## {title}")

    def heading(self, *options: str) -> None:
        """Section heading with a name chosen from synonyms per document."""
        self.h2(self.rng.choice(options))

    def sections(self, *builders) -> None:
        """Run optional section builders in a per-document random order."""
        keep = {"brief": 0.4, "normal": 0.75, "detailed": 1.0}[self.verbosity]
        chosen = [b for b in builders if self.rng.random() < keep]
        self.rng.shuffle(chosen)
        for build in chosen:
            build()

    def related(self, titles: list[str]) -> None:
        if titles:
            self.heading("Related documents", "See also", "References")
            self.bullets(titles)

    def h3(self, title: str) -> None:
        self.parts.append(f"### {title}")

    def para(self, text: str) -> None:
        self.parts.append(text)

    def prose(self, bank: list[str], low: int, high: int) -> None:
        """Add ``low``..``high`` paragraphs drawn from ``bank`` (each used once)."""
        if self.verbosity == "brief":
            high = low
        elif self.verbosity == "detailed":
            low = high
        count = min(len(bank), self.rng.randint(low, high))
        for template in self.rng.sample(bank, count):
            self.para(self.fill(template))

    def bullets(self, items: list[str], numbered: bool = False) -> None:
        self.parts.append("\n".join(
            f"{i}. {item}" if numbered else f"- {item}" for i, item in enumerate(items, start=1)))

    def pick(self, bank: list, low: int, high: int) -> list:
        return self.rng.sample(bank, min(len(bank), self.rng.randint(low, high)))

    def table(self, header: list[str], rows: list[list[str]]) -> None:
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
        self.parts.append("\n".join(lines))

    def key_values(self, pairs: list[tuple[str, str]]) -> None:
        """Render key facts as a table, a bullet list or prose, chosen per document."""
        style = self.rng.choice(["table", "table", "list", "prose"])
        if style == "table":
            self.table(["Item", "Value"], [[k, v] for k, v in pairs])
        elif style == "list":
            self.bullets([f"{k}: {v}" for k, v in pairs])
        else:
            self.para(" ".join(f"{k}: {v}." for k, v in pairs))

    def faq(self, pairs: list[tuple[str, str]]) -> None:
        for question, answer in pairs:
            self.parts.append(f"**Q:** {self.fill(question)}\n**A:** {self.fill(answer)}")

    def date_between(self, start: str, end: str) -> str:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        span = max(1, (last - first).days)
        while True:
            day = (first + timedelta(days=self.rng.randrange(span))).isoformat()
            if day not in RESERVED_DATES:
                return day

    def date_text(self, iso: str) -> str:
        return self.rng.choice([iso, long_date(iso)])

    def render(self) -> str:
        count = {"brief": 1, "normal": 2, "detailed": 3}[self.verbosity]
        about = [t.format(subject=self.subject) for t in self.rng.sample(TB.GENERAL, count)]
        closing = [self.rng.choice(["## About this document", "## Document information",
                                    "## Document control"]), *about]
        text = "\n\n".join(self.parts + closing) + "\n"
        return text.replace("Inc..", "Inc.")  # company names ending a sentence


def datasheet_doc(code: str, p: dict) -> Doc:
    manual = code in MANUALS
    d = Doc(PRODUCT_DOC[code], "datasheet", datasheet_title(code), name=p["name"], kind=p["kind"])
    if manual:
        d.verbosity = "detailed"
    bank = TB.DATASHEET
    many = (3, 6) if manual else (1, 2)
    d.heading("Overview", "Product overview", "Introduction")
    d.para(d.rng.choice([
        f"The {p['name']} is {article(p['kind'])} {p['kind']} sold under SKU {p['sku']}. It was launched in {p['launch_year']}.",
        f"Product overview: the {p['name']} ({p['kind']}) is ordered with SKU {p['sku']} and has been on the market since it was launched in {p['launch_year']}.",
    ]))
    d.prose(bank["overview"], *many)
    if manual:
        d.heading("Safety instructions", "Safety", "For your safety")
        d.prose(bank["safety"], 4, 5)
    d.heading("Technical specifications", "Specifications", "Technical data")
    d.key_values([
        ("Minimum operating temperature", f"{p['min_c']} °C"),
        ("Maximum operating temperature", f"{p['max_c']} °C"),
        ("Supply current", f"{p['current_ma']} mA"),
        ("Communication interface", p["interface"]),
        ("Accuracy", p["accuracy"]),
        ("Enclosure rating", p["ip"]),
        ("Weight (current revision)", f"{p['weight_g']} g"),
    ])
    d.para(d.rng.choice([
        f"The {p['name']} operates between {p['min_c']} °C and {p['max_c']} °C and communicates via {p['interface']}.",
        f"Its operating temperature range is {p['min_c']} °C to {p['max_c']} °C; the supply current is {p['current_ma']} mA.",
    ]))
    d.heading("Ordering information", "Ordering", "Prices and ordering")
    d.table(["SKU", "Current list price", "Warranty"],
            [[p["sku"], f"{p['price_eur']} EUR", f"{p['warranty_years']} years"]])
    d.para(f"The {p['name']} comes with a warranty of {p['warranty_years']} years from delivery.")

    def applications():
        d.heading("Applications", "Application examples", "Typical use cases")
        d.prose(bank["applications"], *many)

    def accessories():
        d.heading("Accessories", "Optional accessories")
        d.bullets(d.pick(bank["accessories"], 3, 7))

    def compliance():
        d.heading("Certifications and compliance", "Compliance", "Approvals")
        d.prose(bank["compliance"], *((2, 4) if manual else (1, 2)))

    def storage():
        d.heading("Storage and transport", "Transport and storage")
        d.prose(bank["storage"], 1, 3)

    def troubleshooting():
        d.heading("Troubleshooting", "Fault finding", "Diagnostics and troubleshooting")
        d.table(["Symptom", "Possible cause", "Remedy"],
                [list(r) for r in d.pick(bank["troubleshooting"], 6 if manual else 3, 10 if manual else 5)])

    def faq():
        d.heading("Frequently asked questions", "FAQ", "Questions and answers")
        d.faq(d.pick(bank["faq"], 4 if manual else 2, 8 if manual else 4))

    d.sections(applications, accessories, compliance, storage)
    install = [(("Mechanical installation", "Mounting"), "mechanical"),
               (("Electrical connection", "Wiring"), "electrical")]
    if manual:
        install += [(("Commissioning", "Start-up"), "commissioning"),
                    (("Configuration", "Device configuration"), "configuration"),
                    (("Operation", "Normal operation"), "operation")]
    for names, key in install:
        d.heading(*names)
        d.prose(bank[key], *((4, 5) if manual else (1, 2)))
    if manual:
        d.heading("Configuration parameters", "Parameter reference")
        d.table(["Parameter", "Description", "Default"], [list(r) for r in d.pick(bank["parameters"], 7, 10)])
    d.heading("Maintenance", "Service and maintenance", "Care and maintenance")
    d.prose(bank["maintenance"], *((4, 5) if manual else (1, 2)))
    if manual:
        troubleshooting()
        faq()
        d.heading("Disposal", "End of life")
        d.prose(bank["disposal"], 2, 2)
    else:
        d.sections(troubleshooting, faq)
    old_price = round(p["price_eur"] * d.rng.uniform(0.78, 0.93))
    old_weight = p["weight_g"] + d.rng.randrange(10, 50, 5)
    y1 = p["launch_year"]
    d.heading("Revision history", "Change history", "Document revisions")
    d.table(["Revision", "Year", "Change"], [
        ["A", y1, f"Initial release. List price {old_price} EUR, weight {old_weight} g."],
        ["B", min(y1 + 1, 2025), f"Housing redesign; weight reduced to {p['weight_g']} g."],
        ["C", min(y1 + 2, 2026), f"Price adjustment; list price now {p['price_eur']} EUR."],
    ])
    d.related([f"{p['family']} Series Firmware Release Notes"]
              + [project_title(x) for x in PROJECTS_BY_PRODUCT[code][:2]])
    return d


def older_versions(rng: random.Random, version: str) -> list[str]:
    major, minor, patch = (int(x) for x in version.split("."))
    options = []
    if patch > 0:
        options.append(f"{major}.{minor}.{rng.randrange(patch)}")
    if minor > 0:
        options.append(f"{major}.{rng.randrange(minor)}.{rng.randrange(10)}")
    if major > 1:
        options.append(f"{rng.randrange(1, major)}.{rng.randrange(10)}.{rng.randrange(10)}")
    unique = sorted(set(options) - {version}, key=lambda v: [int(x) for x in v.split(".")], reverse=True)
    return unique[:2] or [f"{major}.{minor}.{patch}-rc"]


def release_notes_doc(prefix: str, kind: str) -> Doc:
    d = Doc(FAMILY_DOC[prefix], "release_notes", f"{prefix} Series Firmware Release Notes ({kind}s)",
            family=prefix)
    d.heading("About these release notes", "Introduction", "Scope of this document")
    d.prose(TB.RELEASE["intro"], 2, 3)

    def procedure():
        d.heading("Update procedure", "How to update", "Installing updates")
        d.prose(TB.RELEASE["procedure"], 2, 4)

    def compatibility():
        d.heading("Compatibility", "Compatibility notes")
        d.prose(TB.RELEASE["compatibility"], 1, 3)

    def security():
        d.heading("Security", "Security information")
        d.prose(TB.RELEASE["security"], 1, 3)

    def known_issues():
        d.heading("Known issues", "Known limitations")
        d.bullets(d.pick(TB.RELEASE["known_issues"], 2, 5))

    d.sections(procedure, compatibility, security, known_issues)
    d.heading("Firmware history", "Release history", "Versions")
    for p in data.PRODUCTS.values():
        if p["family"] != prefix:
            continue
        d.h3(p["name"])
        rows = [[p["firmware"], d.date_text(p["firmware_date"]), f"Latest release; {p['firmware_change']}."]]
        newest = date.fromisoformat(p["firmware_date"])
        for i, old in enumerate(older_versions(d.rng, p["firmware"]), start=1):
            when = (newest - timedelta(days=90 * i + d.rng.randrange(60))).isoformat()
            rows.append([old, d.date_text(when), d.rng.choice(data.FEATURES).capitalize() + "."])
        d.table(["Version", "Release date", "Notes"], rows)
        d.para(d.rng.choice([
            f"The latest firmware for the {p['name']} is {p['firmware']}, released on {long_date(p['firmware_date'])}.",
            f"{p['name']}: install firmware {p['firmware']} (released {p['firmware_date']}), the latest release.",
        ]))
    return d


def policy_doc(doc_id: str, title: str, facts: list[str], superseded: bool) -> Doc:
    d = Doc(doc_id, "policy", title, title=title.split(" (")[0])
    if superseded:
        d.para("**Status: superseded.** This version is archived and no longer valid.")
    d.heading("Purpose", "Purpose of this policy", "Why this policy exists")
    d.prose(TB.POLICY["purpose"], 1, 3)

    def background():
        d.heading("Background", "Context")
        d.prose(TB.POLICY["background"], 1, 3)

    def scope():
        d.heading("Scope", "Who this applies to", "Applicability")
        d.prose(TB.POLICY["scope"], 1, 3)

    def definitions():
        d.heading("Definitions", "Terms used")
        d.bullets([f"**{term}**: {meaning}" for term, meaning in d.pick(TB.POLICY["definitions"], 3, 6)])

    d.sections(background, scope, definitions)
    d.heading("Rules", "Policy statements", "Requirements", "Policy")
    if d.rng.random() < 0.5:
        d.bullets(facts, numbered=True)
    else:
        d.para(" ".join(facts))

    def procedure():
        d.heading("Procedure", "How to apply this policy", "Process")
        d.bullets([d.fill(step) for step in d.pick(TB.POLICY["procedure"], 3, 6)], numbered=True)

    def responsibilities():
        d.heading("Responsibilities", "Roles and responsibilities")
        if d.rng.random() < 0.5:
            d.bullets([d.fill(t) for t in d.pick(TB.POLICY["responsibilities"], 2, 4)])
        else:
            d.prose(TB.POLICY["responsibilities"], 2, 4)

    def exceptions():
        d.heading("Exceptions", "Deviations")
        d.prose(TB.POLICY["exceptions"], 1, 3)

    def faq():
        d.heading("Frequently asked questions", "FAQ")
        d.faq(d.pick(TB.POLICY["faq"], 2, 4))

    def contacts():
        d.heading("Contacts", "Questions and contacts")
        d.prose(TB.POLICY["contacts"], 1, 2)

    d.sections(procedure, responsibilities, exceptions, faq, contacts)
    others = [t for _i, t, _f in data.POLICIES if t != title]
    d.related(d.pick(others, 1, 3))
    return d


def site_doc(site: dict) -> Doc:
    d = Doc(site["id"], "site", site["name"], name=site["name"], city=site["city"], country=site["country"])
    d.heading("Overview", "About the site", "Introduction")
    d.para(f"The {site['name']} in {site['city']}, {site['country']}, is headed by {site['head']}, "
           f"{site['head_title']}.")
    d.prose(TB.SITE["overview"], 2, 3)
    d.heading("Key facts", "At a glance", "Facts and figures")
    if d.rng.random() < 0.5:
        d.bullets(site["facts"])
    else:
        d.para(" ".join(site["facts"]))

    def history():
        d.heading("History", "Site history")
        d.prose(TB.SITE["history"], 2, 3)

    def teams():
        d.heading("Teams on site", "Who works here")
        d.bullets(d.pick(TB.SITE["teams"], 3, 6))

    def facilities():
        d.heading("Facilities", "Amenities")
        d.bullets(d.pick(TB.SITE["facilities"], 3, 6))

    def visitors():
        d.heading("Visitors", "Visiting the site")
        d.prose(TB.SITE["visitors"], 2, 3)

    def safety():
        d.heading("Safety and emergencies", "Emergency information")
        d.prose(TB.SITE["safety"], 2, 3)

    def sustainability():
        d.heading("Sustainability", "Environment")
        d.prose(TB.SITE["sustainability"], 1, 3)

    def getting_there():
        d.heading("Getting there", "Directions")
        d.prose(TB.SITE["getting_there"], 1, 2)

    d.sections(history, teams, facilities, visitors, safety, sustainability, getting_there)
    return d


def customer_doc(c: dict) -> Doc:
    d = Doc(c["doc"], "customer", f"Customer Profile: {c['name']}", customer=c["name"],
            industry=c["industry"], country=c["country"], article=article(c["industry"]))
    d.heading("Account summary", "Customer overview", "Key account data")
    d.key_values([
        ("Customer ID", c["id"]), ("Company", c["name"]), ("Country", c["country"]),
        ("Industry", c["industry"]), ("Customer since", str(c["since"])),
        ("Current support tier", c["tier"]), ("Current SLA response time", f"{c['sla_hours']} hours"),
        ("Account manager", c["manager"]),
    ])
    d.para(d.rng.choice([
        f"{c['name']} (customer ID {c['id']}) has been a Kestrel customer since {c['since']}. The account is managed by {c['manager']}.",
        f"Customer since {c['since']}, {c['name']} is looked after by account manager {c['manager']}.",
    ]))
    d.heading("History", "Relationship history", "Background")
    d.prose(TB.CUSTOMER["history"] + TB.CUSTOMER["history_more"], 2, 4)
    tier_index = TIERS.index(c["tier"])
    if tier_index > 0:
        old = TIERS[d.rng.randrange(tier_index)]
        year = d.rng.randrange(c["since"], 2026)
        d.heading("Contract history", "Support contract")
        d.para(f"In {year} {c['name']} upgraded its support contract from the {old} tier to the "
               f"{c['tier']} tier. Before the upgrade, the SLA response time was "
               f"{data.TIERS[old]} hours; it is now {c['sla_hours']} hours.")
    projects = PROJECTS_BY_CUSTOMER[c["name"]]
    installed = sorted({x["product"] for x in projects} | set(d.pick(sorted(data.PRODUCTS), 1, 3)))

    def installed_base():
        d.heading("Installed base", "Installed products")
        d.table(["Product", "Measuring points", "In use since"],
                [[f"Kestrel {code}", d.rng.randrange(2, 90), d.rng.randrange(c["since"], 2027)]
                 for code in installed])

    def meetings():
        d.heading("Meeting notes", "Recent meetings", "Account notes")
        for note in d.pick(TB.CUSTOMER["meeting"], 2, 4):
            when = d.date_between("2025-01-10", "2026-10-15")
            d.para(f"**{d.date_text(when)}, meeting with {c['name']}** (notes: {c['manager']})\n{note}")

    def communication():
        d.heading("Communication preferences", "Working with the customer")
        d.prose(TB.CUSTOMER["communication"], 1, 3)

    def next_steps():
        d.heading("Next steps", "Open actions")
        d.bullets(d.pick(TB.CUSTOMER["next_steps"], 2, 5))

    d.sections(installed_base, meetings, communication, next_steps)
    d.related([project_title(x) for x in projects] + [incident_title(x) for x in INCIDENTS_BY_CUSTOMER[c["name"]]])
    return d


def project_doc(p: dict) -> Doc:
    d = Doc(p["doc"], "project", f"Project Brief {p['code']}: {p['name']}", customer=p["customer"],
            project=p["code"], product=f"Kestrel {p['product']}")
    d.heading("Background", "Context", "Introduction")
    d.para(d.rng.choice([
        f"Project {p['code']}, {p['name']}, is delivered for {p['customer']}.",
        f"{p['customer']} commissioned project {p['code']} ({p['name']}).",
    ]))
    d.prose(TB.PROJECT["background"], 2, 3)
    d.heading("Project facts", "Project data", "Key information")
    d.key_values([("Project code", p["code"]), ("Customer", p["customer"]), ("Project lead", p["lead"]),
                  ("Delivery site", p["site"]), ("Product", f"Kestrel {p['product']}"),
                  ("Start date", p["start"]), ("Final deadline", p["deadline"])])
    d.para(f"The project is led by {p['lead']} and run from the {p['site']} site. "
           f"The solution is based on the Kestrel {p['product']}.")
    d.heading("Budget", "Financials")
    estimate = p["budget_eur"] + d.rng.choice([-1, 1]) * d.rng.randrange(10, 60) * 500
    d.para(f"The initial estimate was {estimate} EUR. After the scope review the steering committee "
           f"approved a budget of {p['budget_eur']} EUR.")

    def objectives():
        d.heading("Objectives", "Goals")
        d.bullets(d.pick(TB.PROJECT["objectives"], 3, 6))

    def scope():
        d.heading("Scope", "Scope of delivery")
        d.prose(TB.PROJECT["scope"], 2, 4)

    def milestones():
        d.heading("Milestones", "Schedule", "Timeline")
        rows = [["Kickoff", p["start"]]]
        for label in d.pick(["Design review", "Factory acceptance test", "Site acceptance test",
                             "Pilot installation", "Training"], 2, 4):
            rows.append([label, d.date_between(p["start"], p["deadline"])])
        rows = [rows[0]] + sorted(rows[1:], key=lambda r: r[1])
        rows.append(["Handover (final deadline)", p["deadline"]])
        d.table(["Milestone", "Date"], rows)

    def risks():
        d.heading("Risks", "Risk register")
        d.table(["Risk", "Mitigation"], [list(r) for r in d.pick(TB.PROJECT["risks"], 2, 5)])

    def governance():
        d.heading("Governance", "Project organisation")
        d.prose(TB.PROJECT["governance"], 1, 3)

    def minutes():
        d.heading("Kickoff meeting minutes", "Kickoff notes")
        d.bullets(d.pick(TB.PROJECT["minutes"], 3, 6))

    def communication():
        d.heading("Communication", "Communication plan")
        d.prose(TB.PROJECT["communication"], 1, 3)

    d.sections(objectives, scope, milestones, risks, governance, minutes, communication)
    d.related([f"Customer Profile: {p['customer']}", datasheet_title(p["product"])])
    return d


def incident_doc(i: dict) -> Doc:
    product = data.PRODUCTS[i["product"]]
    d = Doc(i["doc"], "incident", f"Incident Report {i['id']}", incident=i["id"], customer=i["customer"])
    d.heading("Summary", "Incident summary", "Overview")
    d.prose(TB.INCIDENT["summary"], 1, 2)
    d.para(f"Incident {i['id']} occurred on {long_date(i['date'])} at {i['customer']}. The affected "
           f"product was the {product['name']} and the incident was classified as severity {i['severity']}.")
    d.table(["Field", "Value"], [["Incident", i["id"]], ["Date", i["date"]], ["Customer", i["customer"]],
                                 ["Product", product["name"]], ["Severity", str(i["severity"])],
                                 ["Final total downtime", f"{i['downtime_min']} minutes"]])

    def detection():
        d.heading("Detection", "How the incident was detected")
        d.prose(TB.INCIDENT["detection"], 1, 2)

    def timeline():
        d.heading("Timeline", "Chronology")
        estimate = max(5, i["downtime_min"] + d.rng.choice([-1, 1]) * d.rng.randrange(20, 120))
        bank = TB.INCIDENT["timeline"]
        events = sorted(d.pick(bank, 4, 6), key=bank.index)  # keep the natural order
        events.insert(len(events) // 2, f"Initial downtime estimate communicated: {estimate} minutes")
        minutes = sorted(d.rng.sample(range(6 * 60, 22 * 60), len(events)))
        d.table(["Time", "Event"], [[f"{m // 60:02d}:{m % 60:02d}", e] for m, e in zip(minutes, events)])

    def impact():
        d.heading("Impact", "Customer impact")
        d.prose(TB.INCIDENT["impact"], 1, 3)
        d.para(f"The final total downtime was {i['downtime_min']} minutes.")

    d.sections(detection, timeline, impact)
    d.heading("Root cause", "Root cause analysis", "Cause")
    d.para(d.rng.choice([f"The root cause was {i['cause']}.", f"Root cause analysis identified {i['cause']}."]))

    def contributing():
        d.heading("Contributing factors", "Other factors")
        d.bullets(d.pick(TB.INCIDENT["contributing"], 2, 4))

    def communication():
        d.heading("Customer communication", "Communication")
        d.prose(TB.INCIDENT["communication"], 1, 3)

    def lessons():
        d.heading("Lessons learned", "What we learned")
        d.bullets(d.pick(TB.INCIDENT["lessons"], 2, 4))

    def follow_up():
        d.heading("Follow-up", "Next steps")
        d.prose(TB.INCIDENT["follow_up"], 1, 3)

    d.sections(contributing, communication)
    d.heading("Corrective actions", "Actions", "Measures")
    actions = [[f"Ship the fix with firmware {i['fixed_in']}", "Engineering", "done"]]
    actions += [[a, owner, d.rng.choice(["open", "in progress", "done"])]
                for a, owner in d.pick(TB.INCIDENT["actions"], 2, 4)]
    d.table(["Action", "Owner", "Status"], actions)
    d.para(f"The fix shipped with firmware {i['fixed_in']}.")
    d.sections(lessons, follow_up)
    d.related([f"Customer Profile: {i['customer']}", datasheet_title(i["product"])])
    return d


def supplier_doc(s: dict) -> Doc:
    d = Doc(s["doc"], "supplier", f"Supplier Profile: {s['name']}", supplier=s["name"],
            component=s["component"], country=s["country"])
    d.heading("Overview", "Supplier overview", "Introduction")
    d.para(d.rng.choice([f"{s['name']} supplies {s['component']} to Kestrel Instruments.",
                         f"Kestrel buys {s['component']} from {s['name']}."]) + f" {s['name']} is based in {s['country']}.")
    d.prose(TB.SUPPLIER["overview"], 2, 3)
    d.heading("Key data", "Supplier data", "At a glance")
    d.key_values([("Component", s["component"]), ("Country", s["country"]),
                  ("Current standard lead time", f"{s['lead_weeks']} weeks"),
                  ("Framework contract until", str(s["contract_until"]))])
    d.heading("Lead time history", "Delivery performance")
    d.para(f"During the component shortage the lead time temporarily rose to "
           f"{s['lead_weeks'] + d.rng.randrange(3, 12)} weeks. The current standard lead time is "
           f"{s['lead_weeks']} weeks. The current framework contract runs until {s['contract_until']}.")

    def quality():
        d.heading("Quality", "Quality management")
        d.prose(TB.SUPPLIER["quality"], 1, 3)

    def logistics():
        d.heading("Logistics", "Deliveries")
        d.prose(TB.SUPPLIER["logistics"], 1, 3)

    def audits():
        d.heading("Audits", "Audit history")
        d.prose(TB.SUPPLIER["audits"], 1, 3)

    def terms():
        d.heading("Contract terms", "Commercial terms")
        d.prose(TB.SUPPLIER["terms"], 1, 3)

    def sustainability():
        d.heading("Sustainability", "Responsible sourcing")
        d.prose(TB.SUPPLIER["sustainability"], 1, 3)

    def risks():
        d.heading("Risks", "Supply risks")
        d.bullets(d.pick(TB.SUPPLIER["risks"], 2, 4))

    d.sections(quality, logistics, audits, terms, sustainability, risks)
    return d


def course_doc(c: dict) -> Doc:
    d = Doc(c["doc"], "course", f"Training Course: {c['name']}", course=c["name"])
    d.heading("Description", "About the course", "Course overview")
    d.para(d.rng.choice([
        f"The course {c['name']} lasts {c['days']} days and is taught by {c['trainer']}.",
        f"{c['name']}: {c['days']} days, trainer {c['trainer']}.",
    ]))
    d.prose(TB.COURSE["description"], 2, 3)
    d.para(f"It accepts at most {c['max_participants']} participants and costs {c['cost_eur']} EUR per person.")
    d.para(f"Participants must first complete {c['prerequisite']}." if c["prerequisite"]
           else "The course has no prerequisites.")

    def audience():
        d.heading("Target audience", "Who should attend")
        d.prose(TB.COURSE["audience"], 1, 3)

    def objectives():
        d.heading("Learning objectives", "What you will learn")
        d.bullets(d.pick(TB.COURSE["objectives"], 3, 6))

    def materials():
        d.heading("Course materials", "Materials")
        d.prose(TB.COURSE["materials"], 1, 3)

    def assessment():
        d.heading("Assessment", "Certification")
        d.prose(TB.COURSE["assessment"], 1, 2)

    d.sections(audience, objectives, materials, assessment)
    d.heading("Agenda", "Course agenda", "Programme")
    for day in range(1, c["days"] + 1):
        d.h3(f"Day {day}")
        d.bullets(d.pick(TB.COURSE["agenda"], 3, 5))
    d.heading("Practical information", "Logistics", "Course details")
    d.table(["Item", "Details"], [["Duration", f"{c['days']} days"], ["Maximum participants", str(c["max_participants"])],
                                  ["Fee per person", f"{c['cost_eur']} EUR"], ["Trainer", c["trainer"]],
                                  ["Prerequisite", c["prerequisite"] or "none"]])
    if d.rng.random() < 0.7:
        d.heading("Frequently asked questions", "FAQ")
        d.faq(d.pick(TB.COURSE["faq"], 1, 3))
    return d


def discontinued_doc(item: dict) -> Doc:
    d = Doc(item["doc"], "datasheet", f"Kestrel {item['code']} Discontinuation Notice")
    d.para(f"The Kestrel {item['code']}, predecessor of the Kestrel {item['successor']}, was discontinued in "
           f"{item['year']}. The {item['code']} had a maximum operating temperature of {item['max_c']} °C. "
           f"It used SKU {item['sku']} and is no longer stocked.")
    d.h2("Migration")
    d.para(f"Customers still operating a {item['code']} should plan the migration to the Kestrel "
           f"{item['successor']}. The successor uses the same mounting dimensions, but wiring and "
           "configuration must be checked against its current datasheet before replacement.")
    d.bullets(["Check the current datasheet of the successor", "Order mounting adapters if needed",
               "Back up the existing configuration", "Plan the replacement in a maintenance window"])
    return d


def documents() -> dict[str, tuple[str, str]]:
    """Return ``{doc_id: (kind, markdown)}`` for the whole corpus."""
    built: list[Doc] = [datasheet_doc(code, p) for code, p in data.PRODUCTS.items()]
    built += [release_notes_doc(prefix, kind) for prefix, kind, *_rest in data.FAMILIES]
    built += [policy_doc(doc_id, title, [f[0] for f in facts], False) for doc_id, title, facts in data.POLICIES]
    built += [policy_doc(doc_id, title, facts, True) for doc_id, title, facts in data.DISTRACTOR_POLICIES]
    built += [site_doc(s) for s in data.SITES]
    built += [customer_doc(c) for c in data.CUSTOMERS]
    built += [project_doc(p) for p in data.PROJECTS]
    built += [incident_doc(i) for i in data.INCIDENTS]
    built += [supplier_doc(s) for s in data.SUPPLIERS]
    built += [course_doc(c) for c in data.COURSES]
    built += [discontinued_doc(item) for item in data.DISCONTINUED]
    return {d.id: (d.kind, d.render()) for d in sorted(built, key=lambda d: d.id)}


# --- answer pattern helpers --------------------------------------------------

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten", 12: "twelve", 14: "fourteen", 15: "fifteen",
         20: "twenty", 30: "thirty"}
SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥"}


def num(value, *units: str) -> list[list[str]]:
    forms = [str(value)] + ([WORDS[value]] if value in WORDS else [])
    return [[f"{f} {u}" for f in forms for u in units] + [f"{f}{u}" for f in forms for u in units]]


def temp(value: int) -> list[list[str]]:
    return [[f"{value} °C", f"{value}°C", f"{value} degrees", f"{value} ° C"]]


def money(value: float, currency: str) -> list[list[str]]:
    text = f"{value:.2f}"
    alternatives = [text, text.replace(".", ",")]
    if value == int(value):
        whole = str(int(value))
        alternatives += [f"{whole} {currency}", f"{whole}{currency}", f"{currency} {whole}"]
        if currency == "EUR":
            alternatives += [f"{whole} euros", f"{whole} euro"]
        if currency in SYMBOLS:
            alternatives += [f"{SYMBOLS[currency]}{whole}", f"{SYMBOLS[currency]} {whole}"]
    return [alternatives]


def counted(count: int, nouns: str) -> str:
    # The count, up to three words (e.g. "Kestrel KH3"), then the counted noun.
    return rf"re:(?<![\w.]){count}(?:\s+[\w/-]+){{0,3}}\s+(?:{nouns})"


def units(count: int) -> list[list[str]]:
    return [[counted(count, "units?|pieces|sensors?|devices?|items?|in stock|products?")]]


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def tickets(count: int) -> list[list[str]]:
    patterns = [[counted(count, "tickets?|open")]]
    if count == 0:
        patterns[0] += ["no open", "zero", "none"]
    return patterns


def dates(iso: str) -> list[list[str]]:
    d = date.fromisoformat(iso)
    month = MONTHS[d.month - 1]
    suffix = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return [[iso, f"{d.day} {month} {d.year}", f"{month} {d.day}, {d.year}",
             f"{d.day}. {month} {d.year}", f"{month} {d.day}{suffix}, {d.year}",
             f"{d.day}{suffix} {month} {d.year}", f"{month[:3]} {d.day}, {d.year}",
             f"{d.day} {month[:3]} {d.year}", f"{d.day:02d}.{d.month:02d}.{d.year}"]]


def version(v: str) -> list[list[str]]:
    return [[v, f"v{v}"]]


def text(*alternatives: str) -> list[list[str]]:
    return [list(alternatives)]


# --- candidate questions -------------------------------------------------------

P, T = data.PRODUCTS, data.TOOLS
EXT = {e["name"]: e["extension"] for e in data.EMPLOYEES}
CURRENCIES = ["USD", "NOK", "SEK", "DKK", "GBP", "CHF", "PLN", "JPY"]

SITE_QUESTIONS = {
    "KI-201": [("What is the street address of the Kestrel headquarters?", "Fjordveien 18", text("Fjordveien 18")),
               ("About how many employees work at the Oslo headquarters?", "240 employees", num(240, "employees", "people", "staff"))],
    "KI-202": [("How many calibration technicians work at the Bergen lab?", "12 technicians", num(12, "calibration technicians", "technicians")),
               ("To which standard is the Bergen calibration lab accredited?", "ISO/IEC 17025", text("17025"))],
    "KI-203": [("What are the opening hours of the Stavanger warehouse?", "06:00 to 22:00",
                [["06:00", "6:00", "6 am", "6am"], ["22:00", "10 pm", "10pm", "10:00 pm"]]),
               ("What is the carrier cutoff time at the Stavanger warehouse?", "14:00", text("14:00", "2 pm", "2pm", "2:00 pm"))],
    "KI-204": [("How many engineers work at the Trondheim R&D center?", "35 engineers", num(35, "engineers")),
               ("In which year did the Trondheim R&D center open?", "2016", text("2016"))],
    "KI-205": [("Which countries does the Gothenburg sales office serve?", "Sweden and Finland", [["sweden"], ["finland"]]),
               ("In which year did the Gothenburg office open?", "2019", text("2019"))],
    "KI-206": [("Within how many working days are standard repairs completed in Hamburg?", "7 working days",
                num(7, "working days", "business days", "days")),
               ("In which year did the Hamburg service center open?", "2021", text("2021"))],
    "KI-207": [("How many shifts does the Aarhus plant run on working days?", "two shifts", num(2, "shifts")),
               ("To which environmental standard is the Aarhus plant certified?", "ISO 14001", text("14001"))],
    "KI-208": [("Which region does the Houston office serve?", "the Americas", text("americas")),
               ("In which year did the Houston office open?", "2022", text("2022"))],
}

POLICY_ROLE_QUESTIONS = [
    ("KI-105", "Information Security Officer",
     "A laptop was stolen. What is the phone extension of the person it must be reported to under the information security policy?"),
    ("KI-113", "Data Protection Officer",
     "What is the phone extension of the person who receives data protection incident reports?"),
    ("KI-118", "Compliance Officer",
     "What is the phone extension of the person to whom conflicts of interest are disclosed?"),
    ("KI-114", "Fleet Coordinator",
     "A company car was damaged. What is the phone extension of the person it must be reported to?"),
    ("KI-119", "Health and Safety Officer",
     "What is the phone extension of the person to whom near misses are reported?"),
]
POLICY_AMOUNTS = [("KI-102", 55, "the current daily travel per diem"),
                  ("KI-102", 160, "the hotel reimbursement cap per night"),
                  ("KI-104", 300, "the one-time remote work equipment allowance"),
                  ("KI-115", 1200, "the annual training budget per employee"),
                  ("KI-116", 450, "the weekly on-call allowance"),
                  ("KI-118", 50, "the gift value above which gifts must be declined")]


def Q(category, subtype, question, ground_truth, patterns, tools=(), docs=()):
    return dict(category=category, subtype=subtype, question=question,
                ground_truth=ground_truth, answer_patterns=patterns,
                expected_tools=list(tools), gold_doc_ids=list(docs))


def doc_single() -> list[dict]:
    out = []
    for code, p in P.items():
        d, n = [PRODUCT_DOC[code]], p["name"]
        out += [
            Q("doc_single", "product", f"What is the maximum operating temperature of the {n}?", f"{p['max_c']} °C", temp(p["max_c"]), docs=d),
            Q("doc_single", "product", f"What is the minimum operating temperature of the {n}?", f"{p['min_c']} °C", temp(p["min_c"]), docs=d),
            Q("doc_single", "product", f"Which communication interface does the {n} use?", p["interface"],
              text(p["interface"], *(["4-20 mA"] if "4-20" in p["interface"] else [])), docs=d),
            Q("doc_single", "product", f"What is the IP rating of the {n} enclosure?", p["ip"], text(p["ip"], p["ip"].replace("IP", "IP ")), docs=d),
            Q("doc_single", "product", f"How long is the warranty on the {n}?", plural(p["warranty_years"], "year"),
              num(p["warranty_years"], "years", "year", "-year"), docs=d),
            Q("doc_single", "product", f"How much current does the {n} draw?", f"{p['current_ma']} mA", num(p["current_ma"], "mA"), docs=d),
            Q("doc_single", "product", f"What is the current list price of the {n} in EUR?", f"{p['price_eur']} EUR", money(p["price_eur"], "EUR"), docs=d),
            Q("doc_single", "product", f"What is the current weight of the {n}?", f"{p['weight_g']} g", num(p["weight_g"], "g", "grams"), docs=d),
            Q("doc_single", "product", f"In which year was the {n} launched?", str(p["launch_year"]), text(str(p["launch_year"])), docs=d),
            Q("doc_single", "product", f"What is the SKU of the {n}?", p["sku"], text(p["sku"]), docs=d),
        ]
        f = [FAMILY_DOC[p["family"]]]
        out += [
            Q("doc_single", "release_notes", f"What is the latest firmware version for the {n}?", p["firmware"], version(p["firmware"]), docs=f),
            Q("doc_single", "release_notes", f"When was the latest firmware for the {n} released?", p["firmware_date"], dates(p["firmware_date"]), docs=f),
        ]
    for doc_id, _title, facts in data.POLICIES:
        out += [Q("doc_single", "policy", q, gt, pat, docs=[doc_id]) for _s, q, gt, pat in facts]
    for site in data.SITES:
        out.append(Q("doc_single", "site", f"Who heads the {site['name']}?", site["head"], text(site["head"]), docs=[site["id"]]))
        out += [Q("doc_single", "site", q, gt, pat, docs=[site["id"]]) for q, gt, pat in SITE_QUESTIONS[site["id"]]]
    for p in data.PROJECTS:
        d, c = [p["doc"]], p["code"]
        out += [
            Q("doc_single", "project", f"Who leads project {c}?", p["lead"], text(p["lead"]), docs=d),
            Q("doc_single", "project", f"What is the approved budget of project {c}?", f"{p['budget_eur']} EUR", money(p["budget_eur"], "EUR"), docs=d),
            Q("doc_single", "project", f"What is the final deadline of project {c}?", p["deadline"], dates(p["deadline"]), docs=d),
            Q("doc_single", "project", f"For which customer is project {c} delivered?", p["customer"], text(p["customer"]), docs=d),
            Q("doc_single", "project", f"From which Kestrel site is project {c} run?", p["site"], text(p["site"]), docs=d),
            Q("doc_single", "project", f"Which Kestrel product is project {c} based on?", f"Kestrel {p['product']}", text(p["product"]), docs=d),
        ]
    for c in data.CUSTOMERS:
        d, n = [c["doc"]], c["name"]
        out += [
            Q("doc_single", "customer", f"Who is the account manager for {n}?", c["manager"], text(c["manager"]), docs=d),
            Q("doc_single", "customer", f"Which support contract tier does {n} currently hold?", c["tier"], text(c["tier"]), docs=d),
            Q("doc_single", "customer", f"What is the current SLA response time for {n}?", f"{c['sla_hours']} hours",
              num(c["sla_hours"], "hours", "-hour", "h"), docs=d),
            Q("doc_single", "customer", f"Since which year has {n} been a Kestrel customer?", str(c["since"]), text(str(c["since"])), docs=d),
            Q("doc_single", "customer", f"What is the customer ID of {n}?", c["id"], text(c["id"]), docs=d),
            Q("doc_single", "customer", f"In which country is {n} based?", c["country"], text(c["country"]), docs=d),
        ]
    for i in data.INCIDENTS:
        d, k = [i["doc"]], i["id"]
        out += [
            Q("doc_single", "incident", f"What was the root cause of incident {k}?", i["cause"], i["cause_patterns"], docs=d),
            Q("doc_single", "incident", f"How long was the final total downtime of incident {k}?", f"{i['downtime_min']} minutes",
              num(i["downtime_min"], "minutes", "min"), docs=d),
            Q("doc_single", "incident", f"On which date did incident {k} occur?", i["date"], dates(i["date"]), docs=d),
            Q("doc_single", "incident", f"Which product was affected by incident {k}?", f"Kestrel {i['product']}", text(i["product"]), docs=d),
            Q("doc_single", "incident", f"Which severity was incident {k} classified as?", f"severity {i['severity']}",
              text(f"severity {i['severity']}", f"sev {i['severity']}", f"sev{i['severity']}", f"severity level {i['severity']}"), docs=d),
            Q("doc_single", "incident", f"With which firmware version was the fix for incident {k} shipped?", i["fixed_in"], version(i["fixed_in"]), docs=d),
            Q("doc_single", "incident", f"Which customer was affected by incident {k}?", i["customer"], text(i["customer"]), docs=d),
        ]
    for s in data.SUPPLIERS:
        d, n = [s["doc"]], s["name"]
        out += [
            Q("doc_single", "supplier", f"Which component does {n} supply to Kestrel?", s["component"], text(s["component"]), docs=d),
            Q("doc_single", "supplier", f"What is the current standard lead time of {n}?", plural(s["lead_weeks"], "week"),
              num(s["lead_weeks"], "weeks", "-week"), docs=d),
            Q("doc_single", "supplier", f"In which country is {n} based?", s["country"], text(s["country"]), docs=d),
            Q("doc_single", "supplier", f"Until which year does the framework contract with {n} run?", str(s["contract_until"]),
              text(str(s["contract_until"])), docs=d),
        ]
    for c in data.COURSES:
        d, n = [c["doc"]], c["name"]
        out += [
            Q("doc_single", "course", f"How many days does the course {n} last?", plural(c["days"], "day"), num(c["days"], "days", "day", "-day"), docs=d),
            Q("doc_single", "course", f"How many participants can attend the course {n} at most?", f"{c['max_participants']} participants",
              [[counted(c["max_participants"], "participants?|people|attendees|persons|trainees")]], docs=d),
            Q("doc_single", "course", f"How much does the course {n} cost per person?", f"{c['cost_eur']} EUR", money(c["cost_eur"], "EUR"), docs=d),
            Q("doc_single", "course", f"Who teaches the course {n}?", c["trainer"], text(c["trainer"]), docs=d),
        ]
        if c["prerequisite"]:
            out.append(Q("doc_single", "course", f"Which course must be completed before {n}?", c["prerequisite"], text(c["prerequisite"]), docs=d))
    return out


def doc_multi(rng: random.Random) -> list[dict]:
    out = []
    codes = sorted(P)
    pairs = [(a, b) for a in codes for b in codes if a < b]
    rng.shuffle(pairs)
    for a, b in pairs[:400]:
        pa, pb = P[a], P[b]
        d = [PRODUCT_DOC[a], PRODUCT_DOC[b]]
        hi, lo = (pa, pb) if pa["max_c"] > pb["max_c"] else (pb, pa)
        if hi["max_c"] != lo["max_c"]:
            diff = hi["max_c"] - lo["max_c"]
            out.append(Q("doc_multi", "product_temp", f"By how many degrees Celsius does the maximum operating temperature of the {hi['name']} exceed that of the {lo['name']}?",
                         f"{diff} °C", [[f"{diff} °C", f"{diff}°C", f"{diff} degrees", f"{diff} K", f"{diff} kelvin"]], docs=d))
        hi, lo = (pa, pb) if pa["weight_g"] > pb["weight_g"] else (pb, pa)
        if hi["weight_g"] != lo["weight_g"]:
            diff = hi["weight_g"] - lo["weight_g"]
            out.append(Q("doc_multi", "product_weight", f"Based on the current datasheets, how many grams heavier is the {hi['name']} than the {lo['name']}?",
                         f"{diff} g", num(diff, "g", "grams"), docs=d))
        total = pa["price_eur"] + pb["price_eur"]
        out.append(Q("doc_multi", "product_price", f"What is the combined current list price in EUR of one {pa['name']} and one {pb['name']}?",
                     f"{total} EUR", money(total, "EUR"), docs=d))
        hi, lo = (pa, pb) if pa["warranty_years"] > pb["warranty_years"] else (pb, pa)
        if hi["warranty_years"] != lo["warranty_years"]:
            diff = hi["warranty_years"] - lo["warranty_years"]
            out.append(Q("doc_multi", "product_warranty", f"How many years longer is the warranty of the {hi['name']} than that of the {lo['name']}?",
                         plural(diff, "year"), num(diff, "years", "year", "more years", "additional years"), docs=d))
    projects = list(data.PROJECTS)
    for _ in range(80):
        a, b = rng.sample(projects, 2)
        hi, lo = (a, b) if a["budget_eur"] > b["budget_eur"] else (b, a)
        if hi["budget_eur"] != lo["budget_eur"]:
            diff = hi["budget_eur"] - lo["budget_eur"]
            out.append(Q("doc_multi", "project_budget", f"By how much does the approved budget of project {hi['code']} exceed that of project {lo['code']}?",
                         f"{diff} EUR", money(diff, "EUR"), docs=[hi["doc"], lo["doc"]]))
    for _ in range(80):
        a, b = rng.sample(data.CUSTOMERS, 2)
        old, new = (a, b) if a["since"] < b["since"] else (b, a)
        if old["since"] != new["since"]:
            diff = new["since"] - old["since"]
            out.append(Q("doc_multi", "customer_since", f"How many years longer has {old['name']} been a Kestrel customer than {new['name']}?",
                         plural(diff, "year"), num(diff, "years", "year"), docs=[old["doc"], new["doc"]]))
    for _ in range(80):
        a, b = rng.sample(data.INCIDENTS, 2)
        total = a["downtime_min"] + b["downtime_min"]
        out.append(Q("doc_multi", "incident_downtime", f"What was the combined final downtime of incidents {a['id']} and {b['id']} in minutes?",
                     f"{total} minutes", num(total, "minutes", "min"), docs=[a["doc"], b["doc"]]))
    for _ in range(60):
        a, b = rng.sample(data.SUPPLIERS, 2)
        hi, lo = (a, b) if a["lead_weeks"] > b["lead_weeks"] else (b, a)
        if hi["lead_weeks"] != lo["lead_weeks"]:
            diff = hi["lead_weeks"] - lo["lead_weeks"]
            out.append(Q("doc_multi", "supplier_lead", f"How many weeks longer is the current standard lead time of {hi['name']} than that of {lo['name']}?",
                         plural(diff, "week"), num(diff, "weeks", "week"), docs=[hi["doc"], lo["doc"]]))
    return out


def mcp_single(rng: random.Random) -> list[dict]:
    out = []
    for e in data.EMPLOYEES:
        out.append(Q("mcp_single", "directory", f"What is the phone extension of {e['name']}?", e["extension"],
                     text(e["extension"]), tools=["search_directory"]))
    for order_id in data.ORDERS:
        o = T["get_order_status"](order_id)
        status_alts = [o["status"]] + (["canceled"] if o["status"] == "cancelled" else [])
        out.append(Q("mcp_single", "order_status", f"What is the status of order {order_id}, and what is its order date?",
                     f"{o['status']}, {o['date']}", [status_alts, dates(o["date"])[0]], tools=["get_order_status"]))
        out.append(Q("mcp_single", "order_quantity", f"How many units were ordered in order {order_id}?", plural(o["quantity"], "unit"),
                     units(o["quantity"]), tools=["get_order_status"]))
        if o["carrier"]:
            out.append(Q("mcp_single", "order_carrier", f"Which carrier handles order {order_id}?", o["carrier"],
                         text(o["carrier"]), tools=["get_order_status"]))
    for sku, entry in data.STOCK.items():
        if entry["units"]:
            out.append(Q("mcp_single", "stock", f"How many units of SKU {sku} are currently in stock?", plural(entry["units"], "unit"),
                         units(entry["units"]), tools=["get_stock_level"]))
        else:
            out.append(Q("mcp_single", "restock", f"When is SKU {sku} expected to be back in stock?", entry["restock_date"],
                         dates(entry["restock_date"]), tools=["get_stock_level"]))
    for _ in range(120):
        src, dst = rng.sample(["EUR", *CURRENCIES], 2)
        amount = rng.choice([50, 75, 120, 250, 480, 999, 1500, 2750, 4200, 12500])
        value = T["convert_currency"](amount, src, dst)["converted"]
        out.append(Q("mcp_single", "currency", f"Convert {amount} {src} to {dst} using today's company exchange rate.",
                     f"{value:.2f} {dst}", money(value, dst), tools=["convert_currency"]))
    for code in P:
        count = T["count_open_tickets"](code)["open_tickets"]
        out.append(Q("mcp_single", "tickets", f"How many open support tickets are there for the Kestrel {code}?",
                     plural(count, "open ticket"), tickets(count), tools=["count_open_tickets"]))
    for _ in range(120):
        country = rng.choice(sorted(data.SHIPPING))
        weight = rng.choice([0.4, 0.7, 1.2, 2.5, 3.8, 5.0, 7.3, 9.6, 12.0, 18.5, 24.0])
        value = T["get_shipping_quote"](weight, country)["price_eur"]
        out.append(Q("mcp_single", "shipping", f"What does it cost to ship a {weight} kg parcel from the warehouse to {data.COUNTRY_NAMES[country]}?",
                     f"{value:.2f} EUR", money(value, "EUR"), tools=["get_shipping_quote"]))
    for p in data.PROJECTS:
        s = T["get_project_status"](p["code"])
        out.append(Q("mcp_single", "project_spent", f"How much budget has been spent so far on project {p['code']}?",
                     f"{s['budget_spent_eur']} EUR", money(s["budget_spent_eur"], "EUR"), tools=["get_project_status"]))
        out.append(Q("mcp_single", "project_progress", f"What percentage of project {p['code']} is complete, and what is its risk level?",
                     f"{s['percent_complete']} %, {s['risk']} risk",
                     [num(s["percent_complete"], "%", "percent")[0], [s["risk"]]], tools=["get_project_status"]))
    for c in data.CUSTOMERS:
        inv = T["get_invoice_balance"](c["id"])
        out.append(Q("mcp_single", "invoice_balance", f"What is the outstanding invoice balance of customer {c['id']}?",
                     f"{inv['outstanding_eur']:.2f} EUR", money(inv["outstanding_eur"], "EUR"), tools=["get_invoice_balance"]))
        overdue = [[counted(inv["overdue_invoices"], "overdue|invoices?")]]
        if inv["overdue_invoices"] == 0:
            overdue[0] += ["no overdue", "none", "zero"]
        out.append(Q("mcp_single", "invoice_overdue", f"How many overdue invoices does customer {c['id']} have?",
                     plural(inv["overdue_invoices"], "overdue invoice"), overdue, tools=["get_invoice_balance"]))
    return out


def mcp_chain(rng: random.Random) -> list[dict]:
    out = []
    for order_id in data.ORDERS:
        o = T["get_order_status"](order_id)
        stock = T["get_stock_level"](o["sku"])["units"]
        if stock:
            out.append(Q("mcp_chain", "order_stock", f"What is the status of order {order_id}, and how many units of its SKU are in stock?",
                         f"{o['status']}, {plural(stock, 'unit')}", [[o["status"]], units(stock)[0]], tools=["get_order_status", "get_stock_level"]))
        country = rng.choice(sorted(data.SHIPPING))
        value = T["get_shipping_quote"](o["weight_kg"], country)["price_eur"]
        out.append(Q("mcp_chain", "order_shipping", f"What would it cost to ship a parcel with the weight of order {order_id} to {data.COUNTRY_NAMES[country]}?",
                     f"{value:.2f} EUR", money(value, "EUR"), tools=["get_order_status", "get_shipping_quote"]))
        count = T["count_open_tickets"](o["product_code"])["open_tickets"]
        out.append(Q("mcp_chain", "order_tickets", f"How many open support tickets exist for the product ordered in {order_id}?",
                     plural(count, "open ticket"), tickets(count), tools=["get_order_status", "count_open_tickets"]))
        currency = rng.choice(CURRENCIES)
        value = T["convert_currency"](o["order_value_eur"], "EUR", currency)["converted"]
        out.append(Q("mcp_chain", "order_currency", f"What is the value of order {order_id} in {currency}?",
                     f"{value:.2f} {currency}", money(value, currency), tools=["get_order_status", "convert_currency"]))
        inv = T["get_invoice_balance"](o["customer_id"])
        out.append(Q("mcp_chain", "order_invoice", f"What is the outstanding invoice balance of the customer who placed order {order_id}?",
                     f"{inv['outstanding_eur']:.2f} EUR", money(inv["outstanding_eur"], "EUR"), tools=["get_order_status", "get_invoice_balance"]))
    for c in data.CUSTOMERS:
        currency = rng.choice(CURRENCIES)
        inv = T["get_invoice_balance"](c["id"])
        value = T["convert_currency"](inv["outstanding_eur"], "EUR", currency)["converted"]
        out.append(Q("mcp_chain", "invoice_currency", f"What is the outstanding invoice balance of customer {c['id']} in {currency}?",
                     f"{value:.2f} {currency}", money(value, currency), tools=["get_invoice_balance", "convert_currency"]))
    for p in data.PROJECTS:
        currency = rng.choice(CURRENCIES)
        spent = T["get_project_status"](p["code"])["budget_spent_eur"]
        value = T["convert_currency"](spent, "EUR", currency)["converted"]
        out.append(Q("mcp_chain", "project_currency", f"How much has been spent so far on project {p['code']}, expressed in {currency}?",
                     f"{value:.2f} {currency}", money(value, currency), tools=["get_project_status", "convert_currency"]))
    return out


def mixed(rng: random.Random) -> list[dict]:
    out = []
    for code, p in P.items():
        d = [PRODUCT_DOC[code]]
        entry = T["get_stock_level"](p["sku"])
        if entry["units"]:
            out.append(Q("mixed", "product_stock", f"How many {p['name']} are currently in stock?", plural(entry["units"], "unit"),
                         units(entry["units"]), tools=["get_stock_level"], docs=d))
        else:
            out.append(Q("mixed", "product_restock", f"Is the {p['name']} currently in stock, and if not, when is it expected back?",
                         f"out of stock, expected {entry['restock_date']}", dates(entry["restock_date"]), tools=["get_stock_level"], docs=d))
        for currency in rng.sample(CURRENCIES, 3):
            value = T["convert_currency"](p["price_eur"], "EUR", currency)["converted"]
            out.append(Q("mixed", "product_price_currency", f"What is the current list price of the {p['name']} in {currency}?",
                         f"{value:.2f} {currency}", money(value, currency), tools=["convert_currency"], docs=d))
        for _ in range(2):
            count, country = rng.choice([1, 2, 5, 10, 20]), rng.choice(sorted(data.SHIPPING))
            weight = round(p["weight_g"] * count / 1000, 3)
            value = T["get_shipping_quote"](weight, country)["price_eur"]
            what = f"one {p['name']}" if count == 1 else f"{count} units of the {p['name']}"
            out.append(Q("mixed", "product_shipping", f"Based on the current datasheet weight, what does it cost to ship {what} from the warehouse to {data.COUNTRY_NAMES[country]}?",
                         f"{value:.2f} EUR", money(value, "EUR"), tools=["get_shipping_quote"], docs=d))
        count = T["count_open_tickets"](code)["open_tickets"]
        out.append(Q("mixed", "sku_tickets", f"How many open support tickets are there for the product with SKU {p['sku']}?",
                     plural(count, "open ticket"), tickets(count), tools=["count_open_tickets"], docs=d))
    for p in data.PROJECTS:
        d = [p["doc"]]
        out.append(Q("mixed", "project_lead_extension", f"What is the phone extension of the lead of project {p['code']}?",
                     EXT[p["lead"]], text(EXT[p["lead"]]), tools=["search_directory"], docs=d))
        remaining = p["budget_eur"] - T["get_project_status"](p["code"])["budget_spent_eur"]
        out.append(Q("mixed", "project_remaining", f"How much of the approved budget of project {p['code']} is still unspent?",
                     f"{remaining} EUR", money(remaining, "EUR"), tools=["get_project_status"], docs=d))
    for c in data.CUSTOMERS:
        d = [c["doc"]]
        inv = T["get_invoice_balance"](c["id"])
        out.append(Q("mixed", "customer_invoice", f"What is the outstanding invoice balance of {c['name']}?",
                     f"{inv['outstanding_eur']:.2f} EUR", money(inv["outstanding_eur"], "EUR"), tools=["get_invoice_balance"], docs=d))
        out.append(Q("mixed", "account_manager_extension", f"What is the phone extension of the account manager for {c['name']}?",
                     EXT[c["manager"]], text(EXT[c["manager"]]), tools=["search_directory"], docs=d))
    for site in data.SITES:
        out.append(Q("mixed", "site_head_extension", f"What is the phone extension of the person who heads the {site['name']}?",
                     EXT[site["head"]], text(EXT[site["head"]]), tools=["search_directory"], docs=[site["id"]]))
    for i in data.INCIDENTS:
        count = T["count_open_tickets"](i["product"])["open_tickets"]
        out.append(Q("mixed", "incident_tickets", f"How many open support tickets are there for the product affected by incident {i['id']}?",
                     plural(count, "open ticket"), tickets(count), tools=["count_open_tickets"], docs=[i["doc"]]))
        customer = CUSTOMER_BY_NAME[i["customer"]]
        inv = T["get_invoice_balance"](customer["id"])
        out.append(Q("mixed", "incident_customer_invoice", f"What is the outstanding invoice balance of the customer affected by incident {i['id']}?",
                     f"{inv['outstanding_eur']:.2f} EUR", money(inv["outstanding_eur"], "EUR"), tools=["get_invoice_balance"],
                     docs=[i["doc"], customer["doc"]]))
    for doc_id, role, question in POLICY_ROLE_QUESTIONS:
        holder = next(e for e in data.EMPLOYEES if e["title"] == role)
        out.append(Q("mixed", "policy_role_extension", question, holder["extension"], text(holder["extension"]),
                     tools=["search_directory"], docs=[doc_id]))
    for doc_id, amount, label in POLICY_AMOUNTS:
        for currency in ("NOK", "USD", "SEK"):
            value = T["convert_currency"](amount, "EUR", currency)["converted"]
            out.append(Q("mixed", "policy_amount_currency", f"What is {label} in {currency}?", f"{value:.2f} {currency}",
                         money(value, currency), tools=["convert_currency"], docs=[doc_id]))
    return out


def allocate(candidates: list[dict], total: int, rng: random.Random) -> list[dict]:
    """Sample ``total`` questions spread as evenly as possible over sub-types."""
    by_subtype: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_subtype[c["subtype"]].append(c)
    for items in by_subtype.values():
        rng.shuffle(items)
    chosen: list[dict] = []
    remaining = dict(sorted(by_subtype.items()))
    while len(chosen) < total and remaining:
        share = max(1, (total - len(chosen)) // len(remaining))
        for subtype in list(remaining):
            take = remaining[subtype][:share]
            remaining[subtype] = remaining[subtype][share:]
            chosen.extend(take[: total - len(chosen)])
            if not remaining[subtype]:
                del remaining[subtype]
    if len(chosen) < total:
        raise ValueError(f"Only {len(chosen)} candidates for a target of {total}")
    return chosen


def questions() -> list[dict]:
    rng = random.Random("blueprint-mcp-host-questions-v3")
    docs = documents()
    doc_text = {doc_id: normalize_answer(text) for doc_id, (_kind, text) in docs.items()}
    corpus = "\n".join(doc_text.values())

    def keep(q: dict) -> bool:
        if q["category"] == "doc_single":
            return answer_matches_normalized(doc_text[q["gold_doc_ids"][0]], q["answer_patterns"])
        if q["category"] == "doc_multi":
            return True
        return not answer_matches_normalized(corpus, q["answer_patterns"])

    pools = {
        "doc_single": doc_single(),
        "doc_multi": doc_multi(rng),
        "mcp_single": mcp_single(rng),
        "mcp_chain": mcp_chain(rng),
        "mixed": mixed(rng),
    }
    selected: list[dict] = []
    seen: set[str] = set()
    for category, target in CATEGORY_TARGETS.items():
        unique = []
        for q in pools[category]:
            if q["question"] not in seen and keep(q):
                seen.add(q["question"])
                unique.append(q)
        selected += allocate(unique, target, rng)
    rng.shuffle(selected)
    return selected


def main() -> None:
    CORPUS.mkdir(exist_ok=True)
    for stale in CORPUS.glob("*.md"):
        stale.unlink()
    docs = documents()
    for doc_id, (_kind, text) in docs.items():
        (CORPUS / f"{doc_id}.md").write_text(text, encoding="utf-8")
    rows = questions()
    with QUESTIONS.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            metadata = {
                "case_id": f"bmh-{index:04d}",
                "category": row["category"],
                "subtype": row["subtype"],
                "expected_tools": row["expected_tools"],
                "requires_mcp": bool(row["expected_tools"]),
                "gold_doc_ids": row["gold_doc_ids"],
                "answer_patterns": row["answer_patterns"],
            }
            handle.write(json.dumps({
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "context": "",
                "metadata": metadata,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} documents and {len(rows)} questions")


if __name__ == "__main__":
    main()
