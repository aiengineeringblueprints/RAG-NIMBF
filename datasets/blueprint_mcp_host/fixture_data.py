"""Records shared by the corpus generator, the MCP fixture server and tests.

Everything is fictional and generated from a fixed seed, so the corpus, the
questions and the server data are reproducible. Document facts live in the
entity tables (products, sites, policies, projects, customers, incidents,
suppliers, courses); MCP-only facts live in the tool tables (directory
extensions, orders, stock, tickets, project status, invoices, rates,
shipping). Tool functions are pure, so the generator derives ground truths
from the same code the server exposes.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

_rng = random.Random("kestrel-instruments-v3")


def _pick(seq):
    return seq[_rng.randrange(len(seq))]


def _day(start: date, span_days: int) -> str:
    return (start + timedelta(days=_rng.randrange(span_days))).isoformat()


# --- People ---------------------------------------------------------------

FIRST_NAMES = [
    "Anders", "Mette", "Sofia", "Erik", "Ingrid", "Jonas", "Lars", "Kari", "Ola", "Nora",
    "Henrik", "Silje", "Magnus", "Thea", "Sindre", "Ida", "Kristian", "Maren", "Tobias",
    "Hanna", "Fredrik", "Live", "Martin", "Emilie", "Petter", "Julie", "Vegard", "Astrid",
    "Eirik", "Marte", "Johan", "Linnea", "Oskar", "Frida", "Mats", "Elin", "Jens", "Karin",
    "Niklas", "Greta", "Lukas", "Anna", "Felix", "Klara", "Mikkel", "Freja", "Rasmus",
    "Signe", "Paul", "Lena", "Daniel", "Maja", "Simon", "Tuva", "Viktor", "Selma", "Aksel",
    "Ronja", "Emil", "Helga",
]
LAST_NAMES = [
    "Solberg", "Haugen", "Dahl", "Nygaard", "Berg", "Lie", "Moe", "Hansen", "Johansen",
    "Larsen", "Andersen", "Pedersen", "Nilsen", "Kristiansen", "Jensen", "Karlsen",
    "Strand", "Lund", "Eriksen", "Holm", "Bakken", "Aas", "Halvorsen", "Iversen", "Sandvik",
    "Lindqvist", "Sjoberg", "Nordin", "Wallin", "Engstrom", "Hedlund", "Brandt", "Kruger",
    "Vogel", "Hartmann", "Lorenz", "Winkler", "Thomsen", "Mortensen", "Rasmussen",
    "Kjeldsen", "Vestergaard", "Brekke", "Fossum", "Rustad", "Tangen", "Skaug", "Hovland",
    "Myhre", "Ruud", "Aune", "Grondahl", "Bjerke", "Lunde", "Sletten", "Viken", "Ostby",
    "Rognes", "Furu", "Kvam",
]

SITES = [
    dict(id="KI-201", name="Oslo Headquarters", city="Oslo", country="Norway",
         head_title="Chief Executive Officer",
         facts=["The headquarters is located at Fjordveien 18 in Oslo.",
                "About 240 employees work at the Oslo headquarters.",
                "The Oslo headquarters hosts finance, product management and IT."]),
    dict(id="KI-202", name="Bergen Calibration Lab", city="Bergen", country="Norway",
         head_title="Head of Calibration Lab",
         facts=["The Bergen lab employs 12 calibration technicians.",
                "The Bergen lab is accredited to ISO/IEC 17025.",
                "The Bergen lab also hosts the customer support team."]),
    dict(id="KI-203", name="Stavanger Warehouse", city="Stavanger", country="Norway",
         head_title="Warehouse Manager",
         facts=["The Stavanger warehouse is open from 06:00 to 22:00 on working days.",
                "Orders placed before the carrier cutoff at 14:00 leave Stavanger the same day.",
                "All customer shipments leave from the Stavanger warehouse."]),
    dict(id="KI-204", name="Trondheim R&D Center", city="Trondheim", country="Norway",
         head_title="Director of Research",
         facts=["The Trondheim center employs 35 engineers.",
                "Trondheim focuses on wireless gateways and firmware development.",
                "The Trondheim center opened in 2016."]),
    dict(id="KI-205", name="Gothenburg Sales Office", city="Gothenburg", country="Sweden",
         head_title="Sales Director Nordics",
         facts=["The Gothenburg office serves customers in Sweden and Finland.",
                "Eighteen sales and application engineers work in Gothenburg.",
                "The Gothenburg office opened in 2019."]),
    dict(id="KI-206", name="Hamburg Service Center", city="Hamburg", country="Germany",
         head_title="Service Center Manager",
         facts=["The Hamburg service center repairs devices for customers in Central Europe.",
                "Standard repairs in Hamburg are completed within 7 working days.",
                "The Hamburg service center opened in 2021."]),
    dict(id="KI-207", name="Aarhus Production Plant", city="Aarhus", country="Denmark",
         head_title="Plant Manager",
         facts=["The Aarhus plant runs two shifts on working days.",
                "The Aarhus plant assembles all pressure and flow products.",
                "The Aarhus plant is certified to ISO 14001."]),
    dict(id="KI-208", name="Houston Office", city="Houston", country="United States",
         head_title="General Manager Americas",
         facts=["The Houston office handles sales and support for the Americas.",
                "Support in Houston is available from 07:00 to 18:00 Central Time.",
                "The Houston office opened in 2022."]),
]

DEPARTMENT_TITLES = {
    "Sales": ["Account Manager"] * 4 + ["Sales Engineer"],
    "Product": ["Product Manager", "Product Owner"],
    "Engineering": ["Firmware Engineer", "Hardware Engineer", "Test Engineer"],
    "Projects": ["Project Manager"],
    "Customer Support": ["Support Engineer"],
    "Quality": ["Quality Engineer", "Calibration Technician"],
    "Logistics": ["Logistics Coordinator"],
    "Finance": ["Financial Controller"],
    "HR": ["HR Business Partner"],
    "IT": ["IT Administrator"],
    "Training": ["Trainer"],
}
# Roles named in policies; one holder each, found via the directory.
POLICY_ROLES = {
    "Information Security Officer": "IT",
    "Data Protection Officer": "Legal",
    "Compliance Officer": "Legal",
    "Fleet Coordinator": "Logistics",
    "Health and Safety Officer": "Quality",
}


def _build_employees() -> list[dict]:
    names = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    _rng.shuffle(names)
    extensions = _rng.sample(range(1000, 9999), 160)
    employees: list[dict] = []

    def add(title: str, department: str, office: str) -> dict:
        employee = dict(name=names[len(employees)], title=title, department=department,
                        office=office, extension=str(extensions[len(employees)]))
        employees.append(employee)
        return employee

    for site in SITES:
        site["head"] = add(site["head_title"], "Management", site["city"])["name"]
    for title, department in POLICY_ROLES.items():
        add(title, department, "Oslo")
    while len(employees) < 160:
        department = _pick(sorted(DEPARTMENT_TITLES))
        add(_pick(DEPARTMENT_TITLES[department]), department, _pick(SITES)["city"])
    return employees


EMPLOYEES = _build_employees()


def _by_title(title: str) -> list[str]:
    return [e["name"] for e in EMPLOYEES if e["title"] == title]


# --- Products -------------------------------------------------------------

FAMILIES = [
    ("KH", "humidity sensor", "±{} %RH", [0.8, 1.0, 1.5, 2, 3]),
    ("KT", "temperature probe", "±{} K", [0.1, 0.2, 0.3, 0.5]),
    ("KP", "pressure transmitter", "±{} % of full scale", [0.1, 0.25, 0.5]),
    ("KF", "flow meter", "±{} % of reading", [0.5, 1, 2]),
    ("KG", "gas sensor", "±{} ppm", [20, 30, 40, 50]),
    ("KV", "vibration sensor", "±{} % of range", [1, 2, 5]),
    ("KL", "level sensor", "±{} mm", [1, 2, 3, 5]),
    ("KD", "data logger", "±{} % of reading", [0.1, 0.2, 0.5]),
    ("KW", "wireless gateway", "packet loss below {} %", [0.1, 0.5, 1]),
    ("KQ", "power quality meter", "±{} % of reading", [0.1, 0.2, 0.5]),
]
INTERFACES = ["Modbus RTU", "Modbus TCP", "CANopen", "EtherNet/IP", "PROFINET",
              "BACnet MS/TP", "HART", "IO-Link", "4-20 mA current loop", "LoRaWAN"]
FEATURES = ["improves power consumption in sleep mode", "adds TLS 1.3 support",
            "fixes a rare watchdog reset", "adds a configurable alarm hysteresis",
            "improves self-diagnostics", "speeds up the start-up sequence",
            "adds remote configuration backup", "fixes clock drift after power loss"]


def _build_products() -> dict[str, dict]:
    products: dict[str, dict] = {}
    sku = 1001
    for prefix, kind, accuracy_fmt, accuracies in FAMILIES:
        for number in sorted(_rng.sample(range(1, 10), 4)):
            code = f"{prefix}{number}"
            min_c = _pick([-40, -30, -25, -20, -10, 0])
            products[code] = dict(
                code=code, name=f"Kestrel {code}", family=prefix, kind=kind,
                sku=f"KS-{sku}", min_c=min_c,
                max_c=_pick([50, 60, 70, 80, 85, 100, 105, 125]),
                current_ma=_rng.randrange(5, 61), interface=_pick(INTERFACES),
                accuracy=accuracy_fmt.format(_pick(accuracies)),
                weight_g=_rng.randrange(60, 901, 5), price_eur=_rng.randrange(89, 1491),
                warranty_years=_pick([1, 2, 3, 5]),
                ip=_pick(["IP40", "IP54", "IP65", "IP66", "IP67", "IP68"]),
                launch_year=_rng.randrange(2015, 2026),
                firmware=f"{_rng.randrange(1, 6)}.{_rng.randrange(0, 10)}.{_rng.randrange(0, 10)}",
                firmware_date=_day(date(2025, 6, 1), 480), firmware_change=_pick(FEATURES),
            )
            sku += 1
    return products


PRODUCTS = _build_products()


# --- Policies (hand-written facts with their questions) -------------------
# (doc id, title, [(sentence, question, ground truth, answer patterns)])

def _n(value, *units):
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             10: "ten", 12: "twelve", 14: "fourteen", 15: "fifteen", 20: "twenty",
             30: "thirty", 60: "sixty", 90: "ninety"}
    forms = [str(value)] + ([words[value]] if value in words else [])
    return [[f"{f} {u}" for f in forms for u in units] + [f"{f}{u}" for f in forms for u in units]]


POLICIES = [
    ("KI-101", "Leave Policy", [
        ("Employees must give at least 14 days of notice before taking planned leave.",
         "How many days of notice must employees give before planned leave?", "14 days", _n(14, "days")),
        ("Unplanned leave due to illness is reported to the team lead on the first day of absence.",
         "On which day of absence must illness be reported to the team lead?", "the first day", [["first day"]]),
        ("At most 5 unused leave days can be carried over into the next calendar year.",
         "How many unused leave days can be carried over into the next year?", "5 days", _n(5, "days", "leave days", "unused leave days")),
    ]),
    ("KI-102", "Business Travel Policy (current, effective 2025)", [
        ("Flights shorter than 6 hours are booked in economy class.",
         "Below which flight duration are flights booked in economy class?", "6 hours", _n(6, "hours")),
        ("The daily per diem for business travel is 55 EUR.",
         "What is the current daily per diem for business travel?", "55 EUR", _n(55, "EUR", "euros", "€")),
        ("Hotel costs are reimbursed up to 160 EUR per night.",
         "Up to what amount per night are hotel costs reimbursed?", "160 EUR", _n(160, "EUR", "euros", "€")),
    ]),
    ("KI-103", "Expense Policy", [
        ("Expense reports must be submitted within 30 days after the expense was incurred.",
         "Within how many days must expense reports be submitted?", "30 days", _n(30, "days")),
        ("Receipts are required for every expense above 25 EUR.",
         "Above which amount is a receipt required for an expense?", "25 EUR", _n(25, "EUR", "euros", "€")),
        ("Expense reports are approved by the cost center owner.",
         "Who approves expense reports?", "the cost center owner", [["cost center owner", "cost centre owner"]]),
    ]),
    ("KI-104", "Remote Work Policy", [
        ("Employees may work remotely up to 3 days per week.",
         "How many days per week may employees work remotely?", "3 days", _n(3, "days")),
        ("Core hours, during which everyone is reachable, are 10:00 to 15:00.",
         "What are the core hours under the remote work policy?", "10:00 to 15:00",
         [["10:00", "10 am", "10am"], ["15:00", "3 pm", "3pm"]]),
        ("Each remote employee receives a one-time equipment allowance of 300 EUR.",
         "How large is the one-time equipment allowance for remote employees?", "300 EUR", _n(300, "EUR", "euros", "€")),
    ]),
    ("KI-105", "Information Security Policy", [
        ("Passwords must be changed every 180 days.",
         "How often must passwords be changed?", "every 180 days", _n(180, "days")),
        ("Multi-factor authentication is mandatory for all company accounts.",
         "Which authentication method is mandatory for all company accounts?", "multi-factor authentication",
         [["multi-factor", "multifactor", "mfa", "two-factor", "2fa"]]),
        ("A lost or stolen device must be reported to the Information Security Officer within 1 hour.",
         "Within what time must a lost or stolen device be reported?", "1 hour", _n(1, "hour")),
    ]),
    ("KI-106", "Warranty Policy", [
        ("The warranty period of each product is stated in its datasheet.",
         "Where is the warranty period of a product stated?", "in its datasheet", [["datasheet"]]),
        ("Warranty claims must be filed within 60 days after a defect is discovered.",
         "Within how many days after discovering a defect must a warranty claim be filed?", "60 days", _n(60, "days")),
        ("Repairs under warranty are completed within 10 working days.",
         "Within how many working days are warranty repairs completed?", "10 working days", _n(10, "working days", "business days", "days")),
    ]),
    ("KI-107", "Returns Policy", [
        ("Unused products can be returned within 30 days of delivery.",
         "Within how many days of delivery can unused products be returned?", "30 days", _n(30, "days")),
        ("Returned products are subject to a restocking fee of 15 %.",
         "What restocking fee applies to returned products?", "15 %", _n(15, "%", "percent")),
        ("Every return requires an RMA number issued by customer support.",
         "What number does every return require?", "an RMA number", [["rma"]]),
    ]),
    ("KI-108", "Incident Response Procedure", [
        ("Severity 1 incidents must receive a first response within 15 minutes.",
         "How quickly must a Severity 1 incident receive a first response?", "15 minutes", _n(15, "minutes")),
        ("Severity 2 incidents must receive a first response within 4 hours.",
         "How quickly must a Severity 2 incident receive a first response?", "4 hours", _n(4, "hours")),
        ("A written postmortem is due within 5 working days after a Severity 1 incident.",
         "Within how many working days is a postmortem due after a Severity 1 incident?", "5 working days",
         _n(5, "working days", "business days", "days")),
    ]),
    ("KI-109", "Onboarding Guide", [
        ("Every new employee is assigned an onboarding buddy.",
         "Who is assigned to every new employee during onboarding?", "an onboarding buddy", [["buddy"]]),
        ("The probation review takes place after 90 days.",
         "After how many days does the probation review take place?", "90 days", _n(90, "days")),
        ("New employees complete the safety induction within their first 2 weeks.",
         "Within how many weeks must new employees complete the safety induction?", "2 weeks", _n(2, "weeks")),
    ]),
    ("KI-110", "Software Release Process", [
        ("Every release requires two approvals from code owners.",
         "How many code-owner approvals does every release require?", "two approvals", _n(2, "approvals")),
        ("The weekly code freeze starts on Thursday at 12:00.",
         "On which weekday does the weekly code freeze start?", "Thursday", [["thursday"]]),
        ("Hotfixes during the freeze need approval from the on-call engineer.",
         "Who must approve hotfixes during the code freeze?", "the on-call engineer", [["on-call engineer", "on call engineer"]]),
    ]),
    ("KI-111", "Calibration Procedure", [
        ("Reference instruments are recalibrated every 6 months.",
         "How often are reference instruments recalibrated?", "every 6 months", _n(6, "months")),
        ("Calibration certificates are issued within 3 working days after a device is received.",
         "Within how many working days are calibration certificates issued?", "3 working days",
         _n(3, "working days", "business days", "days")),
        ("Each calibration tolerance report is archived for 10 years.",
         "For how many years is a calibration tolerance report archived?", "10 years", _n(10, "years")),
    ]),
    ("KI-112", "Procurement Policy", [
        ("Purchases above 5000 EUR require three competing quotes.",
         "Above which amount do purchases require three competing quotes?", "5000 EUR", [["5000 eur", "5000 euros", "€5000", "eur 5000"]]),
        ("New suppliers are approved by the purchasing board, which meets monthly.",
         "How often does the purchasing board meet?", "monthly", [["monthly", "once a month", "every month"]]),
        ("Framework agreements with suppliers run for at most 3 years.",
         "For how many years can a supplier framework agreement run at most?", "3 years", _n(3, "years")),
    ]),
    ("KI-113", "Data Protection Policy", [
        ("Personal customer data is deleted 24 months after the last contract ends.",
         "How many months after the last contract ends is personal customer data deleted?", "24 months",
         [["24 months", "24-month", "two years", "2 years"]]),
        ("Data protection incidents are reported to the Data Protection Officer within 24 hours.",
         "Within how many hours must data protection incidents be reported?", "24 hours", [["24 hours", "24-hour"]]),
        ("Customer data is stored only in data centers within the European Economic Area.",
         "Where may customer data be stored?", "within the European Economic Area",
         [["european economic area", "eea"]]),
    ]),
    ("KI-114", "Company Vehicle Policy", [
        ("Company cars are replaced after 5 years or 150000 km, whichever comes first.",
         "After how many years are company cars replaced?", "5 years", _n(5, "years")),
        ("Fuel cards may only be used for company vehicles.",
         "For which vehicles may fuel cards be used?", "only company vehicles", [["company vehicle", "company vehicles", "company car", "company cars"]]),
        ("Vehicle damage must be reported to the Fleet Coordinator within 48 hours.",
         "Within how many hours must vehicle damage be reported?", "48 hours", [["48 hours", "48-hour", "two days", "2 days"]]),
    ]),
    ("KI-115", "Training Budget Policy", [
        ("Each employee has an annual training budget of 1200 EUR.",
         "How large is the annual training budget per employee?", "1200 EUR", [["1200 eur", "1200 euros", "€1200", "eur 1200"]]),
        ("External training requires approval from the line manager.",
         "Who approves external training?", "the line manager", [["line manager"]]),
        ("Unused training budget does not carry over to the next year.",
         "Does unused training budget carry over to the next year?", "no, it does not carry over",
         [["not carry", "does not", "doesn't", "no,", "no.", "cannot"]]),
    ]),
    ("KI-116", "On-Call Policy", [
        ("On-call engineers receive a weekly allowance of 450 EUR.",
         "How large is the weekly on-call allowance?", "450 EUR", _n(450, "EUR", "euros", "€")),
        ("An on-call shift lasts 7 days and starts on Monday at 08:00.",
         "How many days does an on-call shift last?", "7 days", _n(7, "days")),
        ("On-call engineers must acknowledge a page within 10 minutes.",
         "Within how many minutes must an on-call engineer acknowledge a page?", "10 minutes", _n(10, "minutes")),
    ]),
    ("KI-117", "Parental Leave Policy", [
        ("Kestrel tops up statutory parental benefits to full salary for 20 weeks.",
         "For how many weeks does Kestrel top up parental benefits to full salary?", "20 weeks", _n(20, "weeks")),
        ("Parental leave must be announced at least 12 weeks in advance.",
         "How many weeks in advance must parental leave be announced?", "12 weeks", _n(12, "weeks")),
        ("Employees returning from parental leave may work 80 % for up to one year.",
         "What working time may employees choose after returning from parental leave?", "80 %", [["80 %", "80%", "80 percent"]]),
    ]),
    ("KI-118", "Code of Conduct", [
        ("Gifts from business partners above 50 EUR must be declined.",
         "Above which value must gifts from business partners be declined?", "50 EUR", _n(50, "EUR", "euros", "€")),
        ("Conflicts of interest are disclosed to the Compliance Officer.",
         "To whom are conflicts of interest disclosed?", "the Compliance Officer", [["compliance officer"]]),
        ("Every employee completes the code of conduct training once a year.",
         "How often must employees complete the code of conduct training?", "once a year",
         [["once a year", "annually", "every year", "yearly"]]),
    ]),
    ("KI-119", "Workplace Safety Policy", [
        ("Safety shoes are mandatory in the Aarhus plant and the Stavanger warehouse.",
         "Where are safety shoes mandatory?", "Aarhus plant and Stavanger warehouse", [["aarhus"], ["stavanger"]]),
        ("Near misses are reported to the Health and Safety Officer within 24 hours.",
         "Within how many hours must near misses be reported?", "24 hours", [["24 hours", "24-hour"]]),
        ("Fire drills take place twice a year at every site.",
         "How often do fire drills take place?", "twice a year", [["twice a year", "two times a year", "2 times a year", "twice per year"]]),
    ]),
    ("KI-120", "Environmental Policy", [
        ("Kestrel aims to cut its scope 1 and 2 emissions by 50 % by 2030.",
         "By what percentage does Kestrel aim to cut scope 1 and 2 emissions by 2030?", "50 %", _n(50, "%", "percent")),
        ("All sites have used renewable electricity since 2023.",
         "Since which year have all sites used renewable electricity?", "2023", [["2023"]]),
        ("Returned devices are refurbished or recycled through a certified partner.",
         "What happens to returned devices under the environmental policy?", "refurbished or recycled",
         [["re:refurbish", "re:recycl"]]),
    ]),
]

DISTRACTOR_POLICIES = [
    ("KI-901", "Business Travel Policy 2022 (superseded)", [
        "This policy was superseded in 2025 and is kept for audit purposes only.",
        "Under this superseded policy the daily per diem was 45 EUR.",
        "Hotel costs were reimbursed up to 130 EUR per night.",
    ]),
    ("KI-902", "Remote Work Policy 2021 (superseded)", [
        "This policy was replaced by the current remote work policy.",
        "It allowed remote work on up to 2 days per week.",
        "The equipment allowance was 200 EUR.",
    ]),
    ("KI-903", "Expense Policy 2020 (superseded)", [
        "This version is no longer valid.",
        "Expense reports had to be submitted within 45 days.",
        "Receipts were required above 50 EUR.",
    ]),
]


# --- Customers, projects, incidents, suppliers, courses --------------------

COUNTRY_SUFFIX = {"AS": ("Norway", "NO"), "AB": ("Sweden", "SE"), "GmbH": ("Germany", "DE"),
                  "ApS": ("Denmark", "DK"), "Oy": ("Finland", "FI"), "Inc.": ("United States", "US")}
SECTORS = {"Aqua": "aquaculture", "Energi": "energy", "Foods": "food processing",
           "Pharma": "pharmaceuticals", "Logistics": "logistics", "Marine": "shipping",
           "Data": "data center", "Bygg": "construction", "Kraft": "hydropower",
           "Mining": "mining"}
NAME_PREFIXES = ["Nordvik", "Brekke", "Holmen", "Lundby", "Vassdal", "Fjellheim", "Bergvik",
                 "Strandli", "Dalsbo", "Elvebakk", "Skogly", "Haugland", "Vikholm", "Nessby",
                 "Torvik", "Aasen", "Ramstad", "Kolbu", "Sandnes", "Lyngby", "Hedberg",
                 "Ostrand", "Falkberg", "Granlund", "Kvist", "Rosen", "Ahlberg", "Wester",
                 "Kronborg", "Halland"]
TIERS = {"Bronze": 48, "Silver": 24, "Gold": 8, "Platinum": 4}


def _build_customers() -> list[dict]:
    managers = _by_title("Account Manager")
    customers = []
    prefixes = _rng.sample(NAME_PREFIXES, 30) + _rng.sample(NAME_PREFIXES, 10)
    seen = set()
    for i, prefix in enumerate(prefixes):
        while True:
            sector = _pick(sorted(SECTORS))
            suffix = _pick(sorted(COUNTRY_SUFFIX))
            name = f"{prefix} {sector} {suffix}"
            if name not in seen:
                seen.add(name)
                break
        country, code = COUNTRY_SUFFIX[suffix]
        tier = _pick(sorted(TIERS))
        customers.append(dict(
            id=f"CUST-{101 + i}", doc=f"KI-{301 + i}", name=name, country=country,
            country_code=code, industry=SECTORS[sector], manager=_pick(managers), tier=tier,
            sla_hours=TIERS[tier], since=_rng.randrange(2008, 2025),
        ))
    return customers


CUSTOMERS = _build_customers()

PROJECT_WORDS = (["Harbor", "Fjord", "Glacier", "Aurora", "Northern", "Coastal", "Arctic",
                  "Pine", "Granite", "Tidal", "Summit", "Polar", "Delta", "Birch"],
                 ["Monitoring", "Metering", "Telemetry", "Automation", "Diagnostics",
                  "Analytics", "Leak Detection", "Climate Control"],
                 ["Upgrade", "Rollout", "Pilot", "Retrofit", "Expansion"])


def _build_projects() -> list[dict]:
    managers = _by_title("Project Manager")
    names = sorted({f"{a} {b} {c}" for a in PROJECT_WORDS[0] for b in PROJECT_WORDS[1]
                    for c in PROJECT_WORDS[2]})
    chosen = _rng.sample(names, 40)
    projects = []
    for i, name in enumerate(chosen):
        start = date(2025, 1, 6) + timedelta(days=_rng.randrange(500))
        projects.append(dict(
            code=f"PRJ-{201 + i}", doc=f"KI-{401 + i}", name=name,
            customer=_pick(CUSTOMERS)["name"], lead=_pick(managers),
            budget_eur=_rng.randrange(80, 1800) * 500, start=start.isoformat(),
            deadline=(start + timedelta(days=_rng.randrange(120, 700))).isoformat(),
            site=_pick(SITES)["city"], product=_pick(sorted(PRODUCTS)),
        ))
    return projects


PROJECTS = _build_projects()

ROOT_CAUSES = [
    ("a failed electrolytic capacitor in the power supply", [["capacitor"]]),
    ("a firmware memory leak", [["memory leak"]]),
    ("water ingress through a damaged cable gland", [["cable gland", "water ingress"]]),
    ("an expired TLS certificate on the gateway", [["certificate"]]),
    ("a misconfigured Modbus address", [["modbus address", "misconfigured"]]),
    ("a corroded connector", [["re:corrod"]]),
    ("a faulty batch of sensing elements from a supplier", [["re:sensing elements?", "faulty batch"]]),
    ("a network switch failure at the customer site", [["switch"]]),
    ("a clock synchronisation error", [["clock", "re:synchroni"]]),
    ("lightning damage to the field wiring", [["lightning"]]),
    ("an incorrect calibration coefficient", [["calibration coefficient"]]),
    ("radio interference from a nearby transmitter", [["interference"]]),
]


def _build_incidents() -> list[dict]:
    incidents = []
    numbers = sorted(_rng.sample(range(1, 400), 40))
    for i, number in enumerate(numbers):
        product = PRODUCTS[_pick(sorted(PRODUCTS))]
        cause, cause_patterns = _pick(ROOT_CAUSES)
        incidents.append(dict(
            id=f"INC-2026-{number:03d}", doc=f"KI-{501 + i}", date=_day(date(2025, 9, 1), 400),
            customer=_pick(CUSTOMERS)["name"], product=product["code"],
            severity=_pick([1, 2, 2, 3, 3]), cause=cause, cause_patterns=cause_patterns,
            downtime_min=_rng.randrange(5, 721),
            fixed_in=f"{_rng.randrange(1, 6)}.{_rng.randrange(0, 10)}.{_rng.randrange(0, 10)}",
        ))
    return incidents


INCIDENTS = _build_incidents()

COMPONENTS = ["humidity sensing elements", "stainless steel housings", "printed circuit boards",
              "LCD displays", "lithium batteries", "radio modules", "cable glands",
              "pressure diaphragms", "ultrasonic transducers", "gas sensing cells",
              "M12 connectors", "power supply units", "microcontrollers", "antennas",
              "gaskets", "aluminium enclosures", "thermistors", "flash memory chips",
              "packaging materials", "calibration gases"]
SUPPLIER_WORDS = (["Nordic", "Baltic", "Alpine", "Rhine", "Danube", "Atlas", "Vega", "Orion",
                   "Helix", "Pioneer"], ["Components", "Industries", "Electronics", "Supply",
                                         "Technik", "Parts"])
SUPPLIER_COUNTRIES = ["Germany", "Poland", "Sweden", "Netherlands", "Czech Republic", "Taiwan",
                      "Japan", "Estonia", "Italy", "Finland"]


def _build_suppliers() -> list[dict]:
    names = _rng.sample(sorted({f"{a} {b}" for a in SUPPLIER_WORDS[0] for b in SUPPLIER_WORDS[1]}), 20)
    return [dict(name=name, doc=f"KI-{601 + i}", component=COMPONENTS[i],
                 country=_pick(SUPPLIER_COUNTRIES), lead_weeks=_rng.randrange(2, 21),
                 contract_until=_rng.randrange(2026, 2031), rating=_pick(["A", "A", "B", "C"]))
            for i, name in enumerate(names)]


SUPPLIERS = _build_suppliers()

COURSE_NAMES = ["Modbus Fundamentals", "Sensor Calibration Basics", "Advanced Calibration",
                "Gateway Configuration", "Firmware Update Workshop", "Industrial Networking",
                "Hazardous Area Installation", "Flow Measurement in Practice",
                "Gas Detection Essentials", "Vibration Analysis", "Data Logger Programming",
                "Power Quality Analysis", "Customer Support Excellence", "Project Management Basics",
                "Root Cause Analysis"]


def _build_courses() -> list[dict]:
    trainers = _by_title("Trainer")
    courses = []
    for i, name in enumerate(COURSE_NAMES):
        courses.append(dict(name=name, doc=f"KI-{701 + i}", days=_rng.randrange(1, 6),
                            max_participants=_rng.randrange(6, 21),
                            cost_eur=_rng.randrange(4, 40) * 50, trainer=_pick(trainers),
                            prerequisite=COURSE_NAMES[i - 1] if i % 3 == 2 else None))
    return courses


COURSES = _build_courses()

DISCONTINUED = [
    dict(doc=f"KI-{911 + i}", code=f"{code}-L", successor=code, year=_rng.randrange(2018, 2024),
         max_c=PRODUCTS[code]["max_c"] - 15, sku=f"KS-0{900 + i}")
    for i, code in enumerate(_rng.sample(sorted(PRODUCTS), 4))
]


# --- MCP-only tables ------------------------------------------------------

ORDER_STATUSES = ["shipped", "delivered", "in production", "awaiting payment", "cancelled",
                  "ready for dispatch", "on hold"]
CARRIERS = ["DHL", "PostNord", "Bring", "UPS", "DB Schenker", "FedEx"]


def _build_orders() -> dict[str, dict]:
    orders = {}
    for i in range(300):
        product = PRODUCTS[_pick(sorted(PRODUCTS))]
        quantity = _rng.randrange(1, 61)
        status = _pick(ORDER_STATUSES)
        orders[f"ORD-{1001 + i}"] = dict(
            customer_id=_pick(CUSTOMERS)["id"], sku=product["sku"], quantity=quantity,
            status=status,
            carrier=_pick(CARRIERS) if status in {"shipped", "delivered"} else None,
            date=_day(date(2026, 6, 1), 150),
            weight_kg=round(quantity * product["weight_g"] / 1000 + 0.3, 1),
        )
    return orders


ORDERS = _build_orders()

_zero_stock = set(_rng.sample(sorted(p["sku"] for p in PRODUCTS.values()), 6))
STOCK = {
    p["sku"]: dict(units=0, restock_date=_day(date(2026, 11, 2), 60))
    if p["sku"] in _zero_stock else dict(units=_rng.randrange(1, 650), restock_date=None)
    for p in PRODUCTS.values()
}

TICKET_ISSUES = ["condensation on the housing", "Modbus timeouts", "node ID resets",
                 "calibration drift", "zero-point offset", "air bubbles distort readings",
                 "slow warm-up", "display flicker", "battery drains quickly", "lost radio link"]
TICKETS = {code: (_rng.randrange(0, 13), _pick(TICKET_ISSUES)) for code in sorted(PRODUCTS)}
TICKETS = {code: (count, issue if count else None) for code, (count, issue) in TICKETS.items()}

PROJECT_STATUS = {
    p["code"]: dict(budget_spent_eur=_rng.randrange(5, 96) * p["budget_eur"] // 100,
                    percent_complete=_rng.randrange(3, 100),
                    hours_booked=_rng.randrange(40, 6000),
                    risk=_pick(["low", "medium", "high"]))
    for p in PROJECTS
}

INVOICES = {
    c["id"]: dict(outstanding_eur=round(_rng.randrange(0, 250000) + _rng.random(), 2),
                  overdue_invoices=_rng.randrange(0, 9))
    for c in CUSTOMERS
}

EUR_RATES = {"EUR": 1.0, "USD": 1.08, "NOK": 11.70, "SEK": 11.45, "DKK": 7.46, "GBP": 0.85,
             "CHF": 0.94, "PLN": 4.31, "JPY": 162.4}

SHIPPING = {  # country code: (base EUR, EUR per kg)
    "NO": (9.0, 1.2), "SE": (14.0, 2.0), "DK": (15.0, 2.1), "FI": (17.0, 2.3),
    "DE": (19.0, 2.5), "NL": (21.0, 2.6), "FR": (23.0, 2.8), "PL": (22.0, 2.4),
    "UK": (26.0, 3.1), "US": (48.0, 6.0), "CA": (52.0, 6.4), "JP": (61.0, 7.2),
}
COUNTRY_NAMES = {"NO": "Norway", "SE": "Sweden", "DK": "Denmark", "FI": "Finland",
                 "DE": "Germany", "NL": "the Netherlands", "FR": "France", "PL": "Poland",
                 "UK": "the United Kingdom", "US": "the United States", "CA": "Canada",
                 "JP": "Japan"}


# --- Tool functions -------------------------------------------------------

class ToolError(ValueError):
    """Raised for unknown inputs; the server reports it as a tool error."""


def product_by_sku(sku: str) -> dict:
    for product in PRODUCTS.values():
        if product["sku"] == sku:
            return product
    raise ToolError(f"Unknown SKU {sku!r}")


def search_directory(query: str) -> list[dict]:
    needle = query.strip().lower()
    hits = [e for e in EMPLOYEES if needle and any(
        needle in str(e[field]).lower() for field in ("name", "title", "department", "office"))]
    if not hits:
        raise ToolError(f"No employee matches {query!r}")
    return hits[:25]


def get_order_status(order_id: str) -> dict:
    key = order_id.strip().upper()
    order = ORDERS.get(key)
    if order is None:
        raise ToolError(f"Unknown order {order_id!r}")
    product = product_by_sku(order["sku"])
    return {"order_id": key, "product": product["name"], "product_code": product["code"],
            **order, "order_value_eur": round(order["quantity"] * product["price_eur"], 2)}


def get_stock_level(sku: str) -> dict:
    key = sku.strip().upper()
    entry = STOCK.get(key)
    if entry is None:
        raise ToolError(f"Unknown SKU {sku!r}; use an SKU like KS-1007")
    return {"sku": key, "warehouse": "Stavanger", **entry}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    src, dst = from_currency.strip().upper(), to_currency.strip().upper()
    if src not in EUR_RATES or dst not in EUR_RATES:
        raise ToolError(f"Supported currencies: {', '.join(EUR_RATES)}")
    rate = EUR_RATES[dst] / EUR_RATES[src]
    return {"amount": amount, "from": src, "to": dst, "rate": round(rate, 6),
            "converted": round(amount * rate, 2)}


def count_open_tickets(product_code: str) -> dict:
    code = product_code.strip().upper().removeprefix("KESTREL ").strip()
    if code not in TICKETS:
        raise ToolError(f"Unknown product {product_code!r}; use a code like KH3")
    count, issue = TICKETS[code]
    return {"product_code": code, "open_tickets": count, "most_common_issue": issue}


def get_shipping_quote(weight_kg: float, destination_country: str) -> dict:
    country = destination_country.strip().upper()
    if country not in SHIPPING:
        raise ToolError(f"Supported destination country codes: {', '.join(SHIPPING)}")
    base, per_kg = SHIPPING[country]
    return {"weight_kg": weight_kg, "destination_country": country,
            "price_eur": round(base + per_kg * weight_kg, 2)}


def get_project_status(project_code: str) -> dict:
    key = project_code.strip().upper()
    if key not in PROJECT_STATUS:
        raise ToolError(f"Unknown project {project_code!r}; use a code like PRJ-201")
    return {"project_code": key, **PROJECT_STATUS[key]}


def get_invoice_balance(customer_id: str) -> dict:
    key = customer_id.strip().upper()
    if key not in INVOICES:
        raise ToolError(f"Unknown customer ID {customer_id!r}; use an ID like CUST-101")
    return {"customer_id": key, **INVOICES[key]}


def get_weather(city: str) -> dict:
    """Decoy tool: no benchmark question needs weather data."""
    return {"city": city, "forecast": "overcast", "temperature_c": 9}


def get_stock_price(ticker: str) -> dict:
    """Decoy tool: no benchmark question needs share prices."""
    return {"ticker": ticker.upper(), "price": 101.5, "currency": "USD"}


TOOLS = {
    fn.__name__: fn
    for fn in (search_directory, get_order_status, get_stock_level, convert_currency,
               count_open_tickets, get_shipping_quote, get_project_status,
               get_invoice_balance, get_weather, get_stock_price)
}
DECOY_TOOLS = frozenset({"get_weather", "get_stock_price"})
