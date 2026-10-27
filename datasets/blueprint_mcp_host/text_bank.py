"""Prose building blocks for the blueprint_mcp_host corpus.

Paragraphs are filled with each document's own names (``{name}``,
``{customer}``, ...), so no paragraph repeats verbatim across documents. The
bank deliberately contains no digits: numbers come only from
``fixture_data``, so filler text can neither contradict a labelled fact nor
leak an MCP-only value into the corpus.
"""

DATASHEET = {
    "overview": [
        "The {name} is designed for continuous operation in demanding industrial environments. "
        "It combines a robust enclosure with a measurement core that Kestrel Instruments develops "
        "and calibrates in-house, so readings stay stable over long service intervals.",
        "Typical applications of the {name} include process monitoring, building automation and "
        "environmental control. The device integrates into existing control systems and does not "
        "need proprietary software for basic operation.",
        "Every {name} leaves the factory with an individual test report. The report documents the "
        "final functional test and the configuration loaded before shipment, which simplifies "
        "acceptance testing on site.",
        "The {name} belongs to the Kestrel {kind} range and shares accessories, mounting kits and "
        "configuration tools with the other models of its family. Spare parts remain available "
        "for the whole support period of the product.",
    ],
    "safety": [
        "Only qualified personnel may install, commission or service the {name}. Read this "
        "document completely before working on the device and keep it available at the place "
        "of installation.",
        "Disconnect the supply voltage before opening the terminal compartment of the {name}. "
        "Work on live circuits is permitted only where national regulations allow it and the "
        "necessary protective measures are in place.",
        "Do not operate the {name} outside the conditions stated in the specification table. "
        "Operation outside these limits can damage the device and void the warranty.",
        "Modifications to the {name} that are not described in this document are not permitted. "
        "Use only original spare parts and accessories supplied by Kestrel Instruments.",
        "If the housing of the {name} is visibly damaged, take the device out of service and "
        "contact customer support before reconnecting it.",
    ],
    "mechanical": [
        "Mount the {name} on a vibration-free surface away from direct sunlight and heat sources. "
        "Leave enough clearance around the housing for cable entry and later maintenance work.",
        "The {name} can be mounted on a wall bracket, a pipe clamp or a standard mounting rail. "
        "Choose the mounting kit that matches the installation and tighten the screws evenly.",
        "Orient the cable entries of the {name} downwards when installing outdoors, so that "
        "condensation cannot collect in the terminal compartment.",
        "Where the {name} is installed in a process line, use the supplied adapter and check the "
        "sealing surfaces for damage before tightening the process connection.",
        "After mechanical installation, check that the {name} is firmly seated and that the "
        "housing cover closes without force.",
    ],
    "electrical": [
        "Route signal cables to the {name} separately from power cables and connect the cable "
        "shield on the controller side only, to avoid ground loops.",
        "Use shielded twisted-pair cable for the communication line of the {name} and terminate "
        "the bus at both ends according to the rules of the fieldbus in use.",
        "Connect the protective earth of the {name} before connecting the supply. Check the "
        "polarity of all terminals against the wiring diagram on the inside of the cover.",
        "The terminals of the {name} accept solid and stranded conductors. Use ferrules for "
        "stranded wire and tighten the terminal screws with a suitable screwdriver.",
        "Install a suitable fuse in the supply line of the {name} and protect the signal lines "
        "against surges where long cable runs leave the building.",
    ],
    "commissioning": [
        "Switch on the supply and wait until the status LED of the {name} indicates normal "
        "operation. The device runs a self-test after every start.",
        "Compare the first readings of the {name} with a reference instrument and document the "
        "result in the commissioning report.",
        "Set the communication address of the {name} before connecting it to a bus with other "
        "devices, otherwise address conflicts can interrupt communication.",
        "Once commissioning is complete, close the housing of the {name}, seal unused cable "
        "entries and record the configuration in the asset management system.",
    ],
    "configuration": [
        "The {name} is configured with the Kestrel Configurator, either locally through the "
        "service interface or remotely over the fieldbus.",
        "Configuration profiles of the {name} can be exported and imported, which speeds up the "
        "commissioning of many identical measuring points.",
        "Alarm limits, filter settings and engineering units of the {name} can be changed during "
        "operation. Changes take effect immediately and are logged in the device event list.",
        "Access to the configuration of the {name} can be protected with a user code. Without the "
        "code, settings are visible but cannot be changed.",
        "A factory reset restores the delivery configuration of the {name}. The calibration data "
        "is kept, but communication settings must be entered again.",
    ],
    "operation": [
        "During normal operation the {name} needs no user intervention. Measured values and "
        "diagnostic information are available continuously through the communication interface.",
        "The status LED of the {name} shows normal operation, maintenance requests and faults "
        "with different colours and blink patterns.",
        "Diagnostic messages of the {name} follow the NAMUR recommendation and distinguish between "
        "failure, function check, out of specification and maintenance required.",
        "Readings of the {name} can be logged in the control system or in a Kestrel data logger "
        "for later analysis.",
    ],
    "maintenance": [
        "Inspect the {name} during scheduled plant shutdowns, clean the housing with a dry cloth, "
        "and record every intervention in the asset log.",
        "Check the cable glands and seals of the {name} for wear during each inspection and "
        "replace them if they have become brittle.",
        "Recalibrate the {name} according to the calibration procedure of Kestrel Instruments or "
        "send it to an accredited calibration lab.",
        "Firmware updates for the {name} are distributed through the Kestrel service portal. "
        "Read the release notes before updating devices in the field.",
        "Do not use solvents or high-pressure cleaners on the {name}; they can damage the seals "
        "and the sensing element.",
    ],
    "disposal": [
        "At the end of its service life, return the {name} to Kestrel Instruments or dispose of it "
        "through a certified electronics recycler in line with local regulations.",
        "Remove batteries and accessories from the {name} before disposal where applicable and "
        "dispose of them separately.",
    ],
    "troubleshooting": [
        ("No reading", "No supply voltage", "Check fuse, wiring and supply polarity"),
        ("Unstable reading", "Electrical interference", "Check shielding and cable routing"),
        ("No communication", "Wrong bus address", "Set a unique address and restart"),
        ("Status LED shows a fault", "Internal self-test failed", "Restart; contact support if it persists"),
        ("Reading drifts slowly", "Sensing element contaminated", "Clean or recalibrate the device"),
        ("Values frozen", "Device in function-check mode", "Leave service mode in the configurator"),
        ("Intermittent communication", "Missing bus termination", "Terminate the bus at both ends"),
        ("Condensation in housing", "Cable entries facing upwards", "Remount with entries facing down"),
        ("Configuration rejected", "User code active", "Enter the user code and retry"),
        ("Alarm without cause", "Alarm limits too tight", "Review limits and hysteresis"),
    ],
    "faq": [
        ("Can the {name} be used outdoors?", "Yes, provided the enclosure rating stated in the "
         "specification table matches the installation conditions and the cable entries face downwards."),
        ("Do I need special software for the {name}?", "No. The device works with standard "
         "control systems; the Kestrel Configurator is only needed for advanced settings."),
        ("How do I find the firmware version of my {name}?", "The configurator shows it on the "
         "device information page; the current release is listed in the firmware release notes."),
        ("Can I replace the sensing element of the {name} myself?", "Only trained service staff "
         "should replace it, because the device must be recalibrated afterwards."),
        ("Where do I order spare parts for the {name}?", "Spare parts are ordered through the "
         "Kestrel sales office responsible for your country."),
        ("Is the {name} suitable for hazardous areas?", "Only variants with an explicit hazardous "
         "area certification may be used there; check the nameplate before installation."),
        ("How is the {name} protected against unauthorised changes?", "A user code can lock the "
         "configuration; without it settings can be read but not changed."),
        ("Can the {name} be mounted next to a frequency converter?", "Keep as much distance as "
         "possible and use shielded cables to limit electromagnetic interference."),
    ],
}

RELEASE = {
    "intro": [
        "These release notes list the firmware history of the {family} series. Install the latest "
        "release unless a project specification requires a validated older version.",
        "Firmware for the {family} series is cumulative: every release contains all fixes of the "
        "earlier releases. Downgrades are possible but not recommended.",
        "The {family} series shares one firmware code base, but each model receives its own "
        "signed image. Images cannot be exchanged between models.",
    ],
    "procedure": [
        "Download the firmware image for the {family} series from the Kestrel service portal and "
        "verify the signature before distributing it.",
        "Devices of the {family} series keep their configuration during an update. Create a "
        "configuration backup anyway before updating devices in the field.",
        "If an update of a {family} device fails, the previous image stays active and the update "
        "can be repeated. Roll back through the service portal if necessary.",
        "Plan updates of {family} devices for maintenance windows, because measurements are "
        "interrupted while the new image is activated.",
    ],
    "known_issues": [
        "Readings may pause briefly after a configuration import.",
        "The event list is cleared after a factory reset.",
        "Older configurator versions show the firmware date in the wrong time zone.",
        "Remote configuration backups fail on networks with very high latency.",
        "The status LED can flash briefly during the self-test after an update.",
        "Some diagnostic texts are only available in English.",
    ],
}

POLICY = {
    "purpose": [
        "This document describes the rules of the {title} at Kestrel Instruments and explains why "
        "they exist. It gives employees and managers a common reference for daily decisions.",
        "The {title} protects employees, customers and the company by setting clear and fair "
        "expectations. It applies equally at every Kestrel site.",
        "The {title} turns legal requirements and company values into practical rules. Where local "
        "law sets stricter requirements, local law takes precedence.",
    ],
    "scope": [
        "The {title} applies to all employees and contractors of Kestrel Instruments unless a local "
        "works agreement states otherwise.",
        "Temporary staff and interns follow the {title} in the same way as permanent employees.",
        "Managers are responsible for making sure that their teams know the {title} and apply it.",
    ],
    "definitions": [
        ("Employee", "a person with an employment contract with Kestrel Instruments"),
        ("Manager", "an employee with formal responsibility for a team or cost center"),
        ("Contractor", "a person working for Kestrel through a service agreement"),
        ("Site", "any office, plant, lab or warehouse operated by Kestrel Instruments"),
        ("Working day", "a weekday that is not a public holiday at the employee's site"),
        ("Intranet", "the internal information portal of Kestrel Instruments"),
    ],
    "responsibilities": [
        "Employees follow the {title} and ask their manager or HR when something is unclear.",
        "Managers apply the {title} consistently and document exceptions they approve.",
        "HR maintains the {title}, answers questions about it and reports on its use to the "
        "management team.",
        "The document owner of the {title} reviews it regularly and publishes changes on the "
        "intranet.",
    ],
    "exceptions": [
        "Exceptions to the {title} require written approval from the responsible manager and HR.",
        "Approved exceptions to the {title} are documented and reviewed when the policy is updated.",
        "Repeated or deliberate violations of the {title} can lead to disciplinary measures.",
    ],
    "faq": [
        ("Where can I find the current version of the {title}?", "The current version is always "
         "published on the intranet; printed copies may be outdated."),
        ("Who answers questions about the {title}?", "Your manager is the first contact; HR helps "
         "with questions the manager cannot answer."),
        ("What happens if the {title} conflicts with local law?", "Local law always takes "
         "precedence, and HR documents the deviation."),
        ("Are earlier versions of the {title} still valid?", "No. Earlier versions are archived "
         "for audit purposes only and must not be applied."),
    ],
}

SITE = {
    "overview": [
        "The {name} is part of the Kestrel Instruments network of sites and works closely with the "
        "headquarters and the other locations.",
        "Teams at the {name} in {city} collaborate daily with colleagues at other sites through "
        "shared tools and regular video meetings.",
        "The {name} has grown steadily since it opened and is an important part of the company's "
        "presence in {country}.",
    ],
    "facilities": [
        "Meeting rooms with video conferencing equipment",
        "Staff canteen and coffee corners",
        "Bicycle parking and showers",
        "Visitor reception and badge office",
        "Quiet rooms for focused work",
        "First aid room and defibrillator",
        "Charging points for electric vehicles",
        "Secure storage for test equipment",
    ],
    "visitors": [
        "Visitors to the {name} register at the reception and receive a badge that must be worn at "
        "all times. Safety instructions are handed out during registration.",
        "Guests of the {name} are collected at the reception by their host and accompanied in "
        "production and laboratory areas.",
        "Deliveries for the {name} are accepted at the goods entrance and registered by the "
        "facilities team.",
    ],
    "getting_there": [
        "The {name} can be reached by public transport from the city center of {city}. Parking for "
        "visitors is available next to the main entrance.",
        "Detailed directions to the {name} are sent with every meeting invitation. Taxis can be "
        "ordered at the reception.",
    ],
}

CUSTOMER = {
    "history": [
        "{customer} is {article} {industry} company based in {country}. The relationship started with "
        "a small pilot installation and has expanded to several plants since then.",
        "{customer} uses Kestrel equipment mainly for monitoring and quality assurance in its "
        "{industry} operations.",
        "Over the years {customer} has standardised several measuring points on Kestrel products, "
        "which simplifies spare parts and training on their side.",
        "{customer} values short response times and local support, which is why the account is "
        "served by a dedicated account manager.",
    ],
    "meeting": [
        "Discussed the operating experience of the installed devices. The customer is satisfied "
        "with stability and asked for a training session for new technicians.",
        "Reviewed open support cases and agreed on a joint maintenance plan for the next season.",
        "Presented the product roadmap. The customer is interested in wireless gateways for remote "
        "measuring points.",
        "Clarified invoicing questions; finance will send an updated account statement.",
        "Agreed to evaluate a firmware update on a test installation before the rollout.",
        "The customer asked for a quotation for spare parts and calibration services.",
    ],
    "next_steps": [
        "Send a quotation for spare parts",
        "Schedule a technician training",
        "Share the latest firmware release notes",
        "Prepare a proposal for a wireless pilot",
        "Invite the customer to the user forum",
        "Review the service contract before renewal",
        "Organise a site visit with product management",
    ],
}

PROJECT = {
    "background": [
        "{customer} commissioned project {project} to modernise measurement and monitoring at one "
        "of its sites. The existing equipment had reached the end of its service life.",
        "Project {project} replaces manual readings with continuous monitoring, so that "
        "{customer} can react to deviations faster.",
        "The goal of project {project} is to improve data quality and reduce maintenance effort for "
        "{customer}.",
    ],
    "objectives": [
        "Replace legacy instruments with current Kestrel devices",
        "Integrate all measuring points into the customer's control system",
        "Reduce manual inspection rounds",
        "Provide remote diagnostics for the service team",
        "Train the customer's technicians",
        "Document the installation for later audits",
        "Hand over a validated configuration backup",
    ],
    "risks": [
        ("Delayed site access", "Agree access windows early with the customer"),
        ("Supply bottlenecks for components", "Order critical parts at project start"),
        ("Interfaces to legacy systems", "Run an interface test in the first project phase"),
        ("Staff availability during holidays", "Plan deputies for all key roles"),
        ("Scope changes", "Handle changes through the standard change process"),
        ("Weather conditions for outdoor work", "Schedule outdoor installation with buffer time"),
    ],
    "governance": [
        "Status reports are shared with {customer} every two weeks. Change requests follow the "
        "standard change process and need written approval.",
        "A steering committee with representatives of {customer} and Kestrel meets at every "
        "milestone of project {project}.",
        "Live figures of project {project} such as budget spent and percent complete are tracked "
        "in the project system, not in this brief.",
    ],
}

INCIDENT = {
    "summary": [
        "This report documents incident {incident} at {customer}. It follows the blameless "
        "postmortem format and focuses on causes and improvements, not on individuals.",
        "Incident {incident} interrupted measurements at {customer} and was handled by customer "
        "support together with engineering.",
    ],
    "timeline": [
        "Customer reported missing readings",
        "Support engineer acknowledged the case",
        "Remote diagnosis started",
        "Engineering joined the investigation",
        "Workaround applied on site",
        "Measurements restored",
        "Customer confirmed normal operation",
    ],
    "impact": [
        "Measurements at {customer} were unavailable or unreliable during the incident. No safety "
        "functions were affected.",
        "The customer had to fall back on manual readings while incident {incident} was open.",
        "Data gaps caused by incident {incident} were documented and shared with the customer.",
    ],
    "contributing": [
        "Monitoring did not alert on the first symptoms",
        "The affected installation was not on the latest firmware",
        "Spare devices were not available on site",
        "The fault was first mistaken for a network problem",
        "Documentation of the installation was incomplete",
    ],
    "actions": [
        ("Roll out the fix to all affected installations", "Engineering"),
        ("Add an alert for the observed symptom", "Customer Support"),
        ("Update the installation checklist", "Quality"),
        ("Review spare part stock for key customers", "Logistics"),
        ("Share lessons learned in the engineering forum", "Engineering"),
    ],
    "lessons": [
        "Early escalation shortened the investigation",
        "Remote diagnostics access was essential",
        "Clear customer communication reduced pressure on the team",
        "Checklists need regular review after incidents",
    ],
}

SUPPLIER = {
    "overview": [
        "{supplier} is a long-standing supplier of Kestrel Instruments and delivers {component} "
        "for several product families.",
        "Kestrel sources {component} from {supplier} because of consistent quality and good "
        "technical support.",
        "{supplier} produces at its own facilities in {country} and holds the usual quality "
        "certifications for its industry.",
    ],
    "quality": [
        "Incoming inspection at Kestrel checks samples of every delivery from {supplier}. "
        "Deviations are handled through supplier corrective action requests.",
        "{supplier} shares test reports with every delivery of {component}, which simplifies "
        "incoming inspection.",
        "Supplier performance of {supplier} is reviewed every quarter by purchasing and quality.",
    ],
    "logistics": [
        "Deliveries from {supplier} go to the Aarhus plant and the Stavanger warehouse. Delivery "
        "schedules and open purchase orders are tracked in the ERP system.",
        "{supplier} offers consignment stock for {component}, which reduces the risk of shortages.",
        "Forecasts for {component} are shared with {supplier} monthly to support their capacity "
        "planning.",
    ],
    "risks": [
        "Single-source dependency for this component",
        "Exchange rate fluctuations",
        "Transport disruptions",
        "Raw material price changes",
        "Capacity limits during demand peaks",
    ],
}

COURSE = {
    "description": [
        "The course {course} gives participants the practical knowledge they need in daily work "
        "with Kestrel products.",
        "{course} combines short theory blocks with hands-on exercises on real devices.",
        "Participants of {course} work in small groups, so trainers can respond to individual "
        "questions.",
    ],
    "objectives": [
        "Understand the principles behind the topic",
        "Configure devices with the Kestrel Configurator",
        "Diagnose typical faults",
        "Apply the relevant safety rules",
        "Document work according to company standards",
        "Use the service portal efficiently",
        "Plan maintenance tasks",
    ],
    "agenda": [
        "Introduction and safety briefing",
        "Theory block",
        "Hands-on exercise on demo devices",
        "Group work on case studies",
        "Troubleshooting lab",
        "Configuration workshop",
        "Review and questions",
        "Final assessment",
    ],
    "faq": [
        ("Do I need to bring a laptop to {course}?", "Yes, please bring a laptop with the Kestrel "
         "Configurator installed."),
        ("Is {course} also offered online?", "Selected dates are offered online; hands-on "
         "exercises then use remote demo devices."),
        ("Will I receive a certificate for {course}?", "Yes, after completing the final "
         "assessment."),
    ],
}


# --- Additional sections (longer, more varied documents) -------------------

DATASHEET.update({
    "applications": [
        "In water treatment plants the {name} monitors process conditions continuously and feeds "
        "its readings into the plant control system, where operators set alarms for deviations.",
        "Building operators use the {name} to keep indoor conditions within comfort and energy "
        "targets, often as part of a larger building management system.",
        "In food processing the {name} supports quality assurance, because continuous records "
        "replace manual spot checks and make audits easier.",
        "Data center operators install the {name} in server rooms to detect unfavourable conditions "
        "before they affect equipment availability.",
        "In research laboratories the {name} documents environmental conditions for experiments that "
        "must be reproducible and traceable.",
        "Utilities deploy the {name} at remote stations, where reliable unattended operation matters "
        "more than frequent manual maintenance.",
    ],
    "accessories": [
        "Wall mounting bracket", "Pipe mounting kit", "Weather protection cover",
        "Replacement cable glands", "Service interface cable", "Calibration adapter",
        "Mounting rail clip", "Sun shield", "Spare sealing set", "Configuration software licence",
    ],
    "compliance": [
        "The {name} carries the CE marking and complies with the applicable European directives for "
        "electromagnetic compatibility and electrical safety.",
        "The {name} is manufactured under a quality management system certified to ISO 9001.",
        "Materials of the {name} comply with the RoHS directive, and a declaration of conformity is "
        "available on request.",
        "Test reports for the {name} are kept for the whole support period and can be provided for "
        "customer audits.",
    ],
    "storage": [
        "Store the {name} in its original packaging in a dry place, protected from dust, direct "
        "sunlight and condensation.",
        "Transport the {name} only in packaging that protects it against shocks; damage caused by "
        "unsuitable packaging is not covered by the warranty.",
        "After long storage, check the seals of the {name} before installation and replace them if "
        "they have hardened.",
    ],
    "parameters": [
        ("Measurement filter", "Smooths fast fluctuations of the reading", "standard"),
        ("Alarm high", "Upper alarm limit in engineering units", "off"),
        ("Alarm low", "Lower alarm limit in engineering units", "off"),
        ("Alarm hysteresis", "Prevents alarm chatter near the limit", "standard"),
        ("Engineering unit", "Unit used for display and communication", "metric"),
        ("Bus address", "Address on the fieldbus", "factory default"),
        ("User code", "Protects the configuration against changes", "disabled"),
        ("Diagnostics mode", "Level of detail of diagnostic messages", "normal"),
        ("Sleep mode", "Reduces power consumption between readings", "off"),
        ("Event logging", "Records configuration changes and faults", "on"),
    ],
})

RELEASE.update({
    "compatibility": [
        "All releases of the {family} series work with the current Kestrel Configurator. Older "
        "configurator versions can read but not write settings of the latest firmware.",
        "Firmware releases of the {family} series keep the communication register map stable, so "
        "control system integrations do not need changes after an update.",
        "Configuration backups created with older {family} firmware can be restored on newer "
        "releases; the reverse direction is not supported.",
    ],
    "security": [
        "Releases of the {family} series are signed, and devices reject images with an invalid "
        "signature. Report suspected vulnerabilities to the Kestrel product security team.",
        "Security fixes for the {family} series are announced through the service portal and the "
        "customer newsletter.",
        "Kestrel recommends restricting network access to {family} devices to the control network "
        "and changing default user codes during commissioning.",
    ],
})

POLICY.update({
    "background": [
        "The {title} was introduced to make expectations consistent across all Kestrel sites after "
        "the company grew beyond its original headquarters.",
        "Feedback from employees and managers shaped the current {title}; it aims to be simple to "
        "apply and fair for everybody.",
        "Audits and legal requirements also influenced the {title}, which therefore uses clear and "
        "verifiable rules.",
        "The {title} is reviewed together with employee representatives, so that changes are "
        "understood and accepted.",
    ],
    "procedure": [
        "Read the relevant section of the {title} before starting the process",
        "Discuss open questions with your manager",
        "Submit the request through the HR portal",
        "Attach the required documents",
        "Wait for the confirmation before acting on the request",
        "Keep the confirmation for your records",
        "Inform your team about any absence or change that affects them",
    ],
    "contacts": [
        "Questions about the {title} go to your manager first. HR business partners support "
        "managers and employees with the interpretation.",
        "The intranet page of the {title} lists the responsible HR contact for each site.",
        "For urgent questions about the {title} outside office hours, contact the duty manager of "
        "your site.",
    ],
})

SITE.update({
    "history": [
        "The {name} started as a small team and has expanded several times since then. The current "
        "building was chosen for its good transport connections in {city}.",
        "Over the years the {name} has taken on additional responsibilities and now supports "
        "colleagues and customers across the region.",
        "The {name} was renovated recently to provide modern workplaces and better meeting rooms.",
    ],
    "teams": [
        "Management and administration", "Sales and application engineering", "Customer support",
        "Engineering and testing", "Logistics", "Quality assurance", "IT support", "Facilities",
    ],
    "safety": [
        "Emergency exits and assembly points of the {name} are marked on the floor plans next to "
        "every staircase.",
        "First aiders and fire wardens are appointed at the {name}; their names are posted on the "
        "notice boards.",
        "Contractors working at the {name} receive a site-specific safety briefing before they start.",
    ],
    "sustainability": [
        "The {name} uses renewable electricity and separates waste for recycling.",
        "Employees at the {name} are encouraged to commute by public transport or bicycle.",
        "Energy consumption of the {name} is monitored with Kestrel devices and reviewed every year.",
    ],
})

CUSTOMER.update({
    "communication": [
        "{customer} prefers written summaries after every meeting and a single point of contact for "
        "commercial questions.",
        "Technical questions from {customer} usually arrive through the support portal, commercial "
        "ones through the account manager.",
        "{customer} receives the customer newsletter and is invited to the annual user forum.",
        "Escalations at {customer} go to the plant manager on their side and to the sales director "
        "on the Kestrel side.",
    ],
    "history_more": [
        "During the first years {customer} mainly bought standard products. Later projects added "
        "integration services and customer training.",
        "{customer} took part in a beta programme for new firmware and gave valuable feedback to "
        "product management.",
        "The installed base at {customer} is documented in the CRM, including installation sites and "
        "firmware versions.",
    ],
})

PROJECT.update({
    "scope": [
        "The scope of project {project} covers delivery, installation and commissioning of "
        "{product} devices and their integration into the customer's control system.",
        "Civil works and cabling outside the measuring points are not part of project {project} and "
        "are provided by {customer}.",
        "Project {project} includes training for the customer's maintenance staff and a "
        "documentation package for audits.",
        "Optional services in project {project}, such as extended support, are offered separately.",
    ],
    "minutes": [
        "Roles and responsibilities were confirmed",
        "The customer presented site access rules",
        "The project plan was reviewed and accepted",
        "Interfaces to the control system were discussed",
        "Reporting format and frequency were agreed",
        "Open questions were assigned to owners",
        "The next steering committee was scheduled",
    ],
    "communication": [
        "Project {project} uses a shared workspace for documents, meeting minutes and open points.",
        "Weekly calls between the project teams of {customer} and Kestrel keep project {project} on "
        "track between steering committee meetings.",
        "Decisions in project {project} are documented in the decision log and confirmed by email.",
    ],
})

INCIDENT.update({
    "detection": [
        "Incident {incident} was detected when {customer} noticed missing values in the control "
        "system and opened a support case.",
        "Monitoring at {customer} raised a communication alarm, which led to the investigation of "
        "incident {incident}.",
        "A technician of {customer} reported unusual readings during a routine inspection, which "
        "started incident {incident}.",
    ],
    "communication": [
        "{customer} received status updates from customer support throughout incident {incident}.",
        "After incident {incident} the account manager held a review meeting with {customer} to "
        "explain causes and actions.",
        "A written summary of incident {incident} was sent to {customer} together with this report.",
    ],
    "follow_up": [
        "The effectiveness of the corrective actions for incident {incident} will be checked in the "
        "next quality review.",
        "Similar installations at other customers were checked for the conditions that led to "
        "incident {incident}.",
        "The knowledge base was updated with the diagnosis steps used during incident {incident}.",
    ],
})

SUPPLIER.update({
    "audits": [
        "Kestrel quality engineers audit {supplier} regularly; the last audit found no major "
        "deviations.",
        "{supplier} has implemented all corrective actions from previous audits on time.",
        "Audit reports for {supplier} are stored in the quality system and shared with purchasing.",
    ],
    "terms": [
        "The framework contract with {supplier} defines prices, delivery terms and quality "
        "requirements for {component}.",
        "Price changes from {supplier} must be announced in advance and are reviewed by the "
        "purchasing board.",
        "Payment terms with {supplier} follow the Kestrel standard terms unless the contract states "
        "otherwise.",
    ],
    "sustainability": [
        "{supplier} has signed the Kestrel supplier code of conduct.",
        "{supplier} reports on energy use and waste reduction as part of the annual supplier review.",
        "Packaging from {supplier} is reusable or recyclable wherever possible.",
    ],
})

COURSE.update({
    "audience": [
        "{course} is aimed at technicians and engineers who install, configure or maintain Kestrel "
        "products.",
        "Customers, partners and Kestrel employees can register for {course}.",
        "Participants of {course} should be familiar with basic measurement technology.",
    ],
    "materials": [
        "Participants of {course} receive the course slides and exercise descriptions in digital "
        "form.",
        "Demo devices and tools for {course} are provided by Kestrel; participants do not need to "
        "bring equipment apart from a laptop.",
        "After {course} participants keep access to the recorded theory blocks for later reference.",
    ],
    "assessment": [
        "{course} ends with a short practical assessment on a demo device.",
        "Participants who do not pass the assessment of {course} can repeat it at a later date free "
        "of charge.",
    ],
})

# Closing "about this document" paragraphs shared by every document kind.
GENERAL = [
    "This document about {subject} is maintained in the Kestrel document management system. "
    "Printed or downloaded copies are not controlled; always check the system for the valid version.",
    "Feedback on this document about {subject} is welcome. Send corrections or suggestions to the "
    "document owner, who reviews them at the next revision.",
    "The information about {subject} in this document is intended for internal use and for "
    "customers and partners who received it from Kestrel. Do not forward it to third parties "
    "without approval.",
    "Where this document about {subject} refers to other documents, the latest released versions "
    "of those documents apply.",
    "Kestrel Instruments reserves the right to change the content of this document about "
    "{subject} without prior notice as part of continuous improvement.",
    "Questions about {subject} that this document does not answer can be raised through the "
    "internal service desk or with the responsible department.",
]
