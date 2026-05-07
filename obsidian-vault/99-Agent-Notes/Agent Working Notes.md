# Agent Working Notes

Use this vault as the first context source for future work.

Workflow for code changes:

1. Read [[Home]] and the relevant topic note before broad source exploration.
2. Inspect source files only where the vault points or where the task requires verification.
3. After changing behavior, update the corresponding vault note in the same turn.
4. If adding a module, create or update a note and add links from [[Repository Map]] and [[Home]] when appropriate.
5. If changing run commands, environment variables, outputs, or services, update [[Operations Runbook]] and [[Configuration Reference]].
6. If changing tests or expected coverage, update [[Testing and Coverage]].

When vault and code conflict:

- Treat source code as authoritative.
- Correct the vault immediately.
- Mention the discrepancy in the final response if it affected the task.

Suggested note naming:

- Architecture notes in `01-Architecture/`.
- Module notes in `02-Modules/`.
- Commands and runbooks in `03-Operations/`.
- Experimental context in `04-Research/`.
- Tests in `05-Testing/`.
- Decisions, risks, and future constraints in `06-Decisions/`.

