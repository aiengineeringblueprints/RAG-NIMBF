# 05: Component-injection policy moves into the adapter infrastructure

**What to build:** The decide-whether-and-what-to-inject policy lives beside the component factory in the adapter infrastructure, not in the orchestrator. The fallback that mutates a frozen configuration object on the way is removed — capability determination comes from the adapter itself, not from overwriting a config field. Tests exercise the policy at the adapter seam.

**Blocked by:** 03 (Orchestrator drives every system through the adapter seam).

**Status:** ready-for-agent

- [ ] Injection policy is resolved inside the adapter infrastructure; the orchestrator hands over the adapter and component bundle and nothing else
- [ ] No code path mutates a frozen config object; the config field remains a user-intent override only, with computed capability derived from the adapter
- [ ] Policy tests cover: adapter supporting all slots, some slots, no slots, and the user-intent override
- [ ] Full pytest passes
