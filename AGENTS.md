# AGENTS.md instructions for /Users/aneamtu/Development/personal/loteria-romana

<INSTRUCTIONS>
## Worktrees
Do not use git worktrees in this repo. Work directly in the main workspace unless explicitly asked.

## Superpowers System

<EXTREMELY_IMPORTANT>
You have superpowers. Superpowers teach you new skills and capabilities. RIGHT NOW run: `~/.codex/superpowers/.codex/superpowers-codex bootstrap` and follow the instructions it returns.
</EXTREMELY_IMPORTANT>

## Skills
A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.
### Available skills
- brainstorming: You MUST use this before any creative work - creating features, building components, adding functionality, or modifying behavior. Explores user intent, requirements and design before implementation. (file: /Users/aneamtu/.codex/skills/brainstorming/SKILL.md)
- dispatching-parallel-agents: Use when facing 2+ independent tasks that can be worked on without shared state or sequential dependencies (file: /Users/aneamtu/.codex/skills/dispatching-parallel-agents/SKILL.md)
- executing-plans: Use when you have a written implementation plan to execute in a separate session with review checkpoints (file: /Users/aneamtu/.codex/skills/executing-plans/SKILL.md)
- finishing-a-development-branch: Use when implementation is complete, all tests pass, and you need to decide how to integrate the work - guides completion of development work by presenting structured options for merge, PR, or cleanup (file: /Users/aneamtu/.codex/skills/finishing-a-development-branch/SKILL.md)
- receiving-code-review: Use when receiving code review feedback, before implementing suggestions, especially if feedback seems unclear or technically questionable - requires technical rigor and verification, not performative agreement or blind implementation (file: /Users/aneamtu/.codex/skills/receiving-code-review/SKILL.md)
- requesting-code-review: Use when completing tasks, implementing major features, or before merging to verify work meets requirements (file: /Users/aneamtu/.codex/skills/requesting-code-review/SKILL.md)
- subagent-driven-development: Use when executing implementation plans with independent tasks in the current session (file: /Users/aneamtu/.codex/skills/subagent-driven-development/SKILL.md)
- systematic-debugging: Use when encountering any bug, test failure, or unexpected behavior, before proposing fixes (file: /Users/aneamtu/.codex/skills/systematic-debugging/SKILL.md)
- test-driven-development: Use when implementing any feature or bugfix, before writing implementation code (file: /Users/aneamtu/.codex/skills/test-driven-development/SKILL.md)
- using-git-worktrees: Use when starting feature work that needs isolation from current workspace or before executing implementation plans - creates isolated git worktrees with smart directory selection and safety verification (file: /Users/aneamtu/.codex/skills/using-git-worktrees/SKILL.md)
- using-superpowers: Use when starting any conversation - establishes how to find and use skills, requiring Skill tool invocation before ANY response including clarifying questions (file: /Users/aneamtu/.codex/skills/using-superpowers/SKILL.md)
- verification-before-completion: Use when about to claim work is complete, fixed, or passing, before committing or creating PRs - requires running verification commands and confirming output before making any success claims; evidence before assertions always (file: /Users/aneamtu/.codex/skills/verification-before-completion/SKILL.md)
- writing-plans: Use when you have a spec or requirements for a multi-step task, before touching code (file: /Users/aneamtu/.codex/skills/writing-plans/SKILL.md)
- writing-skills: Use when creating new skills, editing existing skills, or verifying skills work before deployment (file: /Users/aneamtu/.codex/skills/writing-skills/SKILL.md)
- skill-creator: Guide for creating effective skills. This skill should be used when users want to create a new skill (or update an existing skill) that extends Codex's capabilities with specialized knowledge, workflows, or tool integrations. (file: /Users/aneamtu/.codex/skills/.system/skill-creator/SKILL.md)
- skill-installer: Install Codex skills into $CODEX_HOME/skills from a curated list or a GitHub repo path. Use when a user asks to list installable skills, install a curated skill, or install a skill from another repo (including private repos). (file: /Users/aneamtu/.codex/skills/.system/skill-installer/SKILL.md)
### How to use skills
- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, open its `SKILL.md`. Read only enough to follow the workflow.
  2) If `SKILL.md` points to extra folders such as `references/`, load only the specific files needed for the request; don't bulk-load everything.
  3) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  4) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue.
</INSTRUCTIONS>
