---
title: "AI-native Engineering Method"
slug: ai-native-engineering-method
description: "A method card for AI-native delivery: Discovery → Release → Ops. Spec before code. Humans hold merge and release. Change Ops checkpoints sit across the whole chain."
date: 2026-11-02T12:00:00+08:00
draft: false
tags: ["AI-native", "Engineering Method", "SDLC", "Change Ops", "AI Coding"]
categories: ["开发实践"]
excerpt: "A short method card, not an essay: agents draft, humans hold merge and release. Discovery → Release → Ops, with Change Ops on every band."
toc: true
cover:
  image: "/ai-native-pipeline.png"
  alt: "AI-native Engineering Pipeline: Discovery through Release and Ops, with Change Ops across the chain"
  caption: "Canonical pipeline. Intent verifiable. Artifacts auditable. Humans hold merge and release."
  relative: false
  hiddenInSingle: true
---

This is a method card, not a tutorial. Same line as the longer pipeline essay: **agents may draft anything; humans still hold merge and release.**

The picture below is the whole method. If a change cannot be placed on this chain, it is not ready to ship.

![AI-native Engineering Pipeline: Discovery through PRD, Spec + Rules, Plan, Coding / Test, Security, CR || CI, Preview, Merge, Release, Ops, and Incident, with Change Ops across the whole chain](/ai-native-pipeline.png)

*Canonical pipeline. Intent verifiable · Artifacts auditable · Humans hold merge & release.*

## Three bands, one loop

Read the diagram left to right, then down. It is one delivery loop, not three departments.

1. **Upstream intent** — Discovery → PRD → Spec + Rules → Plan
2. **Build & verify (loop)** — Coding / Test → Security → CR || CI → Preview
3. **Release (human gate)** — Merge → Release → Ops → Incident

**Change Ops (CO)** sits beside the release band on purpose. It is not a late checklist. CO checkpoints run across the whole chain.

## 1. Upstream intent — spec before code

Start with Discovery: what problem, for whom, what evidence, and what must not change. Write a short PRD. Compress that into a **spec plus rules** an agent can follow without guessing. Only then write a plan.

The first hard gate is here: **no spec, no code.** A chat that "sounds like a spec" is not a spec. A spec is versioned, reviewable, and testable. Rules are the constraints the spec forgot to say out loud — security boundaries, merge policy, "do not pay / do not post / do not merge to production."

If intent is not verifiable, later review can only check whether the diff is self-consistent. That is how plausible AI code ships the wrong product.

## 2. Build & verify — a loop, not a one-shot

Coding and tests move together. Security is a station, not a hope. Code review and CI run in parallel (`CR || CI`): humans and machines look at different failure modes. Preview is the last cheap place to see the change as a user would.

Two gates live here:

- **Full tests before PR.** If the agent cannot show the tests it ran, the PR is incomplete.
- **Independent review.** The author — human or model — does not mark its own work done. AI review is useful on the mechanical layer. Intent review stays human.

This band is a loop. Failed security, failed CI, or a preview that does not match the spec sends the work back. Volume is not progress.

## 3. Release is a human gate

Merge, release, ops, and incident are the same story with rising blast radius.

**Humans hold merge and release.** An agent may open the PR, fill the checklist, and recommend ship. It does not press the button. Production deploy, domain changes, payments, and public posting stay on the human side of the line.

Ops is not "someone else's job after launch." If the change has no owner for logs, rollback, and "how we will know it broke," it is not released — it is abandoned.

Incident feeds the loop. A production surprise is a Discovery input, not a chat postmortem that dies in Slack.

## Change Ops across the whole chain

CO is the memory of the pipeline. At each band, ask the same four questions:

| Checkpoint | Ask |
|---|---|
| Intent | Is the spec still the spec we agreed to? |
| Build | Are artifacts auditable — tests, diffs, preview URLs, security notes? |
| Release | Who is the human holder for merge and ship? |
| Ops | What is the rollback, and who is awake for it? |

If you cannot answer one of these, stop. Do not "keep generating." Generation without a checkpoint is inventory, not delivery.

## What this card is not

It is not "vibe coding" without gates. It is not a promise that AI owns production. It is not a role chart for more agents.

The method is deliberately thin: **intent verifiable, artifacts auditable, humans hold merge and release.** Everything else is local practice.

## How to use this card

Keep it next to a dispatch. Before an agent starts: name the band, name the artifact, name the human gate. Before a PR: point to the spec, the tests, and the reviewer who did not write the code. Before ship: name the person who will merge, the person who will watch, and the path back.

That is the whole method. Short on purpose. The diagram is the source of truth; the words only label the gates.
