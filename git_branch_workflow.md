# Git Branch Workflow: Working in the Same Branch After Selective Merges

This document summarizes how you can keep coding on a long‑lived feature branch **_and_** deliver small, review‑friendly pull‑requests (PRs) by selectively merging slices of work into `main`.

---

## Two Common Workflows

| Workflow | Pros | Cons |
|---|---|---|
| **Long‑lived feature branch**  <br>• keep hacking on *one* branch  <br>• cherry‑pick / copy specific commits into short‑lived “delivery” branches  <br>• merge those branches to `main`  <br>• merge `main` back into the feature branch | * Single place for all WIP <br>* Less context‑switching | * Duplicate commits clutter history <br>* Conflicts build up the longer the branch diverges <br>* Easy to push extra WIP to the wrong PR |
| **Short‑lived topic branches&nbsp;(_recommended_)**  <br>• once a PR is open, treat that branch as frozen except for review fixes  <br>• create a fresh branch off `main` (or off the previous one) for the next slice of work | * Each PR shows a stable, focused diff <br>* Conflicts stay small <br>* Cleaner, mostly linear history | * You need to create / checkout new branches more often |

---

## Detailed Steps for the **Long‑Lived Feature Branch** Style

Assume:

* `feature`  – your ongoing branch with **all** work‑in‑progress  
* `slice-a`  – a temporary branch that will contain only the code you want to deliver _now_

1. **Create the delivery branch from `main`:**
   ```bash
   git checkout main
   git pull
   git checkout -b slice-a
   ```

2. **Move *just* the desired changes into `slice-a`:**

   *Cherry‑pick the exact commits*  
   ```bash
   git cherry-pick <sha1> <sha2> ...
   ```

   — *or* —

   *Copy specific files from the feature branch*  
   ```bash
   git checkout feature -- path/to/file
   git commit -m "Add only desired file change"
   ```

3. **Push & open a PR** from `slice-a` into `main`.

4. **Merge the PR** once approved.

5. **Sync the long‑lived branch** so it knows those changes are already in `main`:

   *Option A — rebase (keeps history linear):*
   ```bash
   git checkout feature
   git pull --rebase origin main
   ```

   *Option B — merge (adds a merge commit, but no rewritten history):*
   ```bash
   git checkout feature
   git merge origin/main
   ```

   Git detects that the cherry‑picked patches already exist in `main` and skips them, avoiding duplicate code.

6. **Keep coding on `feature`**.  
   When ready for the next slice, repeat steps 1‑5 with `slice-b`, `slice-c`, etc.

---

## Steps for **Short‑Lived Topic Branches**

1. Finish review fixes on the current PR branch; wait for it to merge.  
2. `git checkout main && git pull`
3. `git checkout -b next-feature`
4. Work, commit, push, PR, merge, repeat.

Each PR is laser‑focused, and your git history stays tidy.

---

## TL;DR

> *Your manager’s plan (“keep working on the feature branch, selectively merge, then merge `main` back”) is perfectly valid in Git.  
> If you prefer neater history and simpler reviews, switch to one branch per PR instead.*

