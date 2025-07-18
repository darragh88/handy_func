# Git Branch Update & Conflict‑Resolution Guide

_Last updated: 1 July 2025_

---
## 1. Scenario & Goal
You have:
* **main** – the shared integration branch (already up‑to‑date with remote).
* **my‑feature** – your working branch with new commits.

**Objective:** Bring everything that is on **main** into **my‑feature** without losing any work.  
You can do this with either a **merge** (adds a merge commit) or a **rebase** (re‑writes your commits onto main’s tip).

---
## 2. Keep `main` Current

### Command‑line
```bash
git checkout main
git pull origin main      # fetch + fast‑forward
```

### PyCharm UI
1. Switch branch: **Git ► Branches ► main** (or from the bottom‑right status bar).
2. Pull: **Git ► Pull…** → make sure _Branch to merge_ is `origin/main`, press **Pull**.

---
## 3. Merge `main` into Your Branch (safe, non‑destructive)

### Command‑line
```bash
git checkout my-feature
git merge main                 # creates a merge commit
#   (< fix conflicts > → git add <files> → git commit)
git push origin my-feature     # upload branch
```

### PyCharm UI
1. Checkout your branch: **Git ► Branches ► my‑feature**.
2. **Git ► Merge Changes…** → select `main`, leave **No fast‑forward** checked, press **Merge**.
3. If conflicts appear, PyCharm opens the **Merge Conflicts** tool window (see §5).
4. Push: **Git ► Push…**.

_Result: main’s commits are added; your commits remain untouched._

---
## 4. Rebase `my‑feature` onto `main` (linear history)

### Command‑line
```bash
git checkout my-feature
git rebase main
#   (< fix conflicts > → git add <files> → git rebase --continue)
git push --force-with-lease origin my-feature
```

### PyCharm UI
1. Checkout `my‑feature`.
2. **Git ► Rebase…** → _Upstream_ = `main`, press **Rebase**.
3. Conflicts open in the **Conflicts** dialog (see §5).
4. After successful rebase, **Git ► Push…** → enable **Force push (–‑‑force‑with‑lease)**.

_Result: history is linear; commit hashes are re‑written._

---
## 5. Handling Conflicts

### What a Conflict Is
Both branches changed the **same lines** or **one deleted while the other modified**.

### Command‑line Markers
```text
<<<<<<< HEAD          # “ours” (current branch)
your version
=======
their version
>>>>>>> main          # “theirs” (incoming)
```

### PyCharm Conflict Resolver
* PyCharm shows a **three‑pane** view: _Left = current (ours)_, _Right = incoming (theirs)_, _Center = result_.
* Buttons:
  * **Accept Yours** – keep your branch’s version.
  * **Accept Theirs** – take main’s version.
  * **Apply** or **Merge** individual changes with checkboxes.
* After finishing every file, press **Apply** (or **Resolve**), then **Commit** (merge) / **Continue Rebasing** (rebase).

---
## 6. Always Prefer Your Branch (“ours strategy”)

### Command‑line (one‑off)
```bash
git merge -X ours main         # in my-feature
# or
git rebase -X ours main
```

### PyCharm UI
There is no single checkbox; resolve each file by repeatedly clicking **Accept Yours** (or use the command‑line merge with `-X ours`).

---
## 7. Prefer Incoming Changes (“theirs strategy”)

```bash
git merge -X theirs main
# or during conflict:
git checkout --theirs path/to/file
```

PyCharm: **Accept Theirs** in the conflict dialog.

---
## 8. Undo & Abort Commands

| Situation | Command |
|-----------|---------|
| Abort merge in progress | `git merge --abort` |
| Abort rebase in progress | `git rebase --abort` |
| Undo last merge (post‑commit) | `git reset --hard ORIG_HEAD` |
| Jump back to any point | `git reflog` → `git reset --hard <hash>` |

PyCharm **Log** tab → right‑click any commit → **Reset Current Branch to Here…** (choose _Hard_).

---
## 9. Automate Conflict Resolution for Specific Files

1. Add rules to `.gitattributes`:
   ```gitattributes
   docs/generated/*.json  merge=ours
   ```
2. Configure driver once:
   ```bash
   git config --global merge.ours.driver true
   ```
After this, those files will always keep _your_ branch’s version during merges/rebases.

---
## 10. Quick Reference Cheat Sheet

| Want to… | Command‑line | PyCharm |
|----------|--------------|---------|
| Update main | `git pull origin main` | **Git ► Pull…** |
| Merge main into feature | `git merge main` | **Git ► Merge Changes…** |
| Rebase feature onto main | `git rebase main` | **Git ► Rebase…** |
| Prefer your lines on conflict | `-X ours` / `Accept Yours` | Conflict dialog → **Accept Yours** |
| Abort current merge | `git merge --abort` | **Merge Conflicts** window → **Abort** |
| Force‑with‑lease push | `git push --force-with-lease` | **Push** dialog → enable **Force push** |

---
### Keep this handy! 
Merging or rebasing frequently (small batches of work) keeps conflicts easy to spot and trivial to resolve.
