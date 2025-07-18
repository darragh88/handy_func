# After Your Pull Request Is Merged: What Happens and How to Clean Up Locally

This guide explains what happens on the remote side (Bitbucket Cloud) when a reviewer merges your pull‑request (PR), and shows exactly how to tidy up branches on your local machine.

---

## 1 ️⃣ What the **Merge** Button Does

* **Destination branch** (`main`, `develop`, etc.) receives the commits from your PR branch.  
  *Depending on the repository settings, Bitbucket performs a **merge commit**, **squash**, or **rebase & fast‑forward** strategy.*
* The PR status flips to **Merged**. Reviewers are done.
* Bitbucket shows an optional **Delete branch** button to remove the *remote* feature branch.

Nothing changes in your local repo until you run Git commands.

---

## 2 ️⃣ Sync Your Local `main`

```bash
git fetch origin              # grab latest objects
git switch main               # or: git checkout main
git pull --ff-only            # fast‑forward to the new tip
```

`git log -1` should now display your PR’s merge (or squash) commit.

---

## 3 ️⃣ Clean Up the PR Branch

### A. Delete the remote branch  *(optional)*

```bash
git push origin --delete my-pr-branch
```
*Or click **Delete branch** on the PR page.*

### B. Delete the local branch

```bash
git branch -d my-pr-branch    # -d = safe delete (will refuse if unmerged)
# use -D to force if you really need to
```

Keeping the branch is harmless, but pruning keeps `git branch` output tidy.

---

## 4 ️⃣ Continue Working

### Short‑lived topic branch workflow

```bash
git switch main
git pull --ff-only
git switch -c next-feature
```

Create a clean branch for the next slice of work.

### Long‑lived feature branch workflow

```bash
git switch feature           # your ongoing WIP branch
git pull --rebase origin main   # or: git merge origin/main
```

If you cherry‑picked commits from `feature` into the PR, Git’s patch‑ID logic
detects duplicates and skips them automatically.

---

## 5 ️⃣ Cheat‑Sheet

```text
# After reviewer merges PR
git fetch origin
git switch main && git pull --ff-only
git push origin --delete <pr-branch>   # remote tidy-up (optional)
git branch -d <pr-branch>              # local tidy-up (optional)

# Next task
git switch -c <new-branch>             # or rebase/merge main into feature
```

---

### Recap

1. Reviewer hits **Merge** → code lands in `main`.  
2. `git fetch` + `git pull` updates your local `main`.  
3. Delete (or keep) the PR branch.  
4. Start new work from a clean, up‑to‑date base.

Stay tidy, stay in sync, and happy coding!
