# Clearing the Final **Red ✖** in a Git Merge/Rebase (PyCharm & CLI)

_Last updated • 1 July 2025_

---

## 1&nbsp;· Find out *what* is still conflicted

```bash
git diff --name-only --diff-filter=U   # lists only unresolved paths
git status                             # shows “unmerged” entries (UU)
```

*If nothing prints,* the file you fixed just isn’t **staged** yet (see §2).  
*If paths appear,* those files still need resolving.

---

## 2&nbsp;· Mark the files you already fixed as *resolved*

| **Command‑line** | **PyCharm UI** |
|------------------|---------------|
| `git add path/to/file`<br>`git add -u`   *(stage everything solved)* | 1. **Alt + 9** → _Version Control_ → **Local Changes**.<br>2. Right‑click the file under **Merge Conflicts** → **Mark Resolved** (or **Add**).<br>3. Alternatively, re‑open with **Resolve Conflicts…** and click **Apply / Mark Resolved**. |

---

## 3&nbsp;· Resolve any remaining files

1. **Git ► Resolve Conflicts…** (or the blue *Resolve conflict* link).  
2. Double‑click each file with a red ✖.  
3. In the three‑way merge editor choose **Accept Yours / Accept Theirs / Merge** for every block.  
4. Press **Apply** → file icon turns green.  
5. Repeat until the dialog says **No conflicts**.

---

## 4&nbsp;· Finish the merge or rebase

```bash
# MERGE
git commit                       # PyCharm may auto‑commit

# REBASE
git rebase --continue
```

PyCharm shows a green **Continue Merge / Commit Changes** button—click it once.

---

## 5&nbsp;· Push your branch

```bash
git push origin my-feature                 # after merge
git push --force-with-lease origin my-feature   # after rebase
```

---

### Quick reference

| Goal | Command |
|------|---------|
| List unresolved files | `git diff --name-only --diff-filter=U` |
| Stage all resolved files | `git add -u` |
| Abort merge | `git merge --abort` |
| Abort rebase | `git rebase --abort` |
| Undo last merge (pre‑push) | `git reset --hard ORIG_HEAD` |

When **`git status`** shows _working tree clean_ **and** PyCharm’s status bar no longer displays _Conflicts_, the red ✖ is gone—you’re done.
