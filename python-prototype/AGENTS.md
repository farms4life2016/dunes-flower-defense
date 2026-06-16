# Dunes Flower Defense - PROTOTYPE

## Purpose

Use Python and Jupyter to quickly verify that Bloons TD is NP-hard.
Specifically, verify the correctness of algorithms needed to transform an NP-complete problem
into an instance of BTD.

This is not meant to be a hard proof, more like a smoke test to see if this reduction is worth pursuing
(i.e. before I fully commit to making Unity game).

## Editing Code

**Do not edit IPYNB files directly.** Only edit the linked `.py` file.
I will be editing notebooks in the web interface provided by `jupyter notebook`,
which has its own syncing mechanism with the file system.

To avoid race conditions, use `jupytext` to sync the `.py` file **BEFORE AND AFTER** you edit:

```bash
JUPYTER_DATA_DIR=.jupyter_data ./venv/bin/python3 -m jupytext --sync mrp3sat_reduction_v2.ipynb
```

You may read unlinked IPYNB files as always, otherwise prioritize reading the linked python file since they are slimmer.

### Testing

When writing new code or making large refactors, always *strive* to test your code to confirm behaviour.
You can run small tests if you paste code into a scratch file (inside `./scratchpad`).
To avoid repeating code, you can copy stable sections of code from the notebook into the scratchpad
and import them into your tests. However, the notebook stays heavy and should *never* reference scratchpad files.

Please keep the scratchpad neat. Clean up stale files unless they are truly useful for later.
Reuse old scratch files instead of deleting and creating new files.

### Installing Packages

There is a current wave of cyber-attacks exploiting the supply-chain.
Whenever you need to install a new Python package via `pip`, **always** follow security best practices:

- check the most recent release from PyPI. Use a release that has been around for several weeks or months.
  *Do not download releases that are young!*
- pin that release to `requirements.txt`
- download that release via `pip` to the venv (`./venv/`)
- check which dependencies were pulled alongside the download and pin those versions to `requirements.txt` as well

Instead of using raw `python3` and `pip3`, **always** use the `./venv/bin/python3` and `./venv/bin/pip3`.

## Relevant Files

- `definitions.tex`: LaTeX file containing many formal definitions and terminologies used in our proof.
  also describes what variable names and data structures I expect you to use.
  Ignore the TODOs -- formal proofs are not in scope of this prototype.
- `mrp3sat_reduction.ipynb`: old attempt at a reduction, mostly coded using AI and create prior to
  the LaTeX definition file. Contains a lot of reusable code that you should strive to copy and adapt.
  However, the AI made too many wrong assumptions about the reduction which ultimately ruined it.
- `mrp3sat_reduction_v2.ipynb`: new attempt at a reduction, I'm started from fresh again.
  Also contains some implementation notes. The definitions in the file may be stale;
  the LaTeX definitions file takes precedence.
- `input*.yaml`: inputs to our notebook file.
  
The PDFs are formal papers that prove that certain problems are or are not NP-Complete.
You *should not* read them, as this will consume too many tokens.

## Version Control

There is a Git repo in the parent folder that pushes to GitHub.
**Only commit when requested, do not commit otherwise!**
I must first confirm the behaviour on Jupyter first to verify that the plots look correct.
Do not create a PR either -- this is a branch for work, not changes.
The commit style is informal, written by a hobbist programmer. Please check recent commits and follow my style.

## Other Notes

If you don't think the reduction will work, just tell me instead of reasoning for a long time
and writing code that is believable working but factually incorrect.
