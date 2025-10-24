A quick way to fix most formatting issues (including “line too long” E501) across all Python files in a project is to use an *auto-formatter* such as **Black** or **autopep8**. These tools parse your Python files and automatically rewrite them so that they conform to PEP 8 (and other stylistic guidelines).

Below are two popular options:

---

## 1. **Using Black**

**Black** is a popular opinionated code formatter. It enforces consistent style (including line lengths) for your entire project.

1. **Install Black** (if you haven’t already):
   ```bash
   pip install black
   ```
2. **Run Black** on your project directory:
   ```bash
   cd pm4py
   black --line-length 79 .
   ```
   - `--line-length 79` matches the PEP 8 recommended max line length of 79 characters (which also aligns with E501).
   - If you prefer a different line length (e.g., 88, which is Black’s default), adjust the `--line-length` accordingly.

Black will automatically reformat all `.py` files in the current directory (and subdirectories) to ensure lines do not exceed `--line-length`, among other style fixes.

---

## 2. **Using autopep8**

**autopep8** is another tool specifically designed to fix style issues reported by [pep8/pycodestyle](https://pypi.org/project/pycodestyle/). It can address line-length issues (E501) as well as many other common PEP 8 deviations.

1. **Install autopep8**:
   ```bash
   pip install autopep8
   ```
2. **Run autopep8** recursively on your project:
   ```bash
   cd pm4py
   autopep8 --in-place --recursive --max-line-length 79 .
   ```
   - `--in-place` modifies the files *in place* (rather than printing to stdout).
   - `--recursive` applies the fix to all Python files in subfolders as well.
   - `--max-line-length 79` sets the desired maximum line length.

> **Tip**: You can add `--aggressive --aggressive` to make autopep8 apply more extensive changes:
> ```bash
> autopep8 --in-place --recursive --aggressive --aggressive --max-line-length 79 .
> ```

**In Summary:**  
- **Black** (opinionated but modern) can format all files in a single command:  
  ```bash
  black --line-length 79 .
  ```
- **autopep8** (closer to PEP 8) can specifically fix E501 line length errors (and more):  
  ```bash
  autopep8 --in-place --recursive --max-line-length 79 .
  ```
