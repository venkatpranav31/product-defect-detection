# 📤 How to Upload This Project to GitHub

## Step 1 — Install Git (if not installed)
```bash
# Check if git is installed
git --version

# Install if needed:
# Windows: https://git-scm.com/download/win
# Mac:     brew install git
# Ubuntu:  sudo apt install git
```

---

## Step 2 — Configure Git (first time only)
```bash
git config --global user.name  "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Step 3 — Create a GitHub Repository
1. Go to https://github.com/new
2. Name it: `product-defect-detection`
3. Set visibility: **Public** (shows on your resume/portfolio)
4. ❌ Do NOT initialize with README, .gitignore, or license (we already have them)
5. Click **Create repository**
6. Copy the repo URL (e.g. `https://github.com/yourname/product-defect-detection.git`)

---

## Step 4 — Initialize & Push the Project

Open a terminal inside the project folder:

```bash
# Navigate to project folder
cd path/to/product-defect-detection

# Initialize git
git init

# Stage all files
git add .

# First commit
git commit -m "feat: initial project — ResNet-50 defect detection with transfer learning"

# Link to GitHub repo
git remote add origin https://github.com/<your-username>/product-defect-detection.git

# Push
git branch -M main
git push -u origin main
```

---

## Step 5 — Verify on GitHub
Visit `https://github.com/<your-username>/product-defect-detection`
You should see all your files and the README rendered at the bottom.

---

## Step 6 — Recommended Follow-up Actions

### Add a LICENSE file
```bash
# On GitHub: go to repo → Add file → Create new file → name it LICENSE
# Click "Choose a license template" → pick MIT
```

### Enable GitHub Actions (CI) — optional but impressive
Create `.github/workflows/test.yml` to auto-run your tests on every push.

### Use Git LFS for large model files
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "chore: track model files with Git LFS"
```

---

## Commit Message Best Practices
Use conventional commits — it looks professional:

| Prefix   | Use for                        |
|----------|-------------------------------|
| `feat:`  | New feature or file           |
| `fix:`   | Bug fix                       |
| `docs:`  | README or documentation       |
| `chore:` | Config, CI, dependencies      |
| `refactor:` | Code cleanup without new logic |
| `test:`  | Adding/fixing tests           |

Example:
```bash
git commit -m "feat: add real-time inference pipeline with 20fps benchmarking"
git commit -m "docs: update README with evaluation metrics table"
git commit -m "fix: correct class weight calculation in dataset.py"
```

---

## Making the Repo Portfolio-Ready

✅ Checklist:
- [ ] README with results table (accuracy, FPS, etc.)
- [ ] `requirements.txt` present
- [ ] `.gitignore` excludes data/models
- [ ] Clean folder structure
- [ ] At least 1 commit per major feature
- [ ] License file added
- [ ] Description + tags set on GitHub repo page
  - Go to repo → ⚙️ (Settings gear on right) → add Description + Topics
  - Suggested topics: `computer-vision`, `pytorch`, `resnet`, `transfer-learning`, `manufacturing`, `quality-control`, `deep-learning`
