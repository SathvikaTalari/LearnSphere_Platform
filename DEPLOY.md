# 🚀 LearnSphere Deployment Guide

## ⚠️ Why Not Vercel?
Vercel only supports **Node.js and frontend** frameworks (Next.js, React, etc.).  
LearnSphere uses **Flask (Python)** — you need a platform that supports Python backends.

---

## 🏆 Option 1: Render (RECOMMENDED — Free & Easy)

**Free tier:** 750 hrs/month · Sleeps after 15 min inactivity · Auto-deploys from GitHub

### Step 1 — Push to GitHub

```bash
# In your learnsphere folder:
git init
git add .
git commit -m "Initial commit"

# Create a new repo at github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/learnsphere.git
git push -u origin main
```

### Step 2 — Deploy on Render

1. Go to **[render.com](https://render.com)** → Sign up free
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account → Select your **learnsphere** repo
4. Fill in these settings:

| Field | Value |
|-------|-------|
| **Name** | learnsphere |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120` |
| **Instance Type** | Free |

5. Click **"Advanced"** → **"Add Environment Variable"**:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | your_groq_api_key_here |
| `SECRET_KEY` | any_random_long_string_here |

6. Click **"Create Web Service"**
7. Wait ~3 minutes — Render builds and deploys automatically
8. Your app is live at `https://learnsphere.onrender.com` 🎉

### Auto-Deploy
Every time you push to GitHub, Render redeploys automatically!

---

## 🚂 Option 2: Railway (No Sleep — $5 Free Credit)

**Free tier:** $5/month credit · No sleeping · Faster cold starts

### Step 1 — Push to GitHub (same as above)

### Step 2 — Deploy on Railway

1. Go to **[railway.app](https://railway.app)** → Sign up with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your **learnsphere** repository
4. Railway auto-detects Python and deploys!
5. Go to **"Variables"** tab → Add:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | your_groq_api_key_here |
| `SECRET_KEY` | any_random_long_string_here |

6. Go to **"Settings"** → **"Domains"** → **"Generate Domain"**
7. App is live in ~2 minutes 🎉

---

## 🐍 Option 3: PythonAnywhere (Always On — No Sleeping!)

**Free tier:** Always-on · 1 web app · Manual deploys · Best for always-available demo

### Steps

1. Go to **[pythonanywhere.com](https://www.pythonanywhere.com)** → Sign up free
2. Go to **"Files"** tab → Upload your project ZIP or use:
   ```bash
   # In PythonAnywhere Bash console:
   git clone https://github.com/YOUR_USERNAME/learnsphere.git
   ```
3. Go to **"Consoles"** → Open a **Bash console**:
   ```bash
   cd learnsphere
   pip install flask groq gunicorn --user
   ```
4. Go to **"Web"** tab → **"Add a new web app"**
5. Choose **"Manual configuration"** → **Python 3.10**
6. Set **Source code**: `/home/YOUR_USERNAME/learnsphere`
7. Set **WSGI file** — click the link and replace contents with:
   ```python
   import sys
   sys.path.insert(0, '/home/YOUR_USERNAME/learnsphere')
   
   import os
   os.environ['GROQ_API_KEY'] = 'your_groq_api_key_here'
   os.environ['SECRET_KEY'] = 'your_secret_key_here'
   
   from app import app as application
   ```
8. Click **"Reload"** → App is live at `YOUR_USERNAME.pythonanywhere.com` 🎉

---

## 🔑 Getting Your Free GROQ API Key

1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up free (no credit card needed)
3. Go to **"API Keys"** → **"Create API Key"**
4. Copy the key and paste it as `GROQ_API_KEY` in your deployment platform

---

## 📁 Files Added for Deployment

| File | Purpose |
|------|---------|
| `requirements.txt` | Python packages to install |
| `Procfile` | Tells Render/Railway how to start the app |
| `render.yaml` | Render auto-configuration |
| `railway.json` | Railway auto-configuration |
| `.gitignore` | Prevents committing secrets & database |

---

## ❓ FAQ

**Q: Why does my app take 30 seconds to load on Render free tier?**  
A: Free tier instances "sleep" after 15 minutes of inactivity. The first request wakes it up. Upgrade to "Starter" ($7/mo) to keep it always-on, or use PythonAnywhere free tier.

**Q: Will my database data persist on Render?**  
A: SQLite data is lost on every redeploy on Render free tier. For production, add a free PostgreSQL database from Render's dashboard.

**Q: Can multiple users use it at the same time?**  
A: Yes — gunicorn runs with 2 threads so it handles concurrent requests.

**Q: How do I update my deployed app?**  
A: Just push to GitHub: `git add . && git commit -m "update" && git push` — Render/Railway redeploys automatically.
