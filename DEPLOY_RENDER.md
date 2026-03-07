# 🚀 Deploying LearnSphere on Render (Free Tier) — Complete Guide

---

## What You Need Before Starting

| Item | Where to get it | Cost |
|------|----------------|------|
| GitHub account | github.com | Free |
| Render account | render.com | Free |
| Groq API key | console.groq.com | Free |

---

## Step 1 — Get a Free Groq API Key

LearnSphere uses Groq (not OpenAI) so it runs completely free.

1. Go to **https://console.groq.com**
2. Sign up / log in with Google or GitHub
3. Click **"API Keys"** in the left sidebar
4. Click **"Create API Key"**
5. Copy the key — it looks like: `gsk_xxxxxxxxxxxxxxxxxxxx`
6. **Save it** — you'll need it in Step 4

---

## Step 2 — Upload Your Code to GitHub

Render deploys directly from a GitHub repository.

### 2a. Create a new GitHub repository

1. Go to **https://github.com/new**
2. Repository name: `learnsphere` (or any name)
3. Set to **Public** (free Render works with both, but Public is easier)
4. Click **"Create repository"**

### 2b. Upload your files

**Option A — Upload via GitHub website (easiest, no Git needed):**

1. Open your new empty repository on GitHub
2. Click **"uploading an existing file"** link
3. Drag and drop the **entire contents** of your `learnsphere/` folder
   - Make sure you upload: `app.py`, `requirements.txt`, `render.yaml`, `Procfile`, `static/`, `templates/`
   - Do **NOT** upload `learnsphere.db` (the database file)
4. Scroll down, add commit message: `Initial LearnSphere deployment`
5. Click **"Commit changes"**

**Option B — Using Git on your computer:**
```bash
cd learnsphere
git init
git add .
git commit -m "Initial LearnSphere deployment"
git remote add origin https://github.com/YOUR_USERNAME/learnsphere.git
git push -u origin main
```

---

## Step 3 — Create a Web Service on Render

1. Go to **https://dashboard.render.com**
2. Click **"New +"** → select **"Web Service"**
3. Click **"Connect a repository"**
4. Find and select your `learnsphere` repository
5. Click **"Connect"**

---

## Step 4 — Configure the Service

Fill in these settings on the configuration page:

| Setting | Value |
|---------|-------|
| **Name** | `learnsphere` (or anything you like) |
| **Region** | Singapore (closest for India) or US |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120` |
| **Instance Type** | **Free** |

### Add Environment Variables

Scroll down to **"Environment Variables"** and add these:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | `gsk_xxxx` ← paste your Groq key from Step 1 |
| `SECRET_KEY` | Click **"Generate"** button (Render auto-creates a secure key) |

> ⚠️ **Important:** Never put your API key in the code or commit it to GitHub.

---

## Step 5 — Deploy!

1. Click **"Create Web Service"** at the bottom
2. Render will start building — you'll see live logs
3. Build takes **2–3 minutes** (installing Python packages)
4. When you see **"Your service is live 🎉"**, it's done!

Your app will be at: **`https://learnsphere.onrender.com`** (or similar URL shown at the top)

---

## 🔑 Default Login Credentials

When the app first starts, it creates these accounts automatically:

| Role | Username | Password |
|------|----------|----------|
| Educator | `educator` | `admin123` |
| Student | `student` | `student123` |

> 🔒 **Change these passwords** after first login via the profile settings.

---

## ⚠️ Important: Free Tier Limitations

### Spin-down (most important!)
- Free Render services **sleep after 15 minutes** of no traffic
- First visit after sleeping takes **30–60 seconds** to wake up
- This is normal — once awake, it's fast

### Database resets
- SQLite on Render's free tier uses `/tmp` storage
- **Data is lost when the service restarts or redeploys**
- This is fine for testing; for production use, see "Persistent Database" below

### Resource limits
- 512 MB RAM, shared CPU
- 750 free hours/month (enough for one always-on service)

---

## 🗄️ Making Data Persist (Optional Upgrade)

The free tier resets data on restart. To keep data permanently:

### Option A — Render PostgreSQL (requires code changes)
Render has a free PostgreSQL database tier. Would require migrating from SQLite.

### Option B — Keep SQLite, use Render Disk (paid, $1/month)
Add a disk in Render settings and set:
```
DB_PATH = /var/data/learnsphere.db
```

### Option C — Stay free, back up regularly
Since this is an educational app, data loss on restart is acceptable for demos.

---

## 🔄 How to Update Your App

After making changes locally:

1. Upload the changed files to GitHub (drag-drop or `git push`)
2. Render **automatically detects** the push and redeploys
3. New version is live in ~2 minutes

---

## 🐛 Troubleshooting

### Build fails
- Check the **Logs** tab in Render dashboard
- Most common cause: missing package in `requirements.txt`

### "Application error" on page load
- Check logs for Python errors
- Usually means `GROQ_API_KEY` is not set correctly

### AI features not working
- Verify your Groq API key is correct (no extra spaces)
- Check if Groq is rate-limiting you at console.groq.com

### App is slow / timing out
- Free tier has limited CPU — normal for first load
- AI responses on free tier may take 10–20 seconds

---

## 📋 File Checklist

Make sure these files are in your GitHub repo root:

```
learnsphere/
├── app.py              ✅ Main Flask application
├── requirements.txt    ✅ Python dependencies  
├── render.yaml         ✅ Render configuration
├── Procfile            ✅ Process file (backup)
├── static/
│   ├── css/style.css   ✅ Stylesheet
│   └── js/main.js      ✅ JavaScript
└── templates/          ✅ All HTML templates
    ├── base.html
    ├── login.html
    └── ... (all other templates)
```

---

## ✅ Quick Checklist

- [ ] Got Groq API key from console.groq.com
- [ ] Created GitHub repository
- [ ] Uploaded all project files to GitHub
- [ ] Created Render Web Service
- [ ] Set `GROQ_API_KEY` environment variable
- [ ] Set `SECRET_KEY` environment variable  
- [ ] Selected **Free** instance type
- [ ] App is live at your Render URL
- [ ] Logged in with `educator / admin123`

---

*Need help? The Render documentation is at docs.render.com*
