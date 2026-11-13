# 🗄️ Setting Up & Deploying a Free Cloud Database for LearnSphere

This guide walks you through setting up a **free cloud PostgreSQL database** (using **Supabase** or **Neon**) and connecting it to your LearnSphere deployment on **Vercel**.

---

## ❓ Why Did Data Reset on Vercel Previously?
LearnSphere was initially configured with **SQLite** (`/tmp/learnsphere.db`).
On Vercel, Python runs as **serverless functions (AWS Lambda)**. Every time the serverless container stops or restarts (cold start), the temporary disk is wiped clean. This meant all registered users, course progress, and quiz results were erased.

With the new **Cloud Database Adapter** (`db_adapter.py`), LearnSphere automatically connects to a **cloud-hosted PostgreSQL database** whenever `DATABASE_URL` is set in your environment variables, keeping all your data persistent and secure.

---

## 🚀 Option 1: Supabase (Recommended — 100% Free)

### Step 1: Create a Free Supabase Project
1. Go to **[supabase.com](https://supabase.com)** and click **Start your project** (sign up with GitHub).
2. Click **"New Project"**.
3. Fill in:
   - **Name:** `learnsphere-db`
   - **Database Password:** Enter a strong password (and **save it**!).
   - **Region:** Choose the region closest to you (e.g. `Singapore` or `East US`).
4. Click **"Create new project"** (takes ~1-2 minutes to provision).

### Step 2: Copy the Connection String
1. In your Supabase dashboard, click the **Connect** button at the top (or go to **Project Settings** → **Database**).
2. Under **Connection string**, select **URI**.
3. Choose the **Session pooler** or **Direct connection** (both work):
   ```
   postgresql://postgres.[project-ref]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres
   ```
4. Replace `[YOUR-PASSWORD]` with the password you created in Step 1.

---

## ⚡ Option 2: Neon (Fast Serverless Postgres — 100% Free)

1. Go to **[neon.tech](https://neon.tech)** and sign up with GitHub.
2. Click **"Create Project"** → Name it `learnsphere`.
3. On the project dashboard, copy the **Connection string** (select `psql` or `Connection string`):
   ```
   postgresql://user:password@ep-xyz.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

---

## 🔗 Step 3: Add `DATABASE_URL` to Vercel

1. Go to your **[Vercel Dashboard](https://vercel.com/dashboard)**.
2. Click on your **LearnSphere** project.
3. Go to **Settings** → **Environment Variables**.
4. Add the following variables:

| Key | Value |
|---|---|
| `DATABASE_URL` | *Paste your Supabase or Neon connection string from above* |
| `GROQ_API_KEY` | *Your Groq API key (`gsk_...`)* |
| `SECRET_KEY` | *Any random secure string, e.g. `learnsphere_prod_secret_2026`* |

5. Click **Save**.

---

## 🚀 Step 4: Deploy to Vercel

Once you have pushed the latest code to GitHub, redeploy on Vercel:

```bash
cd LearnSphere_Platform
git add .
git commit -m "Add PostgreSQL cloud database adapter and Vercel configuration"
git push origin main
```

1. Vercel will trigger a new deployment automatically.
2. When the app starts up, `init_db()` will automatically:
   - Create all **40+ database tables** in your cloud PostgreSQL database.
   - Seed the default educator demo account (`educator@learnsphere.com` / `admin123`).
   - Seed the default student demo account (`student@learnsphere.com` / `student123`).
   - Populate the initial courses and lecture curriculum.
3. Every new user registration, quiz score, and course enrollment is now permanently stored in the cloud and will **never** disappear!
