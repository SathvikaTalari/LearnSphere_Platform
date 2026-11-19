from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response, send_from_directory
from groq import Groq
import json, os, sqlite3, hashlib, uuid, random
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import parse_qs, urlencode, urlparse

try:
    from dotenv import load_dotenv
    # Load from LearnSphere_Platform/.env and parent c:\LearnSPhere\.env
    _cur_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    _parent_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(_cur_env):
        load_dotenv(_cur_env, override=True)
    if os.path.exists(_parent_env):
        load_dotenv(_parent_env, override=False)
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(BASE_DIR, 'public', 'static') if os.path.isdir(os.path.join(BASE_DIR, 'public', 'static')) else os.path.join(BASE_DIR, 'static')
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=_STATIC_DIR,
    static_url_path='/static'
)

class VercelPathFixMiddleware:
    """
    Restores the real URL path when running behind Vercel serverless rewrites.
    Works whether Vercel runs app.py directly or through api/index.py.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        qs = environ.get('QUERY_STRING', '')
        if '__path__' in qs:
            params = parse_qs(qs, keep_blank_values=True)
            if '__path__' in params:
                raw_target = params.pop('__path__')[0]
                clean = '/' + raw_target.lstrip('/')
                environ['PATH_INFO'] = clean
                environ['QUERY_STRING'] = urlencode(params, doseq=True)
                return self.wsgi_app(environ, start_response)

        matched = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_FORWARDED_URI') or ''
        if matched and not matched.startswith('/api/index'):
            parsed = urlparse(matched)
            environ['PATH_INFO'] = '/' + parsed.path.lstrip('/')
            return self.wsgi_app(environ, start_response)

        path_info = environ.get('PATH_INFO', '')
        if path_info in ('/api/index', '/api/index/'):
            environ['PATH_INFO'] = '/'
        elif path_info.startswith('/api/index/'):
            environ['PATH_INFO'] = '/' + path_info[len('/api/index'):].lstrip('/')

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

@app.route('/static/<path:filename>')
def serve_static(filename):
    for s_dir in [
        os.path.join(BASE_DIR, 'public', 'static'),
        os.path.join(BASE_DIR, 'static')
    ]:
        if os.path.exists(os.path.join(s_dir, filename)):
            return send_from_directory(s_dir, filename)
    return ('File not found', 404)
app.secret_key = os.environ.get('SECRET_KEY', 'learnsphere_secret_2024_xyz_dev_only')

# ─── GROQ API SETUP (Free & Fast) ─────────────────────────────────────────
# Get free API key at: https://console.groq.com
CANDIDATE_GROQ_MODELS = [
    'openai/gpt-oss-120b',
    'qwen/qwen3.8-27b',
    'openai/gpt-oss-20b',
    'llama-3.3-70b-versatile',
    'llama-3.1-70b-versatile'
]
GROQ_MODEL = 'openai/gpt-oss-120b'

groq_client = None

def get_groq_client():
    global groq_client
    key = os.environ.get('GROQ_API_KEY', '').strip()
    if not key or key == 'YOUR_GROQ_API_KEY':
        for p in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        ]:
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip().startswith('GROQ_API_KEY='):
                                candidate = line.strip().split('=', 1)[1].strip()
                                if candidate and not candidate.startswith('YOUR_'):
                                    os.environ['GROQ_API_KEY'] = candidate
                                    key = candidate
                                    break
                except Exception:
                    pass
    if key and not key.startswith('YOUR_'):
        if groq_client is None or getattr(groq_client, '_api_key', None) != key:
            groq_client = Groq(api_key=key)
        return groq_client
    return None

def ask_groq(prompt, system="You are LearnSphere, an expert ML tutor.", temperature=0.7, max_tokens=8192):
    """Call Groq API with automatic model fallback."""
    client = get_groq_client()
    if not client:
        return "Please configure a valid GROQ_API_KEY in .env to enable AI features."

    for m in CANDIDATE_GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if 'model_not_found' in err_str or '404' in err_str:
                continue
            raise e
    raise RuntimeError("No available Groq model succeeded")

def clean_and_parse_json(text):
    """
    Robustly parse JSON from LLM output.
    Handles: markdown fences, control characters, bad escapes, code blocks inside JSON.
    """
    text = text.strip()

    # Remove markdown code fences
    import re
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find outermost JSON object
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    # Try again after extracting
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fix common issues:
    # 1. Remove actual newlines/tabs inside string values (keep JSON structure newlines)
    # 2. Fix unescaped backslashes in code strings
    # 3. Replace control characters
    def fix_json_string(m):
        s = m.group(0)
        # Replace real newlines inside strings with \n
        s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        # Fix double-escaped: \\n that became \\\\n
        s = s.replace('\\\\n', '\\n').replace('\\\\t', '\\t')
        return s

    # Fix string values that span multiple lines
    fixed = re.sub(r'"(?:[^"\\]|\\.)*"', fix_json_string, text, flags=re.DOTALL)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # Last resort: use a very permissive approach
    # Remove all control characters except proper JSON whitespace
    clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    try:
        return json.loads(clean)
    except Exception as e:
        raise ValueError(f"Could not parse JSON after all attempts. Raw error: {e}\nText preview: {text[:200]}")


def ask_groq_json(prompt, system="You are LearnSphere, an expert ML tutor. Always respond with valid JSON only. Never include raw newlines inside JSON string values - use \\n instead.", temperature=0.3):
    """Call Groq and parse JSON response reliably."""
    text = ask_groq(prompt, system=system, temperature=temperature)
    return clean_and_parse_json(text)

# Database adapter (Supports PostgreSQL in production and SQLite in local development)
from db_adapter import get_db, DB_PATH

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'student',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP, level TEXT DEFAULT 'beginner',
        total_queries INTEGER DEFAULT 0, correct_attempts INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS courses (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT, category TEXT,
        difficulty TEXT DEFAULT 'beginner', educator_id TEXT NOT NULL, educator_name TEXT,
        thumbnail_color TEXT DEFAULT '#10B981', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        is_published INTEGER DEFAULT 1, FOREIGN KEY (educator_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS lectures (
        id TEXT PRIMARY KEY, course_id TEXT NOT NULL, title TEXT NOT NULL,
        content TEXT, lecture_type TEXT DEFAULT 'text', video_url TEXT, code_example TEXT,
        diagram_svg TEXT, order_num INTEGER DEFAULT 0, duration_mins INTEGER DEFAULT 10,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (course_id) REFERENCES courses(id))''')
    try:
        c.execute("ALTER TABLE lectures ADD COLUMN diagram_svg TEXT")
    except Exception:
        pass
    c.execute('''CREATE TABLE IF NOT EXISTS enrollments (
        id TEXT PRIMARY KEY, student_id TEXT NOT NULL, course_id TEXT NOT NULL,
        enrolled_at TEXT DEFAULT CURRENT_TIMESTAMP, progress INTEGER DEFAULT 0,
        UNIQUE(student_id, course_id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS completed_lectures (
        id TEXT PRIMARY KEY, student_id TEXT NOT NULL, lecture_id TEXT NOT NULL,
        completed_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(student_id, lecture_id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS notes (
        id TEXT PRIMARY KEY, student_id TEXT NOT NULL, lecture_id TEXT NOT NULL,
        content TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS otps (
        email TEXT PRIMARY KEY,
        otp TEXT NOT NULL,
        expires_at TEXT NOT NULL)''')

    # AI Learning DNA — tracks per-concept performance
    c.execute('''CREATE TABLE IF NOT EXISTS learning_dna (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        concept TEXT NOT NULL,
        attempts INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        avg_response_ms INTEGER DEFAULT 0,
        last_seen TEXT,
        next_review TEXT,
        ease_factor REAL DEFAULT 2.5,
        interval_days INTEGER DEFAULT 1,
        style_pref TEXT DEFAULT 'balanced',
        UNIQUE(user_id, concept))''')

    # Battle mode scores
    c.execute('''CREATE TABLE IF NOT EXISTS battle_scores (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        username TEXT,
        score INTEGER DEFAULT 0,
        ai_score INTEGER DEFAULT 0,
        topic TEXT,
        played_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    # ── Focus Reward Engine ──────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS focus_sessions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        started_at TEXT DEFAULT CURRENT_TIMESTAMP,
        ended_at TEXT,
        duration_mins INTEGER DEFAULT 0,
        coins_earned INTEGER DEFAULT 0,
        topic TEXT DEFAULT \'General Study\',
        status TEXT DEFAULT \'active\')''')

    c.execute('''CREATE TABLE IF NOT EXISTS user_rewards (
        user_id TEXT PRIMARY KEY,
        coins INTEGER DEFAULT 0,
        total_focus_mins INTEGER DEFAULT 0,
        streak_days INTEGER DEFAULT 0,
        last_study_date TEXT,
        badges TEXT DEFAULT \'[]\')''')

    # ── Events & Opportunities ───────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        platform TEXT,
        url TEXT,
        event_type TEXT DEFAULT \'hackathon\',
        description TEXT,
        deadline TEXT,
        prize TEXT,
        tags TEXT DEFAULT \'[]\')''')

    # ── Career Tracker ───────────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS career_entries (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        entry_type TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        date_earned TEXT DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT DEFAULT \'{}\',
        verified INTEGER DEFAULT 0)''')

    # ── Assessments / Quizzes ────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS assessments (
        id TEXT PRIMARY KEY,
        course_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        pass_score INTEGER DEFAULT 60,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS assessment_questions (
        id TEXT PRIMARY KEY,
        assessment_id TEXT NOT NULL,
        question_type TEXT NOT NULL,
        question_text TEXT NOT NULL,
        options TEXT DEFAULT \'[]\',
        correct_answer TEXT,
        explanation TEXT,
        points INTEGER DEFAULT 10,
        order_num INTEGER DEFAULT 0)''')

    c.execute('''CREATE TABLE IF NOT EXISTS assessment_attempts (
        id TEXT PRIMARY KEY,
        assessment_id TEXT NOT NULL,
        student_id TEXT NOT NULL,
        score INTEGER DEFAULT 0,
        max_score INTEGER DEFAULT 0,
        passed INTEGER DEFAULT 0,
        answers TEXT DEFAULT \'{}\',
        attempted_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    # ── Certificates ─────────────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS certificates (
        id TEXT PRIMARY KEY,
        student_id TEXT NOT NULL,
        course_id TEXT NOT NULL,
        student_name TEXT,
        course_title TEXT,
        issued_at TEXT DEFAULT CURRENT_TIMESTAMP,
        cert_code TEXT UNIQUE,
        UNIQUE(student_id, course_id))''')

    # ── Course Ratings ────────────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS course_ratings (
        id TEXT PRIMARY KEY,
        course_id TEXT NOT NULL,
        student_id TEXT NOT NULL,
        rating INTEGER DEFAULT 5,
        review TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(course_id, student_id))''')

    # ── Community & Social Features ──────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS course_forums (
        id TEXT PRIMARY KEY,
        course_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        username TEXT NOT NULL,
        content TEXT NOT NULL,
        parent_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS study_groups (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        course_id TEXT,
        created_by TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS study_group_members (
        group_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        username TEXT,
        joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (group_id, user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS study_group_notes (
        id TEXT PRIMARY KEY,
        group_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        username TEXT,
        content TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS mock_interviews (
        id TEXT PRIMARY KEY,
        host_id TEXT NOT NULL,
        host_name TEXT NOT NULL,
        peer_id TEXT,
        peer_name TEXT,
        topic TEXT NOT NULL,
        meeting_url TEXT,
        scheduled_at TEXT NOT NULL,
        status TEXT DEFAULT 'open',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    # Upgrade courses table if needed
    for col_sql in [
        "ALTER TABLE courses ADD COLUMN price REAL DEFAULT 0.0",
        "ALTER TABLE courses ADD COLUMN features TEXT DEFAULT '[]'",
        "ALTER TABLE courses ADD COLUMN duration_hours REAL DEFAULT 0.0",
        "ALTER TABLE users ADD COLUMN college TEXT DEFAULT ''",
        "ALTER TABLE users ADD COLUMN region TEXT DEFAULT ''",
        "ALTER TABLE users ADD COLUMN bio TEXT DEFAULT ''",
        "ALTER TABLE users ADD COLUMN headline TEXT DEFAULT ''",
        "ALTER TABLE users ADD COLUMN interests TEXT DEFAULT ''",
        "ALTER TABLE users ADD COLUMN preferences TEXT DEFAULT '{}'",
    ]:
        try: c.execute(col_sql)
        except Exception: pass

    admin_id = str(uuid.uuid4())
    admin_pw = hash_password('admin123')
    try:
        c.execute("INSERT INTO users (id,username,email,password,role) VALUES (?,?,?,?,?)",
                  (admin_id,'educator','educator@learnsphere.com',admin_pw,'educator'))
        # Sample student
        stu_id = str(uuid.uuid4())
        c.execute("INSERT INTO users (id,username,email,password,role) VALUES (?,?,?,?,?)",
                  (stu_id,'student','student@learnsphere.com',hash_password('student123'),'student'))

        cid = str(uuid.uuid4())
        c.execute("""INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (cid,'Machine Learning Fundamentals',
                   'A complete beginner-friendly course covering core ML concepts from linear regression to neural networks.',
                   'Machine Learning','beginner',admin_id,'Prof. Educator','#10B981'))

        lectures_data = [
            (1,'Introduction to Machine Learning','text',15,
             '''Machine Learning is a subset of Artificial Intelligence that enables computers to learn from data without being explicitly programmed.

## What is Machine Learning?
Machine Learning allows systems to automatically learn and improve from experience. Instead of writing rules manually, we feed data to algorithms that discover patterns themselves.

## Three Core Types of ML

**1. Supervised Learning**
The algorithm learns from labeled training data. You provide inputs AND expected outputs.
- Examples: Email spam detection, image classification, price prediction

**2. Unsupervised Learning**
The algorithm finds hidden patterns in unlabeled data on its own.
- Examples: Customer segmentation, anomaly detection, recommendation systems

**3. Reinforcement Learning**
An agent learns by interacting with an environment, receiving rewards/penalties.
- Examples: Game-playing AI (Chess, Go), robotics, autonomous vehicles

## Why ML Matters Today
- Powers Netflix/YouTube recommendations
- Enables voice assistants (Siri, Alexa, Google)
- Drives self-driving car systems
- Detects fraud in banking in real-time
- Diagnoses diseases from medical images

## The Standard ML Workflow
1. **Collect Data** — gather relevant, quality data
2. **Prepare Data** — clean, normalize, split train/test
3. **Choose Algorithm** — pick based on problem type
4. **Train Model** — fit model to training data
5. **Evaluate** — measure accuracy on test data
6. **Deploy** — serve predictions in production''',
             '# No code in intro lecture'),

            (2,'Linear Regression','text',25,
             '''Linear Regression is the most fundamental supervised learning algorithm for predicting continuous numeric values.

## Core Concept
Find the best straight line (or hyperplane) that fits through data points to make predictions.

**The Formula:**
```
y = w₁x₁ + w₂x₂ + ... + b
```
- y = predicted value
- x = input features
- w = weights (learned from data)
- b = bias/intercept

## How Training Works
1. Initialize weights randomly
2. Make predictions on training data
3. Calculate loss (how wrong predictions are)
4. Use gradient descent to adjust weights
5. Repeat until loss stops decreasing

**Loss Function — Mean Squared Error:**
```
MSE = (1/n) Σ (y_predicted - y_actual)²
```

## Real-World Applications
- House price prediction (size → price)
- Sales forecasting (ad spend → revenue)
- Student performance (study hours → grade)
- Medical dosage (weight → dose)

## Evaluation Metrics
- **R² Score**: How well model explains variance (1.0 = perfect)
- **MSE**: Average squared prediction error
- **RMSE**: Root MSE (same units as target)''',
             '''import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset: Study hours vs exam scores
hours = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
scores = np.array([35,42,57,62,70,78,82,88,92,96])

X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"R² Score:   {r2_score(y_test, predictions):.3f}")
print(f"MSE:        {mean_squared_error(y_test, predictions):.2f}")
print(f"Slope:      {model.coef_[0]:.2f}")
print(f"Intercept:  {model.intercept_:.2f}")
print(f"\\nPrediction for 7.5 hours: {model.predict([[7.5]])[0]:.1f}")'''),

            (3,'Decision Trees & Random Forests','text',30,
             '''Decision Trees are intuitive ML models that make predictions by asking a series of yes/no questions.

## How a Decision Tree Works
Think of it like 20 questions:
- "Is the tumor size > 3cm?" → Yes → "Is patient age > 50?" → ...

**Key Components:**
- **Root Node**: First split (most important feature)
- **Internal Nodes**: Subsequent questions
- **Leaf Nodes**: Final prediction/class
- **Splitting Criterion**: Gini Impurity or Information Gain

## Pros & Cons
✅ Easy to understand and visualize
✅ No feature scaling needed
✅ Handles mixed data types
❌ Prone to overfitting
❌ Unstable with small data changes

## Random Forest = Ensemble of Trees
Combines 100–1000 decision trees, each trained on random data subsets. Takes majority vote for final prediction.

**Why it's better:**
- Reduces overfitting dramatically
- More stable and accurate
- Provides feature importance scores
- Works well out of the box

## When to Use
- Classification (spam, disease, fraud)
- Feature importance analysis
- When you need interpretable results
- Tabular/structured data problems''',
             '''from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.2%}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2%}")

print("\\nFeature Importances:")
for name, imp in zip(iris.feature_names, rf.feature_importances_):
    print(f"  {name}: {imp:.3f}")'''),

            (4,'Neural Networks Deep Dive','text',45,
             '''Neural Networks are the foundation of modern AI — loosely inspired by the human brain's structure.

## Architecture
```
Input Layer → Hidden Layers → Output Layer
[x1,x2,x3] →  [neurons]   →   [output]
```

## How a Single Neuron Works
1. Receive inputs: x₁, x₂, x₃...
2. Multiply by weights: w₁, w₂, w₃...
3. Sum + add bias: z = Σwᵢxᵢ + b
4. Apply activation function: a = f(z)
5. Pass result to next layer

## Common Activation Functions
| Function | Formula | Use Case |
|----------|---------|---------|
| ReLU | max(0, x) | Hidden layers (most common) |
| Sigmoid | 1/(1+e⁻ˣ) | Binary output (0-1) |
| Softmax | eˣⁱ/Σeˣʲ | Multi-class output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Hidden layers (-1 to 1) |

## Training Process
1. **Forward Pass** → compute predictions
2. **Loss Calculation** → measure error
3. **Backpropagation** → compute gradients
4. **Weight Update** → gradient descent step
5. **Repeat** for many epochs

## Types of Neural Networks
- **MLP**: Basic feedforward (tabular data)
- **CNN**: Images and spatial data
- **RNN/LSTM**: Sequences (text, time-series)
- **Transformer**: State-of-the-art NLP (GPT, BERT)
- **GAN**: Generate new data (images, music)''',
             '''import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lr = lr

    def relu(self, x): return np.maximum(0, x)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        dz2 = self.a2 - y
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0, keepdims=True)
        dz1 = (dz2 @ self.W2.T) * (self.a1 > 0)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0, keepdims=True)
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000):
        for e in range(epochs):
            out = self.forward(X)
            self.backward(X, y)
            if e % 200 == 0:
                loss = -np.mean(y*np.log(out+1e-8)+(1-y)*np.log(1-out+1e-8))
                print(f"Epoch {e:4d}: Loss={loss:.4f}")

# XOR problem demo
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(2, 4, 1, lr=0.1)
nn.train(X, y, epochs=1000)
print("\\nPredictions:", nn.forward(X).round(2))'''),
        ]

        for order, title, ltype, dur, content, code in lectures_data:
            lid = str(uuid.uuid4())
            c.execute("""INSERT INTO lectures (id,course_id,title,content,lecture_type,code_example,order_num,duration_mins)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (lid, cid, title, content, ltype, code, order, dur))

        # Course 2
        cid2 = str(uuid.uuid4())
        c.execute("""INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (cid2,'Deep Learning with Python',
                   'Advanced course covering CNNs, RNNs, Transformers, and hands-on deep learning projects with TensorFlow & PyTorch.',
                   'Deep Learning','advanced',admin_id,'Prof. Educator','#7C3AED'))
        dl_lid = str(uuid.uuid4())
        c.execute("""INSERT INTO lectures (id,course_id,title,content,lecture_type,code_example,order_num,duration_mins)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (dl_lid,cid2,'Introduction to Deep Learning',
                   '''Deep Learning is a subset of ML using multi-layered neural networks to learn hierarchical representations from large amounts of data.

## Why Deep Learning Exploded
Three converging factors:
1. **Big Data** — internet-scale labeled datasets (ImageNet, Common Crawl)
2. **GPUs** — parallel computing made training feasible
3. **Better Algorithms** — ReLU, Dropout, BatchNorm, Adam optimizer

## Deep vs Shallow Learning
- **Shallow ML**: handcrafted features → simple model
- **Deep Learning**: raw data → automatic feature learning → prediction

## Major Applications
- Computer Vision (image classification, object detection, segmentation)
- NLP (translation, summarization, question answering)
- Speech Recognition & synthesis
- Drug discovery & protein folding (AlphaFold)
- Game playing (AlphaGo, OpenAI Five)

## Popular Frameworks
- **TensorFlow/Keras** (Google) — production ready, easy API
- **PyTorch** (Meta) — research favorite, dynamic graphs
- **JAX** (Google) — high performance, functional style''',
                   'text',
                   '''import tensorflow as tf

# Simple deep neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
# Train: model.fit(X_train, y_train, epochs=10, validation_split=0.1)''',1,20))

        # Course 3
        cid3 = str(uuid.uuid4())
        c.execute("""INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (cid3,'Natural Language Processing',
                   'Learn how machines understand text: from tokenization and embeddings to Transformers, BERT, and GPT.',
                   'NLP','intermediate',admin_id,'Prof. Educator','#10B981'))
        nlp_lid = str(uuid.uuid4())
        c.execute("""INSERT INTO lectures (id,course_id,title,content,lecture_type,code_example,order_num,duration_mins)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (nlp_lid,cid3,'Text Preprocessing Fundamentals',
                   '''Before feeding text to ML models, we must convert raw text into numerical representations.

## NLP Pipeline
```
Raw Text → Tokenize → Clean → Vectorize → Model
```

## Step 1: Tokenization
Split text into words, subwords, or characters.
- "Hello world" → ["Hello", "world"]
- "don't" → ["don", "'", "t"]  or  ["don't"]

## Step 2: Text Cleaning
- Lowercase: "Hello" → "hello"
- Remove punctuation, special characters
- Remove stop words: "the", "a", "is"
- Stemming/Lemmatization: "running" → "run"

## Step 3: Vectorization
Convert tokens to numbers:

**Bag of Words**: Count word frequencies (ignores order)

**TF-IDF**: Weight words by importance across documents

**Word Embeddings**: Dense vectors capturing semantic meaning
- Word2Vec, GloVe, FastText
- "king" - "man" + "woman" ≈ "queen"

**Modern**: BERT/GPT tokenizers (subword BPE)

## Why This Matters
Bad preprocessing = bad model performance
Good features = 80% of NLP success''',
                   'text',
                   '''import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample text data
texts = [
    "Machine learning is amazing and powerful",
    "Deep learning uses neural networks",
    "Python is great for data science",
    "Natural language processing enables text analysis",
    "Neural networks learn from examples"
]
labels = [1, 1, 0, 1, 1]  # 1=ML topic, 0=other

def preprocess(text):
    text = text.lower()
    text = re.sub(r\'[^a-z\\s]\', \'\', text)
    return text

processed = [preprocess(t) for t in texts]

# TF-IDF + Naive Bayes Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=1000)),
    ("clf", MultinomialNB())
])
pipeline.fit(processed, labels)

test = ["recurrent neural networks handle sequences"]
print("Prediction:", "ML topic" if pipeline.predict([preprocess(test[0])])[0] else "Other")''',1,20))

        # ── Course 4: Computer Vision with CNNs ──────────────────────────────
        cid4 = str(uuid.uuid4())
        c.execute("""INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (cid4,'Computer Vision with CNNs',
                   'Master image classification, object detection, and segmentation using Convolutional Neural Networks. Build real-world CV applications.',
                   'Computer Vision','intermediate',admin_id,'Prof. Educator','#EC4899'))
        cv_lectures = [
            (1,'Image Classification Fundamentals','text',25,
             '''Image Classification is the task of assigning a label to an image from a predefined set of categories.

## How Computers See Images
Images are stored as 3D arrays of numbers:
- Height × Width × Channels (RGB)
- Each pixel: 0-255 intensity value
- A 224×224 color image = 224 × 224 × 3 = 150,528 numbers

## The CNN Architecture
```
Input → [Conv → ReLU → Pool] × N → Flatten → Dense → Output
```

**Convolutional Layer**: Slides small filters across the image to detect features
- Early layers: edges, corners, textures
- Deep layers: eyes, wheels, faces

**Pooling Layer**: Reduces spatial dimensions (Max Pool most common)

**Fully Connected**: Final classification layers

## Key CNN Architectures
| Model | Year | Params | Top-1 Accuracy |
|-------|------|--------|----------------|
| AlexNet | 2012 | 60M | 63.3% |
| VGG-16 | 2014 | 138M | 74.4% |
| ResNet-50 | 2015 | 25M | 76.1% |
| EfficientNet | 2019 | 5.3M | 77.1% |

## Applications
- Medical imaging (X-ray diagnosis)
- Self-driving cars (road recognition)
- Face recognition systems
- Quality control in manufacturing''',
             '''import tensorflow as tf
from tensorflow.keras import layers, models

# Build a CNN for image classification
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Train
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))'''),
            (2,'Object Detection with YOLO','text',35,
             '''Object Detection localizes AND classifies multiple objects within an image simultaneously.

## Detection vs Classification
- **Classification**: "This image contains a cat"
- **Detection**: "There is a cat at coordinates (50,30,200,180) and a dog at (300,100,450,350)"

## YOLO (You Only Look Once)
The most popular real-time detection framework.

### How YOLO Works
1. Divide image into S×S grid
2. Each cell predicts B bounding boxes + confidence
3. Each cell predicts C class probabilities
4. Non-max suppression removes duplicates

### Key Metrics
- **IoU** (Intersection over Union): Measures overlap between predicted and ground truth boxes
- **mAP** (mean Average Precision): Standard detection metric
- **FPS**: Frames per second (real-time = 30+ FPS)

## YOLO Versions
- YOLOv3: Balanced speed/accuracy
- YOLOv5: PyTorch native, easy to use
- YOLOv8: Latest, state-of-the-art

## Applications
- Autonomous driving
- Security surveillance
- Retail analytics
- Sports analysis''',
             '''from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano model for speed

# Run inference on an image
# results = model("path/to/image.jpg")

# Train on custom dataset
# model.train(data="custom.yaml", epochs=50, imgsz=640)

# Example: Process results
# for result in results:
#     boxes = result.boxes
#     for box in boxes:
#         x1, y1, x2, y2 = box.xyxy[0]
#         conf = box.conf[0]
#         cls = box.cls[0]
#         print(f"Class: {cls}, Confidence: {conf:.2f}")
#         print(f"Bounding Box: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})")

print("YOLOv8 loaded successfully!")
print("Model has", sum(p.numel() for p in model.model.parameters()), "parameters")'''),
            (3,'Transfer Learning & Fine-tuning','text',30,
             '''Transfer Learning leverages pre-trained models to solve new tasks with minimal data and training.

## Why Transfer Learning?
- Training from scratch needs millions of images
- Pre-trained models already know edges, textures, shapes
- Fine-tuning only adjusts top layers for your task

## The Process
```
Pre-trained Model (ImageNet) → Freeze base layers → Add custom head → Fine-tune
```

### Step 1: Choose a backbone
- ResNet, EfficientNet, VGG, MobileNet

### Step 2: Freeze early layers
- Keep learned features from ImageNet
- Only train the last few layers

### Step 3: Add your classifier
- Replace the final classification layer
- Add Dense layers for your specific classes

### Step 4: Fine-tune
- Unfreeze some deeper layers
- Train with very low learning rate (1e-5)

## When to Use What
| Your Data | Strategy |
|-----------|----------|
| Small, similar to ImageNet | Freeze all, train head only |
| Small, different domain | Freeze early, fine-tune late |
| Large dataset | Fine-tune entire network |

## Data Augmentation
Artificially expand your training data:
- Random rotations, flips, crops
- Color jittering, brightness changes
- Mixup, CutMix, RandAugment''',
             '''import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# Load pre-trained ResNet50 (without top classification layer)
base_model = ResNet50(weights="imagenet", include_top=False,
                      input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Build custom classifier on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(5, activation="softmax")  # 5 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"Total params: {model.count_params():,}")
print(f"Trainable params: {sum(p.numpy().size for p in model.trainable_weights):,}")

# Fine-tuning: unfreeze last 20 layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])
print("\\nAfter fine-tuning setup:")
print(f"Trainable params: {sum(p.numpy().size for p in model.trainable_weights):,}")'''),
            (4,'Image Segmentation','text',30,
             '''Image Segmentation assigns a class label to every single pixel in an image.

## Types of Segmentation
1. **Semantic Segmentation**: Label every pixel with a class (all cars = same label)
2. **Instance Segmentation**: Distinguish between individual objects (car1, car2)
3. **Panoptic Segmentation**: Combines both semantic + instance

## U-Net Architecture
The most popular segmentation architecture:
```
Encoder (Downsampling) → Bottleneck → Decoder (Upsampling)
        ↓ Skip connections ↑
```

### Encoder Path
- Convolutional blocks + max pooling
- Captures WHAT is in the image

### Decoder Path
- Transposed convolutions (upsampling)
- Skip connections from encoder
- Recovers WHERE objects are

### Skip Connections
- Connect encoder layers to decoder layers
- Preserve fine spatial details
- Critical for precise boundaries

## Loss Functions
- **Cross-Entropy**: Standard pixel-wise loss
- **Dice Loss**: Measures overlap between prediction and ground truth
- **Focal Loss**: Handles class imbalance

## Applications
- Medical image analysis (tumor segmentation)
- Autonomous driving (road, pedestrian, vehicle)
- Satellite imagery (land use mapping)
- Video editing (background removal)''',
             '''import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(256, 256, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, 3, activation="relu", padding="same")(c3)

    # Decoder
    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c2])  # Skip connection
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(u1)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c1])  # Skip connection
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(c5)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c5)

    return models.Model(inputs, outputs)

model = build_unet()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("U-Net built successfully!")
model.summary()'''),
        ]
        for order, title, ltype, dur, content, code in cv_lectures:
            lid = str(uuid.uuid4())
            c.execute("""INSERT INTO lectures (id,course_id,title,content,lecture_type,code_example,order_num,duration_mins)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (lid, cid4, title, content, ltype, code, order, dur))

        # ── Course 5: Reinforcement Learning ─────────────────────────────────
        cid5 = str(uuid.uuid4())
        c.execute("""INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (cid5,'Reinforcement Learning Mastery',
                   'From MDPs to Deep RL — learn how agents learn optimal decisions through trial and error. Build game-playing AI.',
                   'Reinforcement Learning','advanced',admin_id,'Prof. Educator','#EF4444'))
        rl_lectures = [
            (1,'MDPs & Foundations of RL','text',30,
             '''Reinforcement Learning is about agents learning optimal behavior through interaction with an environment.

## The RL Framework
```
Agent → Action → Environment → Reward + New State → Agent
```

## Markov Decision Process (MDP)
The mathematical foundation of RL:
- **S**: Set of states
- **A**: Set of actions
- **P(s'|s,a)**: Transition probability
- **R(s,a)**: Reward function
- **γ**: Discount factor (0-1)

## Key Concepts
**Policy (π)**: Strategy mapping states to actions
- π(a|s) = probability of taking action a in state s

**Value Function V(s)**: Expected cumulative reward from state s
- V(s) = E[R₁ + γR₂ + γ²R₃ + ...]

**Q Function Q(s,a)**: Expected reward for taking action a in state s

## Exploration vs Exploitation
The fundamental dilemma:
- **Exploit**: Use current best strategy
- **Explore**: Try new actions to find better strategies
- **ε-greedy**: Exploit (1-ε)%, Explore ε% of the time

## Types of RL
1. **Model-Based**: Learn the environment model, then plan
2. **Model-Free**: Learn directly from experience
   - Value-based (Q-Learning, DQN)
   - Policy-based (REINFORCE, PPO)
   - Actor-Critic (A2C, SAC)''',
             '''import numpy as np

class GridWorld:
    """Simple 4x4 grid environment"""
    def __init__(self):
        self.size = 4
        self.state = (0, 0)
        self.goal = (3, 3)
        self.actions = [(0,1),(0,-1),(1,0),(-1,0)]  # right,left,down,up

    def step(self, action):
        dx, dy = self.actions[action]
        nx = max(0, min(self.size-1, self.state[0]+dx))
        ny = max(0, min(self.size-1, self.state[1]+dy))
        self.state = (nx, ny)
        if self.state == self.goal:
            return self.state, 10.0, True
        return self.state, -0.1, False

    def reset(self):
        self.state = (0, 0)
        return self.state

# Value Iteration
V = np.zeros((4,4))
gamma = 0.9
for _ in range(100):
    for i in range(4):
        for j in range(4):
            if (i,j) == (3,3): continue
            values = []
            for a in range(4):
                ni = max(0,min(3,i+[0,0,1,-1][a]))
                nj = max(0,min(3,j+[1,-1,0,0][a]))
                r = 10.0 if (ni,nj)==(3,3) else -0.1
                values.append(r + gamma * V[ni,nj])
            V[i,j] = max(values)

print("Optimal Value Function:")
print(np.round(V, 2))'''),
            (2,'Q-Learning Algorithm','text',35,
             '''Q-Learning is a model-free, off-policy RL algorithm that learns the optimal Q-function directly.

## The Q-Learning Update Rule
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
```

Where:
- α: Learning rate (how fast to update)
- γ: Discount factor (how much to value future rewards)
- r: Immediate reward
- max_a' Q(s',a'): Best possible future value

## Algorithm Steps
1. Initialize Q-table with zeros
2. For each episode:
   a. Start in initial state
   b. Choose action (ε-greedy)
   c. Take action, observe reward and new state
   d. Update Q-value using the formula
   e. Move to new state
   f. Repeat until terminal state

## Key Properties
- **Off-policy**: Learns optimal Q regardless of current policy
- **Tabular**: Works with discrete state/action spaces
- **Convergence**: Guaranteed with sufficient exploration
- **Model-free**: No need to know environment dynamics

## Hyperparameters
| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| α (learning rate) | 0.01-0.5 | Speed vs stability |
| γ (discount) | 0.9-0.99 | Short vs long-term focus |
| ε (exploration) | 0.1-1.0 (decaying) | Explore vs exploit |''',
             '''import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Training loop
agent = QLearningAgent(n_states=16, n_actions=4)

for episode in range(500):
    state = 0  # start state
    total_reward = 0
    for step in range(50):
        action = agent.choose_action(state)
        # Simulate environment step
        next_state = min(15, max(0, state + [-4,-1,1,4][action]))
        reward = 10.0 if next_state == 15 else -0.1
        done = next_state == 15
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done: break
    agent.decay_epsilon()

print("Training complete!")
print("Q-table sample (states 0-3):")
print(np.round(agent.Q[:4], 2))'''),
            (3,'Deep Q-Networks (DQN)','text',40,
             '''DQN combines Q-Learning with deep neural networks to handle large/continuous state spaces.

## Why DQN?
- Q-tables can not handle large state spaces
- 1 Atari frame = 210×160×3 pixels = billions of possible states
- Use a neural network to approximate Q(s,a)

## Key Innovations

### 1. Experience Replay
- Store transitions (s, a, r, s') in a replay buffer
- Sample random mini-batches for training
- Breaks correlation between consecutive samples
- More data-efficient learning

### 2. Target Network
- Separate network for computing targets
- Updated periodically (every N steps)
- Stabilizes training significantly

### 3. Architecture
```
State (pixels/features) → CNN/MLP → Q-values for each action
```

## DQN Algorithm
```
1. Initialize replay buffer D, Q-network, target network Q'
2. For each episode:
   a. Get state s
   b. Choose action a (ε-greedy from Q)
   c. Execute a, observe r, s'
   d. Store (s,a,r,s') in D
   e. Sample mini-batch from D
   f. Compute target: y = r + γ·max Q'(s',a')
   g. Update Q by minimizing (Q(s,a) - y)²
   h. Periodically update Q' ← Q
```

## Advanced Variants
- **Double DQN**: Use Q to select action, Q' to evaluate → reduces overestimation
- **Dueling DQN**: Separate value and advantage streams
- **Rainbow**: Combines 6 improvements into one agent''',
             '''import numpy as np
from collections import deque
import random

class DQNAgent:
    """Simplified DQN Agent (without neural network for demo)"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # In real DQN: self.model = build_neural_network()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # In real DQN: return np.argmax(self.model.predict(state))
        return 0

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * 1.0  # simplified
            # In real DQN: self.model.fit(state, target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Demo
agent = DQNAgent(state_size=4, action_size=2)

# Simulate training
for episode in range(100):
    state = np.random.randn(4)
    for t in range(50):
        action = agent.act(state)
        next_state = np.random.randn(4)
        reward = float(np.random.rand() > 0.5)
        done = t == 49
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done: break
    agent.replay()

print(f"DQN Agent trained for 100 episodes")
print(f"Final epsilon: {agent.epsilon:.4f}")
print(f"Memory size: {len(agent.memory)}")'''),
        ]
        for order, title, ltype, dur, content, code in rl_lectures:
            lid = str(uuid.uuid4())
            c.execute("""INSERT INTO lectures (id,course_id,title,content,lecture_type,code_example,order_num,duration_mins)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (lid, cid5, title, content, ltype, code, order, dur))

        # ── Course 6: Data Science with Python ───────────────────────────────
        cid6 = str(uuid.uuid4())
        c.execute("""INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (cid6,'Data Science with Python',
                   'Learn data analysis from scratch with Pandas, visualization with Matplotlib & Seaborn, and statistical analysis for real-world datasets.',
                   'Data Science','beginner',admin_id,'Prof. Educator','#8B5CF6'))
        ds_lectures = [
            (1,'Pandas & Data Wrangling','text',25,
             '''Pandas is the backbone of data science in Python — a powerful library for data manipulation and analysis.

## Core Data Structures
- **Series**: 1D labeled array (like a column)
- **DataFrame**: 2D table (rows + columns)

## Essential Operations

### Reading Data
```python
df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx")
df = pd.read_json("data.json")
```

### Exploring Data
- `df.head()` — first 5 rows
- `df.info()` — column types + null counts
- `df.describe()` — summary statistics
- `df.shape` — (rows, columns)

### Selection
- `df["column"]` — single column
- `df[["col1","col2"]]` — multiple columns
- `df.loc[row, col]` — by label
- `df.iloc[row, col]` — by position

### Filtering
```python
df[df["age"] > 25]
df[(df["age"] > 25) & (df["city"] == "NYC")]
```

### Data Cleaning
- `df.dropna()` — remove rows with nulls
- `df.fillna(0)` — fill nulls with value
- `df.duplicated()` — find duplicates
- `df.drop_duplicates()` — remove duplicates

### Grouping & Aggregation
```python
df.groupby("category").mean()
df.groupby("region")["sales"].agg(["sum","mean","count"])
```''',
             '''import pandas as pd
import numpy as np

# Create sample sales dataset
np.random.seed(42)
n = 100
data = {
    "product": np.random.choice(["Laptop","Phone","Tablet","Watch"], n),
    "region": np.random.choice(["North","South","East","West"], n),
    "sales": np.random.randint(100, 5000, n),
    "quantity": np.random.randint(1, 20, n),
    "date": pd.date_range("2024-01-01", periods=n, freq="D")
}
df = pd.DataFrame(data)

# Data exploration
print("Shape:", df.shape)
print("\\nColumn types:\\n", df.dtypes)
print("\\nSummary Statistics:\\n", df.describe())

# Grouping
print("\\nSales by Product:")
print(df.groupby("product")["sales"].agg(["mean","sum","count"]).round(0))

# Filtering
high_sales = df[df["sales"] > 3000]
print(f"\\nHigh-value sales (>3000): {len(high_sales)} records")

# Time-based analysis
df["month"] = df["date"].dt.month_name()
monthly = df.groupby("month")["sales"].sum().sort_values(ascending=False)
print("\\nTop 3 months by sales:")
print(monthly.head(3))'''),
            (2,'Data Visualization with Matplotlib & Seaborn','text',30,
             '''Data Visualization transforms raw numbers into visual insights — critical for both exploration and communication.

## Matplotlib Basics
The foundational plotting library in Python.

### Common Plot Types
- **Line plot**: Trends over time
- **Bar chart**: Comparing categories
- **Scatter plot**: Relationships between variables
- **Histogram**: Distribution of values
- **Pie chart**: Proportions of a whole

### Anatomy of a Plot
```
Figure → Axes → Plot Elements (lines, bars, labels)
```

## Seaborn: Statistical Visualization
Built on top of Matplotlib with beautiful default styles.

### Key Seaborn Plots
- `sns.histplot()` — Distribution with KDE
- `sns.boxplot()` — Box-and-whisker
- `sns.heatmap()` — Correlation matrices
- `sns.pairplot()` — All variable pairs
- `sns.countplot()` — Category counts

## Visualization Best Practices
1. Choose the right chart type for your data
2. Label axes clearly with units
3. Use color purposefully (not decoratively)
4. Remove chart junk (unnecessary gridlines, borders)
5. Tell a story with your visualization
6. Use consistent color palettes
7. Consider your audience''',
             '''import matplotlib.pyplot as plt
import numpy as np

# Create sample data
categories = ["ML", "Deep Learning", "NLP", "Data Science", "Computer Vision"]
scores = [85, 72, 90, 78, 65]
colors = ["#10B981", "#7C3AED", "#059669", "#F59E0B", "#EC4899"]

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].barh(categories, scores, color=colors)
axes[0].set_xlabel("Progress (%)")
axes[0].set_title("Course Progress Overview")
axes[0].set_xlim(0, 100)

# Scatter plot
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 30)
exam_scores = 30 + 6 * study_hours + np.random.normal(0, 5, 30)
axes[1].scatter(study_hours, exam_scores, c="#10B981", alpha=0.7, s=60)
axes[1].set_xlabel("Study Hours")
axes[1].set_ylabel("Exam Score")
axes[1].set_title("Study Hours vs Exam Score")

# Add trend line
z = np.polyfit(study_hours, exam_scores, 1)
p = np.poly1d(z)
axes[1].plot(sorted(study_hours), p(sorted(study_hours)), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig("visualization_demo.png", dpi=150)
print("Chart saved as visualization_demo.png")
print(f"Correlation: {np.corrcoef(study_hours, exam_scores)[0,1]:.3f}")'''),
            (3,'Exploratory Data Analysis (EDA)','text',35,
             '''EDA is the detective work of data science — understanding your data before building models.

## The EDA Process
1. **Load & Inspect**: Shape, types, head/tail
2. **Summary Statistics**: Mean, median, std, quartiles
3. **Missing Values**: Identify patterns of missingness
4. **Distributions**: Histograms, KDE plots for each feature
5. **Correlations**: Heatmaps, pairplots
6. **Outliers**: Box plots, IQR method
7. **Feature Relationships**: Scatter plots, grouped analyses

## Key Questions to Ask
- What does each column represent?
- Are there missing values? How many? Pattern?
- What is the distribution of the target variable?
- Which features are correlated?
- Are there outliers? Are they real or errors?
- Do I need to transform any features?

## Common Patterns to Look For
- **Skewed distributions** → Consider log transform
- **High correlation** between features → Multicollinearity risk
- **Class imbalance** → May need SMOTE or weighted loss
- **Temporal patterns** → Time-series features needed

## Tools for EDA
- `pandas-profiling` — Automated EDA reports
- `sweetviz` — Beautiful comparison reports
- `dtale` — Interactive data exploration
- `missingno` — Missing data visualization''',
             '''import pandas as pd
import numpy as np

# Create a realistic dataset
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "age": np.random.randint(18, 65, n),
    "income": np.random.lognormal(10.5, 0.8, n).astype(int),
    "education_years": np.random.randint(8, 22, n),
    "experience": np.random.randint(0, 40, n),
    "department": np.random.choice(["Engineering","Marketing","Sales","HR","Finance"], n),
    "satisfaction": np.random.randint(1, 11, n),
})
# Add some missing values
df.loc[np.random.choice(n, 15, replace=False), "income"] = np.nan
df.loc[np.random.choice(n, 10, replace=False), "satisfaction"] = np.nan

print("=== EXPLORATORY DATA ANALYSIS ===\\n")
print(f"Shape: {df.shape}")
print(f"\\nColumn Types:\\n{df.dtypes}\\n")

# Missing values
print("Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Summary statistics
print(f"\\nSummary Statistics:\\n{df.describe().round(1)}\\n")

# Correlation matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
print("Correlation Matrix:")
print(corr.round(3))

# Outlier detection (IQR method)
for col in ["income", "age", "experience"]:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"\\n{col}: {len(outliers)} outliers detected")

# Group analysis
print("\\nAverage salary by department:")
print(df.groupby("department")["income"].mean().sort_values(ascending=False).round(0))'''),
        ]
        for order, title, ltype, dur, content, code in ds_lectures:
            lid = str(uuid.uuid4())
            c.execute("""INSERT INTO lectures (id,course_id,title,content,lecture_type,code_example,order_num,duration_mins)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (lid, cid6, title, content, ltype, code, order, dur))

        # ── Course 7: Generative AI & LLMs ──────────────────────────────────
        cid7 = str(uuid.uuid4())
        c.execute("""INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (cid7,'Generative AI & Large Language Models',
                   'Deep dive into transformer architecture, LLMs, prompt engineering, RAG, and fine-tuning. Build AI-powered applications.',
                   'Generative AI','advanced',admin_id,'Prof. Educator','#F59E0B'))
        gen_lectures = [
            (1,'Transformer Architecture','text',40,
             '''The Transformer is the architecture behind GPT, BERT, and all modern LLMs.

## The Attention Mechanism
"Attention Is All You Need" (Vaswani et al., 2017)

### Self-Attention
Each token looks at ALL other tokens to understand context:
```
"The cat sat on the mat"
For "sat": attends most to "cat" (who sat) and "mat" (where)
```

### Multi-Head Attention
Multiple attention heads capture different relationships:
- Head 1: Syntactic relationships
- Head 2: Semantic similarities
- Head 3: Positional relationships

## Transformer Architecture
```
Input → Embedding + Positional Encoding → N × [Multi-Head Attention → Feed-Forward] → Output
```

### Key Components
1. **Token Embedding**: Words → vectors (768 or 1024 dims)
2. **Positional Encoding**: Inject position information
3. **Layer Normalization**: Stabilize training
4. **Residual Connections**: Enable deep networks

## Encoder vs Decoder
- **Encoder-only**: BERT — bidirectional understanding
- **Decoder-only**: GPT — autoregressive generation
- **Encoder-Decoder**: T5 — sequence-to-sequence tasks

## Scaling Laws
```
Performance ∝ log(Parameters) × log(Data) × log(Compute)
```
- GPT-3: 175B params → emergent abilities
- GPT-4: ~1.7T params (estimated)
- Llama 3: 70B params, open-source''',
             '''import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """Core attention mechanism"""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights

# Demo: Simple self-attention
np.random.seed(42)
seq_len, d_model = 4, 8

# Simulate token embeddings for "The cat sat"
tokens = np.random.randn(seq_len, d_model)

# Query, Key, Value projections
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1

Q = tokens @ W_q
K = tokens @ W_k
V = tokens @ W_v

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Input shape:", tokens.shape)
print("Output shape:", output.shape)
print("\\nAttention weights (who attends to whom):")
print(np.round(attention_weights, 3))
print("\\nToken 3 attends most to token:", np.argmax(attention_weights[2]))'''),
            (2,'Prompt Engineering Mastery','text',30,
             '''Prompt Engineering is the art and science of designing effective instructions for LLMs.

## Prompt Anatomy
```
[System Message] + [Context/Examples] + [User Query] + [Output Format]
```

## Core Techniques

### 1. Zero-Shot Prompting
Direct instruction without examples:
"Classify the following review as positive or negative: ..."

### 2. Few-Shot Prompting
Provide examples before the task:
```
Review: "Great product!" → Positive
Review: "Terrible quality" → Negative
Review: "It works okay" → ???
```

### 3. Chain-of-Thought (CoT)
Ask the model to reason step by step:
"Let's think step by step..."
- Dramatically improves math & logic tasks
- 20-40% accuracy improvement on complex questions

### 4. Role Prompting
"You are an expert data scientist. Explain..."

### 5. Output Formatting
"Respond in JSON format with keys: answer, confidence, reasoning"

## Advanced Techniques
- **Self-Consistency**: Generate multiple CoT paths, take majority vote
- **Tree of Thoughts**: Explore multiple reasoning branches
- **Retrieval-Augmented Generation (RAG)**: Inject external knowledge
- **Constitutional AI**: Self-critique and revision

## Common Pitfalls
❌ Vague instructions
❌ Missing context
❌ No output format specification
❌ Asking too many things at once''',
             '''# Prompt Engineering Examples

# 1. Zero-Shot Classification
zero_shot = """Classify the following text into one of these categories: 
[Technology, Sports, Politics, Entertainment]

Text: "The new GPU architecture delivers 3x faster training for neural networks."
Category:"""

# 2. Few-Shot with Chain of Thought
few_shot_cot = """Solve the math problem step by step.

Question: If a train travels 120 km in 2 hours, what is its speed?
Reasoning: Speed = Distance / Time = 120 / 2 = 60 km/h
Answer: 60 km/h

Question: A store has a 25% discount on a $80 item. What is the final price?
Reasoning: Discount = 80 * 0.25 = $20. Final = 80 - 20 = $60
Answer: $60

Question: If 3 workers can build a wall in 12 hours, how long for 6 workers?
Reasoning:"""

# 3. Structured Output
structured = """Extract entities from the following text and return as JSON.

Text: "OpenAI released GPT-4 in March 2023, which can process both text and images."

Return format:
{
    "organizations": [],
    "products": [],
    "dates": [],
    "capabilities": []
}"""

# 4. RAG-style prompt
rag_prompt = """Use the following context to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context: {retrieved_documents}

Question: {user_question}

Answer:"""

print("=== Prompt Engineering Templates ===")
print("\\n1. Zero-Shot:", len(zero_shot), "chars")
print("2. Few-Shot CoT:", len(few_shot_cot), "chars")
print("3. Structured Output:", len(structured), "chars")
print("4. RAG Template:", len(rag_prompt), "chars")
print("\\nPrompt engineering is about clarity, context, and constraints.")'''),
            (3,'RAG: Retrieval-Augmented Generation','text',35,
             '''RAG combines the power of LLMs with external knowledge retrieval for accurate, up-to-date responses.

## Why RAG?
LLMs have knowledge cutoffs and can hallucinate. RAG solves this by:
1. Retrieving relevant documents from a knowledge base
2. Injecting them as context into the prompt
3. LLM generates answer grounded in real data

## RAG Architecture
```
User Query → Embedding → Vector Search → Top-K Documents → LLM Prompt → Answer
```

### Step 1: Indexing (Offline)
- Split documents into chunks (500-1000 tokens)
- Generate embeddings for each chunk
- Store in vector database (Pinecone, Weaviate, ChromaDB)

### Step 2: Retrieval (Online)
- Convert user query to embedding
- Find most similar chunks via cosine similarity
- Return top-K relevant chunks

### Step 3: Generation
- Insert retrieved chunks into prompt context
- LLM generates answer referencing the context

## Vector Databases
| Database | Type | Best For |
|----------|------|----------|
| ChromaDB | Open-source | Prototyping |
| Pinecone | Managed | Production |
| Weaviate | Open-source | Hybrid search |
| FAISS | Library | Speed |

## Chunking Strategies
- Fixed-size chunks (simple but may break context)
- Sentence-based (preserves meaning)
- Recursive splitting (respects document structure)
- Semantic chunking (AI-based boundaries)

## Evaluation Metrics
- **Faithfulness**: Is the answer grounded in retrieved docs?
- **Relevance**: Are retrieved docs relevant to the query?
- **Answer correctness**: Is the final answer correct?''',
             '''import numpy as np

class SimpleRAG:
    """Simplified RAG system for demonstration"""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_document(self, text, metadata=None):
        # In production: use OpenAI/Sentence-Transformers embeddings
        embedding = self._simple_embed(text)
        self.documents.append({"text": text, "metadata": metadata or {}})
        self.embeddings.append(embedding)

    def _simple_embed(self, text):
        """Simple TF-based embedding (demo only)"""
        words = text.lower().split()
        vocab = list(set(words))
        vec = np.zeros(100)
        for i, w in enumerate(words[:100]):
            vec[hash(w) % 100] += 1
        return vec / (np.linalg.norm(vec) + 1e-8)

    def retrieve(self, query, top_k=3):
        query_emb = self._simple_embed(query)
        scores = []
        for emb in self.embeddings:
            score = np.dot(query_emb, emb)
            scores.append(score)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i]["text"], scores[i]) for i in top_indices]

    def generate(self, query):
        retrieved = self.retrieve(query)
        context = "\\n".join([f"[Doc {i+1}]: {doc}" for i, (doc, _) in enumerate(retrieved)])
        prompt = f"Context:\\n{context}\\n\\nQuestion: {query}\\nAnswer:"
        return prompt, retrieved

# Demo
rag = SimpleRAG()
rag.add_document("Machine learning uses algorithms to learn patterns from data.")
rag.add_document("Deep learning is a subset of ML using neural networks with many layers.")
rag.add_document("Natural language processing enables computers to understand human language.")
rag.add_document("Reinforcement learning trains agents through reward signals.")
rag.add_document("Computer vision allows machines to interpret visual information.")

prompt, docs = rag.generate("What is deep learning?")
print("Retrieved Documents:")
for doc, score in docs:
    print(f"  Score: {score:.3f} | {doc[:60]}...")
print(f"\\nGenerated Prompt:\\n{prompt}")'''),
        ]
        for order, title, ltype, dur, content, code in gen_lectures:
            lid = str(uuid.uuid4())
            c.execute("""INSERT INTO lectures (id,course_id,title,content,lecture_type,code_example,order_num,duration_mins)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (lid, cid7, title, content, ltype, code, order, dur))

        # ── Course 8: MLOps & Model Deployment ──────────────────────────────
        cid8 = str(uuid.uuid4())
        c.execute("""INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (cid8,'MLOps & Model Deployment',
                   'Learn to deploy, monitor, and maintain ML models in production. Docker, CI/CD, model serving, and A/B testing.',
                   'MLOps','intermediate',admin_id,'Prof. Educator','#06B6D4'))
        mlops_lectures = [
            (1,'Model Serving with Flask & FastAPI','text',30,
             '''Deploying an ML model means making it accessible via an API so applications can get predictions.

## Serving Options
1. **REST API** (Flask/FastAPI) — Most common, simple
2. **gRPC** — High performance, binary protocol
3. **Batch Inference** — Process large datasets offline
4. **Edge Deployment** — Run on devices (TFLite, ONNX)

## Flask vs FastAPI
| Feature | Flask | FastAPI |
|---------|-------|---------|
| Speed | Standard | 3x faster |
| Async | Manual | Built-in |
| Docs | Manual | Auto-generated |
| Validation | Manual | Pydantic |

## API Design Best Practices
- Use POST for prediction endpoints
- Accept JSON input, return JSON output
- Include model version in response
- Add health check endpoint
- Handle errors gracefully
- Use input validation

## Model Loading Strategies
- **On startup**: Load once, reuse → most common
- **Lazy loading**: Load on first request
- **Model registry**: Pull latest version from registry

## Production Considerations
- Logging & monitoring
- Rate limiting
- Authentication
- Input validation
- Batch prediction support
- A/B testing support''',
             '''from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Simulated model (in production: load from pickle/joblib/ONNX)
class SimpleModel:
    def predict(self, features):
        return [sum(f) * 0.1 for f in features]

    def version(self):
        return "1.0.0"

model = SimpleModel()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_version": model.version()})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = data.get("features", [])
        if not features:
            return jsonify({"error": "No features provided"}), 400

        predictions = model.predict(features)
        return jsonify({
            "predictions": predictions,
            "model_version": model.version(),
            "count": len(predictions)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.json
    features = data.get("features", [])
    predictions = model.predict(features)
    return jsonify({
        "predictions": predictions,
        "batch_size": len(features)
    })

# FastAPI equivalent (shown as comment):
# from fastapi import FastAPI
# from pydantic import BaseModel
# app = FastAPI()
# class PredictionRequest(BaseModel):
#     features: list[list[float]]
# @app.post("/predict")
# async def predict(req: PredictionRequest):
#     return {"predictions": model.predict(req.features)}

if __name__ == "__main__":
    print("ML API Server starting...")
    print(f"Model version: {model.version()}")
    print("Endpoints: /health, /predict, /predict/batch")
    # app.run(host="0.0.0.0", port=5000)'''),
            (2,'Docker for ML Applications','text',25,
             '''Docker packages your ML application with all dependencies into a portable container.

## Why Docker for ML?
- **Reproducibility**: Same environment everywhere
- **Dependency isolation**: No "it works on my machine"
- **Easy deployment**: Ship container to any cloud
- **Scaling**: Run multiple instances easily

## Dockerfile Anatomy
```dockerfile
FROM python:3.10-slim          # Base image
WORKDIR /app                   # Set working directory
COPY requirements.txt .        # Copy dependency list
RUN pip install -r requirements.txt  # Install dependencies
COPY . .                       # Copy application code
COPY model.pkl .               # Copy trained model
EXPOSE 8000                    # Declare port
CMD ["python", "serve.py"]     # Start command
```

## Docker Commands
```bash
docker build -t my-ml-app .           # Build image
docker run -p 8000:8000 my-ml-app     # Run container
docker push registry/my-ml-app:v1     # Push to registry
```

## Multi-Stage Builds for ML
```dockerfile
# Stage 1: Training
FROM python:3.10 AS trainer
COPY train.py requirements-train.txt ./
RUN pip install -r requirements-train.txt
RUN python train.py  # outputs model.pkl

# Stage 2: Serving (smaller image)
FROM python:3.10-slim
COPY --from=trainer /app/model.pkl .
COPY serve.py requirements-serve.txt ./
RUN pip install -r requirements-serve.txt
CMD ["python", "serve.py"]
```

## Best Practices
- Use slim base images (python:3.10-slim)
- Multi-stage builds to reduce image size
- Cache pip install layer separately
- Don't include training data in production image
- Use .dockerignore file
- Pin dependency versions''',
             '''# Dockerfile Generator for ML Projects

def generate_dockerfile(project_type="api", python_version="3.10",
                       port=8000, use_gpu=False):
    """Generate a Dockerfile for ML projects"""

    base = f"python:{python_version}-slim"
    if use_gpu:
        base = f"nvidia/cuda:11.8.0-runtime-ubuntu22.04"

    lines = [
        f"FROM {base}",
        "",
        "# Set working directory",
        "WORKDIR /app",
        "",
        "# Install system dependencies",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\\",
        "    build-essential && rm -rf /var/lib/apt/lists/*",
        "",
        "# Install Python dependencies (cached layer)",
        "COPY requirements.txt .",
        "RUN pip install --no-cache-dir -r requirements.txt",
        "",
        "# Copy application code",
        "COPY . .",
        "",
    ]

    if project_type == "api":
        lines.extend([
            f"EXPOSE {port}",
            "",
            "# Health check",
            f\'HEALTHCHECK CMD curl -f http://localhost:{port}/health || exit 1\',
            "",
            f\'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]\',
        ])
    elif project_type == "batch":
        lines.extend([
            \'CMD ["python", "batch_predict.py"]\',
        ])

    return "\\n".join(lines)

# Generate Dockerfile
dockerfile = generate_dockerfile(project_type="api", port=8000)
print("=== Generated Dockerfile ===")
print(dockerfile)
print("\\n=== Docker Commands ===")
print("docker build -t ml-api .")
print("docker run -p 8000:8000 ml-api")
print("docker push myregistry/ml-api:v1")'''),
            (3,'ML Pipeline CI/CD','text',30,
             '''CI/CD for ML automates the process of testing, training, and deploying models whenever code or data changes.

## ML CI/CD Pipeline
```
Code Push → Tests → Train → Evaluate → Register → Deploy → Monitor
```

## Key Differences from Software CI/CD
| Software CI/CD | ML CI/CD |
|---------------|----------|
| Test code | Test code + data + model |
| Build artifact | Build model artifact |
| Deploy code | Deploy model + serving code |
| Unit tests | Unit tests + model quality tests |

## Pipeline Components

### 1. Data Validation
- Check data schema (columns, types)
- Detect data drift
- Validate data quality (nulls, outliers)

### 2. Model Training
- Reproducible training with versioned data
- Hyperparameter tracking (MLflow, W&B)
- GPU resource management

### 3. Model Evaluation
- Performance metrics (accuracy, F1, AUC)
- Compare against baseline/production model
- Bias and fairness checks
- Latency benchmarks

### 4. Model Registry
- Version control for models
- Stage management (staging → production)
- Metadata: metrics, training data version, params

### 5. Deployment
- Canary deployment (route 5% traffic to new model)
- A/B testing (compare two models)
- Shadow mode (run new model alongside, don't serve)

## Tools
- **MLflow**: Experiment tracking + model registry
- **DVC**: Data version control
- **GitHub Actions**: CI/CD automation
- **Kubeflow**: ML pipelines on Kubernetes
- **Weights & Biases**: Experiment tracking
- **Seldon/BentoML**: Model serving''',
             '''import json
from datetime import datetime

class MLPipeline:
    """Simplified ML CI/CD Pipeline"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.stages = []
        self.metrics = {}

    def validate_data(self, data_stats):
        """Stage 1: Data validation"""
        checks = {
            "row_count": data_stats.get("rows", 0) > 100,
            "no_nulls": data_stats.get("null_pct", 100) < 5,
            "schema_match": data_stats.get("schema_valid", False),
        }
        passed = all(checks.values())
        self.stages.append(("data_validation", passed, checks))
        return passed

    def train_model(self, params):
        """Stage 2: Model training"""
        # Simulated training
        accuracy = 0.85 + (params.get("epochs", 10) * 0.005)
        self.metrics = {"accuracy": min(accuracy, 0.98), "f1": accuracy - 0.02}
        self.stages.append(("training", True, params))
        return self.metrics

    def evaluate(self, baseline_accuracy=0.80):
        """Stage 3: Model evaluation"""
        improved = self.metrics.get("accuracy", 0) > baseline_accuracy
        self.stages.append(("evaluation", improved, {
            "current": self.metrics.get("accuracy", 0),
            "baseline": baseline_accuracy,
            "improved": improved
        }))
        return improved

    def deploy(self, strategy="canary"):
        """Stage 4: Deploy"""
        self.stages.append(("deployment", True, {"strategy": strategy}))
        return {"status": "deployed", "strategy": strategy}

    def report(self):
        print(f"\\n{'='*50}")
        print(f"ML Pipeline Report: {self.model_name}")
        print(f"{'='*50}")
        for stage, passed, details in self.stages:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\\n{status} | {stage}")
            for k, v in details.items():
                print(f"       {k}: {v}")

# Run pipeline
pipeline = MLPipeline("sentiment-classifier-v2")

# Stage 1
pipeline.validate_data({"rows": 5000, "null_pct": 1.2, "schema_valid": True})

# Stage 2
pipeline.train_model({"epochs": 20, "lr": 0.001, "batch_size": 32})

# Stage 3
pipeline.evaluate(baseline_accuracy=0.82)

# Stage 4
pipeline.deploy(strategy="canary")

pipeline.report()'''),
        ]
        for order, title, ltype, dur, content, code in mlops_lectures:
            lid = str(uuid.uuid4())
            c.execute("""INSERT INTO lectures (id,course_id,title,content,lecture_type,code_example,order_num,duration_mins)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (lid, cid8, title, content, ltype, code, order, dur))

    except Exception:
        pass

    try:
        from events_catalog import init_events_db
        init_events_db(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

def educator_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        if session.get('role') != 'educator':
            return redirect(url_for('student_dashboard'))
        return f(*args, **kwargs)
    return decorated

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('student_dashboard') if session.get('role')=='student' else url_for('educator_dashboard'))
    return render_template('index.html')

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('student_dashboard') if session.get('role')=='student' else url_for('educator_dashboard'))
    return render_template('login.html')

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json or {}
    conn = get_db()
    login_id = (data.get('email') or data.get('username') or '').strip()
    pw = data.get('password', '')
    user = conn.execute("SELECT * FROM users WHERE (email=? OR username=?) AND password=?",
                        (login_id, login_id, hash_password(pw))).fetchone()
    conn.close()
    if user:
        session['user_id']=user['id']; session['username']=user['username']
        session['role']=user['role']; session['email']=user['email']
        try:
            session['college']=user['college']
        except (IndexError, KeyError):
            session['college']=''
        try:
            session['region']=user['region']
        except (IndexError, KeyError):
            session['region']=''
        return jsonify({'success':True,'role':user['role'],'username':user['username']})
    return jsonify({'success':False,'error':'Invalid email or password'})



@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    conn = get_db()
    try:
        uid = str(uuid.uuid4())
        conn.execute("INSERT INTO users (id,username,email,password,role,college,region) VALUES (?,?,?,?,?,?,?)",
                     (uid, data['username'], data['email'], hash_password(data['password']), data.get('role','student'), data.get('college',''), data.get('region','')))
        conn.commit(); conn.close()
        return jsonify({'success':True})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'success':False,'error':'Email or username already exists'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/profile')
@app.route('/student/profile')
@login_required
def user_profile():
    return render_template('student_profile.html')

@app.route('/api/user/profile', methods=['GET', 'POST', 'PUT'])
@login_required
def api_user_profile():
    conn = get_db()
    uid = session.get('user_id')
    if not uid:
        conn.close()
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    if request.method in ['POST', 'PUT']:
        data = request.json or {}
        new_username = (data.get('username') or session.get('username', '')).strip()
        headline = (data.get('headline') or '').strip()
        college = (data.get('college') or '').strip()
        region = (data.get('region') or '').strip()
        level = (data.get('level') or 'beginner').strip().lower()
        interests = (data.get('interests') or '').strip()
        bio = (data.get('bio') or '').strip()

        if not new_username:
            conn.close()
            return jsonify({'success': False, 'error': 'Username cannot be empty'}), 400

        existing = conn.execute("SELECT id FROM users WHERE username=? AND id!=?", (new_username, uid)).fetchone()
        if existing:
            conn.close()
            return jsonify({'success': False, 'error': 'Username is already taken by another account'}), 400

        conn.execute("""
            UPDATE users 
            SET username=?, headline=?, college=?, region=?, level=?, interests=?, bio=?
            WHERE id=?
        """, (new_username, headline, college, region, level, interests, bio, uid))
        conn.commit()
        session['username'] = new_username
        session['level'] = level

    user = conn.execute("""
        SELECT id, username, email, role, level, college, region, bio, headline, interests, created_at, total_queries, correct_attempts
        FROM users WHERE id=?
    """, (uid,)).fetchone()

    enrolled_count = conn.execute("SELECT COUNT(*) FROM enrollments WHERE student_id=?", (uid,)).fetchone()[0]
    completed_lecs = conn.execute("SELECT COUNT(DISTINCT lecture_id) FROM completed_lectures WHERE student_id=?", (uid,)).fetchone()[0]
    certs = conn.execute("""
        SELECT id, course_id, course_title, cert_code, issued_at 
        FROM certificates 
        WHERE student_id=? 
        ORDER BY issued_at DESC
    """, (uid,)).fetchall()

    conn.close()

    return jsonify({
        'success': True,
        'profile': dict(user) if user else {},
        'stats': {
            'enrolled_courses': enrolled_count,
            'completed_lectures': completed_lecs,
            'certificates_count': len(certs)
        },
        'certificates': [dict(c) for c in certs]
    })

@app.route('/settings')
@login_required
def user_settings_page():
    return render_template('settings.html')

@app.route('/api/user/settings', methods=['GET', 'POST'])
@login_required
def api_user_settings():
    conn = get_db()
    uid = session.get('user_id')
    if not uid:
        conn.close()
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    if request.method == 'POST':
        data = request.json or {}
        action = data.get('action')

        if action == 'change_password':
            curr_pw = data.get('current_password', '')
            new_pw = data.get('new_password', '')
            if not curr_pw or not new_pw:
                conn.close()
                return jsonify({'success': False, 'error': 'Current and new password are required'}), 400
            if len(new_pw) < 6:
                conn.close()
                return jsonify({'success': False, 'error': 'New password must be at least 6 characters long'}), 400

            user = conn.execute("SELECT password FROM users WHERE id=?", (uid,)).fetchone()
            if not user or user['password'] != hash_password(curr_pw):
                conn.close()
                return jsonify({'success': False, 'error': 'Current password is incorrect'}), 400

            conn.execute("UPDATE users SET password=? WHERE id=?", (hash_password(new_pw), uid))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'message': 'Password updated successfully'})

        elif action == 'update_preferences':
            prefs = {
                'daily_goal_mins': int(data.get('daily_goal_mins', 30)),
                'ai_tutor_style': str(data.get('ai_tutor_style', 'balanced')),
                'study_reminders': bool(data.get('study_reminders', True)),
                'email_updates': bool(data.get('email_updates', True)),
                'preferred_theme': str(data.get('preferred_theme', 'day'))
            }
            conn.execute("UPDATE users SET preferences=? WHERE id=?", (json.dumps(prefs), uid))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'message': 'Preferences saved successfully', 'preferences': prefs})

    # GET request: fetch current settings
    user = conn.execute("SELECT username, email, role, preferences FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()

    prefs = {}
    if user and user['preferences']:
        try:
            prefs = json.loads(user['preferences'])
        except Exception:
            prefs = {}

    default_prefs = {
        'daily_goal_mins': 30,
        'ai_tutor_style': 'balanced',
        'study_reminders': True,
        'email_updates': True,
        'preferred_theme': 'day'
    }
    default_prefs.update(prefs)

    return jsonify({
        'success': True,
        'email': user['email'] if user else '',
        'username': user['username'] if user else '',
        'role': user['role'] if user else 'student',
        'preferences': default_prefs
    })

@app.route('/api/user/export_data')
@login_required
def api_user_export_data():
    conn = get_db()
    uid = session.get('user_id')
    user = conn.execute("SELECT id, username, email, role, level, college, region, bio, headline, interests, created_at FROM users WHERE id=?", (uid,)).fetchone()
    enrollments = conn.execute("SELECT e.*, c.title as course_title FROM enrollments e JOIN courses c ON e.course_id = c.id WHERE e.student_id=?", (uid,)).fetchall()
    completed = conn.execute("SELECT * FROM completed_lectures WHERE student_id=?", (uid,)).fetchall()
    certs = conn.execute("SELECT * FROM certificates WHERE student_id=?", (uid,)).fetchall()
    battles = conn.execute("SELECT * FROM battle_scores WHERE user_id=?", (uid,)).fetchall()
    conn.close()

    export_obj = {
        'platform': 'LearnSphere',
        'exported_at': datetime.now().isoformat(),
        'profile': dict(user) if user else {},
        'enrolled_courses': [dict(e) for e in enrollments],
        'completed_lectures': [dict(cl) for cl in completed],
        'certificates': [dict(c) for c in certs],
        'battle_history': [dict(b) for b in battles]
    }
    resp = make_response(json.dumps(export_obj, indent=2))
    resp.headers['Content-Type'] = 'application/json'
    resp.headers['Content-Disposition'] = f'attachment; filename=learnsphere_data_{session.get("username", "user")}.json'
    return resp

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    if session.get('role')=='educator': return redirect(url_for('educator_dashboard'))
    return render_template('student_dashboard.html')

@app.route('/student/courses')
@login_required
def student_courses():
    return render_template('student_courses.html')

@app.route('/student/course/<course_id>')
@login_required
def student_course_view(course_id):
    return render_template('student_course_view.html', course_id=course_id)

@app.route('/student/lecture/<lecture_id>')
@login_required
def student_lecture(lecture_id):
    return render_template('student_lecture.html', lecture_id=lecture_id)

@app.route('/learn')
@login_required
def learn():
    return render_template('learn.html')

@app.route('/code')
@login_required
def code():
    return render_template('code.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/educator/dashboard')
@educator_required
def educator_dashboard():
    return render_template('educator_dashboard.html')

@app.route('/educator/courses')
@educator_required
def educator_courses():
    return render_template('educator_courses.html')

@app.route('/educator/course/new')
@educator_required
def educator_new_course():
    return render_template('educator_course_form.html', course_id=None)

@app.route('/educator/course/<course_id>/edit')
@educator_required
def educator_edit_course(course_id):
    return render_template('educator_course_form.html', course_id=course_id)

@app.route('/educator/course/<course_id>/lectures')
@educator_required
def educator_lectures(course_id):
    return render_template('educator_lectures.html', course_id=course_id)

@app.route('/educator/lecture/new/<course_id>')
@educator_required
def educator_new_lecture(course_id):
    return render_template('educator_lecture_form.html', course_id=course_id, lecture_id=None)

@app.route('/educator/lecture/<lecture_id>/edit')
@educator_required
def educator_edit_lecture(lecture_id):
    return render_template('educator_lecture_form.html', course_id=None, lecture_id=lecture_id)

# ─── COURSE API ───────────────────────────────────────────────────────────────
@app.route('/api/courses', methods=['GET'])
@login_required
def get_courses():
    conn = get_db()
    courses = conn.execute("SELECT * FROM courses WHERE is_published=1 ORDER BY created_at DESC").fetchall()
    result = []
    for cr in courses:
        lc = conn.execute("SELECT COUNT(*) FROM lectures WHERE course_id=?",(cr['id'],)).fetchone()[0]
        enrolled = conn.execute("SELECT 1 FROM enrollments WHERE student_id=? AND course_id=?",(session['user_id'],cr['id'])).fetchone()
        ec = conn.execute("SELECT COUNT(*) FROM enrollments WHERE course_id=?",(cr['id'],)).fetchone()[0]
        # Ratings
        rating_rows = conn.execute("SELECT rating FROM course_ratings WHERE course_id=?",(cr['id'],)).fetchall()
        avg_rating = round(sum(r['rating'] for r in rating_rows)/len(rating_rows),1) if rating_rows else 0
        # Progress for enrolled student
        progress = 0
        if enrolled and lc > 0:
            done = conn.execute(
                "SELECT COUNT(*) FROM completed_lectures cl JOIN lectures l ON cl.lecture_id=l.id WHERE cl.student_id=? AND l.course_id=?",
                (session['user_id'],cr['id'])
            ).fetchone()[0]
            progress = round(done/lc*100)
        result.append({**dict(cr),'lecture_count':lc,'enrolled':enrolled is not None,
                       'enrollment_count':ec,'avg_rating':avg_rating,'rating_count':len(rating_rows),'progress':progress})
    conn.close()
    return jsonify(result)

@app.route('/api/educator/courses', methods=['GET'])
@educator_required
def get_educator_courses():
    conn = get_db()
    courses = conn.execute("SELECT * FROM courses WHERE educator_id=? ORDER BY created_at DESC",(session['user_id'],)).fetchall()
    result = []
    for cr in courses:
        lc = conn.execute("SELECT COUNT(*) FROM lectures WHERE course_id=?",(cr['id'],)).fetchone()[0]
        ec = conn.execute("SELECT COUNT(*) FROM enrollments WHERE course_id=?",(cr['id'],)).fetchone()[0]
        result.append({**dict(cr),'lecture_count':lc,'enrollment_count':ec})
    conn.close()
    return jsonify(result)

@app.route('/api/courses', methods=['POST'])
@educator_required
def create_course():
    data = request.json
    conn = get_db()
    cid = str(uuid.uuid4())
    conn.execute("INSERT INTO courses (id,title,description,category,difficulty,educator_id,educator_name,thumbnail_color) VALUES (?,?,?,?,?,?,?,?)",
                 (cid,data['title'],data.get('description',''),data.get('category','ML'),data.get('difficulty','beginner'),
                  session['user_id'],session['username'],data.get('thumbnail_color','#10B981')))
    conn.commit(); conn.close()
    return jsonify({'success':True,'id':cid})

@app.route('/api/courses/<course_id>', methods=['GET'])
@login_required
def get_course(course_id):
    conn = get_db()
    course = conn.execute("SELECT * FROM courses WHERE id=?",(course_id,)).fetchone()
    lectures = conn.execute("SELECT * FROM lectures WHERE course_id=? ORDER BY order_num",(course_id,)).fetchall()
    if not course: return jsonify({'error':'Not found'}),404
    completed = []
    enrolled = False
    if session.get('role')=='student':
        rows = conn.execute("SELECT lecture_id FROM completed_lectures WHERE student_id=?",(session['user_id'],)).fetchall()
        completed = [r['lecture_id'] for r in rows]
        is_enrolled = conn.execute("SELECT 1 FROM enrollments WHERE student_id=? AND course_id=?",(session['user_id'],course_id)).fetchone()
        if is_enrolled: enrolled = True
    conn.close()
    return jsonify({'course':dict(course),'lectures':[dict(l) for l in lectures],'completed':completed,'enrolled':enrolled})

@app.route('/api/courses/<course_id>', methods=['PUT'])
@educator_required
def update_course(course_id):
    data = request.json
    conn = get_db()
    conn.execute("UPDATE courses SET title=?,description=?,category=?,difficulty=?,thumbnail_color=? WHERE id=? AND educator_id=?",
                 (data['title'],data.get('description',''),data.get('category','ML'),data.get('difficulty','beginner'),
                  data.get('thumbnail_color','#10B981'),course_id,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'success':True})

@app.route('/api/courses/<course_id>', methods=['DELETE'])
@educator_required
def delete_course(course_id):
    conn = get_db()
    conn.execute("DELETE FROM lectures WHERE course_id=?",(course_id,))
    conn.execute("DELETE FROM courses WHERE id=? AND educator_id=?",(course_id,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'success':True})

# ─── LECTURE API ──────────────────────────────────────────────────────────────
@app.route('/api/lectures/<lecture_id>', methods=['GET'])
@login_required
def get_lecture(lecture_id):
    conn = get_db()
    lecture = conn.execute("SELECT * FROM lectures WHERE id=?",(lecture_id,)).fetchone()
    if not lecture: return jsonify({'error':'Not found'}),404
    completed = conn.execute("SELECT 1 FROM completed_lectures WHERE student_id=? AND lecture_id=?",(session['user_id'],lecture_id)).fetchone()
    note = conn.execute("SELECT content FROM notes WHERE student_id=? AND lecture_id=? ORDER BY created_at DESC LIMIT 1",(session['user_id'],lecture_id)).fetchone()
    all_lecs = conn.execute("SELECT id,title FROM lectures WHERE course_id=? ORDER BY order_num",(lecture['course_id'],)).fetchall()
    ids = [l['id'] for l in all_lecs]
    idx = ids.index(lecture_id) if lecture_id in ids else 0
    conn.close()
    return jsonify({'lecture':dict(lecture),'completed':completed is not None,'note':note['content'] if note else '',
                    'prev_id':ids[idx-1] if idx>0 else None,'next_id':ids[idx+1] if idx<len(ids)-1 else None,
                    'all_lectures':[dict(l) for l in all_lecs]})

@app.route('/api/lectures', methods=['POST'])
@educator_required
def create_lecture():
    data = request.json
    conn = get_db()
    lid = str(uuid.uuid4())
    max_order = conn.execute("SELECT MAX(order_num) FROM lectures WHERE course_id=?",(data['course_id'],)).fetchone()[0] or 0
    conn.execute("INSERT INTO lectures (id,course_id,title,content,lecture_type,video_url,code_example,order_num,duration_mins) VALUES (?,?,?,?,?,?,?,?,?)",
                 (lid,data['course_id'],data['title'],data.get('content',''),data.get('lecture_type','text'),
                  data.get('video_url',''),data.get('code_example',''),max_order+1,data.get('duration_mins',15)))
    conn.commit(); conn.close()
    return jsonify({'success':True,'id':lid})

@app.route('/api/lectures/<lecture_id>', methods=['PUT'])
@educator_required
def update_lecture(lecture_id):
    data = request.json
    conn = get_db()
    conn.execute("UPDATE lectures SET title=?,content=?,lecture_type=?,video_url=?,code_example=?,duration_mins=? WHERE id=?",
                 (data['title'],data.get('content',''),data.get('lecture_type','text'),
                  data.get('video_url',''),data.get('code_example',''),data.get('duration_mins',15),lecture_id))
    conn.commit(); conn.close()
    return jsonify({'success':True})

@app.route('/api/lectures/<lecture_id>', methods=['DELETE'])
@educator_required
def delete_lecture(lecture_id):
    conn = get_db()
    conn.execute("DELETE FROM lectures WHERE id=?",(lecture_id,))
    conn.commit(); conn.close()
    return jsonify({'success':True})

# ─── ENROLLMENT & PROGRESS ───────────────────────────────────────────────────
@app.route('/api/enroll/<course_id>', methods=['POST'])
@login_required
def enroll(course_id):
    conn = get_db()
    try:
        conn.execute("INSERT INTO enrollments (id,student_id,course_id) VALUES (?,?,?)",
                     (str(uuid.uuid4()),session['user_id'],course_id))
        conn.commit()
    except sqlite3.IntegrityError: pass
    conn.close()
    return jsonify({'success':True})

@app.route('/api/complete_lecture/<lecture_id>', methods=['POST'])
@login_required
def complete_lecture(lecture_id):
    conn = get_db()
    conn.execute("INSERT OR IGNORE INTO completed_lectures (id,student_id,lecture_id) VALUES (?,?,?)",
                 (str(uuid.uuid4()), session['user_id'], lecture_id))
    lec = conn.execute("SELECT course_id FROM lectures WHERE id=?", (lecture_id,)).fetchone()
    progress_pct = 0
    course_completed = False
    if lec:
        cid = lec['course_id']
        total = conn.execute("SELECT COUNT(*) FROM lectures WHERE course_id=?", (cid,)).fetchone()[0]
        done = conn.execute("""SELECT COUNT(DISTINCT cl.lecture_id) FROM completed_lectures cl 
                               JOIN lectures l ON cl.lecture_id=l.id
                               WHERE cl.student_id=? AND l.course_id=?""", (session['user_id'], cid)).fetchone()[0]
        progress_pct = int(round(done / total * 100)) if total else 0
        
        # Ensure enrollment exists and update progress
        enrolled = conn.execute("SELECT id FROM enrollments WHERE student_id=? AND course_id=?", (session['user_id'], cid)).fetchone()
        if not enrolled:
            conn.execute("INSERT INTO enrollments (id, student_id, course_id, progress) VALUES (?,?,?,?)",
                         (str(uuid.uuid4()), session['user_id'], cid, progress_pct))
        else:
            conn.execute("UPDATE enrollments SET progress=? WHERE student_id=? AND course_id=?",
                         (progress_pct, session['user_id'], cid))
            
        # Check course completion
        if done >= total and total > 0:
            course_completed = True
            cert = conn.execute("SELECT id FROM certificates WHERE student_id=? AND course_id=?", (session['user_id'], cid)).fetchone()
            if not cert:
                user = conn.execute("SELECT username FROM users WHERE id=?", (session['user_id'],)).fetchone()
                course = conn.execute("SELECT title FROM courses WHERE id=?", (cid,)).fetchone()
                uname = user['username'] if user else 'Student'
                ctitle = course['title'] if course else 'Course'
                code = 'CERT-' + uuid.uuid4().hex[:8].upper()
                conn.execute("""INSERT INTO certificates (id, student_id, course_id, student_name, course_title, cert_code)
                                VALUES (?,?,?,?,?,?)""", (str(uuid.uuid4()), session['user_id'], cid, uname, ctitle, code))
                conn.execute("""INSERT INTO career_entries (id, user_id, title, entry_type, description)
                                VALUES (?,?,?,?,?)""",
                             (str(uuid.uuid4()), session['user_id'], f"Completed Course: {ctitle}",
                              "certificate", f"Successfully completed all lectures and curriculum in {ctitle}."))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'progress': progress_pct, 'course_completed': course_completed})

@app.route('/api/toggle_lecture/<lecture_id>', methods=['POST'])
@login_required
def toggle_lecture(lecture_id):
    conn = get_db()
    existing = conn.execute("SELECT id FROM completed_lectures WHERE student_id=? AND lecture_id=?",
                            (session['user_id'], lecture_id)).fetchone()
    if existing:
        conn.execute("DELETE FROM completed_lectures WHERE id=?", (existing['id'],))
        is_done = False
    else:
        conn.execute("INSERT OR IGNORE INTO completed_lectures (id,student_id,lecture_id) VALUES (?,?,?)",
                     (str(uuid.uuid4()), session['user_id'], lecture_id))
        is_done = True
        
    lec = conn.execute("SELECT course_id FROM lectures WHERE id=?", (lecture_id,)).fetchone()
    progress_pct = 0
    course_completed = False
    if lec:
        cid = lec['course_id']
        total = conn.execute("SELECT COUNT(*) FROM lectures WHERE course_id=?", (cid,)).fetchone()[0]
        done = conn.execute("""SELECT COUNT(DISTINCT cl.lecture_id) FROM completed_lectures cl 
                               JOIN lectures l ON cl.lecture_id=l.id
                               WHERE cl.student_id=? AND l.course_id=?""", (session['user_id'], cid)).fetchone()[0]
        progress_pct = int(round(done / total * 100)) if total else 0
        
        enrolled = conn.execute("SELECT id FROM enrollments WHERE student_id=? AND course_id=?", (session['user_id'], cid)).fetchone()
        if not enrolled:
            conn.execute("INSERT INTO enrollments (id, student_id, course_id, progress) VALUES (?,?,?,?)",
                         (str(uuid.uuid4()), session['user_id'], cid, progress_pct))
        else:
            conn.execute("UPDATE enrollments SET progress=? WHERE student_id=? AND course_id=?",
                         (progress_pct, session['user_id'], cid))
            
        if done >= total and total > 0:
            course_completed = True
            cert = conn.execute("SELECT id FROM certificates WHERE student_id=? AND course_id=?", (session['user_id'], cid)).fetchone()
            if not cert:
                user = conn.execute("SELECT username FROM users WHERE id=?", (session['user_id'],)).fetchone()
                course = conn.execute("SELECT title FROM courses WHERE id=?", (cid,)).fetchone()
                uname = user['username'] if user else 'Student'
                ctitle = course['title'] if course else 'Course'
                code = 'CERT-' + uuid.uuid4().hex[:8].upper()
                conn.execute("""INSERT INTO certificates (id, student_id, course_id, student_name, course_title, cert_code)
                                VALUES (?,?,?,?,?,?)""", (str(uuid.uuid4()), session['user_id'], cid, uname, ctitle, code))
                conn.execute("""INSERT INTO career_entries (id, user_id, title, entry_type, description)
                                VALUES (?,?,?,?,?)""",
                             (str(uuid.uuid4()), session['user_id'], f"Completed Course: {ctitle}",
                              "certificate", f"Successfully completed all lectures and curriculum in {ctitle}."))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'completed': is_done, 'progress': progress_pct, 'course_completed': course_completed})

@app.route('/api/save_note', methods=['POST'])
@login_required
def save_note():
    data = request.json
    conn = get_db()
    existing = conn.execute("SELECT id FROM notes WHERE student_id=? AND lecture_id=?",(session['user_id'],data['lecture_id'])).fetchone()
    if existing:
        conn.execute("UPDATE notes SET content=? WHERE id=?",(data['content'],existing['id']))
    else:
        conn.execute("INSERT INTO notes (id,student_id,lecture_id,content) VALUES (?,?,?,?)",
                     (str(uuid.uuid4()),session['user_id'],data['lecture_id'],data['content']))
    conn.commit(); conn.close()
    return jsonify({'success':True})

@app.route('/api/my_enrollments')
@login_required
def my_enrollments():
    conn = get_db()
    rows = conn.execute("""
        SELECT 
            e.id AS enrollment_id,
            c.id AS course_id,
            c.id AS id,
            c.title,
            c.description,
            c.category,
            c.difficulty,
            c.thumbnail_color,
            c.educator_name,
            e.enrolled_at,
            e.progress AS stored_progress,
            COUNT(DISTINCT l.id) AS total_lectures,
            COUNT(DISTINCT cl.lecture_id) AS completed_lectures,
            CASE 
                WHEN COUNT(DISTINCT l.id) > 0 
                THEN CAST(ROUND(COUNT(DISTINCT cl.lecture_id) * 100.0 / COUNT(DISTINCT l.id)) AS INTEGER)
                ELSE 0 
            END AS real_progress
        FROM enrollments e
        JOIN courses c ON e.course_id = c.id
        LEFT JOIN lectures l ON l.course_id = c.id
        LEFT JOIN completed_lectures cl ON cl.lecture_id = l.id AND cl.student_id = e.student_id
        WHERE e.student_id = ?
        GROUP BY e.id, c.id
        ORDER BY e.enrolled_at DESC
    """, (session['user_id'],)).fetchall()
    
    result = []
    for r in rows:
        d = dict(r)
        real_prog = d['real_progress']
        d['progress'] = real_prog
        if d['stored_progress'] != real_prog:
            conn.execute("UPDATE enrollments SET progress=? WHERE id=?", (real_prog, d['enrollment_id']))
        if real_prog == 100 and d['total_lectures'] > 0:
            cert = conn.execute("SELECT id FROM certificates WHERE student_id=? AND course_id=?", 
                                (session['user_id'], d['course_id'])).fetchone()
            if not cert:
                user = conn.execute("SELECT username FROM users WHERE id=?", (session['user_id'],)).fetchone()
                uname = user['username'] if user else 'Student'
                code = 'CERT-' + uuid.uuid4().hex[:8].upper()
                conn.execute("""INSERT INTO certificates (id, student_id, course_id, student_name, course_title, cert_code)
                                VALUES (?,?,?,?,?,?)""", 
                             (str(uuid.uuid4()), session['user_id'], d['course_id'], uname, d['title'], code))
                conn.execute("""INSERT INTO career_entries (id, user_id, title, entry_type, description)
                                VALUES (?,?,?,?,?)""",
                             (str(uuid.uuid4()), session['user_id'], f"Completed Course: {d['title']}",
                              "certificate", f"Successfully completed all lectures and curriculum in {d['title']}."))
        result.append(d)
    conn.commit()
    conn.close()
    return jsonify(result)

@app.route('/api/student_stats')
@login_required
def student_stats():
    conn = get_db()
    enrolled = conn.execute("SELECT COUNT(*) FROM enrollments WHERE student_id=?",(session['user_id'],)).fetchone()[0]
    completed = conn.execute("SELECT COUNT(DISTINCT lecture_id) FROM completed_lectures WHERE student_id=?",(session['user_id'],)).fetchone()[0]
    user = conn.execute("SELECT * FROM users WHERE id=?",(session['user_id'],)).fetchone()
    conn.close()
    return jsonify({'enrolled_courses':enrolled,'completed_lectures':completed,
                    'level':user['level'] if user else 'beginner','username':user['username'] if user else '',
                    'total_queries':user['total_queries'] if user else 0})

@app.route('/api/educator_stats')
@educator_required
def educator_stats():
    conn = get_db()
    courses = conn.execute("SELECT COUNT(*) FROM courses WHERE educator_id=?",(session['user_id'],)).fetchone()[0]
    lectures = conn.execute("SELECT COUNT(*) FROM lectures l JOIN courses c ON l.course_id=c.id WHERE c.educator_id=?",(session['user_id'],)).fetchone()[0]
    students = conn.execute("SELECT COUNT(DISTINCT e.student_id) FROM enrollments e JOIN courses c ON e.course_id=c.id WHERE c.educator_id=?",(session['user_id'],)).fetchone()[0]
    conn.close()
    return jsonify({'courses':courses,'lectures':lectures,'students':students,'username':session['username']})

# ─── AI API ──────────────────────────────────────────────────────────────────

def _write_dna_entry(concept, correct=True, resp_ms=2000, style='balanced'):
    """Write/update a learning_dna row for the current user. Safe to call anywhere."""
    try:
        uid  = session.get('user_id', '')
        if not uid:
            return
        conn = get_db()
        existing = conn.execute(
            "SELECT * FROM learning_dna WHERE user_id=? AND concept=?", (uid, concept)
        ).fetchone()
        now = datetime.now().isoformat()
        from datetime import timedelta

        if existing:
            attempts = existing['attempts'] + 1
            correct_count = existing['correct'] + (1 if correct else 0)
            ef  = existing['ease_factor']
            q   = 5 if correct and resp_ms < 3000 else (4 if correct else 2)
            ef2 = max(1.3, ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            iv  = existing['interval_days']
            niv = (6 if iv == 1 else round(iv * ef2)) if q >= 3 else 1
            avg_ms = round((existing['avg_response_ms'] * (attempts - 1) + resp_ms) / attempts)
            nrev = (datetime.now() + timedelta(days=niv)).isoformat()
            conn.execute(
                """UPDATE learning_dna SET attempts=?,correct=?,avg_response_ms=?,last_seen=?,
                   next_review=?,ease_factor=?,interval_days=?,style_pref=?
                   WHERE user_id=? AND concept=?""",
                (attempts, correct_count, avg_ms, now, nrev, ef2, niv, style, uid, concept)
            )
        else:
            nrev = (datetime.now() + timedelta(days=1)).isoformat()
            conn.execute(
                """INSERT INTO learning_dna
                   (id,user_id,concept,attempts,correct,avg_response_ms,last_seen,next_review,ease_factor,interval_days,style_pref)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), uid, concept, 1, 1 if correct else 0,
                 resp_ms, now, nrev, 2.5, 1, style)
            )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Never crash the parent request due to DNA tracking

def get_user_profile():
    if 'profile' not in session:
        session['profile']={'level':'beginner','topics_learned':[],'weak_topics':[],'strong_topics':[],'total_queries':0,'correct_attempts':0,'history':[]}
    return session['profile']

def update_profile(topic, success=True):
    profile = get_user_profile()
    profile['total_queries'] += 1
    if topic not in profile['topics_learned']: profile['topics_learned'].append(topic)
    if success:
        profile['correct_attempts'] += 1
        if topic not in profile['strong_topics']: profile['strong_topics'].append(topic)
        if topic in profile['weak_topics']: profile['weak_topics'].remove(topic)
    else:
        if topic not in profile['weak_topics']: profile['weak_topics'].append(topic)
    r = profile['correct_attempts'] / max(profile['total_queries'],1)
    if profile['total_queries']>5: profile['level']='advanced' if r>0.8 else ('intermediate' if r>0.5 else 'beginner')
    try:
        conn=get_db(); conn.execute("UPDATE users SET level=?,total_queries=?,correct_attempts=? WHERE id=?",(profile['level'],profile['total_queries'],profile['correct_attempts'],session.get('user_id','')))
        conn.commit(); conn.close()
    except: pass
    session['profile']=profile; session.modified=True

@app.route('/api/explain', methods=['POST'])
@login_required
def explain():
    data = request.json or {}
    concept = data.get('concept', '').strip()
    mode = data.get('mode', 'full')
    level = get_user_profile().get('level', 'beginner')

    if not concept:
        return jsonify({'success': False, 'error': 'Please enter a concept to explain.'})

    if mode == 'simple':
        prompt = f'''Explain the ML concept "{concept}" in the simplest possible way for a complete beginner.
Respond with ONLY a JSON object. Use \\n for newlines inside strings. No raw newlines inside string values.
{{"explanation":"simple 2-3 sentence plain English explanation","analogy":"a fun everyday life analogy that makes this concept clear","emoji_summary":"3-5 relevant emojis"}}'''

    elif mode == 'technical':
        prompt = f'''Give a deep technical explanation of "{concept}" in ML for an advanced practitioner.
Respond with ONLY a JSON object. Use \\n for newlines inside strings. No raw newlines inside string values.
{{"explanation":"in-depth technical explanation in 3-4 sentences","mathematics":"key mathematical foundations and formulas","implementation_notes":"key implementation tips and tricks","pitfalls":["common pitfall 1","common pitfall 2","common pitfall 3"]}}'''

    else:  # full
        prompt = f'''Explain the ML concept "{concept}" for a {level} level learner.
Respond with ONLY a JSON object. Use \\n for newlines inside strings. No raw newlines inside string values.
For the code_example field, write Python code on a single conceptual line using \\n between lines.

{{"title":"{concept}","simple_explanation":"1-2 plain English sentences","detailed_explanation":"3-4 paragraphs for {level} level learner. Use \\n\\n between paragraphs.","real_world_example":"one concrete real-world use case","formula":"math formula as plain text string, or null","key_points":["key point 1","key point 2","key point 3","key point 4"],"code_example":"import numpy as np\\nfrom sklearn.cluster import KMeans\\n\\n# Example code here\\nX = np.array([[1,2],[3,4]])\\nmodel = KMeans(n_clusters=2)\\nmodel.fit(X)","visual_description":"describe in words how to visualize this concept","difficulty":"beginner","related_topics":["related topic 1","related topic 2","related topic 3"],"quiz_question":"A good multiple choice question about {concept}?","quiz_options":["A) first option","B) second option","C) third option","D) fourth option"],"quiz_answer":"A"}}'''

    result = None
    try:
        result = ask_groq_json(prompt)
        if result and isinstance(result, dict):
            if 'code_example' in result and isinstance(result['code_example'], str):
                result['code_example'] = result['code_example'].replace('\\n', '\n').replace('\\t', '\t')
    except Exception:
        # Fall back to curated curriculum and dynamic engine
        from ml_knowledge import get_fallback_explanation
        result = get_fallback_explanation(concept, mode=mode, level=level)

    if not result:
        from ml_knowledge import get_fallback_explanation
        result = get_fallback_explanation(concept, mode=mode, level=level)

    try:
        p = get_user_profile()
        p['history'].append({'type': 'concept', 'topic': concept, 'time': datetime.now().isoformat()})
        if len(p['history']) > 50:
            p['history'] = p['history'][-50:]
        session['profile'] = p
        session.modified = True
        update_profile(concept)
        # Also update learning_dna so DNA page shows real concept tracking
        _write_dna_entry(concept, correct=True, resp_ms=2000, style='balanced')
    except Exception:
        pass

    return jsonify({'success': True, 'data': result})


# ─── CODE LAB EXECUTION & CHALLENGE API ──────────────────────────────────────

@app.route('/api/run_code', methods=['POST'])
@login_required
def run_code_endpoint():
    import sys, subprocess, time
    data = request.json or {}
    code_text = data.get('code', '').strip()
    if not code_text:
        return jsonify({'success': False, 'error': 'No code provided to execute.'})

    forbidden = ['subprocess', 'os.system', 'os.popen', 'shutil.rmtree', '__import__("os").system']
    for bad in forbidden:
        if bad in code_text:
            return jsonify({
                'success': False,
                'error': f'Security policy: {bad} is restricted in Code Lab.',
                'stdout': '',
                'stderr': '',
                'elapsed_ms': 0,
                'plots': []
            })

    wrapper = '''
import sys, io, base64
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _has_mpl = True
except ImportError:
    _has_mpl = False

_plots = []
def _custom_show(*args, **kwargs):
    if _has_mpl and plt.get_fignums():
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        _plots.append(base64.b64encode(buf.read()).decode())
        plt.close()

if _has_mpl:
    plt.show = _custom_show

# User code starts
''' + code_text + '''
# User code ends

if _has_mpl and plt.get_fignums():
    _custom_show()

for _p in _plots:
    print('__PLOT_B64__:' + _p)
'''

    t0 = time.time()
    try:
        res = subprocess.run([sys.executable, '-c', wrapper], capture_output=True, text=True, timeout=7)
        elapsed = round((time.time() - t0) * 1000, 1)

        stdout = res.stdout or ''
        stderr = res.stderr or ''
        plots = []
        clean_lines = []
        for line in stdout.splitlines():
            if line.startswith('__PLOT_B64__:'):
                plots.append(line.replace('__PLOT_B64__:', '').strip())
            else:
                clean_lines.append(line)

        return jsonify({
            'success': True,
            'stdout': '\n'.join(clean_lines),
            'stderr': stderr,
            'elapsed_ms': elapsed,
            'plots': plots
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'Execution timed out (7s limit reached). Check for infinite loops or long computation.',
            'stdout': '',
            'stderr': '',
            'elapsed_ms': 7000,
            'plots': []
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Execution failed: {str(e)}',
            'stdout': '',
            'stderr': '',
            'elapsed_ms': 0,
            'plots': []
        })


@app.route('/api/run_challenge', methods=['POST'])
@login_required
def run_challenge_endpoint():
    import sys, subprocess, time
    data = request.json or {}
    challenge_id = data.get('challenge_id', '').strip()
    user_code = data.get('code', '').strip()
    if not challenge_id or not user_code:
        return jsonify({'success': False, 'error': 'Challenge ID and code are required.'})

    test_harnesses = {
        'softmax': '''
import numpy as np
try:
    s1 = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(np.sum(s1), 1.0), "Test 1 Failed: Probabilities must sum to 1.0"
    assert s1[2] > s1[1] > s1[0], "Test 1 Failed: Higher inputs must yield higher probabilities"
    s2 = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert not np.isnan(s2).any(), "Test 2 Failed: Numerical overflow! Subtract max(z) for stability"
    assert np.isclose(np.sum(s2), 1.0), "Test 2 Failed: Must sum to 1.0 with large inputs"
    s3 = softmax(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert s3.shape == (2, 2), "Test 3 Failed: Should support 2D batch dimension"
    assert np.allclose(np.sum(s3, axis=-1), [1.0, 1.0]), "Test 3 Failed: Each batch item must sum to 1.0"
    print("__CHALLENGE_PASSED__")
except AssertionError as ae:
    print(f"__CHALLENGE_FAILED__: {str(ae)}")
except Exception as ex:
    print(f"__CHALLENGE_ERROR__: {type(ex).__name__}: {str(ex)}")
''',
        'mse_loss': '''
import numpy as np
try:
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.5, 2.5, 2.0])
    loss, grad = mse_and_grad(yt, yp)
    assert np.isclose(loss, 0.5), f"Test 1 Failed: Expected MSE 0.5, got {loss}"
    exp_grad = (2.0 / 3.0) * (yp - yt)
    assert np.allclose(grad, exp_grad), f"Test 2 Failed: Expected gradient {exp_grad}, got {grad}"
    yt2 = np.array([5.0])
    yp2 = np.array([5.0])
    l2, g2 = mse_and_grad(yt2, yp2)
    assert np.isclose(l2, 0.0) and np.isclose(g2[0], 0.0), "Test 3 Failed: Zero error must give zero loss and grad"
    print("__CHALLENGE_PASSED__")
except AssertionError as ae:
    print(f"__CHALLENGE_FAILED__: {str(ae)}")
except Exception as ex:
    print(f"__CHALLENGE_ERROR__: {type(ex).__name__}: {str(ex)}")
''',
        'bce_loss': '''
import numpy as np
try:
    yt = np.array([1.0, 0.0, 1.0])
    yp = np.array([0.9, 0.1, 0.8])
    loss = binary_cross_entropy(yt, yp)
    assert loss > 0, "Test 1 Failed: BCE Loss must be positive"
    loss_extreme = binary_cross_entropy(np.array([1.0]), np.array([0.0]))
    assert not np.isnan(loss_extreme) and not np.isinf(loss_extreme), "Test 2 Failed: Output contains NaN/Inf! Clip predictions to [eps, 1-eps]"
    print("__CHALLENGE_PASSED__")
except AssertionError as ae:
    print(f"__CHALLENGE_FAILED__: {str(ae)}")
except Exception as ex:
    print(f"__CHALLENGE_ERROR__: {type(ex).__name__}: {str(ex)}")
''',
        'pairwise_dist': '''
import numpy as np
try:
    A = np.array([[0.0, 0.0], [3.0, 4.0]])
    B = np.array([[0.0, 0.0], [0.0, 4.0]])
    D = euclidean_distances(A, B)
    assert D.shape == (2, 2), f"Test 1 Failed: Expected shape (2, 2), got {D.shape}"
    assert np.isclose(D[0, 0], 0.0), "Test 2 Failed: Distance to itself must be 0"
    assert np.isclose(D[0, 1], 4.0), "Test 3 Failed: Distance from (0,0) to (0,4) must be 4.0"
    assert np.isclose(D[1, 0], 5.0), "Test 4 Failed: Distance from (3,4) to (0,0) must be 5.0"
    print("__CHALLENGE_PASSED__")
except AssertionError as ae:
    print(f"__CHALLENGE_FAILED__: {str(ae)}")
except Exception as ex:
    print(f"__CHALLENGE_ERROR__: {type(ex).__name__}: {str(ex)}")
''',
        'one_hot': '''
import numpy as np
try:
    labels = np.array([0, 2, 1])
    mat = one_hot_encode(labels, 3)
    assert mat.shape == (3, 3), f"Test 1 Failed: Expected shape (3, 3), got {mat.shape}"
    assert np.array_equal(mat, [[1, 0, 0], [0, 0, 1], [0, 1, 0]]), "Test 2 Failed: Encoding mismatch"
    labels2 = np.array([1, 1, 0])
    mat2 = one_hot_encode(labels2, 2)
    assert np.array_equal(mat2, [[0, 1], [0, 1], [1, 0]]), "Test 3 Failed: Binary one-hot mismatch"
    print("__CHALLENGE_PASSED__")
except AssertionError as ae:
    print(f"__CHALLENGE_FAILED__: {str(ae)}")
except Exception as ex:
    print(f"__CHALLENGE_ERROR__: {type(ex).__name__}: {str(ex)}")
''',
        'mini_batch': '''
import numpy as np
try:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    batches = list(get_batches(X, y, 4))
    assert len(batches) == 3, f"Test 1 Failed: 10 items in batches of 4 should yield 3 batches, got {len(batches)}"
    assert batches[0][0].shape == (4, 2), "Test 2 Failed: Batch 0 X shape should be (4, 2)"
    assert batches[-1][0].shape == (2, 2), "Test 3 Failed: Final batch should contain remaining 2 items"
    print("__CHALLENGE_PASSED__")
except AssertionError as ae:
    print(f"__CHALLENGE_FAILED__: {str(ae)}")
except Exception as ex:
    print(f"__CHALLENGE_ERROR__: {type(ex).__name__}: {str(ex)}")
'''
    }

    harness = test_harnesses.get(challenge_id)
    if not harness:
        return jsonify({'success': False, 'error': f'Challenge "{challenge_id}" not found.'})

    full_code = user_code + '\n' + harness
    t0 = time.time()
    try:
        res = subprocess.run([sys.executable, '-c', full_code], capture_output=True, text=True, timeout=6)
        elapsed = round((time.time() - t0) * 1000, 1)

        stdout = res.stdout or ''
        stderr = res.stderr or ''

        passed = '__CHALLENGE_PASSED__' in stdout
        feedback = ''
        if passed:
            feedback = 'All unit tests passed successfully!'
            try:
                _write_dna_entry('code_challenge_' + challenge_id, correct=True, resp_ms=int(elapsed), style='practical')
            except Exception:
                pass
        elif '__CHALLENGE_FAILED__:' in stdout:
            feedback = stdout.split('__CHALLENGE_FAILED__:')[1].strip()
        elif '__CHALLENGE_ERROR__:' in stdout:
            feedback = stdout.split('__CHALLENGE_ERROR__:')[1].strip()
        elif stderr:
            feedback = stderr.strip().splitlines()[-1] if stderr.strip() else 'Runtime error occurred.'
        else:
            feedback = 'Tests did not complete. Check function signature and returns.'

        return jsonify({
            'success': True,
            'passed': passed,
            'feedback': feedback,
            'elapsed_ms': elapsed,
            'stdout': stdout.replace('__CHALLENGE_PASSED__', '').strip(),
            'stderr': stderr
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'passed': False,
            'feedback': 'Execution timed out (6s limit). Check for infinite loops.',
            'elapsed_ms': 6000
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'passed': False,
            'feedback': f'Execution failed: {str(e)}',
            'elapsed_ms': 0
        })


@app.route('/api/explain_code', methods=['POST'])
@login_required
def explain_code_endpoint():
    data = request.json or {}
    code_text = data.get('code', '').strip()
    if not code_text:
        return jsonify({'success': False, 'error': 'No code provided to explain.'})

    prompt = (
        'You are an expert Python ML educator.\n'
        'Analyze this code and explain it clearly for a student.\n'
        'Return ONLY valid JSON with no markdown:\n'
        '{"summary":"1-2 sentence high-level overview of what this script accomplishes.",'
        '"time_complexity":"Time complexity with Big-O notation and explanation.",'
        '"space_complexity":"Space complexity with Big-O notation.",'
        '"line_by_line":[{"step":"Imports & Data Prep","detail":"..."},{"step":"Model Training / Logic","detail":"..."},{"step":"Evaluation / Output","detail":"..."}],'
        '"edge_cases":["Edge case 1 to watch out for","Edge case 2 to watch out for"]}\n\n'
        f'Code:\n"""\n{code_text}\n"""'
    )

    try:
        explanation = ask_groq_json(prompt)
        return jsonify({'success': True, 'data': explanation})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Code explanation failed: {str(e)}'})

@app.route('/api/generate_code', methods=['POST'])
@login_required
def generate_code():
    import re
    data  = request.json or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'success': False, 'error': 'Please enter a description of the code you want.'})
    level = get_user_profile().get('level', 'beginner')

    # ── Step 1: metadata JSON (NO code block inside) ──────────────
    meta_prompt = (
        f'You are an expert ML engineer. The user is at {level} level and wants to implement: "{query}"\n'
        'Respond with ONLY a valid JSON object. Use \\n for newlines inside strings. No code in this JSON.\n'
        'Required keys: title (string), explanation (string, 3-4 sentences), '
        'libraries_needed (array of strings), expected_output (string), '
        'variations (string), common_mistakes (array of strings).\n'
        'Example: {"title":"Linear Regression","explanation":"We fit a line...","libraries_needed":["numpy","sklearn"],'
        '"expected_output":"Prints MSE and coefficients","variations":"Try Ridge or Lasso","common_mistakes":["Not scaling features","Ignoring outliers"]}'
    )

    # ── Step 2: actual Python code (plain text, no JSON) ──────────
    code_prompt = (
        f'Write complete, well-commented Python code for: "{query}"\n'
        f'The user is at {level} level.\n'
        'Rules: Return ONLY raw Python code. Start with imports. No markdown fences. No JSON. No explanation text.'
    )

    try:
        meta      = ask_groq_json(meta_prompt)
        code_text = ask_groq(
            code_prompt,
            system="You are an expert Python ML engineer. Output ONLY raw Python code with no markdown, no explanation.",
            temperature=0.2
        )
        # Strip any accidental markdown fences
        code_text = re.sub(r'```python\s*', '', code_text.strip())
        code_text = re.sub(r'```',           '', code_text).strip()

        # Guarantee required keys exist
        meta.setdefault('title',           query.title())
        meta.setdefault('explanation',     'See code comments for details.')
        meta.setdefault('libraries_needed', [])
        meta.setdefault('expected_output', '')
        meta.setdefault('variations',      '')
        meta.setdefault('common_mistakes', [])
        meta['code'] = code_text

        update_profile(query)
        return jsonify({'success': True, 'data': meta})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Code generation failed: {str(e)}'})


@app.route('/api/analyze_code', methods=['POST'])
@login_required
def analyze_code():
    import re
    data       = request.json or {}
    code_input = data.get('code', '').strip()
    if not code_input:
        return jsonify({'success': False, 'error': 'Please paste some Python code to analyze.'})

    # ── Step 1: analysis JSON (NO code inside the JSON) ───────────
    meta_prompt = (
        'You are an expert Python ML code reviewer.\n'
        'Analyze the following Python code and respond with ONLY a valid JSON object.\n'
        'Use \\n for newlines inside strings. Do NOT put code inside the JSON.\n'
        'Required keys: overall_assessment (string), score (integer 0-100), '
        'errors (array of objects with keys line, issue, fix), '
        'improvements (array of strings), explanation (string), '
        'best_practices (array of strings), complexity (string).\n'
        'If there are no errors use an empty array for errors.\n\n'
        'Code to analyze:\n'
        '"""\n'
        f'{code_input}\n'
        '"""'
    )

    # ── Step 2: optimised code (plain text) ───────────────────────
    opt_prompt = (
        'Rewrite the following Python ML code as a clean, optimised, well-commented version.\n'
        'Return ONLY the improved Python code. No markdown. No explanation. No JSON.\n\n'
        f'{code_input}'
    )

    try:
        meta     = ask_groq_json(meta_prompt)
        opt_code = ask_groq(
            opt_prompt,
            system="You are an expert Python ML engineer. Output ONLY clean optimised Python code.",
            temperature=0.2
        )
        # Strip accidental fences
        opt_code = re.sub(r'```python\s*', '', opt_code.strip())
        opt_code = re.sub(r'```',           '', opt_code).strip()

        # Guarantee required keys
        meta.setdefault('overall_assessment', 'Code reviewed successfully.')
        meta.setdefault('score',              70)
        meta.setdefault('errors',             [])
        meta.setdefault('improvements',       [])
        meta.setdefault('explanation',        '')
        meta.setdefault('best_practices',     [])
        meta.setdefault('complexity',         '')
        # Coerce score to int in case model returns a string
        try:
            meta['score'] = int(meta['score'])
        except Exception:
            meta['score'] = 70
        meta['optimized_code'] = opt_code

        return jsonify({'success': True, 'data': meta})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Code analysis failed: {str(e)}'})

@app.route('/api/generate_diagram', methods=['POST'])
@login_required
def generate_diagram():
    data = request.json or {}
    concept = data.get('concept', '')
    vd = data.get('visual_description', '')
    prompt = f'''Create a diagram data structure for the ML concept "{concept}".
Visual description hint: {vd}

Return ONLY valid JSON, no markdown fences. Use x values between 80-720 and y values between 60-360:
{{"type":"flowchart","title":"diagram title","nodes":[{{"id":"1","label":"Node Label","x":200,"y":150,"color":"#10B981"}},{{"id":"2","label":"Node 2","x":400,"y":150,"color":"#7C3AED"}}],"edges":[{{"from":"1","to":"2","label":"connection"}}],"description":"what this diagram illustrates"}}

Create at least 4 nodes and meaningful connections that explain {concept} visually.'''
    try:
        result = ask_groq_json(prompt)
        if result and isinstance(result, dict) and 'nodes' in result and len(result['nodes']) > 0:
            return jsonify({'success': True, 'data': result})
    except Exception:
        pass
    
    from ml_knowledge import get_fallback_diagram
    return jsonify({'success': True, 'data': get_fallback_diagram(concept, vd)})

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.json or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])
    p = get_user_profile()
    if not message:
        return jsonify({'success': False, 'error': 'Message cannot be empty.'})

    system = f"You are LearnSphere, a friendly and expert ML tutor. The student is at {p.get('level', 'beginner')} level. Topics they've studied: {', '.join(p.get('topics_learned', [])[-5:]) or 'none yet'}. Give concise, helpful, encouraging answers. Use examples when helpful."
    messages = [{"role": "system", "content": system}]
    for h in history[-6:]:
        messages.append({"role": h.get('role', 'user'), "content": h.get('content', '')})
    messages.append({"role": "user", "content": message})
    try:
        client = get_groq_client()
        if client:
            for m in CANDIDATE_GROQ_MODELS:
                try:
                    response = client.chat.completions.create(
                        model=m, messages=messages, temperature=0.7, max_tokens=1024
                    )
                    return jsonify({'success': True, 'response': response.choices[0].message.content})
                except Exception as me:
                    if 'model_not_found' in str(me) or '404' in str(me):
                        continue
                    raise me
        raise RuntimeError("No Groq client available")
    except Exception:
        from ml_knowledge import get_fallback_chat
        reply = get_fallback_chat(message, history, level=p.get('level', 'beginner'), topics_learned=p.get('topics_learned', []))
        return jsonify({'success': True, 'response': reply})

@app.route('/api/profile')
@login_required
def get_profile(): return jsonify(get_user_profile())

@app.route('/api/quiz_check', methods=['POST'])
@login_required
def quiz_check():
    data = request.json or {}
    ans = str(data.get('answer', '')).strip().upper()
    correct = str(data.get('correct', '')).strip().upper()
    topic = data.get('topic', '')

    # Normalize e.g. "A" from "A) Mean Squared Error"
    if len(ans) > 1 and ans[0].isalpha() and ans[1] in [')', '.', ':', ' ']:
        ans = ans[0]
    if len(correct) > 1 and correct[0].isalpha() and correct[1] in [')', '.', ':', ' ']:
        correct = correct[0]

    ok = (ans == correct)
    update_profile(topic, ok)
    _write_dna_entry(topic, correct=ok, resp_ms=2500, style='balanced')

    # Look up explanation if available
    from ml_knowledge import resolve_topic_key, ML_CURRICULUM
    explanation = ""
    canonical = resolve_topic_key(topic)
    if canonical in ML_CURRICULUM:
        explanation = ML_CURRICULUM[canonical].get('quiz_explanation', '')

    msg = "🎉 Correct! Outstanding work! +15 XP added." if ok else f"Not quite. The correct answer is ({correct})."
    if explanation:
        msg += f" {explanation}"

    return jsonify({'correct': ok, 'message': msg, 'correct_answer': correct})

@app.route('/api/suggest_topics')
@login_required
def suggest_topics():
    p = get_user_profile()
    prompt = f'''Suggest 6 machine learning topics for a {p.get('level', 'beginner')} level learner.
They have already studied: {", ".join(p.get('topics_learned', [])[-10:]) or "nothing yet"}.
Suggest topics that build naturally on what they know.
Return ONLY valid JSON, no markdown fences:
{{"suggestions":[{{"topic":"topic name","reason":"why this topic is good to learn next","difficulty":"beginner or intermediate or advanced"}},{{"topic":"topic2","reason":"reason","difficulty":"difficulty"}}]}}'''
    try:
        result = ask_groq_json(prompt)
        if result and isinstance(result, dict) and 'suggestions' in result and len(result['suggestions']) > 0:
            return jsonify({'success': True, 'data': result})
    except Exception:
        pass

    from ml_knowledge import get_fallback_suggestions
    return jsonify({'success': True, 'data': {'suggestions': get_fallback_suggestions(p.get('level', 'beginner'), p.get('topics_learned', []))}})

@app.route('/api/audio_script', methods=['POST'])
@login_required
def audio_script():
    data = request.json or {}
    concept = data.get('concept', '')
    exp = data.get('explanation', '')
    prompt = f'''Write a natural, engaging audio narration script that teaches the ML concept "{concept}".
Base it on this explanation: {exp[:500]}
The script should sound natural when spoken aloud — conversational, clear, 100-150 words.
Return ONLY valid JSON, no markdown fences:
{{"script":"the complete narration text that sounds natural when read aloud","duration_estimate":"estimated X seconds"}}'''
    try:
        result = ask_groq_json(prompt)
        if result and isinstance(result, dict) and 'script' in result:
            return jsonify({'success': True, 'data': result})
    except Exception:
        pass

    from ml_knowledge import get_fallback_audio
    return jsonify({'success': True, 'data': get_fallback_audio(concept, exp)})

@app.route('/api/ai_explain_lecture', methods=['POST'])
@login_required
def ai_explain_lecture():
    data=request.json; title=data.get('title',''); content=data.get('content','')[:600]; p=get_user_profile()
    prompt=f'''You are tutoring a {p["level"]} level student who is reading a lecture titled "{title}".
Here is a summary of the lecture content: {content}

Give a helpful, encouraging 3-4 sentence explanation in plain English. 
Be conversational and make it easy to understand. Mention one concrete example if helpful.'''
    try:
        response = ask_groq(prompt, system="You are a friendly, expert ML tutor who explains things clearly and encouragingly.")
        return jsonify({'success':True,'response':response})
    except Exception as e: return jsonify({'success':False,'error':str(e)})


# ─── LEARNING DNA ROUTES ─────────────────────────────────────────────────────

@app.route('/dna')
@login_required
def dna_page():
    return render_template('dna.html')

@app.route('/compete')
@login_required
def battle_page():
    return render_template('compete.html')

@app.route('/api/dna/update', methods=['POST'])
@login_required
def update_dna():
    """Update learning DNA after any interaction"""
    data = request.json
    concept   = data.get('concept', '')
    correct   = data.get('correct', True)
    resp_ms   = data.get('response_ms', 3000)
    style     = data.get('style', 'balanced')
    uid       = session['user_id']
    conn      = get_db()

    existing = conn.execute(
        "SELECT * FROM learning_dna WHERE user_id=? AND concept=?", (uid, concept)
    ).fetchone()

    now = datetime.now().isoformat()

    if existing:
        attempts = existing['attempts'] + 1
        correct_count = existing['correct'] + (1 if correct else 0)
        # SM-2 spaced repetition
        ef = existing['ease_factor']
        q  = 5 if correct and resp_ms < 3000 else (4 if correct else 2)
        ef_new = max(1.3, ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        interval = existing['interval_days']
        if q >= 3:
            if interval == 1:
                new_interval = 6
            else:
                new_interval = round(interval * ef_new)
        else:
            new_interval = 1
        from datetime import timedelta
        next_rev = (datetime.now() + timedelta(days=new_interval)).isoformat()
        avg_ms = round((existing['avg_response_ms'] * (attempts - 1) + resp_ms) / attempts)

        conn.execute("""UPDATE learning_dna
            SET attempts=?, correct=?, avg_response_ms=?, last_seen=?,
                next_review=?, ease_factor=?, interval_days=?, style_pref=?
            WHERE user_id=? AND concept=?""",
            (attempts, correct_count, avg_ms, now, next_rev, ef_new, new_interval, style, uid, concept))
    else:
        from datetime import timedelta
        next_rev = (datetime.now() + timedelta(days=1)).isoformat()
        conn.execute("""INSERT INTO learning_dna
            (id,user_id,concept,attempts,correct,avg_response_ms,last_seen,next_review,ease_factor,interval_days,style_pref)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), uid, concept, 1, 1 if correct else 0,
             resp_ms, now, next_rev, 2.5, 1, style))

    conn.commit(); conn.close()
    return jsonify({'success': True})


@app.route('/api/dna/profile')
@login_required
def get_dna_profile():
    uid  = session['user_id']
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM learning_dna WHERE user_id=? ORDER BY last_seen DESC", (uid,)
    ).fetchall()
    now  = datetime.now()

    dna = []
    for r in rows:
        accuracy = round(r['correct'] / r['attempts'] * 100) if r['attempts'] else 0
        speed_label = (
            'Fast Learner' if r['avg_response_ms'] < 3000 else
            'Average'      if r['avg_response_ms'] < 7000 else
            'Takes Time'
        )
        # Decay: days since last seen
        days_ago = 0
        if r['last_seen']:
            try:
                last = datetime.fromisoformat(r['last_seen'])
                days_ago = (now - last).days
            except Exception:
                pass

        # Forgetting curve: retention = e^(-days/interval)
        import math
        retention = round(100 * math.exp(-days_ago / max(r['interval_days'], 1)))
        retention = max(5, min(100, retention))

        due_review = False
        if r['next_review']:
            try:
                due_review = datetime.fromisoformat(r['next_review']) <= now
            except Exception:
                pass

        dna.append({
            'concept':       r['concept'],
            'attempts':      r['attempts'],
            'accuracy':      accuracy,
            'speed_label':   speed_label,
            'avg_ms':        r['avg_response_ms'],
            'last_seen':     r['last_seen'],
            'next_review':   r['next_review'],
            'retention':     retention,
            'due_review':    due_review,
            'interval_days': r['interval_days'],
            'ease_factor':   round(r['ease_factor'], 2),
            'style_pref':    r['style_pref'],
            'days_ago':      days_ago,
        })

    # Compute summary stats
    total_concepts   = len(dna)
    mastered         = [d for d in dna if d['accuracy'] >= 80]
    weak             = [d for d in dna if d['accuracy'] < 60]
    due_now          = [d for d in dna if d['due_review']]
    avg_retention    = round(sum(d['retention'] for d in dna) / total_concepts) if dna else 0
    preferred_style  = max(set(d['style_pref'] for d in dna), key=lambda x: sum(1 for d in dna if d['style_pref']==x)) if dna else 'balanced'

    conn.close()
    return jsonify({
        'dna': dna,
        'summary': {
            'total_concepts':  total_concepts,
            'mastered_count':  len(mastered),
            'weak_count':      len(weak),
            'due_review_count':len(due_now),
            'avg_retention':   avg_retention,
            'preferred_style': preferred_style,
        }
    })


@app.route('/api/dna/predict_decay')
@login_required
def predict_decay():
    """Return concepts at risk of being forgotten"""
    uid  = session['user_id']
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM learning_dna WHERE user_id=? ORDER BY next_review ASC", (uid,)
    ).fetchall()
    now  = datetime.now()

    import math
    predictions = []
    for r in rows:
        days_ago = 0
        if r['last_seen']:
            try:
                days_ago = (now - datetime.fromisoformat(r['last_seen'])).days
            except Exception:
                pass
        retention = round(100 * math.exp(-days_ago / max(r['interval_days'], 1)))
        retention = max(5, min(100, retention))

        days_until_forget = 0
        # days until retention drops below 50%
        import math as m
        if r['interval_days'] > 0:
            days_until_forget = max(0, round(r['interval_days'] * m.log(2) - days_ago))

        predictions.append({
            'concept':           r['concept'],
            'retention':         retention,
            'days_until_forget': days_until_forget,
            'urgency':           'critical' if retention < 30 else ('warning' if retention < 60 else 'good'),
            'last_seen':         r['last_seen'],
        })

    predictions.sort(key=lambda x: x['retention'])
    conn.close()
    return jsonify({'predictions': predictions[:10]})


# ─── BATTLE MODE ROUTES ───────────────────────────────────────────────────────

@app.route('/api/battle/question', methods=['POST'])
@login_required
def battle_question():
    """Generate a truly unique quiz question for battle mode using subtopic rotation."""
    import random as _rnd

    data            = request.json or {}
    topic           = data.get('topic', 'Machine Learning')
    difficulty      = data.get('difficulty', 'medium')
    q_num           = data.get('question_num', 1)
    asked_questions = data.get('asked_questions', [])

    # Rich subtopic pool for common ML topics — forces variety
    subtopic_pools = {
        'Machine Learning': [
            'overfitting vs underfitting','bias-variance tradeoff','cross-validation','gradient descent',
            'learning rate','regularization (L1 L2)','feature engineering','dimensionality reduction',
            'ensemble methods','decision trees','random forests','support vector machines',
            'k-nearest neighbours','naive bayes','logistic regression','evaluation metrics (F1 AUC ROC)',
            'confusion matrix','train/test split','hyperparameter tuning','model selection'
        ],
        'Deep Learning': [
            'backpropagation','activation functions','vanishing gradient','batch normalisation',
            'dropout regularization','convolutional layers','pooling layers','recurrent networks',
            'LSTM vs GRU','attention mechanism','transformers','transfer learning',
            'optimizers (Adam SGD RMSProp)','weight initialisation','loss functions',
            'ResNet skip connections','embedding layers','encoder-decoder','GANs','autoencoders'
        ],
        'Neural Networks': [
            'perceptron','multi-layer perceptron','forward pass','backward pass','weight update',
            'sigmoid tanh ReLU','softmax','cross-entropy loss','mean squared error','mini-batch',
            'epoch vs iteration','neuron activation','architecture depth vs width',
            'universal approximation theorem','output layer design'
        ],
        'Natural Language Processing': [
            'tokenisation','word embeddings','Word2Vec GloVe','BERT GPT','attention mechanism',
            'sequence-to-sequence','named entity recognition','sentiment analysis','POS tagging',
            'n-grams','TF-IDF','cosine similarity','language model perplexity','fine-tuning'
        ],
        'Computer Vision': [
            'convolution operation','feature maps','object detection YOLO','image segmentation',
            'data augmentation','ResNet VGG Inception','image classification','anchor boxes',
            'non-max suppression','semantic vs instance segmentation','feature pyramid networks'
        ],
    }

    pool = subtopic_pools.get(topic, subtopic_pools['Machine Learning'])

    # Pick a subtopic that has NOT been covered in asked questions
    used_subtopics = set()
    for q in asked_questions:
        for sub in pool:
            if sub.lower() in q.lower():
                used_subtopics.add(sub)

    available = [s for s in pool if s not in used_subtopics]
    subtopic = _rnd.choice(available if available else pool)

    # Rotate question styles to force lexical variety
    styles = [
        'Give a definition question about',
        'Ask how to apply or implement',
        'Ask what happens when or why in the context of',
        'Create a scenario-based question involving',
        'Ask about the difference between two aspects of',
        'Ask about a common mistake or pitfall with',
        'Ask to interpret a result or metric related to',
        'Ask about the mathematics or formula behind',
        'Ask when to use or not use',
        'Create a code-reading question about',
    ]
    style = styles[(q_num - 1) % len(styles)]

    avoid_block = ''
    if asked_questions:
        recent = asked_questions[-8:]
        avoid_block = (
            'FORBIDDEN — these questions were ALREADY asked, do NOT repeat or paraphrase them:\n'
            + '\n'.join(f'  × {q}' for q in recent)
            + '\n\n'
        )

    pts = 10 if difficulty == 'easy' else (20 if difficulty == 'medium' else 30)

    import time as _time
    seed = int(_time.time() * 1000) % 99991   # unique per request

    prompt = (
        f'[SESSION_SEED:{seed}] Create a quiz question for a Machine Learning student.\n'
        f'This is question #{q_num} in a battle session. Generate a COMPLETELY DIFFERENT question each time.\n\n'
        f'REQUIRED SUBTOPIC: {subtopic} (within {topic})\n'
        f'QUESTION STYLE: {style} "{subtopic}"\n'
        f'DIFFICULTY: {difficulty}\n\n'
        f'{avoid_block}'
        f'Rules:\n'
        f'- The question MUST specifically test knowledge of: {subtopic}\n'
        f'- Make all 4 options plausible (no obviously wrong answers)\n'
        f'- Each question must have a UNIQUE angle — never re-use the same phrasing\n'
        f'- Explanation must be educational (2-3 sentences)\n\n'
        f'Respond with ONLY this JSON (no markdown):\n'
        f'{{"question":"question text here","options":["A) ...","B) ...","C) ...","D) ..."],'
        f'"answer":"A","explanation":"clear explanation of why that answer is correct","difficulty":"{difficulty}","points":{pts}}}'
    )

    try:
        # Temperature 0.95 + unique seed forces true variety per question
        result = ask_groq_json(prompt, system=(
            "You are an expert ML quiz designer. Generate UNIQUE, VARIED questions. "
            f"Seed:{seed}. Never repeat patterns. Always match the required subtopic exactly."
        ), temperature=0.95)
        if result and isinstance(result, dict) and 'question' in result and 'options' in result:
            if result.get('answer') and len(result['answer']) > 1:
                result['answer'] = result['answer'][0].upper()
            return jsonify({'success': True, 'data': result})
    except Exception:
        pass

    # Seamless reliable fallback from battle_engine
    from battle_engine import get_fallback_battle_question
    q = get_fallback_battle_question(topic=topic, difficulty=difficulty, q_num=q_num, asked_questions=asked_questions)
    return jsonify({'success': True, 'data': q})


@app.route('/api/battle/save_score', methods=['POST'])
@login_required
def save_battle_score():
    data = request.json or {}
    conn = get_db()
    score = int(data.get('score', 0))
    ai_score = int(data.get('ai_score', 0))
    topic = data.get('topic', 'Machine Learning')
    uid = session.get('user_id', '')
    uname = session.get('username', 'Player')

    # 1. Insert authentic game match record
    match_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO battle_scores (id, user_id, username, score, ai_score, topic) VALUES (?,?,?,?,?,?)",
        (match_id, uid, uname, score, ai_score, topic)
    )

    # 2. Award real platform coins based on score (1 coin per 10 points)
    coins_earned = max(0, score // 10)
    if coins_earned > 0:
        existing_reward = conn.execute("SELECT coins FROM user_rewards WHERE user_id=?", (uid,)).fetchone()
        if existing_reward:
            conn.execute("UPDATE user_rewards SET coins = coins + ? WHERE user_id=?", (coins_earned, uid))
        else:
            conn.execute("INSERT INTO user_rewards (user_id, coins) VALUES (?, ?)", (uid, coins_earned))

    # 3. Update learning DNA with real match interaction
    try:
        dna_row = conn.execute("SELECT * FROM learning_dna WHERE user_id=?", (uid,)).fetchone()
        if dna_row:
            conn.execute("""
                UPDATE learning_dna 
                SET total_interactions = total_interactions + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (uid,))
    except Exception:
        pass

    conn.commit()
    conn.close()
    return jsonify({'success': True, 'coins_earned': coins_earned, 'match_id': match_id})


@app.route('/api/battle/leaderboard')
@login_required
def battle_leaderboard():
    conn = get_db()
    rows = conn.execute("""
        SELECT 
            COALESCE(u.username, bs.username) AS username,
            MAX(bs.score) AS best_score,
            COUNT(*) AS games_played,
            SUM(CASE WHEN bs.score > bs.ai_score THEN 1 ELSE 0 END) AS wins
        FROM battle_scores bs
        LEFT JOIN users u ON bs.user_id = u.id
        GROUP BY COALESCE(u.username, bs.username)
        ORDER BY best_score DESC, wins DESC LIMIT 20
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/battle/my_history')
@login_required
def battle_history():
    conn = get_db()
    uid = session.get('user_id', '')
    uname = session.get('username', '')
    rows = conn.execute(
        "SELECT * FROM battle_scores WHERE user_id=? OR username=? ORDER BY played_at DESC LIMIT 20",
        (uid, uname)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/battle/my_stats')
@login_required
def battle_stats():
    conn = get_db()
    uid = session.get('user_id', '')
    uname = session.get('username', '')
    total = conn.execute("SELECT COUNT(*) FROM battle_scores WHERE user_id=? OR username=?", (uid, uname)).fetchone()[0]
    wins = conn.execute("SELECT COUNT(*) FROM battle_scores WHERE (user_id=? OR username=?) AND score > ai_score", (uid, uname)).fetchone()[0]
    best = conn.execute("SELECT MAX(score) FROM battle_scores WHERE user_id=? OR username=?", (uid, uname)).fetchone()[0] or 0
    total_score = conn.execute("SELECT SUM(score) FROM battle_scores WHERE user_id=? OR username=?", (uid, uname)).fetchone()[0] or 0
    conn.close()
    win_rate = int((wins / total * 100)) if total > 0 else 0
    return jsonify({
        'success': True,
        'total_matches': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'best_score': best,
        'total_score': total_score
    })


# ─── RESEARCH & VIRTUAL LAB ROUTES ──────────────────────────────────────────

@app.route('/research')
@login_required
def research():
    return render_template('research.html')

@app.route('/vlab')
@login_required
def vlab():
    return render_template('vlab.html')

@app.route('/api/research_paper', methods=['POST'])
@login_required
def research_paper():
    data       = request.json or {}
    concept    = (data.get('concept') or data.get('topic') or '').strip()
    paper_type = data.get('paper_type', 'survey').strip().lower()
    if not concept:
        return jsonify({'success': False, 'error': 'Please enter a research topic or ML concept.'})
    
    level_prof = get_user_profile().get('level', 'beginner')
    level = data.get('level', level_prof)

    import re as _re

    type_titles = {
        'survey': 'Comprehensive Academic Survey',
        'benchmark': 'Empirical Benchmark and Comparative Study',
        'proposal': 'Novel Architecture and Paradigm Proposal',
        'system': 'System Implementation and Scalability Analysis'
    }
    type_desc = type_titles.get(paper_type, 'Academic Research Paper')

    paper_prompt = (
        f'Write a publication-quality IEEE-style {type_desc} about "{concept}" for a {level}-level ML researcher.\n'
        f'Respond with ONLY valid JSON with no markdown:\n'
        f'{{"title":"Formal academic title for {concept}",'
        f'"abstract":"Rigorous 100-140 word abstract summarizing background, motivation, methodology, and key results.",'
        f'"introduction":"120-160 word introduction covering technical challenges, previous bottlenecks, and 3 explicit contributions.",'
        f'"theory_math":"110-150 word theoretical formulation with loss functions, optimization bounds, or probability models.",'
        f'"methodology":"130-170 word architecture pipeline, design decisions, and algorithmic complexity.",'
        f'"results":"120-160 word empirical benchmarks, baseline comparative evaluations, and ablation studies.",'
        f'"conclusion":"90-120 word conclusion with practical deployment trade-offs and future research avenues.",'
        f'"keywords":["{concept}","machine learning","deep learning","neural networks","optimization"],'
        f'"references":["[1] Vaswani, A., et al. Attention Is All You Need. NeurIPS, 2017.","[2] He, K., et al. Deep Residual Learning. CVPR, 2016.","[3] Goodfellow, I., et al. Deep Learning. MIT Press, 2016.","[4] Devlin, J., et al. BERT. NAACL, 2019.","[5] Kingma, D. P., and Ba, J. Adam. ICLR, 2015."],'
        f'"key_takeaways":["Key finding 1","Key finding 2","Key finding 3"],'
        f'"bibtex":"@article{{author2026, title={{...}}, journal={{IEEE TA}}, year={{2026}}}}"}}'
    )

    try:
        doc = ask_groq_json(paper_prompt)

        raw_title = doc.get('title') or f'{concept}: A {type_desc}'
        raw_title = _re.sub(r'[#\[\]`*]', '', raw_title).strip()

        clean_slug = _re.sub(r'[^a-zA-Z0-9]', '', concept).lower()[:12] or 'paper'
        fallback_bib = (
            f"@article{{{clean_slug}2026survey,\n"
            f"  title={{{raw_title}}},\n"
            f"  author={{LearnSphere Research Lab and Academic Contributors}},\n"
            f"  journal={{IEEE Transactions on Artificial Intelligence}},\n"
            f"  volume={{14}},\n"
            f"  number={{2}},\n"
            f"  pages={{105--124}},\n"
            f"  year={{2026}}\n"
            f"}}"
        )

        result = {
            'title':         raw_title,
            'paper_type':    paper_type,
            'abstract':      doc.get('abstract') or f'This paper presents a formal analysis of {concept} in machine learning systems.',
            'introduction':  doc.get('introduction') or f'The field of {concept} has experienced rapid development with foundational algorithmic breakthroughs.',
            'theory_math':   doc.get('theory_math') or f'Theoretical formulation establishes the loss minimization objective and probabilistic bounds for {concept}.',
            'methodology':   doc.get('methodology') or f'The system architecture leverages parameterized optimization and neural feature representations.',
            'results':       doc.get('results') or f'Empirical evaluation benchmarks demonstrate significant gains in convergence speed and generalization.',
            'conclusion':    doc.get('conclusion') or f'In conclusion, this investigation synthesizes critical advancements in {concept} and charts directions for scalable deployment.',
            'keywords':      doc.get('keywords', [concept, 'machine learning', 'deep learning', 'neural networks', 'optimization']),
            'references':    doc.get('references', [
                '[1] Vaswani, A., et al. Attention Is All You Need. NeurIPS, 2017.',
                '[2] He, K., et al. Deep Residual Learning for Image Recognition. CVPR, 2016.',
                '[3] Goodfellow, I., et al. Deep Learning. MIT Press, 2016.',
                '[4] Sutton, R. S., and Barto, A. G. Reinforcement Learning: An Introduction. MIT Press, 2018.',
                '[5] Kingma, D. P., and Ba, J. Adam: A Method for Stochastic Optimization. ICLR, 2015.'
            ]),
            'key_takeaways': doc.get('key_takeaways', [
                f'Establishes key theoretical bounds and objective functions for {concept}.',
                'Demonstrates competitive empirical performance across standardized benchmark tasks.',
                'Identifies scalability limitations and outlines actionable future research avenues.'
            ]),
            'bibtex': doc.get('bibtex') or fallback_bib
        }

        update_profile(concept)
        _write_dna_entry(concept, correct=True, resp_ms=5000, style='balanced')
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': 'Research generation failed: ' + str(e)})


@app.route('/api/research/arxiv_search', methods=['GET', 'POST'])
@login_required
def research_arxiv_search():
    import urllib.request
    import urllib.parse
    import ssl
    import xml.etree.ElementTree as ET

    data  = (request.json if request.is_json else request.args) or {}
    query = (data.get('query') or data.get('q') or '').strip()
    limit = int(data.get('limit', 6))

    # Landmark curated catalogue for instant reference and offline safety
    landmark_papers = [
        {
            'arxiv_id': '1706.03762',
            'title': 'Attention Is All You Need',
            'authors': 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones et al.',
            'published': '2017-06-12',
            'summary': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
            'categories': ['cs.CL', 'cs.LG'],
            'pdf_url': 'https://arxiv.org/pdf/1706.03762.pdf',
            'link': 'https://arxiv.org/abs/1706.03762'
        },
        {
            'arxiv_id': '1512.03385',
            'title': 'Deep Residual Learning for Image Recognition (ResNet)',
            'authors': 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun',
            'published': '2015-12-10',
            'summary': 'Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously by explicitly reformulating layers as learning residual functions with reference to layer inputs.',
            'categories': ['cs.CV'],
            'pdf_url': 'https://arxiv.org/pdf/1512.03385.pdf',
            'link': 'https://arxiv.org/abs/1512.03385'
        },
        {
            'arxiv_id': '2106.09685',
            'title': 'LoRA: Low-Rank Adaptation of Large Language Models',
            'authors': 'Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li et al.',
            'published': '2021-06-17',
            'summary': 'An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks. LoRA freezes pre-trained model weights and injects trainable rank decomposition matrices into each Transformer layer.',
            'categories': ['cs.CL', 'cs.AI', 'cs.LG'],
            'pdf_url': 'https://arxiv.org/pdf/2106.09685.pdf',
            'link': 'https://arxiv.org/abs/2106.09685'
        },
        {
            'arxiv_id': '2312.00752',
            'title': 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
            'authors': 'Albert Gu, Tri Dao',
            'published': '2023-12-01',
            'summary': 'Foundation models now power most applications in AI. We propose Mamba, a new class of selective state space models that solves the computational inefficiency of Transformers on long sequences with fast linear-time sequence processing.',
            'categories': ['cs.LG', 'cs.AI'],
            'pdf_url': 'https://arxiv.org/pdf/2312.00752.pdf',
            'link': 'https://arxiv.org/abs/2312.00752'
        },
        {
            'arxiv_id': '2112.10752',
            'title': 'High-Resolution Image Synthesis with Latent Diffusion Models',
            'authors': 'Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer',
            'published': '2021-12-20',
            'summary': 'By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models achieve state-of-the-art synthesis results. We apply them in the latent space of powerful pretrained autoencoders (Stable Diffusion).',
            'categories': ['cs.CV', 'cs.LG'],
            'pdf_url': 'https://arxiv.org/pdf/2112.10752.pdf',
            'link': 'https://arxiv.org/abs/2112.10752'
        },
        {
            'arxiv_id': '2305.18290',
            'title': 'Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model',
            'authors': 'Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn',
            'published': '2023-05-29',
            'summary': 'While large-scale language models learn broad world knowledge, controlling their behavior is difficult. DPO demonstrates that the constrained RLHF problem can be optimized exactly with a simple binary cross-entropy loss, bypassing reward model fitting entirely.',
            'categories': ['cs.LG', 'cs.AI'],
            'pdf_url': 'https://arxiv.org/pdf/2305.18290.pdf',
            'link': 'https://arxiv.org/abs/2305.18290'
        }
    ]

    if not query:
        return jsonify({'success': True, 'query': '', 'papers': landmark_papers[:limit]})

    # Try live arXiv API search
    try:
        ctx = ssl._create_unverified_context()
        enc_q = urllib.parse.quote(query)
        url = f'http://export.arxiv.org/api/query?search_query=all:{enc_q}&start=0&max_results={limit}'
        req = urllib.request.Request(url, headers={'User-Agent': 'LearnSphere/1.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=7) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            papers = []
            for e in entries:
                t = e.find('atom:title', ns)
                s = e.find('atom:summary', ns)
                p = e.find('atom:published', ns)
                i = e.find('atom:id', ns)
                if t is None or s is None or i is None:
                    continue
                title = t.text.strip().replace('\n', ' ')
                summary = s.text.strip().replace('\n', ' ')
                published = p.text.strip()[:10] if p is not None else ''
                link = i.text.strip()
                raw_id = link.split('/abs/')[-1] if '/abs/' in link else link.split('/')[-1]
                authors = [a.find('atom:name', ns).text.strip() for a in e.findall('atom:author', ns) if a.find('atom:name', ns) is not None]
                papers.append({
                    'arxiv_id': raw_id,
                    'title': title,
                    'summary': summary,
                    'published': published,
                    'link': link,
                    'pdf_url': f'https://arxiv.org/pdf/{raw_id}.pdf',
                    'authors': ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else '')
                })
            if papers:
                return jsonify({'success': True, 'query': query, 'papers': papers})
    except Exception:
        pass

    # If live search fails or has no hits, filter curated catalog
    matched = [p for p in landmark_papers if query.lower() in p['title'].lower() or query.lower() in p['summary'].lower()]
    return jsonify({'success': True, 'query': query, 'papers': (matched or landmark_papers)[:limit]})


@app.route('/api/research/explain_paper', methods=['POST'])
@login_required
def research_explain_paper():
    data     = request.json or {}
    title    = data.get('title', '').strip()
    abstract = data.get('abstract', '').strip()
    if not title:
        return jsonify({'success': False, 'error': 'Paper title is required.'})

    prompt = (
        f'You are an expert ML educator. Explain this academic research paper for a computer science student:\n'
        f'Title: "{title}"\n'
        f'Abstract: "{abstract}"\n\n'
        f'Respond with ONLY valid JSON with no markdown:\n'
        f'{{"core_intuition":"2-3 clear sentences explaining the central intuition in plain English without jargon.",'
        f'"architectural_breakthrough":"1-2 sentences on what mathematical or engineering technique was invented.",'
        f'"strengths":["Key practical advantage 1","Key practical advantage 2"],'
        f'"limitations":["Important constraint or limitation 1","Important constraint or limitation 2"],'
        f'"experiment_idea":"A small reproducible PyTorch or Scikit-learn experiment a student can code today to verify this idea."}}'
    )

    try:
        explanation = ask_groq_json(prompt)
        return jsonify({'success': True, 'data': explanation})
    except Exception as e:
        return jsonify({'success': False, 'error': 'Paper explanation failed: ' + str(e)})


@app.route('/api/research/compare', methods=['POST'])
@login_required
def research_compare():
    data      = request.json or {}
    concept_a = data.get('concept_a', '').strip()
    concept_b = data.get('concept_b', '').strip()
    if not concept_a or not concept_b:
        return jsonify({'success': False, 'error': 'Please provide two concepts or architectures to compare.'})

    prompt = (
        f'Generate an academic comparison matrix and literature synthesis between "{concept_a}" and "{concept_b}".\n'
        f'Return ONLY valid JSON matching this schema with no markdown:\n'
        f'{{"concept_a":"{concept_a}",'
        f'"concept_b":"{concept_b}",'
        f'"summary":"2-3 sentence technical overview comparing both approaches.",'
        f'"dimensions":['
        f'{{"factor":"Time Complexity & Scalability","a":"...","b":"...","advantage":"{concept_a} or {concept_b}"}},'
        f'{{"factor":"Memory & Hardware Footprint","a":"...","b":"...","advantage":"..."}},'
        f'{{"factor":"Sample Efficiency & Training Stability","a":"...","b":"...","advantage":"..."}},'
        f'{{"factor":"Inductive Bias & Generalization","a":"...","b":"...","advantage":"..."}},'
        f'{{"factor":"Inference Latency & Deployment","a":"...","b":"...","advantage":"..."}}],'
        f'"when_to_use_a":"Clear engineering recommendations on when to choose {concept_a}.",'
        f'"when_to_use_b":"Clear engineering recommendations on when to choose {concept_b}."}}'
    )

    try:
        comparison = ask_groq_json(prompt)
        return jsonify({'success': True, 'data': comparison})
    except Exception as e:
        return jsonify({'success': False, 'error': 'Comparative matrix failed: ' + str(e)})


@app.route('/api/research/proposal', methods=['POST'])
@login_required
def research_proposal():
    data      = request.json or {}
    topic     = data.get('topic', '').strip()
    objective = data.get('objective', '').strip()
    if not topic:
        return jsonify({'success': False, 'error': 'Please provide a research question or topic.'})

    prompt = (
        f'Generate a formal ML academic thesis and experiment proposal for:\n'
        f'Topic: "{topic}"\n'
        f'Objective / Problem Statement: "{objective}"\n\n'
        f'Return ONLY valid JSON with no markdown:\n'
        f'{{"title":"Formal academic thesis title",'
        f'"h0":"Null Hypothesis (H0) formulated precisely",'
        f'"h1":"Alternative Hypothesis (H1) formulated precisely",'
        f'"independent_vars":["Variable 1","Variable 2"],'
        f'"dependent_vars":["Metric 1","Metric 2"],'
        f'"datasets":["Benchmark Dataset 1","Benchmark Dataset 2"],'
        f'"baselines":["Baseline Algorithm 1","Baseline Model 2"],'
        f'"ablation_plan":["Ablation study 1","Ablation study 2","Ablation study 3"],'
        f'"metrics":["Evaluation metric 1","Evaluation metric 2"],'
        f'"starter_code":"# Complete minimal PyTorch experiment scaffolding code\\nimport torch\\nimport torch.nn as nn\\n..."}}'
    )

    try:
        proposal = ask_groq_json(prompt)
        return jsonify({'success': True, 'data': proposal})
    except Exception as e:
        return jsonify({'success': False, 'error': 'Proposal generation failed: ' + str(e)})


@app.route('/api/virtual_lab', methods=['POST'])
@login_required
def virtual_lab():
    import re as _re
    data       = request.json or {}
    experiment = data.get('experiment', '').strip()
    if not experiment:
        return jsonify({'success': False, 'error': 'Please describe an experiment.'})
    level = get_user_profile().get('level', 'beginner')

    meta_prompt = (
        'You are designing a virtual ML lab experiment for: "' + experiment + '"\n'
        'Student level: ' + level + '\n'
        'Respond with ONLY a valid JSON object. Use \\n for newlines inside strings. No code in this JSON.\n'
        'Required keys:\n'
        'title: experiment title\n'
        'objective: 1-2 sentences on what student will learn\n'
        'dataset_description: describe synthetic dataset (size, features, labels)\n'
        'hypothesis: what we expect to observe\n'
        'interpretation: 2-3 paragraphs explaining results meaning\n'
        'key_observations: array of 4-5 specific things to notice\n'
        'conclusion: 1-2 sentence summary\n'
        'further_experiments: array of 2-3 follow-up experiments\n'
        'Example: {"title":"Overfitting Demo","objective":"Observe model complexity effect","dataset_description":"200 samples with Gaussian noise","hypothesis":"High complexity will overfit","interpretation":"Training error falls while validation rises...","key_observations":["Training loss drops steadily","Validation loss rises after epoch 10"],"conclusion":"Regularization prevents overfitting","further_experiments":["Try dropout","Add more training data"]}'
    )

    code_prompt = (
        'Write complete runnable Python code for this ML lab experiment: "' + experiment + '"\n'
        'Student level: ' + level + '\n'
        'Rules:\n'
        '1. Generate synthetic dataset using only numpy and sklearn (NO file I/O)\n'
        '2. Train model(s) demonstrating the concept clearly\n'
        '3. Print clearly labelled metrics\n'
        '4. Create matplotlib figure, then convert to base64 exactly like this:\n'
        '   import io, base64\n'
        '   buf = io.BytesIO()\n'
        '   plt.tight_layout()\n'
        '   plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")\n'
        '   buf.seek(0)\n'
        '   img_b64 = base64.b64encode(buf.read()).decode()\n'
        '   print("PLOT_B64:" + img_b64)\n'
        '5. Only use: numpy, sklearn, matplotlib, io, base64\n'
        '6. Return ONLY raw Python code, no markdown fences, no explanation\n'
        '7. Add clear student-friendly comments on each step\n'
    )

    try:
        meta = ask_groq_json(meta_prompt)
        code_text = ask_groq(
            code_prompt,
            system="You are an expert ML educator. Write clean well-commented Python experiment code. Output ONLY raw Python, no markdown.",
            temperature=0.2
        )
        code_text = _re.sub(r"```python\s*", "", code_text.strip())
        code_text = _re.sub(r"```", "", code_text).strip()

        # Execute code in isolated subprocess to generate live Matplotlib plot
        plot_b64 = None
        try:
            import subprocess, sys
            exec_code = "import matplotlib\nmatplotlib.use('Agg')\n" + code_text
            res = subprocess.run(
                [sys.executable, '-c', exec_code],
                capture_output=True,
                text=True,
                timeout=7
            )
            stdout = res.stdout or ""
            if "PLOT_B64:" in stdout:
                parts = stdout.split("PLOT_B64:")
                plot_b64 = parts[1].strip().split()[0]
        except Exception:
            plot_b64 = None

        meta.setdefault('title',               experiment)
        meta.setdefault('objective',           '')
        meta.setdefault('dataset_description', '')
        meta.setdefault('hypothesis',          '')
        meta.setdefault('interpretation',      '')
        meta.setdefault('key_observations',    [])
        meta.setdefault('conclusion',          '')
        meta.setdefault('further_experiments', [])
        meta['code'] = code_text
        meta['plot_b64'] = plot_b64

        update_profile(experiment)
        return jsonify({'success': True, 'data': meta})
    except Exception as e:
        return jsonify({'success': False, 'error': 'Virtual lab failed: ' + str(e)})


# ─── FOCUS REWARD ENGINE ──────────────────────────────────────────────────────

@app.route('/focus')
@login_required
def focus_page():
    return render_template('focus.html')

@app.route('/events')
@login_required
def events_page():
    return render_template('events.html')

@app.route('/career')
@login_required
def career_page():
    return render_template('career.html')

# ── Focus session API ─────────────────────────────────────────────────────────

@app.route('/api/focus/start', methods=['POST'])
@login_required
def focus_start():
    data  = request.json or {}
    topic = data.get('topic', 'General Study')
    sid   = str(uuid.uuid4())
    conn  = get_db()
    conn.execute(
        "INSERT INTO focus_sessions (id,user_id,topic,status) VALUES (?,?,?,?)",
        (sid, session['user_id'], topic, 'active')
    )
    # Ensure reward row exists
    conn.execute(
        "INSERT OR IGNORE INTO user_rewards (user_id) VALUES (?)",
        (session['user_id'],)
    )
    conn.commit(); conn.close()
    return jsonify({'success': True, 'session_id': sid})

@app.route('/api/focus/end', methods=['POST'])
@login_required
def focus_end():
    data    = request.json or {}
    sid     = data.get('session_id', '')
    mins    = int(data.get('duration_mins', 0))
    # 1 coin per minute, bonus 10 coins for 25+ min (Pomodoro)
    coins   = mins + (10 if mins >= 25 else 0)

    conn    = get_db()
    conn.execute(
        "UPDATE focus_sessions SET ended_at=datetime('now'), duration_mins=?, coins_earned=?, status='done' WHERE id=? AND user_id=?",
        (mins, coins, sid, session['user_id'])
    )
    conn.execute(
        "UPDATE user_rewards SET coins=coins+?, total_focus_mins=total_focus_mins+?, last_study_date=date('now') WHERE user_id=?",
        (coins, mins, session['user_id'])
    )

    # Badge logic
    row = conn.execute("SELECT * FROM user_rewards WHERE user_id=?", (session['user_id'],)).fetchone()
    badges = json.loads(row['badges']) if row and row['badges'] else []
    total_mins = (row['total_focus_mins'] or 0) if row else 0
    new_badges = []
    if total_mins >= 60  and 'Focus Novice'     not in badges: new_badges.append('Focus Novice')
    if total_mins >= 300 and 'Focus Pro'        not in badges: new_badges.append('Focus Pro')
    if total_mins >= 600 and 'Focus Master'     not in badges: new_badges.append('Focus Master')
    if mins       >= 25  and 'Pomodoro Champ'   not in badges: new_badges.append('Pomodoro Champ')
    if badges.count('session') >= 7 and 'Week Warrior' not in badges: new_badges.append('Week Warrior')
    if new_badges:
        badges.extend(new_badges)
        conn.execute("UPDATE user_rewards SET badges=? WHERE user_id=?", (json.dumps(badges), session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'coins': coins, 'new_badges': new_badges})

@app.route('/api/focus/stats')
@login_required
def focus_stats():
    conn = get_db()
    row  = conn.execute("SELECT * FROM user_rewards WHERE user_id=?", (session['user_id'],)).fetchone()
    sessions = conn.execute(
        "SELECT * FROM focus_sessions WHERE user_id=? ORDER BY started_at DESC LIMIT 10",
        (session['user_id'],)
    ).fetchall()
    # Leaderboard
    lb = conn.execute("""
        SELECT u.username, r.coins, r.total_focus_mins, r.badges
        FROM user_rewards r JOIN users u ON r.user_id=u.id
        ORDER BY r.coins DESC LIMIT 10
    """).fetchall()
    conn.close()
    return jsonify({
        'success': True,
        'rewards': dict(row) if row else {'coins':0,'total_focus_mins':0,'streak_days':0,'badges':'[]'},
        'sessions': [dict(s) for s in sessions],
        'leaderboard': [dict(r) for r in lb]
    })

# ── Events / Opportunities API ────────────────────────────────────────────────

@app.route('/api/events')
@login_required
def get_events():
    """Return real curated events list from database / catalog."""
    from events_catalog import REAL_EVENTS, init_events_db
    conn = get_db()
    try:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if count == 0:
            init_events_db(conn)
        rows = conn.execute("SELECT * FROM events ORDER BY rowid ASC").fetchall()
        events_list = [dict(r) for r in rows]
    except Exception:
        events_list = []
    finally:
        conn.close()

    if not events_list:
        events_list = REAL_EVENTS

    return jsonify({'success': True, 'events': events_list})


@app.route('/api/events/recommend', methods=['POST'])
@login_required
def recommend_events():
    """AI recommends which real competitions and hackathons suit this student."""
    from events_catalog import get_personalized_event_recommendations
    p = get_user_profile()
    topics = p.get('topics_learned', [])[-10:]
    weak = p.get('weak_topics', [])[:5]
    level = p.get('level', 'beginner')
    prompt = (
        f"A {level} ML student has studied: {', '.join(topics) or 'Foundational ML'}. "
        f"Weak areas: {', '.join(weak) or 'none'}. "
        "Recommend 4-5 real, specific competitions/hackathons/challenges they should enter right now from Kaggle, MLH, LeetCode, Devpost, or HackerRank. "
        "For each provide: title, platform, url, reason, difficulty, and type. "
        "Respond with ONLY valid JSON: "
        '{"recommendations":[{"title":"Kaggle: Titanic — Machine Learning from Disaster","platform":"Kaggle","url":"https://www.kaggle.com/competitions/titanic","reason":"Directly applies classification to tabular data","type":"competition","difficulty":"beginner"}]}'
    )
    try:
        result = ask_groq_json(prompt)
        if result and isinstance(result, dict) and 'recommendations' in result and len(result['recommendations']) > 0:
            return jsonify({'success': True, 'data': result})
    except Exception:
        pass

    # Seamless reliable fallback using profile-aware heuristic recommender
    fallback_recs = get_personalized_event_recommendations(p)
    return jsonify({'success': True, 'data': {'recommendations': fallback_recs}})

# ── Career Tracker API ────────────────────────────────────────────────────────

@app.route('/api/career/entries')
@login_required
def career_entries():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM career_entries WHERE user_id=? ORDER BY date_earned DESC",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return jsonify({'success': True, 'entries': [dict(r) for r in rows]})

@app.route('/api/career/add', methods=['POST'])
@login_required
def career_add():
    data = request.json or {}
    eid  = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO career_entries (id,user_id,entry_type,title,description,date_earned,metadata) VALUES (?,?,?,?,?,?,?)",
        (eid, session['user_id'], data.get('entry_type','skill'), data.get('title',''),
         data.get('description',''), data.get('date_earned', datetime.now().strftime('%Y-%m-%d')),
         json.dumps(data.get('metadata', {})))
    )
    conn.commit(); conn.close()
    return jsonify({'success': True, 'id': eid})

@app.route('/api/career/delete/<entry_id>', methods=['DELETE'])
@login_required
def career_delete(entry_id):
    conn = get_db()
    conn.execute("DELETE FROM career_entries WHERE id=? AND user_id=?", (entry_id, session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'success': True})

@app.route('/api/career/ai_summary')
@login_required
def career_ai_summary():
    """AI generates a professional career growth summary."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM career_entries WHERE user_id=? ORDER BY date_earned DESC",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    entries = [dict(r) for r in rows]
    p = get_user_profile()
    if not entries and not p.get('topics_learned'):
        return jsonify({'success': True, 'summary': 'No career entries yet. Start by adding your skills, certificates, and competitions!'})
    prompt = (
        'Generate a professional career growth summary for an ML student. '
        'Topics studied: ' + ', '.join(p.get('topics_learned', [])[-15:]) + '. '
        'Career entries: ' + ', '.join([e['title'] for e in entries[:10]]) + '. '
        'Level: ' + p.get('level', 'beginner') + '. '
        'Write 2-3 paragraphs highlighting strengths, growth, and next career steps. Be encouraging and specific.'
    )
    try:
        summary = ask_groq(prompt, system="You are a career counselor for ML engineers. Write professional, encouraging career summaries.")
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─── ASSESSMENT API ───────────────────────────────────────────────────────────

@app.route('/api/assessments/<course_id>', methods=['GET'])
@login_required
def get_assessments(course_id):
    conn = get_db()
    rows = conn.execute("SELECT * FROM assessments WHERE course_id=? ORDER BY created_at", (course_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/assessments', methods=['POST'])
@educator_required
def create_assessment():
    data = request.json or {}
    aid  = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO assessments (id,course_id,title,description,pass_score) VALUES (?,?,?,?,?)",
        (aid, data['course_id'], data.get('title','Quiz'), data.get('description',''), data.get('pass_score',60))
    )
    conn.commit(); conn.close()
    return jsonify({'success': True, 'id': aid})

@app.route('/api/assessments/<assessment_id>/questions', methods=['GET'])
@login_required
def get_questions(assessment_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM assessment_questions WHERE assessment_id=? ORDER BY order_num", (assessment_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/assessments/<assessment_id>/questions', methods=['POST'])
@educator_required
def add_question(assessment_id):
    data = request.json or {}
    qid  = str(uuid.uuid4())
    conn = get_db()
    # Get next order_num
    max_order = conn.execute(
        "SELECT MAX(order_num) FROM assessment_questions WHERE assessment_id=?", (assessment_id,)
    ).fetchone()[0] or 0
    conn.execute(
        """INSERT INTO assessment_questions
           (id,assessment_id,question_type,question_text,options,correct_answer,explanation,points,order_num)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (qid, assessment_id, data.get('question_type','single_choice'), data.get('question_text',''),
         json.dumps(data.get('options', [])), data.get('correct_answer',''),
         data.get('explanation',''), data.get('points', 10), max_order + 1)
    )
    conn.commit(); conn.close()
    return jsonify({'success': True, 'id': qid})

@app.route('/api/questions/<question_id>', methods=['DELETE'])
@educator_required
def delete_question(question_id):
    conn = get_db()
    conn.execute("DELETE FROM assessment_questions WHERE id=?", (question_id,))
    conn.commit(); conn.close()
    return jsonify({'success': True})

@app.route('/api/assessments/<assessment_id>/submit', methods=['POST'])
@login_required
def submit_assessment(assessment_id):
    data    = request.json or {}
    answers = data.get('answers', {})  # {question_id: student_answer}

    conn = get_db()
    assessment = conn.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
    questions  = conn.execute(
        "SELECT * FROM assessment_questions WHERE assessment_id=? ORDER BY order_num", (assessment_id,)
    ).fetchall()

    score = 0; max_score = 0; results = {}
    for q in questions:
        max_score += q['points']
        student_ans = str(answers.get(q['id'], '')).strip().lower()
        correct_ans = str(q['correct_answer'] or '').strip().lower()
        correct = False

        if q['question_type'] in ('single_choice', 'true_false'):
            correct = student_ans == correct_ans
        elif q['question_type'] == 'multiple_choice':
            # Both are comma-separated sorted lists
            s_set = set(x.strip() for x in student_ans.split(',') if x.strip())
            c_set = set(x.strip() for x in correct_ans.split(',') if x.strip())
            correct = s_set == c_set
        elif q['question_type'] in ('text_answer', 'fill_blank'):
            correct = student_ans in correct_ans or correct_ans in student_ans
        else:
            correct = student_ans == correct_ans

        if correct:
            score += q['points']
        results[q['id']] = {
            'correct': correct,
            'student_answer': answers.get(q['id'], ''),
            'correct_answer': q['correct_answer'],
            'explanation': q['explanation'],
            'points_earned': q['points'] if correct else 0,
        }

    pass_score = assessment['pass_score'] if assessment else 60
    passed     = max_score > 0 and (score / max_score * 100) >= pass_score
    pct        = round(score / max_score * 100) if max_score > 0 else 0

    attempt_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO assessment_attempts (id,assessment_id,student_id,score,max_score,passed,answers) VALUES (?,?,?,?,?,?,?)",
        (attempt_id, assessment_id, session['user_id'], score, max_score, 1 if passed else 0, json.dumps(results))
    )
    conn.commit()

    # Check if all lectures completed AND quiz passed → issue certificate
    cert_issued = False
    if passed and assessment:
        course_id  = assessment['course_id']
        total_lecs = conn.execute("SELECT COUNT(*) FROM lectures WHERE course_id=?", (course_id,)).fetchone()[0]
        done_lecs  = conn.execute(
            "SELECT COUNT(*) FROM completed_lectures cl JOIN lectures l ON cl.lecture_id=l.id WHERE cl.student_id=? AND l.course_id=?",
            (session['user_id'], course_id)
        ).fetchone()[0]

        if done_lecs >= total_lecs and total_lecs > 0:
            existing_cert = conn.execute(
                "SELECT id FROM certificates WHERE student_id=? AND course_id=?",
                (session['user_id'], course_id)
            ).fetchone()
            if not existing_cert:
                import hashlib as _hl
                cert_code = 'LS-' + _hl.sha256(f"{session['user_id']}{course_id}{datetime.now().isoformat()}".encode()).hexdigest()[:12].upper()
                course = conn.execute("SELECT title FROM courses WHERE id=?", (course_id,)).fetchone()
                user   = conn.execute("SELECT username FROM users WHERE id=?", (session['user_id'],)).fetchone()
                conn.execute(
                    "INSERT OR IGNORE INTO certificates (id,student_id,course_id,student_name,course_title,cert_code) VALUES (?,?,?,?,?,?)",
                    (str(uuid.uuid4()), session['user_id'], course_id,
                     user['username'] if user else 'Student',
                     course['title'] if course else 'Course', cert_code)
                )
                conn.commit()
                cert_issued = True

    conn.close()
    return jsonify({
        'success': True,
        'score': score, 'max_score': max_score, 'percentage': pct,
        'passed': passed, 'pass_score': pass_score,
        'results': results, 'cert_issued': cert_issued
    })

# ─── CERTIFICATE API ──────────────────────────────────────────────────────────

@app.route('/api/certificate/<course_id>')
@login_required
def get_certificate(course_id):
    conn = get_db()
    cert = conn.execute(
        "SELECT * FROM certificates WHERE student_id=? AND course_id=?",
        (session['user_id'], course_id)
    ).fetchone()
    conn.close()
    if cert:
        return jsonify({'success': True, 'certificate': dict(cert)})
    return jsonify({'success': False, 'error': 'No certificate found'})

@app.route('/api/my_certificates')
@login_required
def my_certificates():
    conn = get_db()
    certs = conn.execute(
        "SELECT * FROM certificates WHERE student_id=? ORDER BY issued_at DESC",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return jsonify({'success': True, 'certificates': [dict(c) for c in certs]})

@app.route('/verify/<cert_code>')
def verify_certificate(cert_code):
    conn = get_db()
    cert = conn.execute("SELECT * FROM certificates WHERE cert_code=?", (cert_code,)).fetchone()
    if not cert:
        conn.close()
        return "<h2>Certificate Not Found</h2><p>Code: " + cert_code + " is not valid.</p>", 404

    course = conn.execute("SELECT * FROM courses WHERE id=?", (cert['course_id'],)).fetchone()
    user = conn.execute("SELECT * FROM users WHERE id=?", (cert['student_id'],)).fetchone()
    conn.close()

    cert_data = dict(cert)
    issued_date = cert_data.get('issued_at', '')
    formatted_date = issued_date
    try:
        if issued_date:
            from datetime import datetime
            dt = datetime.strptime(issued_date[:10], '%Y-%m-%d')
            formatted_date = dt.strftime('%B %d, %Y')
    except Exception:
        pass

    return render_template('certificate_verify.html', 
                           cert=cert_data, 
                           course=dict(course) if course else {},
                           user=dict(user) if user else {},
                           formatted_date=formatted_date)

@app.route('/certificate/<course_id>/download')
@login_required
def download_certificate(course_id):
    """Generate and return a printable HTML certificate."""
    conn = get_db()
    cert = conn.execute(
        "SELECT * FROM certificates WHERE student_id=? AND course_id=?",
        (session['user_id'], course_id)
    ).fetchone()
    if not cert:
        conn.close()
        return "Certificate not found. Complete all lessons and pass the quiz first.", 404

    course = conn.execute("SELECT * FROM courses WHERE id=?", (course_id,)).fetchone()
    user = conn.execute("SELECT * FROM users WHERE id=?", (session['user_id'],)).fetchone()
    conn.close()

    cert_data = dict(cert)
    issued_date = cert_data.get('issued_at', '')
    formatted_date = issued_date
    try:
        if issued_date:
            from datetime import datetime
            dt = datetime.strptime(issued_date[:10], '%Y-%m-%d')
            formatted_date = dt.strftime('%B %d, %Y')
    except Exception:
        pass

    return render_template('certificate.html', 
                           cert=cert_data, 
                           course=dict(course) if course else {},
                           user=dict(user) if user else {},
                           formatted_date=formatted_date)

# ─── RATINGS API ─────────────────────────────────────────────────────────────

@app.route('/api/courses/<course_id>/rate', methods=['POST'])
@login_required
def rate_course(course_id):
    data = request.json or {}
    rid  = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO course_ratings (id,course_id,student_id,rating,review) VALUES (?,?,?,?,?)",
        (rid, course_id, session['user_id'], int(data.get('rating', 5)), data.get('review', ''))
    )
    conn.commit(); conn.close()
    return jsonify({'success': True})

@app.route('/api/courses/<course_id>/rating', methods=['GET'])
@login_required
def get_course_rating(course_id):
    conn = get_db()
    rows  = conn.execute("SELECT rating FROM course_ratings WHERE course_id=?", (course_id,)).fetchall()
    my    = conn.execute("SELECT rating,review FROM course_ratings WHERE course_id=? AND student_id=?",
                         (course_id, session['user_id'])).fetchone()
    conn.close()
    avg = round(sum(r['rating'] for r in rows) / len(rows), 1) if rows else 0
    return jsonify({'avg': avg, 'count': len(rows), 'my_rating': dict(my) if my else None})

# ─── EDUCATOR PAGES ───────────────────────────────────────────────────────────

@app.route('/educator/course/<course_id>/assessments')
@educator_required
def educator_assessments(course_id):
    return render_template('educator_assessments.html', course_id=course_id)

@app.route('/student/course/<course_id>/assessment/<assessment_id>')
@login_required
def take_assessment(course_id, assessment_id):
    return render_template('take_assessment.html', course_id=course_id, assessment_id=assessment_id)

# ─── COMMUNITY & SOCIAL PAGES ───────────────────────────────────────────────

@app.route('/community')
@login_required
def community_dashboard():
    return render_template('community.html')

@app.route('/u/<username>')
def public_profile(username):
    conn = get_db()
    user = conn.execute("SELECT id, username, level, college, region, bio, headline, interests, created_at, role FROM users WHERE username=?", (username,)).fetchone()
    if not user:
        conn.close()
        return "User not found", 404
    
    uid = user['id']
    certs = conn.execute("SELECT course_title, issued_at FROM certificates WHERE student_id=?", (uid,)).fetchall()
    battles = conn.execute("SELECT * FROM battle_scores WHERE user_id=? OR username=? ORDER BY played_at DESC LIMIT 5", (uid, username)).fetchall()
    careers = conn.execute("SELECT * FROM career_entries WHERE user_id=? ORDER BY date_earned DESC", (uid,)).fetchall()
    conn.close()
    
    return render_template('public_profile.html', 
                           profile=dict(user), 
                           certs=[dict(c) for c in certs], 
                           battles=[dict(b) for b in battles], 
                           careers=[dict(ce) for ce in careers])

# ─── COMMUNITY API ──────────────────────────────────────────────────────────

@app.route('/api/community/leaderboard')
def api_leaderboard():
    loc = request.args.get('filter', 'global')
    conn = get_db()
    query = """
        SELECT COALESCE(u.username, bs.username) AS username, 
               MAX(bs.score) AS score,
               COUNT(*) AS games_played,
               SUM(CASE WHEN bs.score > bs.ai_score THEN 1 ELSE 0 END) AS wins
        FROM battle_scores bs
        LEFT JOIN users u ON (bs.user_id = u.id OR bs.username = u.username)
        GROUP BY COALESCE(u.username, bs.username)
        ORDER BY score DESC, wins DESC LIMIT 15
    """
    params = ()
    if loc == 'college' and session.get('college'):
        query = """
            SELECT COALESCE(u.username, bs.username) AS username, 
                   MAX(bs.score) AS score,
                   COUNT(*) AS games_played,
                   SUM(CASE WHEN bs.score > bs.ai_score THEN 1 ELSE 0 END) AS wins
            FROM battle_scores bs 
            JOIN users u ON (bs.user_id = u.id OR bs.username = u.username) 
            WHERE u.college = ? 
            GROUP BY COALESCE(u.username, bs.username) 
            ORDER BY score DESC, wins DESC LIMIT 15
        """
        params = (session['college'],)
    elif loc == 'region' and session.get('region'):
        query = """
            SELECT COALESCE(u.username, bs.username) AS username, 
                   MAX(bs.score) AS score,
                   COUNT(*) AS games_played,
                   SUM(CASE WHEN bs.score > bs.ai_score THEN 1 ELSE 0 END) AS wins
            FROM battle_scores bs 
            JOIN users u ON (bs.user_id = u.id OR bs.username = u.username) 
            WHERE u.region = ? 
            GROUP BY COALESCE(u.username, bs.username) 
            ORDER BY score DESC, wins DESC LIMIT 15
        """
        params = (session['region'],)
    
    try:
        rows = conn.execute(query, params).fetchall()
    except Exception:
        rows = []
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/community/groups', methods=['GET', 'POST'])
@login_required
def api_groups():
    conn = get_db()
    if request.method == 'POST':
        data = request.json or {}
        name = (data.get('name') or '').strip()
        if not name:
            conn.close()
            return jsonify({'success': False, 'error': 'Group name is required'}), 400
        gid = str(uuid.uuid4())
        conn.execute("INSERT INTO study_groups (id, name, description, course_id, created_by) VALUES (?,?,?,?,?)",
                     (gid, name, data.get('description',''), data.get('course_id',''), session['user_id']))
        conn.execute("INSERT INTO study_group_members (group_id, user_id, username) VALUES (?,?,?)",
                     (gid, session['user_id'], session['username']))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'id': gid})
    else:
        # Check if groups are empty, seed starter study circles
        cnt = conn.execute("SELECT COUNT(*) FROM study_groups").fetchone()[0]
        if cnt == 0:
            starter_groups = [
                ("Machine Learning & AI Guild", "Weekly peer discussions on supervised models, neural architectures, and Kaggle benchmarks.", "Machine Learning"),
                ("Deep Learning & PyTorch Builders", "Hands-on neural network implementations, CNNs, Transformers, and paper implementations.", "Deep Learning"),
                ("DSA & Placement Interview Prep", "Collaborative problem solving, coding contest debriefs, and mock interview practice.", "Algorithms")
            ]
            for gname, gdesc, gcat in starter_groups:
                sg_id = str(uuid.uuid4())
                conn.execute("INSERT INTO study_groups (id, name, description, course_id, created_by) VALUES (?,?,?,?,?)",
                             (sg_id, gname, gdesc, gcat, 'system'))
                conn.execute("INSERT INTO study_group_members (group_id, user_id, username) VALUES (?,?,?)",
                             (sg_id, 'system', 'LearnSphere Moderator'))
            conn.commit()

        groups = conn.execute("""
            SELECT g.*, (SELECT COUNT(*) FROM study_group_members WHERE group_id=g.id) as count 
            FROM study_groups g 
            ORDER BY g.rowid DESC
        """).fetchall()
        my_groups = conn.execute("SELECT group_id FROM study_group_members WHERE user_id=?", (session['user_id'],)).fetchall()
        conn.close()
        return jsonify({'groups': [dict(g) for g in groups], 'my_groups': [r['group_id'] for r in my_groups]})

@app.route('/api/community/groups/<group_id>/join', methods=['POST'])
@login_required
def join_group(group_id):
    conn = get_db()
    try:
        exists = conn.execute("SELECT 1 FROM study_group_members WHERE group_id=? AND user_id=?", (group_id, session['user_id'])).fetchone()
        if not exists:
            conn.execute("INSERT INTO study_group_members (group_id, user_id, username) VALUES (?,?,?)",
                         (group_id, session['user_id'], session['username']))
            conn.commit()
    except Exception:
        pass
    conn.close()
    return jsonify({'success': True})

@app.route('/api/community/groups/<group_id>/leave', methods=['POST'])
@login_required
def leave_group(group_id):
    conn = get_db()
    conn.execute("DELETE FROM study_group_members WHERE group_id=? AND user_id=?", (group_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/community/forums/<course_id>', methods=['GET', 'POST'])
@login_required
def api_forums(course_id):
    conn = get_db()
    if request.method == 'POST':
        data = request.json or {}
        fid = str(uuid.uuid4())
        conn.execute("INSERT INTO course_forums (id, course_id, user_id, username, content, parent_id) VALUES (?,?,?,?,?,?)",
                     (fid, course_id, session['user_id'], session['username'], data.get('content',''), data.get('parent_id')))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'id': fid})
    else:
        rows = conn.execute("SELECT * FROM course_forums WHERE course_id=? ORDER BY created_at ASC", (course_id,)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])

@app.route('/api/community/interviews', methods=['GET', 'POST'])
@login_required
def api_interviews():
    conn = get_db()
    if request.method == 'POST':
        data = request.json or {}
        topic = (data.get('topic') or '').strip()
        scheduled_at = (data.get('scheduled_at') or '').strip()
        meeting_url = (data.get('meeting_url') or '').strip()
        if not topic or not scheduled_at:
            conn.close()
            return jsonify({'success': False, 'error': 'Topic and scheduled time are required.'}), 400
        if not meeting_url:
            meeting_url = 'https://meet.google.com/new'
        iid = str(uuid.uuid4())
        conn.execute("INSERT INTO mock_interviews (id, host_id, host_name, topic, meeting_url, scheduled_at) VALUES (?,?,?,?,?,?)",
                     (iid, session['user_id'], session['username'], topic, meeting_url, scheduled_at))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'id': iid})
    else:
        open_rows = conn.execute("SELECT * FROM mock_interviews WHERE status='open' ORDER BY scheduled_at ASC").fetchall()
        my_rows = conn.execute("SELECT * FROM mock_interviews WHERE (peer_id=? OR host_id=?) ORDER BY scheduled_at DESC LIMIT 5",
                               (session['user_id'], session['user_id'])).fetchall()
        conn.close()
        return jsonify({
            'open_interviews': [dict(r) for r in open_rows],
            'my_interviews': [dict(r) for r in my_rows]
        })

@app.route('/api/community/interviews/<interview_id>/book', methods=['POST'])
@login_required
def book_interview(interview_id):
    conn = get_db()
    interview = conn.execute("SELECT host_id, status FROM mock_interviews WHERE id=?", (interview_id,)).fetchone()
    if not interview:
        conn.close()
        return jsonify({'success': False, 'error': 'Interview slot not found.'}), 404
    if interview['host_id'] == session['user_id']:
        conn.close()
        return jsonify({'success': False, 'error': 'You cannot book your own hosted slot.'}), 400
    if interview['status'] != 'open':
        conn.close()
        return jsonify({'success': False, 'error': 'This interview slot is already booked.'}), 400
    conn.execute("UPDATE mock_interviews SET peer_id=?, peer_name=?, status='booked' WHERE id=?",
                 (session['user_id'], session['username'], interview_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

try:
    init_db()  # ← runs on startup, initializes DB safely
except Exception as _init_err:
    print(f"[LearnSphere Startup] Notice during init_db: {_init_err}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
