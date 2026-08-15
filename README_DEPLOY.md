# 🌐 Free Cloud Hosting Guide: Pre-Market India Predictor

Your project is already git-initialized with `Procfile` and `requirements.txt` ready for 1-click cloud deployment.

---

## ⚡ Option 1: Permanent 24/7 Free Hosting on Render.com (Recommended)

1. **Create a free GitHub repository**:
   - Go to [github.com/new](https://github.com/new).
   - Repository name: `premarket-india-predictor`
   - Keep it **Public** or **Private** and click **Create repository**.

2. **Push your code to GitHub** (run this in terminal):
   ```bash
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/premarket-india-predictor.git
   git push -u origin main
   ```

3. **Deploy on Render**:
   - Sign up for free at [render.com](https://render.com/) (Sign in with GitHub).
   - Click **"New +" ➔ "Web Service"**.
   - Select your repository `premarket-india-predictor`.
   - Settings:
     - **Name:** `premarket-india-predictor`
     - **Environment:** `Python 3`
     - **Region:** `Singapore` (Lowest latency for India)
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
     - **Plan:** Free ($0 / month)
   - Click **"Deploy Web Service"**.
   - Your permanent website is live at `https://premarket-india-predictor.onrender.com`!

---

## 🚀 Option 2: Free 24/7 Hosting on Hugging Face Spaces (16 GB RAM)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Select **Blank / Docker** and give it a name.
3. Push your repository to the Hugging Face git remote.
4. Your space will build and run 24/7 with a free permanent `.hf.space` URL!
