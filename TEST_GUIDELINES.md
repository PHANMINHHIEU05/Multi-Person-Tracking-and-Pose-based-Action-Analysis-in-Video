Terminal 1 (backend)
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .venv/bin/activate
python -m uvicorn web.backend.app:app --reload --host 0.0.0.0 --port 8000

Terminal 2 (frontend)
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video/web/frontend
npm run dev
