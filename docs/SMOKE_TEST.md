# Spot-AI-fy manual chat smoke test (~2 minutes)

## Setup

1. Start the backend (`cd backend && uvicorn spot_backend.app:app --reload --port 8765` or your usual command).
2. Start the frontend (`cd frontend && npm run dev`) and open the Vite URL (typically http://127.0.0.1:5173).
3. Complete setup if needed, then **Connect Spotify** and sign in.
4. Open Spotify on a phone, desktop, or web player so at least one **Connect** device is available; save that device in **Model & Spotify → Device**.

## Steps (run in order)

| # | You type | Expected result |
|---|----------|-----------------|
| 1 | `Play John Mayer` | Music starts on your device; within ~3s the now-playing bar shows a John Mayer track. |
| 2 | `What's playing?` | The assistant names the **same** track shown in the bar. |
| 3 | `Skip` | The track changes; the bar updates to the new song. |
| 4 | `Pause` | Playback stops; the bar shows paused (play icon). |
| 5 | `Resume` | Playback resumes; the bar shows playing again. |
| 6 | `What are my playlists?` | The reply lists your real Spotify playlists (names from your account). |
| 7 | `Queue Gravity by John Mayer` | Expand the bar (click the bar body); **Gravity** appears in the up-next queue list. |
| 8 | Press **previous**, **play/pause**, and **next** on the bar once each | Each control works; send another short chat message afterward and confirm chat still responds. |
