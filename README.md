# RealJaenLab

RealJaenLab is a clean internal development application for football-video research.

All previous football-analysis research has been removed from the active tree. Development is restarting from zero; Git history remains the archive for earlier work.

The clean Supervisor shell contains three development workspaces:

- **VIDEO**: local media library, playback, seeking, and frame-by-frame navigation;
- **PREVIEW**: generic space for future result visualization;
- **CALIBRACIÓN**: generic space reserved for a future calibration approach.

- local media library and match/video management;
- lossless MP4 uploads and registration of local input videos;
- video playback, seeking, and frame-by-frame navigation.

No football-analysis or computer-vision functionality is currently implemented.

## Run locally

Install the dependencies from `supervisor/requirements.txt`, then run:

```bash
uvicorn supervisor.app:app --reload
```

Open the local URL printed by Uvicorn and import or organize an MP4 from the media library.
