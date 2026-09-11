# RealJaenLab

RealJaenLab is a clean internal development application for football-video research.

All previous football-analysis research has been removed from the active tree. Development is restarting from zero; Git history remains the archive for earlier work.

Only **VIDEO FOUNDATION** exists today:

- local media library and match/video management;
- lossless MP4 uploads and registration of local input videos;
- video playback, seeking, and frame-by-frame navigation.

The intended future order is:

`VIDEO FOUNDATION → FIELD → PEOPLE → POSITION → ROLE → ANALYTICS`

No FIELD, PEOPLE, POSITION, ROLE, or ANALYTICS implementation currently exists.

## Run locally

Install the dependencies from `supervisor/requirements.txt`, then run:

```bash
uvicorn supervisor.app:app --reload
```

Open the local URL printed by Uvicorn and import or organize an MP4 from the media library.
