# media/

Final course visualizations, produced by `manim_studio.ipynb` — one notebook
renders both the videos and the static images (the image scenes use Manim's
`-s` save-last-frame flag).

Structure:

    media/
      README.md
      session_01/
        perceptron_boundary.mp4     (video)  — perceptron learning rule
        mlp_warp.mp4                (video)  — hidden layer warps the space
        activation_functions.png    (image)  — activations + derivatives
        standardization.png         (image)  — loss-contour conditioning
        mlp_architecture.png        (image)  — 48-256-256-12 with param counts

Only the files under `session_XX/` are "real" assets: they are the clean,
final files referenced by the session markdown via `<video>`/`<img>` tags,
and the same files you drop into PowerPoint later.

## Scratch folders (safe to delete)

When you render in the notebook, Manim recreates auto-generated working
folders next to this one:

    media/videos/   media/jupyter/   media/images/   media/texts/

These are raw renders, inline-preview copies, and a text cache — none of them
are referenced by the site. They can be deleted any time; the export cell in
`manim_studio.ipynb` copies the finished files into `session_XX/` for you.

## git note

`.gitignore` currently ignores all of `media/`, so nothing here is committed
(videos are large and were intentionally kept local for now). If you later
want the **images** to appear on the deployed GitHub Pages site, track them
explicitly, e.g. add to `.gitignore`:

    media/*
    !media/session_01/
    media/session_01/*.mp4      # still skip the large videos
