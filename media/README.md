# media/

Finale Kurs-Visualisierungen, erzeugt von den `session_XX_manim.ipynb`-Notebooks
(ein Notebook pro Session, das **direkt im jeweiligen `media/session_XX/`-Ordner**
neben seinen Ausgaben liegt). Jede Szenen-Zelle ruft am Ende `render_scene(...)` auf
und speichert ihr Ergebnis direkt in denselben `session_XX/`-Ordner — kein separater
Export-Schritt. Zelle erneut ausführen = Datei wird aktualisiert. Die Setup-Zelle wechselt
zu Beginn in die Repo-Wurzel, daher funktionieren die Pfade unabhängig vom Speicherort.

Struktur:

    media/
      README.md
      .manim_cache/              # Manim-Arbeitsverzeichnis (Zwischendateien, von git ignoriert)
      session_01/
        session_01_manim.ipynb       (Quell-Notebook für alle Visualisierungen dieser Session)
        feature_vector_1.png
        feature_vectors_2.png
        feature_space_1d.png
        feature_space_2d.png
        feature_space_3d.png
        standardization.png
        perceptron_forward.mp4       (Video)  — Forward-Pass des Perceptrons
        mlp.mp4                      (Video)  — Forward-Pass durch ein 6→2→1-MLP (Eingabe→Hidden Layer→Ausgabe)
        image_mlp.mp4                (Video)  — ganzes Foto → Pixel, Katzen-Pixel hervorgehoben → MLP, Klasse Katze
        loss_landscape.png           (Standbild aus session_01.pptx — Teaser, keine Manim-Szene)
        xor_feature_space.png        (aus session_01.pptx extrahiert — noch kein Notebook-Quellcode)

Die Bild-/Video-Dateien unter `session_XX/` sind die "echten" Assets: die sauberen,
finalen Dateien, die von der Session-Markdown (`<video>`/`<img>`) referenziert und in
PowerPoint übernommen werden. Die `.ipynb` daneben ist der **Quellcode**, der sie erzeugt.

## Scratch-Ordner (kann jederzeit gelöscht werden)

Beim Rendern legt Manim sein Arbeitsverzeichnis unter `media/.manim_cache/` an
(Zwischen-Renders, Teil-Videos, Text-Cache). Dieser Ordner ist in `.gitignore`
ausgenommen und kann jederzeit gelöscht werden; beim nächsten Rendern wird er
neu erzeugt.

## git-Hinweis

Die Dateien unter `media/session_01/` werden eingecheckt (inklusive der Videos,
die bisher klein sind). Falls die Videos später zu groß werden, können sie über
`.gitignore` ausgenommen werden, z. B.:

    media/session_01/*.mp4

Hinweis: Eine pauschale Regel wie `media/session_01/*` würde auch das Quell-Notebook
ausschließen — daher nur gezielt einzelne Endungen (z. B. `*.mp4`) ignorieren.
