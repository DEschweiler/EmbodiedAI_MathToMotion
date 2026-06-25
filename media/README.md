# media/

Finale Kurs-Visualisierungen, erzeugt von den `session_XX_manim.ipynb`-Notebooks
(ein Notebook pro Session). Jede Szenen-Zelle ruft am Ende `render_scene(...)` auf
und speichert ihr Ergebnis direkt in den passenden `session_XX/`-Ordner — kein
separater Export-Schritt. Zelle erneut ausführen = Datei wird aktualisiert.

Struktur:

    media/
      README.md
      .manim_cache/              # Manim-Arbeitsverzeichnis (Zwischendateien, von git ignoriert)
      session_01/
        feature_vector_1.png
        feature_vectors_2.png
        feature_space_1d.png
        feature_space_2d.png
        feature_space_3d.png
        standardization.png
        perceptron_forward.mp4       (Video)  — Forward-Pass des Perceptrons
        xor_feature_space.png        (aus session_01.pptx extrahiert — noch kein Notebook-Quellcode)

Nur die Dateien unter `session_XX/` sind "echte" Assets: die sauberen, finalen
Dateien, die von der Session-Markdown (`<video>`/`<img>`) referenziert und in
PowerPoint übernommen werden.

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
