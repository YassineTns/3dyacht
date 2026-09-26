# Yacht Club Morocco — Sport Fishing tee · rendu 3D (Blender, 100 % Python)

Pipeline headless qui part du mockup `Tee_1.png` et produit un t-shirt oversize photoréaliste :
extraction des prints → patron en panneaux → couture et drapé par simulation de tissu →
matériaux coton jersey lourd → studio → rendus.

```
Tee_1.png                  mockup de référence (face + dos)
assets/prints/             print_front.png, print_back.png (+ prints.json : placement)
assets/textures/           maille jersey + bord-côte (hauteur + normal map, générées)
scripts/extract_prints.py  étape 1 (Python + OpenCV/Pillow, hors Blender)
scripts/build_tee.py       étapes 2-3 + rendu (bpy)
scripts/tee3d/             patron, mannequin, simulation, textures, matériaux, studio
renders/                   rendus PNG
output/tee.blend           scène finale
```

## Installer Blender (CLI)

Testé avec **Blender 5.2 LTS** et **4.5 LTS** (minimum 4.2).

| OS | Installation | Rendre `blender` accessible en CLI |
|----|--------------|------------------------------------|
| macOS | `brew install --cask blender` ou le `.dmg` de blender.org | `sudo ln -s /Applications/Blender.app/Contents/MacOS/Blender /usr/local/bin/blender` |
| Windows | `winget install BlenderFoundation.Blender` ou l'installeur de blender.org | ajouter `C:\Program Files\Blender Foundation\Blender 5.2` au `PATH` |
| Linux | `sudo snap install blender --classic`, ou l'archive `.tar.xz` de blender.org | lien vers `blender` dans `~/.local/bin` |

Vérifier : `blender --version`.

Sans Blender installé, le module officiel `bpy` (PyPI) fait tourner le même script :
`pip install bpy==5.2.*` (Python 3.13) ou `pip install bpy==4.5.*` (Python 3.11), puis
`python scripts/build_tee.py`.

## Étape 1 — extraire les prints

```bash
pip install -r requirements.txt
python scripts/extract_prints.py          # --scale 2 --back-scale 4 --ink "#RRGGBB"
```

- `print_front.png` (x2) : libellule + canne + ligne. Le fond écru est retiré par un
  « colour to alpha » en lumière linéaire, contre une estimation locale du tissu (les plis du
  mockup et la tache verte derrière la canne partent avec). Les ailes gardent leur alpha partiel.
  La canne et la ligne, coupées par le bord du mockup, sont prolongées (fond perdu).
- `print_back.png` (x4) : logo une couleur. L'alpha est la couverture d'encre mesurée sur la
  luminance, puis ré-affûtée pour des bords d'encre nets ; la couleur est l'encre mesurée
  (≈ `#360306`, forçable avec `--ink`).
- `renders/print_*_check.png` : chaque print sur damier / fond sombre / écru pour juger le détourage.

Pour utiliser les fichiers sources HD à la place : les déposer sous les mêmes noms dans
`assets/prints/` (PNG transparents) et relancer l'étape 2.

## Étapes 2-3 — modèle, simulation, matériaux, rendu

```bash
blender -b -P scripts/build_tee.py -- --quality final
# options : --quality preview|final  --views front,back,three_quarter  --no-render
#           --frames N (simulation)  --debug-renders (vues workbench du drapé)
```

Sorties : `output/tee.blend`, `renders/tee_{front,back,three_quarter}.png` (fond transparent),
`renders/tee_*_white.png` (fond blanc + ombre douce) et `renders/tee_mockup.png` (face + dos
disposés comme le mockup d'origine).
