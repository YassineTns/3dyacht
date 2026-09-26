# Yacht Club Morocco — Sport Fishing tee · rendu 3D (Blender, 100 % Python)

Pipeline headless qui part du mockup `Tee_1.png` et produit un t-shirt oversize photoréaliste :
extraction des prints → patron en panneaux → couture et drapé par simulation de tissu →
matériaux coton jersey lourd → studio → rendus. Aucune manipulation dans l'interface.

```
Tee_1.png                  mockup de référence (face + dos)
assets/prints/             print_front.png, print_back.png, prints.json (placement)
assets/textures/           maille jersey + bord-côte : hauteur + normal map (générées)
scripts/extract_prints.py  étape 1 (Python + OpenCV/Pillow, hors Blender)
scripts/build_tee.py       étapes 2-3 + rendus (bpy)
scripts/tee3d/             pattern, mannequin, garment (simulation), textures, materials, scene
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

Le même script tourne aussi avec le module officiel `bpy` de PyPI, sans installer Blender :
`pip install bpy==5.2.*` (Python 3.13) ou `pip install bpy==4.5.*` (Python 3.11), puis
`python scripts/build_tee.py`. C'est ce qui a produit les rendus du dépôt.

## Étape 1 — extraire les prints

```bash
pip install -r requirements.txt
python scripts/extract_prints.py          # options : --scale 2 --back-scale 4 --ink "#RRGGBB"
```

- `print_front.png` (x2, 495×1656) : libellule + canne + ligne. Le fond écru est retiré par un
  « colour to alpha » en lumière linéaire contre une estimation locale du tissu. Les plis du mockup
  et la tache verte derrière la canne partent avec le fond. Les ailes gardent leur alpha partiel.
  La canne et la ligne, coupées par le bord du mockup, sont prolongées (fond perdu sous l'ourlet).
- `print_back.png` (x4, 1643×549) : logo une couleur. L'alpha est la couverture d'encre mesurée
  sur la luminance, puis ré-affûtée pour avoir des bords d'encre nets. La couleur est l'encre mesurée
  (≈ `#360306`, forçable avec `--ink`).
- `renders/print_*_check.png` : chaque print sur damier / fond sombre / écru, pour juger le détourage.
- `prints.json` : position des prints mesurée sur le mockup, relative au t-shirt. `build_tee.py`
  s'en sert pour placer les prints.

Avec les fichiers sources HD, les déposer sous les mêmes noms dans `assets/prints/` (PNG transparents)
et relancer l'étape 2 : le placement reste celui de `prints.json`.

## Étapes 2-3 — modèle, simulation, matériaux, rendu

```bash
blender -b -P scripts/build_tee.py -- --quality final
# --quality preview|final   preview : rendus à 50 %, 48 échantillons (~10 min au total)
# --views front,back,three_quarter,detail
# --no-render               construit et sauvegarde seulement output/tee.blend
# --frames N                durée de la passe de couture (110 par défaut)
# --debug-renders           vues workbench du drapé dans renders/_debug
```

**Patron** (`tee3d/pattern.py`, cotes en cm dans `TeeSpec`) : coupe oversize/boxy, 60 cm de large
à plat, 74 cm de long depuis l'épaule. Épaules tombantes (58 cm couture à couture), manches courtes
et larges (23 cm, ouverture 48 cm, s'arrêtent au-dessus du coude), bord-côte de col de 3 cm posé
à 90 % de l'encolure. La hauteur de tête de manche (12 cm) fixe l'angle auquel la manche tombe :
une tête basse, typique des épaules tombantes, est faite pour des bras presque horizontaux. Avec les
bras du mannequin à 50°, elle laisserait ~12 cm de tissu pendre sous le bras. Chaque panneau (devant, dos, 2 manches, bord-côte) est un maillage quad structuré.
Les coutures sont appariées point à point et les UV sont le patron à plat, sans déformation.

**Simulation** (`tee3d/garment.py`) : un seul objet tissu. Chaque couture est une rangée d'arêtes
libres que le solveur transforme en *sewing springs*. Une shape key « Flat » (le patron à plat) sert
de forme au repos. Le vêtement ne garde donc pas la forme de départ, il cherche à retrouver ses
dimensions de patron.
1. Devant, dos et manches sont posés autour du mannequin fantôme (bras à 50°), cousus sans
   gravité, puis drapés (110 frames). Chaque manche part d'un anneau d'ourlet incliné autour du bras,
   placé pour que le dessus et le dessous de manche démarrent à leur longueur de patron. Une légère
   pression interne remplit les manches, comme sur un mannequin fantôme.
2. Le bord-côte est posé debout sur l'encolure drapée, cousu, et tout se repose 40 frames.

Les coutures sont ensuite soudées. S'y ajoutent Subdivision (niveau 2 au rendu) et Solidify :
1,5 mm pour le corps, 3 mm pour les ourlets repliés, 4 mm pour le bord-côte doublé. Le mannequin
reste dans le `.blend` (collection *Simulation*, exclue) mais n'est jamais rendu.

**Matériaux** (`tee3d/materials.py`) : Principled BSDF coton écru `#F2EFE6`, très mat, avec sheen
de fibre. Maille jersey procédurale (12 colonnes × 16 rangs/cm) en normal map + cavités, et
bord-côte 1x1 sur le col. Double surpiqûre des ourlets et surpiqûre du col en relief. Les prints
sont mélangés à la couleur du tissu (l'encre suit la maille et lisse un peu la fibre). L'intérieur
du vêtement a son propre matériau, sans print.

**Rendu** (`tee3d/scene.py`) : Cycles, studio à grandes sources douces (key/fill/top/rims), vue
*Khronos PBR Neutral*, exposition calibrée automatiquement pour que le tissu éclairé ressorte
en `#F2EFE6`.

Sorties :
- `output/tee.blend`
- `renders/tee_{front,back,three_quarter,detail}.png` (fond transparent)
- `renders/tee_*_white.png` (fond blanc + ombre douce)
- `renders/tee_mockup.png` (face + dos disposés comme le mockup d'origine)
