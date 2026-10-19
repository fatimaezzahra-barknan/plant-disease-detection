import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import gradio as gr

# ---------- 1. Charger le modèle (sur CPU) ----------
checkpoint = torch.load("../models/meilleur_modele.pth", map_location="cpu")
noms_classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(noms_classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- 2. Noms en français ----------
NOMS_FR = {
    "Apple___Apple_scab": "Pommier – Tavelure",
    "Apple___Black_rot": "Pommier – Pourriture noire",
    "Apple___Cedar_apple_rust": "Pommier – Rouille grillagée",
    "Apple___healthy": "Pommier – Sain",
    "Blueberry___healthy": "Myrtillier – Sain",
    "Cherry_(including_sour)___Powdery_mildew": "Cerisier – Oïdium",
    "Cherry_(including_sour)___healthy": "Cerisier – Sain",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Maïs – Cercosporiose",
    "Corn_(maize)___Common_rust_": "Maïs – Rouille commune",
    "Corn_(maize)___Northern_Leaf_Blight": "Maïs – Helminthosporiose",
    "Corn_(maize)___healthy": "Maïs – Sain",
    "Grape___Black_rot": "Vigne – Black rot",
    "Grape___Esca_(Black_Measles)": "Vigne – Esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Vigne – Brûlure des feuilles",
    "Grape___healthy": "Vigne – Saine",
    "Orange___Haunglongbing_(Citrus_greening)": "Oranger – Huanglongbing (verdissement)",
    "Peach___Bacterial_spot": "Pêcher – Tache bactérienne",
    "Peach___healthy": "Pêcher – Sain",
    "Pepper,_bell___Bacterial_spot": "Poivron – Tache bactérienne",
    "Pepper,_bell___healthy": "Poivron – Sain",
    "Potato___Early_blight": "Pomme de terre – Alternariose",
    "Potato___Late_blight": "Pomme de terre – Mildiou",
    "Potato___healthy": "Pomme de terre – Saine",
    "Raspberry___healthy": "Framboisier – Sain",
    "Soybean___healthy": "Soja – Sain",
    "Squash___Powdery_mildew": "Courge – Oïdium",
    "Strawberry___Leaf_scorch": "Fraisier – Brûlure des feuilles",
    "Strawberry___healthy": "Fraisier – Sain",
    "Tomato___Bacterial_spot": "Tomate – Tache bactérienne",
    "Tomato___Early_blight": "Tomate – Alternariose",
    "Tomato___Late_blight": "Tomate – Mildiou",
    "Tomato___Leaf_Mold": "Tomate – Cladosporiose",
    "Tomato___Septoria_leaf_spot": "Tomate – Septoriose",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomate – Acariens (tétranyques)",
    "Tomato___Target_Spot": "Tomate – Tache cible (corynesporiose)",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomate – Virus TYLCV (enroulement jaune)",
    "Tomato___Tomato_mosaic_virus": "Tomate – Virus de la mosaïque",
    "Tomato___healthy": "Tomate – Saine",
}
noms_fr = [NOMS_FR.get(n, n) for n in noms_classes]
SEUIL_CONFIANCE = 0.60
CARTE_VIDE = "<div class='carte vide'>⬅️ Chargez une photo de feuille pour lancer le diagnostic.</div>"

# ---------- 3. Fonction de diagnostic ----------
def diagnostiquer(image):
    if image is None:
        return CARTE_VIDE, {}
    tenseur = transform_eval(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probas = F.softmax(model(tenseur), dim=1)[0]

    top_p, top_i = probas.topk(3)
    meilleur = noms_fr[top_i[0]]
    confiance = float(top_p[0])
    plante, etat = meilleur.split(" – ")
    est_saine = "healthy" in noms_classes[top_i[0]]

    if confiance < SEUIL_CONFIANCE:
        classe_css, icone, titre = "incertain", "❓", "Diagnostic incertain"
        message = ("Le modèle hésite. Prenez une photo plus nette d'une seule feuille, "
                   "bien éclairée, ou vérifiez que la plante fait partie de la liste supportée.")
    elif est_saine:
        classe_css, icone, titre = "saine", "✅", etat
        message = "Aucun symptôme de maladie détecté."
    else:
        classe_css, icone, titre = "malade", "⚠️", etat
        message = "Symptômes compatibles avec cette maladie. Confirmez auprès d'un spécialiste avant tout traitement."

    carte = f"""
    <div class='carte {classe_css}'>
        <div class='icone'>{icone}</div>
        <div class='plante'>{plante}</div>
        <div class='titre'>{titre}</div>
        <div class='jauge'><div class='remplissage' style='width:{confiance*100:.0f}%'></div></div>
        <div class='confiance'>Confiance : <b>{confiance:.1%}</b></div>
        <div class='message'>{message}</div>
    </div>"""
    top3 = {noms_fr[i]: float(p) for p, i in zip(top_p, top_i)}
    return carte, top3

# ---------- 4. Style ----------
CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#entete {text-align: center; padding: 18px 0 6px;}
#entete h1 {font-size: 2.1em; margin-bottom: 4px;}
#entete p {opacity: 0.75; font-size: 1.05em;}
.carte {border-radius: 16px; padding: 22px; text-align: center; border: 2px solid;}
.carte.vide {border-style: dashed; opacity: 0.6; padding: 50px 20px; border-color: #9ca3af;}
.carte.saine {background: rgba(34,197,94,0.10); border-color: #22c55e;}
.carte.malade {background: rgba(249,115,22,0.10); border-color: #f97316;}
.carte.incertain {background: rgba(148,163,184,0.15); border-color: #94a3b8;}
.carte .icone {font-size: 2.6em;}
.carte .plante {text-transform: uppercase; letter-spacing: 1px; font-size: 0.85em; opacity: 0.7; margin-top: 6px;}
.carte .titre {font-size: 1.6em; font-weight: 700; margin: 4px 0 14px;}
.carte .jauge {height: 10px; background: rgba(148,163,184,0.3); border-radius: 99px; overflow: hidden;}
.carte .remplissage {height: 100%; background: #16a34a; border-radius: 99px;}
.carte.malade .remplissage {background: #f97316;}
.carte.incertain .remplissage {background: #94a3b8;}
.carte .confiance {margin-top: 8px;}
.carte .message {margin-top: 12px; font-size: 0.95em; opacity: 0.85;}
footer {display: none !important;}
"""
theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald", radius_size="lg")
version_majeure = int(gr.__version__.split(".")[0])
options_blocks = {} if version_majeure >= 6 else {"theme": theme, "css": CSS}
options_launch = {"theme": theme, "css": CSS} if version_majeure >= 6 else {}

# ---------- 5. Interface ----------
exemples = sorted([[f] for f in os.listdir(".") if f.startswith("exemple_")])
plantes = sorted({n.split(" – ")[0] for n in noms_fr})

with gr.Blocks(title="Diagnostic des plantes", **options_blocks) as app:
    gr.HTML("""
    <div id='entete'>
        <h1>🌿 Diagnostic des maladies des plantes</h1>
        <p>Déposez la photo d'une feuille : l'intelligence artificielle identifie la maladie en quelques secondes.</p>
    </div>""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="📷 Photo de la feuille", height=340,
                                sources=["upload", "webcam", "clipboard"])
            with gr.Row():
                bouton = gr.Button("🔍 Analyser", variant="primary", size="lg")
                effacer = gr.ClearButton(value="🗑️ Effacer", size="lg")
            if exemples:
                gr.Examples(examples=exemples, inputs=image_in, label="Exemples à essayer")

        with gr.Column(scale=1):
            carte_out = gr.HTML(CARTE_VIDE)
            label_out = gr.Label(num_top_classes=3, label="Top 3 des diagnostics possibles")

    with gr.Accordion("🌱 Plantes supportées (14 espèces, 38 classes)", open=False):
        gr.Markdown(" · ".join(plantes) +
                    "\n\n⚠️ Le modèle ne connaît que ces plantes. Pour une autre espèce, "
                    "il donnera quand même une réponse, qui sera fausse.")

    with gr.Accordion("ℹ️ À propos du modèle", open=False):
        gr.Markdown("""
- **Architecture** : ResNet18 pré-entraîné sur ImageNet, ajusté par *transfer learning*
- **Données** : PlantVillage (54 305 images, 38 classes)
- **Performance sur le test** : 98,5 % d'accuracy, macro F1-score de 0,981
- **Limite** : entraîné sur des photos de laboratoire (une feuille, fond uni). Les photos prises au champ sont moins bien reconnues.
""")

    bouton.click(diagnostiquer, inputs=image_in, outputs=[carte_out, label_out])
    image_in.upload(diagnostiquer, inputs=image_in, outputs=[carte_out, label_out])
    effacer.add([image_in, label_out])
    effacer.click(lambda: CARTE_VIDE, outputs=carte_out)

app.launch(**options_launch)
