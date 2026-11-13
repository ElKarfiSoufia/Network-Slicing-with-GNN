// Présentation de soutenance (pptxgenjs). Les chiffres viennent de results/deck_data.json,
// écrit par la dernière cellule du notebook ; les figures de figures/*.png ; le discours
// (notes de l'orateur, minuté) de presentation/discours.js.
// Usage : NODE_PATH=<dossier node_modules contenant pptxgenjs> node presentation/build_deck.js
// Sorties : Presentation_B5G_Slicing_GNN.pptx et discours_soutenance.md (même dossier).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "figures", f);
const DATA = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "deck_data.json"), "utf8"));
const LOGO = path.resolve(ROOT, "..", "INPT.png");
const OUT = path.join(__dirname, "Presentation_B5G_Slicing_GNN.pptx");
const APPLY_THEME = process.env.APPLY_THEME;

const THEME = {
  name: "SMARTICT Slicing",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14213D", lt1: "FFFFFF", dk2: "0E1A30", lt2: "F2F4F7",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "6250D6",
    accent5: "A3A29C", accent6: "EDA100", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = THEME.colors;
const INK = HEX.dk1, MUTED = "52514E", LIGHT = "C9D3E3", GRID = "E4E3DF";

// ---------- utilitaires ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(slide, file, x, y, w, h, alt) {
  // image centrée dans la boîte (x, y, w, h), proportions conservées
  const s = pngSize(file);
  const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r, ih = s.h * r;
  slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, altText: alt });
}
const fr = (v, nd = 3) => Number(v).toFixed(nd).replace(".", ",");
const pct = (v, nd = 0) => (100 * v).toFixed(nd).replace(".", ",") + " %";
function cmp(model, sub, key) {
  const r = DATA.comparison.find((d) => d["modèle"] === model && d.sous_ensemble === sub);
  return r ? r[key] : NaN;
}
const MODELS = ["Naïf", "Ridge", "Gradient Boosting", "MLP", "GNN"];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13,333 × 7,5 pouces
pres.title = "AI-Based Dynamic Resource Allocation for B5G Network Slicing";
pres.author = "Projet SMARTICT – INPT";
pres.company = "INPT";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- discours : une entrée de discours.js par diapositive, dans l'ordre ----------
const SPEECH = require("./discours.js");
const mmss = (t) => `${Math.floor(t / 60)} min ${String(Math.round(t % 60)).padStart(2, "0")}`;
let slideIdx = 0, elapsed = 0;
const _addSlide = pres.addSlide.bind(pres);
pres.addSlide = (opts) => {
  const sl = _addSlide(opts);
  const sp = SPEECH[slideIdx++];
  if (!sp) throw new Error(`pas de discours pour la diapositive ${slideIdx}`);
  elapsed += sp.sec;
  sl.addNotes(`[${sp.titre} · ≈ ${sp.sec} s · cumul ${mmss(elapsed)}]\n\n${sp.texte}`);
  return sl;
};

// ---------- dispositions (layouts) ----------
pres.defineSlideMaster({
  title: "TITRE",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.7, fontSize: 40, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.15, w: 11.7, h: 1.0, fontSize: 20, color: C.accent5, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 2.3, y: 2.9, w: 10.2, h: 1.2, fontSize: 38, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 2.3, y: 4.1, w: 10.2, h: 0.9, fontSize: 18, color: C.accent5, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENU",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.85, fontSize: 30, bold: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle" }, text: "" } },
    { text: { text: "Projet SMARTICT · Network slicing B5G et GNN", options: { x: 0.6, y: 7.02, w: 8, h: 0.3, fontSize: 10, color: MUTED } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

// ---------- briques visuelles ----------
function title(slide, t) { slide.addText(t, { placeholder: "title" }); }
function stepDot(slide, x, y, label, color, size = 0.5) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: size, h: size, fill: { color }, line: { color }, objectName: "pastille" });
  slide.addText(label, { x, y, w: size, h: size, fontSize: size > 0.6 ? 18 : 14, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function card(slide, x, y, w, h, fill = HEX.lt2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: "carte" });
}
function bullets(slide, items, x, y, w, h, fontSize = 16) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontSize, color: INK, paraSpaceAfter: 8, valign: "top", isTextBox: true });
}
function stat(slide, x, y, w, value, label, color = HEX.accent1) {
  slide.addText(value, { x, y, w, h: 0.9, fontSize: 40, bold: true, color, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.9, w, h: 0.6, fontSize: 13, color: MUTED, margin: 0, valign: "top", isTextBox: true });
}
function caption(slide, t, x, y, w) {
  slide.addText(t, { x, y, w, h: 0.35, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
}
const chartBase = (extra) => Object.assign({
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10,
  dataLabelColor: INK, valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" },
  legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: MUTED, titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: INK,
}, extra);

// =====================================================================
// 1. Titre
// =====================================================================
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITRE", sectionTitle: "Introduction" });
s.addImage({ path: LOGO, x: 10.6, y: 0.45, w: 2.1, h: 2.1 * pngSize(LOGO).h / pngSize(LOGO).w, altText: "Logo INPT" });
s.addText("INPT · FILIÈRE SMART ICT · PROJET SMARTICT", { x: 0.8, y: 0.6, w: 9, h: 0.4, fontSize: 13, color: HEX.accent5, bold: true, charSpacing: 2, isTextBox: true });
s.addText("AI-Based Dynamic Resource Allocation for B5G Network Slicing", { placeholder: "title" });
s.addText("Prédire la QoS des slices avec un GNN et tester sa robustesse aux changements brusques du réseau", { placeholder: "body" });
s.addText([
  { text: "El Mehdi LAZAAR · EL KARFI Soufia · ABOUBAKAR Sidiki · Nizar BENSELLOUM", options: { breakLine: true } },
  { text: "Encadrant : M. Hassan EL GHAZI", options: { color: HEX.accent5 } },
], { x: 0.8, y: 5.75, w: 11.7, h: 0.8, fontSize: 15, color: "FFFFFF", isTextBox: true });

// =====================================================================
// 2. Question et réponse courte
// =====================================================================
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Introduction" });
title(s, "Une question, une réponse courte");
card(s, 0.6, 1.4, 12.1, 1.6, HEX.lt2);
s.addText([
  { text: "Question centrale du guide", options: { fontSize: 13, color: MUTED, breakLine: true } },
  { text: "« Peut-on apprendre une prédiction utile pour gérer les ressources des slices, et reste-t-elle fiable lors de changements brusques du réseau ? »", options: { fontSize: 20, italic: true, color: INK } },
], { x: 0.9, y: 1.5, w: 11.5, h: 1.4, valign: "middle", isTextBox: true });
const gnnG = cmp("GNN", "global", "MAE_delay"), hgbG = cmp("Gradient Boosting", "global", "MAE_delay");
const gnnE = cmp("GNN", "événement", "MAE_delay"), hgbE = cmp("Gradient Boosting", "événement", "MAE_delay");
const naiveG = cmp("Naïf", "global", "MAE_delay");
const ans = [
  ["1", HEX.accent1, `Oui pour la prédiction : le GNN prédit le délai de chaque flux avec une erreur moyenne de ${fr(gnnG, 3)} décade (soit ×${(10 ** gnnG).toFixed(1).replace(".", ",")}), contre ${fr(naiveG, 3)} pour la référence naïve.`],
  ["2", HEX.accent2, `Fiabilité partielle : pendant les événements, l'erreur du GNN passe à ${fr(gnnE, 3)} décade (×${(gnnE / gnnG).toFixed(2).replace(".", ",")}) ; celle du Gradient Boosting est multipliée par ${(hgbE / hgbG).toFixed(1).replace(".", ",")}.`],
  ["3", HEX.accent3, `Le graphe apporte un gain réel : ${pct((hgbG - gnnG) / hgbG)} d'erreur en moins que le meilleur modèle sans graphe (${pct((hgbE - gnnE) / hgbE)} pendant les événements).`],
];
ans.forEach(([n, c, t], i) => {
  const y = 3.35 + i * 1.15;
  stepDot(s, 0.75, y + 0.12, n, c, 0.6);
  s.addText(t, { x: 1.6, y, w: 11.0, h: 0.95, fontSize: 17, color: INK, valign: "middle", isTextBox: true });
});

// =====================================================================
// 3. Démarche
// =====================================================================
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Introduction" });
title(s, "Démarche : sept livrables, une chaîne reproductible");
const steps = [
  ["L1", "Exploration", "7 127 simulations, dictionnaire des variables, anomalies"],
  ["L2", "Problème", "QoS par flux = jumeau numérique du slicing"],
  ["L3", "Baselines", "naïf, Ridge, Gradient Boosting, MLP"],
  ["L4", "Événements", "écart robuste au régime habituel"],
  ["L5", "GNN", "graphe flux / files WFQ / liens"],
  ["L6", "Comparaison", "global, événements, ablation, décalage"],
  ["L7", "Soutenance", "défense des choix et limites"],
];
steps.forEach(([k, t, d], i) => {
  const x = 0.6 + i * 1.75;
  s.addShape(pres.shapes.LINE, { x: x + 0.55, y: 2.15, w: i < steps.length - 1 ? 1.75 : 0, h: 0, line: { color: GRID, width: 2 } });
  stepDot(s, x + 0.25, 1.85, k, i === 4 ? HEX.accent1 : INK, 0.6);
  s.addText(t, { x, y: 2.65, w: 1.65, h: 0.45, fontSize: 15, bold: true, color: INK, align: "center", isTextBox: true });
  s.addText(d, { x, y: 3.1, w: 1.65, h: 1.2, fontSize: 12, color: MUTED, align: "center", valign: "top", isTextBox: true });
});
card(s, 0.6, 4.75, 12.1, 1.85);
s.addText([
  { text: "Livré : ", options: { bold: true } },
  { text: "notebook exécuté (projet_B5G_slicing_GNN.ipynb), code commenté (code/), rapport LaTeX, cette présentation. Toutes les étapes longues sont mises en cache ; le GNN tourne sur CPU ou GPU." },
], { x: 0.9, y: 4.9, w: 11.5, h: 1.55, fontSize: 15, color: INK, valign: "middle", isTextBox: true });

// =====================================================================
// État de l'art
// =====================================================================
function arrow(slide, x1, y1, x2, y2, color, width = 2) {
  // flèche de (x1, y1) vers (x2, y2) : une ligne avec pointe, retournée selon le sens
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1, line: { color, width, endArrowType: "triangle" } });
}
pres.addSection({ title: "État de l'art" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "État de l'art" });
stepDot(s, 0.8, 3.05, "1", HEX.accent1, 0.9);
s.addText("État de l'art", { placeholder: "title" });
s.addText("Network slicing, allocation des ressources, réseaux de neurones sur graphes", { placeholder: "body" });

// --- network slicing ---
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "État de l'art" });
title(s, "Le network slicing : trois réseaux logiques, une infrastructure");
const bands = [
  ["Slice mMTC", "jusqu'à 10⁶ objets / km², faible débit, tolérante au délai", HEX.accent3],
  ["Slice URLLC", "latence 1 ms, fiabilité 99,999 % (industrie, véhicules)", HEX.accent2],
  ["Slice eMBB", "débit crête 20 Gbit/s, 100 Mbit/s perçus (vidéo, XR)", HEX.accent1],
];
bands.forEach(([t, d, c], i) => {
  const y = 1.5 + i * 1.0;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 7.6, h: 0.82, rectRadius: 0.08, fill: { color: c }, line: { color: c } });
  s.addText([{ text: t + "   ", options: { bold: true } }, { text: d }], { x: 0.8, y, w: 7.3, h: 0.82, fontSize: 15, color: "FFFFFF", valign: "middle", isTextBox: true });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.65, w: 7.6, h: 0.82, rectRadius: 0.08, fill: { color: HEX.lt2 }, line: { color: HEX.accent5 } });
s.addText("Infrastructure physique partagée : antennes, liens de transport, cœur de réseau", { x: 0.8, y: 4.65, w: 7.3, h: 0.82, fontSize: 15, color: INK, valign: "middle", isTextBox: true });
caption(s, "Usages définis par l'UIT (ITU-R M.2083) ; exigences URLLC du 3GPP (TR 38.913).", 0.6, 5.65, 7.6);
card(s, 8.6, 1.5, 4.1, 1.75);
s.addText([{ text: "Normalisation 3GPP", options: { bold: true, breakLine: true } },
  { text: "TS 23.501 : chaque slice porte un type de service. SST 1 = eMBB, 2 = URLLC, 3 = MIoT : les trois slices du dataset.", options: { color: MUTED, fontSize: 13.5 } }],
  { x: 8.8, y: 1.55, w: 3.75, h: 1.65, fontSize: 15, color: INK, valign: "middle", isTextBox: true });
card(s, 8.6, 3.45, 4.1, 1.35);
s.addText([{ text: "Briques techniques", options: { bold: true, breakLine: true } },
  { text: "virtualisation des fonctions (NFV) et réseaux définis par logiciel (SDN).", options: { color: MUTED, fontSize: 13.5 } }],
  { x: 8.8, y: 3.5, w: 3.75, h: 1.25, fontSize: 15, color: INK, valign: "middle", isTextBox: true });
card(s, 8.6, 5.0, 4.1, 1.35, "DCE8F7");
s.addText([{ text: "Le problème : ", options: { bold: true } }, { text: "quelle part des ressources partagées réserver à chaque slice ?" }],
  { x: 8.8, y: 5.05, w: 3.75, h: 1.25, fontSize: 15, color: INK, valign: "middle", isTextBox: true });

// --- WFQ et saturation ---
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "État de l'art" });
title(s, "Allouer les ressources : WFQ, et une QoS qui sature");
card(s, 0.6, 1.4, 5.6, 2.3);
s.addText([
  { text: "Partage d'un port : WFQ ≈ GPS (Parekh et Gallager, 1993)", options: { bold: true, breakLine: true } },
  { text: "débit garanti de la file i :  gᵢ = φᵢ / Σⱼ φⱼ × C", options: { fontSize: 17, color: HEX.accent1, bold: true, breakLine: true } },
  { text: "la capacité inutilisée est redistribuée aux autres files. Dans le dataset : une file par slice à chaque antenne, poids φ = réservation de la slice.", options: { color: MUTED, fontSize: 13.5 } },
], { x: 0.85, y: 1.5, w: 5.2, h: 2.1, fontSize: 15, color: INK, valign: "middle", isTextBox: true });
card(s, 0.6, 3.95, 5.6, 2.55, "DCE8F7");
s.addText([
  { text: "Théorie des files (Kleinrock, 1975)", options: { bold: true, breakLine: true } },
  { text: "M/M/1 : délai  T = 1 / (μ − λ) = (1/μ) · 1 / (1 − ρ)", options: { fontSize: 16, color: HEX.accent1, bold: true, breakLine: true } },
  { text: "le délai explose quand la charge ρ → 1 ; au-delà, la file finie est pleine et perd des paquets. Dans un réseau, les files sont couplées : plus de formule simple.", options: { color: MUTED, fontSize: 13.5 } },
], { x: 0.85, y: 4.05, w: 5.2, h: 2.35, fontSize: 15, color: INK, valign: "middle", isTextBox: true });
const rhos = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98];
s.addChart(pres.charts.LINE, [{ name: "délai relatif 1 / (1 − ρ)", labels: rhos.map((r) => r.toFixed(2).replace(".", ",")), values: rhos.map((r) => 1 / (1 - r)) }],
  chartBase({ x: 6.5, y: 1.35, w: 6.2, h: 5.2, chartColors: [HEX.accent2], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    showLegend: false, showTitle: true, title: "File M/M/1 : délai moyen relatif selon la charge ρ", showValue: false,
    showCatAxisTitle: true, catAxisTitle: "charge ρ = λ / μ", catAxisTitleColor: MUTED, catAxisTitleFontSize: 11,
    showValAxisTitle: true, valAxisTitle: "délai × μ", valAxisTitleColor: MUTED, valAxisTitleFontSize: 11 }));

// --- GNN : passage de messages ---
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "État de l'art" });
title(s, "Les GNN : apprendre par passage de messages");
const nodes = { v: [3.2, 3.4], u1: [1.6, 2.1], u2: [1.4, 4.7], u3: [4.9, 2.0], u4: [5.0, 4.8], w: [0.55, 3.4] };
const R = 0.33;
[["u1", "w"], ["u2", "w"]].forEach(([a, b]) => s.addShape(pres.shapes.LINE, { x: Math.min(nodes[a][0], nodes[b][0]), y: Math.min(nodes[a][1], nodes[b][1]),
  w: Math.abs(nodes[a][0] - nodes[b][0]), h: Math.abs(nodes[a][1] - nodes[b][1]), flipV: (nodes[b][1] < nodes[a][1]) !== (nodes[b][0] < nodes[a][0]), line: { color: HEX.accent5, width: 1.5 } }));
["u1", "u2", "u3", "u4"].forEach((k) => {
  const [x1, y1] = nodes[k], [x2, y2] = nodes.v, L = Math.hypot(x2 - x1, y2 - y1);
  arrow(s, x1 + (x2 - x1) * R / L, y1 + (y2 - y1) * R / L, x2 - (x2 - x1) * R / L, y2 - (y2 - y1) * R / L, HEX.accent2, 2.5);
});
Object.entries(nodes).forEach(([k, [x, y]]) => {
  const c = k === "v" ? HEX.accent1 : (k === "w" ? "D3D2CD" : "E9E8E4");
  s.addShape(pres.shapes.OVAL, { x: x - R, y: y - R, w: 2 * R, h: 2 * R, fill: { color: c }, line: { color: MUTED, width: 1 } });
  s.addText(k.replace("u", "u").replace(/(\d)/, "$1"), { x: x - R, y: y - R, w: 2 * R, h: 2 * R, fontSize: 14, bold: true, italic: true, color: k === "v" ? "FFFFFF" : INK, align: "center", valign: "middle", margin: 0, isTextBox: true });
});
caption(s, "itération 1 : v agrège u₁…u₄ ; itération 2 : l'information de w atteint v.", 0.6, 5.45, 5.6);
card(s, 6.4, 1.4, 6.3, 1.5, "DCE8F7");
s.addText([{ text: "À chaque itération (Gilmer et al., 2017) : ", options: { bold: true } },
  { text: "messages des voisins → agrégation insensible à l'ordre → mise à jour de l'état. Après T itérations : voisinage à T sauts." }],
  { x: 6.6, y: 1.45, w: 5.9, h: 1.4, fontSize: 14.5, color: INK, valign: "middle", isTextBox: true });
const fam = [["GCN", "moyenne normalisée", "graphe homogène"], ["GraphSAGE", "agrégation d'un ensemble", "inductif"],
  ["GAT / GATv2", "attention apprise", "pondère les voisins"], ["GIN", "somme + MLP", "le plus expressif"]];
fam.forEach(([n, a, d], i) => {
  const x = 6.4 + (i % 2) * 3.2, y = 3.1 + Math.floor(i / 2) * 1.35;
  card(s, x, y, 3.0, 1.2);
  s.addText([{ text: n, options: { bold: true, breakLine: true } }, { text: `${a} · ${d}`, options: { color: MUTED, fontSize: 12.5 } }],
    { x: x + 0.15, y, w: 2.75, h: 1.2, fontSize: 15, color: INK, valign: "middle", isTextBox: true });
});
caption(s, "Limites connues : sur-lissage (couches empilées), sur-compression, généralisation à des graphes plus grands.", 6.4, 5.85, 6.3);

// --- GNN pour les réseaux et positionnement ---
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "État de l'art" });
title(s, "Les GNN pour les réseaux, et notre positionnement");
card(s, 0.6, 1.4, 5.95, 5.2);
s.addText("Ce qui existe", { x: 0.85, y: 1.5, w: 5.5, h: 0.5, fontSize: 19, bold: true, color: INK, isTextBox: true });
bullets(s, [
  "RouteNet (Rusek et al., 2020) : graphe flux ↔ liens ; un GRU parcourt le chemin de chaque flux ; généralise à des topologies inédites.",
  "RouteNet-Fermi (2023) : nœuds « file » pour les ordonnancements multi-files (FIFO, priorité, WFQ, DRR).",
  "GNN pour la gestion radio (Eisen et Ribeiro, 2020 ; Shen et al., 2021).",
  "Briques de jumeaux numériques de réseau (Almasan et al., 2022).",
], 0.85, 2.05, 5.5, 4.4, 14);
card(s, 6.75, 1.4, 5.95, 5.2, "DCE8F7");
s.addText("Ce qui manque → notre travail", { x: 7.0, y: 1.5, w: 5.5, h: 0.5, fontSize: 19, bold: true, color: HEX.accent1, isTextBox: true });
const gaps = [["Réservations de slices", "graphe avec les files WFQ par slice, où s'applique δ"],
  ["Fiabilité hors régime", "événements définis et évaluation dédiée"],
  ["Références exigeantes", "Gradient Boosting et MLP nourris de features physiques"]];
gaps.forEach(([t, d], i) => {
  const y = 2.2 + i * 1.4;
  stepDot(s, 7.0, y + 0.15, String(i + 1), HEX.accent1, 0.55);
  s.addText([{ text: t, options: { bold: true, breakLine: true } }, { text: d, options: { color: MUTED, fontSize: 13.5 } }],
    { x: 7.7, y, w: 4.8, h: 1.15, fontSize: 15.5, color: INK, valign: "middle", isTextBox: true });
});

// =====================================================================
// 4. Données
// =====================================================================
pres.addSection({ title: "Données" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Données" });
stepDot(s, 0.8, 3.05, "2", HEX.accent1, 0.9);
s.addText("Le dataset B5G Network Slicing", { placeholder: "title" });
s.addText("Farreras et al., Data in Brief 2024 · simulations paquet OMNeT++ sur topologies réelles", { placeholder: "body" });

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Données" });
title(s, "Un réseau simulé : des slices qui partagent antennes et liens");
const sm = DATA.summary;
stat(s, 0.6, 1.45, 2.9, sm.n_sims.toLocaleString("fr-FR"), "simulations stables (régime établi)");
stat(s, 3.6, 1.45, 2.9, (sm.n_flows / 1e6).toFixed(2).replace(".", ",") + " M", "flux, regroupés en " + sm.n_classes.toLocaleString("fr-FR") + " classes", HEX.accent2);
stat(s, 6.6, 1.45, 2.9, String(sm.n_topologies), "topologies réelles (16 à 765 nœuds)", HEX.accent3);
stat(s, 9.6, 1.45, 3.1, "3", "types de slices : eMBB, URLLC, mMTC", HEX.accent4);
// schéma : terminal -> antenne (WFQ) -> transport -> destination
const boxes = [["Terminaux", "origin_*", HEX.lt2], ["Antenne", "WFQ : 1 file par slice", "DCE8F7"], ["Transport", "FIFO, 32 paquets", HEX.lt2], ["Destination", "par slice", HEX.lt2]];
boxes.forEach(([t, d, f], i) => {
  const x = 0.6 + i * 3.15;
  card(s, x, 3.75, 2.6, 1.35, f);
  s.addText([{ text: t, options: { bold: true, fontSize: 17, breakLine: true } }, { text: d, options: { fontSize: 13, color: MUTED } }],
    { x, y: 3.75, w: 2.6, h: 1.35, align: "center", valign: "middle", color: INK, isTextBox: true });
  if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.68, y: 4.25, w: 0.4, h: 0.35, fill: { color: HEX.accent5 }, line: { color: HEX.accent5 } });
});
s.addText([
  { text: "Levier de l'opérateur : ", options: { bold: true } },
  { text: "le poids WFQ de chaque file d'antenne = part de capacité réservée à la slice, fixée par δ ∈ [0, 1] (réservation entre 90 % et 120 % du débit nominal)." },
], { x: 0.6, y: 5.45, w: 12.1, h: 0.9, fontSize: 15, color: INK, isTextBox: true });

// =====================================================================
// 6. Anomalies
// =====================================================================
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Données" });
title(s, "Quatre pièges dans les données, tous vérifiés");
const anomalies = [
  ["42 %", "des chemins du fichier de slices diffèrent du routage réellement simulé", "chemins reconstruits depuis le routage ; contrôle : charge offerte et utilisation mesurée corrélées à 0,98", HEX.accent2],
  ["8 %", "des flux traversent un port où leur slice n'a aucune réservation", "files quasi affamées : effet réel du dataset, conservé", HEX.accent4],
  ["×1,25", "débit réel eMBB et mMTC par rapport au nominal", "champ « taille de paquet » incohérent ; 91 % des flux ont un débit offert supérieur à leur débit réservé", HEX.accent1],
  ["93 %", "des flux ont un jumeau aux entrées identiques", "regroupés en classes pondérées : métriques par flux exactes, calcul 6× plus léger", HEX.accent3],
];
anomalies.forEach(([v, t, d, c], i) => {
  const y = 1.4 + i * 1.33;
  card(s, 0.6, y, 12.1, 1.18);
  s.addText(v, { x: 0.8, y, w: 1.9, h: 1.18, fontSize: 30, bold: true, color: c, fontFace: THEME.headFontFace, valign: "middle", isTextBox: true });
  s.addText([{ text: t, options: { bold: true, breakLine: true } }, { text: d, options: { color: MUTED, fontSize: 13 } }],
    { x: 2.8, y, w: 9.7, h: 1.18, fontSize: 15, color: INK, valign: "middle", isTextBox: true });
});

// =====================================================================
// 7. EDA : coude
// =====================================================================
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Données" });
title(s, "La QoS change de régime quand le goulot sature");
fitImage(s, FIG("fig_knee.png"), 0.6, 1.3, 8.3, 4.9, "QoS en fonction de la charge du goulot");
caption(s, "b = débit offert / capacité de la ressource la plus chargée du flux (file WFQ ou lien), calculé à partir des entrées.", 0.6, 6.3, 8.3);
bullets(s, [
  "Délai sur 8 ordres de grandeur, bimodal : µs sans congestion, ms à s derrière une file pleine → cible en log10.",
  "Coude net en b ≈ 1 : sans pertes en dessous, pertes et délai qui explosent au-dessus.",
  "Aucune feature seule n'explique la QoS : elle dépend des autres flux qui partagent les ressources.",
], 9.1, 1.4, 3.6, 4.8, 15);

// =====================================================================
// 8. Méthode
// =====================================================================
pres.addSection({ title: "Méthode" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Méthode" });
stepDot(s, 0.8, 3.05, "3", HEX.accent1, 0.9);
s.addText("Formulation, protocole et événements", { placeholder: "title" });
s.addText("Ce que le modèle prédit, et comment on l'évalue sans tricher", { placeholder: "body" });

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Méthode" });
title(s, "La cible : un jumeau numérique de la QoS des slices");
const chain = [["Configuration", "topologie, routage, demandes des slices, poids WFQ (δ)"], ["Modèle", "Ridge, Gradient Boosting, MLP, GNN"], ["QoS prédite par flux", "log10 délai, taux de pertes"], ["Décision", "SLA respecté ? admission ? quel δ ?"]];
chain.forEach(([t, d], i) => {
  const x = 0.6 + i * 3.15;
  card(s, x, 1.5, 2.6, 1.6, i === 1 ? "DCE8F7" : HEX.lt2);
  s.addText([{ text: t, options: { bold: true, fontSize: 16, breakLine: true } }, { text: d, options: { fontSize: 12.5, color: MUTED } }],
    { x: x + 0.1, y: 1.5, w: 2.4, h: 1.6, align: "center", valign: "middle", color: INK, isTextBox: true });
  if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.68, y: 2.12, w: 0.4, h: 0.35, fill: { color: HEX.accent5 }, line: { color: HEX.accent5 } });
});
bullets(s, [
  "Pourquoi c'est de la gestion de slices : l'opérateur doit savoir avant de déployer si une allocation respecte les SLA. Une simulation paquet prend de quelques minutes à 15 h ; le modèle répond en millisecondes.",
  "Pourquoi pas « prédire l'allocation optimale » : le dataset ne contient que des allocations aléatoires et leurs conséquences, aucune étiquette d'optimalité.",
  "SLA de l'article : eMBB 10 ms (59 % de violations), URLLC 0,5 ms (51 %), mMTC 10 s (1 %).",
], 0.6, 3.5, 12.1, 3.2, 16);

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Méthode" });
title(s, "Un protocole sans fuite d'information");
const prot = [
  ["Découpage par topologie", "≈ 44 simulations par topologie : train / val / test sur des topologies disjointes (70/15/15 %). Le modèle ne peut pas mémoriser un réseau.", HEX.accent1],
  ["Entrées seulement", "utilisations, pertes et délais mesurés ne sont jamais des features ; normalisations et hyperparamètres fixés sans le test.", HEX.accent2],
  ["Métriques par flux", "MAE et MSE pondérées par la multiplicité des classes, sur les deux cibles ; test lu une seule fois.", HEX.accent3],
  ["Décalage de distribution", "second découpage : entraînement sur petits et moyens réseaux, test sur les plus grands (≥ 212 nœuds).", HEX.accent4],
];
prot.forEach(([t, d, c], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.45 + Math.floor(i / 2) * 2.6;
  card(s, x, y, 5.95, 2.35);
  stepDot(s, x + 0.3, y + 0.3, String(i + 1), c, 0.55);
  s.addText(t, { x: x + 1.05, y: y + 0.25, w: 4.7, h: 0.6, fontSize: 18, bold: true, color: INK, valign: "middle", isTextBox: true });
  s.addText(d, { x: x + 0.3, y: y + 1.0, w: 5.4, h: 1.25, fontSize: 14.5, color: MUTED, valign: "top", isTextBox: true });
});

// événements
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Méthode" });
title(s, "Événement = écart brusque au régime habituel du réseau");
fitImage(s, FIG("fig_events_z.png"), 0.6, 1.3, 12.1, 3.5, "Distribution de z et QoS selon z");
const ev = Object.fromEntries(DATA.events.map((e) => [e.event_type, e]));
bullets(s, [
  "Pas d'axe temporel : chaque topologie est simulée ≈ 44 fois, ce qui définit son régime habituel.",
  "Mesure : z = écart robuste (médiane, MAD) de log10 b par topologie et par slice. Règle de Hampel |z| ≥ 3.",
  `Surcharge ${pct(ev["surcharge"].part, 1)} des flux (pertes médianes ${pct(ev["surcharge"].pertes_median)}), délestage ${pct(ev["délestage"].part, 1)} (pertes nulles). Le masque ne sert qu'à l'évaluation.`,
], 0.6, 4.95, 12.1, 1.9, 15);

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Méthode" });
title(s, "Vérification : les flux marqués ont une QoS réellement différente");
fitImage(s, FIG("fig_events_topo.png"), 0.6, 1.3, 12.1, 4.6, "Vérification visuelle des événements sur une topologie");
caption(s, "Le masque est calculé à partir des seules entrées ; la QoS mesurée confirme le changement de régime.", 0.6, 6.15, 12.1);

// =====================================================================
// GNN
// =====================================================================
pres.addSection({ title: "GNN" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "GNN" });
stepDot(s, 0.8, 3.05, "4", HEX.accent1, 0.9);
s.addText("Représenter le réseau par un graphe", { placeholder: "title" });
s.addText("Qui partage quoi : flux, files WFQ et liens", { placeholder: "body" });

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "GNN" });
title(s, "Un graphe hétérogène : flux, files WFQ, liens");
fitImage(s, FIG("fig_graph.png"), 0.5, 1.25, 6.4, 5.6, "Graphe hétérogène d'une simulation");
bullets(s, [
  "Nœuds flux : slice, débit offert, taille de paquet, δ, nombre de sauts.",
  "Nœuds files WFQ : poids, débit réservé, charge, délai de file pleine.",
  "Nœuds liens : capacité, charge offerte, nombre de flux, type.",
  "Arêtes : chemin ordonné flux → liens, flux → file, file → port.",
  "Pourquoi pas un vecteur : il ignore qui partage chaque lien et combien de trafic arrive après les pertes en amont.",
], 7.1, 1.4, 5.6, 5.3, 15);

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "GNN" });
title(s, "SliceGNN : passage de messages le long des chemins");
const arch = [["Encodage", "3 MLP : flux, files, liens → états de dimension 64"], ["× 4 itérations", "1. GRU le long du chemin (le flux lit liens et file)\n2. liens ← moyenne pondérée des flux\n3. files ← moyenne pondérée des flux"], ["Lecture", "MLP → [log10 délai, pertes]"]];
arch.forEach(([t, d], i) => {
  const x = 0.6 + i * 4.15, w = 3.7;
  card(s, x, 1.45, w, 2.4, i === 1 ? "DCE8F7" : HEX.lt2);
  s.addText([{ text: t, options: { bold: true, fontSize: 17, breakLine: true } }, { text: d, options: { fontSize: 13.5, color: MUTED } }],
    { x: x + 0.15, y: 1.5, w: w - 0.3, h: 2.3, valign: "middle", color: INK, isTextBox: true });
  if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + w + 0.05, y: 2.45, w: 0.35, h: 0.35, fill: { color: HEX.accent5 }, line: { color: HEX.accent5 } });
});
bullets(s, [
  "Graphe hétérogène et biparti : un GCN homogène suppose des voisins de même nature.",
  "L'ordre du chemin compte : un flux qui perd des paquets en amont charge moins l'aval. Le GRU le modélise ; GraphSAGE et GAT agrègent un ensemble sans ordre.",
  "Dépendances circulaires flux ↔ liens : des itérations avec mémoire approchent ce point fixe.",
  "De 1 à plusieurs centaines de flux par lien : moyenne pondérée stable. Surapprentissage contrôlé par validation sur topologies inédites et arrêt précoce.",
], 0.6, 4.1, 12.1, 2.7, 14.5);

// =====================================================================
// Résultats
// =====================================================================
pres.addSection({ title: "Résultats" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Résultats" });
stepDot(s, 0.8, 3.05, "5", HEX.accent1, 0.9);
s.addText("Comparaison", { placeholder: "title" });
s.addText("Global, événements, SLA, ablation, décalage de distribution", { placeholder: "body" });

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Résultats" });
title(s, "Le GNN est le plus précis, surtout pendant les événements");
const mk = (sub, key) => MODELS.map((m) => cmp(m, sub, key));
s.addChart(pres.charts.BAR, [
  { name: "global", labels: MODELS, values: mk("global", "MAE_delay") },
  { name: "événement", labels: MODELS, values: mk("événement", "MAE_delay") },
], chartBase({ x: 0.6, y: 1.3, w: 6.6, h: 4.6, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent1, HEX.accent2],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "t",
  showTitle: true, title: "MAE log10 délai (décades, test)", valAxisLabelFormatCode: "0.0" }));
s.addChart(pres.charts.BAR, [
  { name: "global", labels: MODELS, values: mk("global", "MAE_loss") },
  { name: "événement", labels: MODELS, values: mk("événement", "MAE_loss") },
], chartBase({ x: 7.3, y: 1.3, w: 5.4, h: 4.6, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent1, HEX.accent2],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "t",
  showTitle: true, title: "MAE taux de pertes (test)", valAxisLabelFormatCode: "0.00" }));
const seedsStd = DATA.seeds.find((r) => r.index === "écart-type");
caption(s, `Topologies jamais vues. Variabilité du GNN entre 3 initialisations : ± ${fr(seedsStd["MAE_delay global"], 3)} décade (délai). MSE et détails par sous-ensemble dans le rapport.`, 0.6, 6.05, 12.1);

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Résultats" });
title(s, "Où les modèles se trompent : les ruptures de régime");
fitImage(s, FIG("fig_errors_z.png"), 0.6, 1.3, 12.1, 3.9, "Erreur selon l'écart z au régime habituel");
bullets(s, [
  "L'erreur croît avec |z| pour tous les modèles : prédire loin du régime habituel reste difficile.",
  "Les surcharges extrêmes (files affamées) concentrent les plus grosses erreurs ; le GNN les réduit le plus.",
], 0.6, 5.35, 12.1, 1.5, 15);

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Résultats" });
title(s, "Utilité opérationnelle : détecter les violations de SLA");
const slaRows = [["Modèle", "Exactitude équilibrée (global)", "Exactitude équilibrée (événement)", "F1 violation (global)", "F1 violation (événement)"]];
DATA.sla.forEach((r) => slaRows.push([r["modèle"], fr(r["exactitude_équilibrée global"]), fr(r["exactitude_équilibrée événement"]), fr(r["F1_violation global"]), fr(r["F1_violation événement"])]));
s.addTable(slaRows.map((row, i) => row.map((c, j) => ({ text: c, options: {
  bold: i === 0 || (row[0] === "GNN"), color: i === 0 ? "FFFFFF" : INK, fill: { color: i === 0 ? INK : (row[0] === "GNN" ? "DCE8F7" : (i % 2 ? "FFFFFF" : HEX.lt2)) },
  align: j === 0 ? "left" : "center", fontSize: 14 } }))), { x: 0.6, y: 1.5, w: 12.1, colW: [2.9, 2.3, 2.5, 2.1, 2.3], rowH: 0.5, border: { type: "solid", pt: 0.5, color: GRID } });
caption(s, "Décision « délai prédit > SLA ? » pour eMBB (10 ms) et URLLC (0,5 ms), flux du test pondérés.", 0.6, 4.85, 12.1);
card(s, 0.6, 5.35, 12.1, 1.35, "DCE8F7");
s.addText([
  { text: "À retenir : ", options: { bold: true } },
  { text: "loin du seuil, tous les modèles non linéaires décident correctement ; le GNN se distingue près du seuil et sur l'ampleur des dégradations, ce qui compte pour choisir de combien augmenter une réservation." },
], { x: 0.85, y: 5.4, w: 11.6, h: 1.25, fontSize: 15, color: INK, valign: "middle", isTextBox: true });

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Résultats" });
title(s, "Ablation : ce qui compte vraiment");
const KEEP = ["GNN|T = 0 (sans interactions)", "GNN|sans nœuds file", "GNN|sans allocation", "GNN|sans charge", "GNN|sans réservation", "GNN|sans physique", "Gradient Boosting|sans allocation", "Gradient Boosting|sans charge"];
const ab = KEEP.map((k) => DATA.ablation.find((r) => `${r["modèle"]}|${r.variante}` === k));
s.addChart(pres.charts.BAR, [{ name: "Δ MAE délai", labels: ab.map((r) => `${r["modèle"]} – ${r.variante}`), values: ab.map((r) => r["Δ MAE délai"]) }],
  chartBase({ x: 0.6, y: 1.25, w: 8.0, h: 5.6, barDir: "bar", chartColors: ab.map((r) => (r["modèle"] === "GNN" ? HEX.accent1 : HEX.accent5)),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00;-0.00", showLegend: false, showTitle: true,
    title: "Hausse de la MAE délai quand on retire… (décades)", catAxisLabelFontSize: 10, valAxisLabelFormatCode: "0.0", catAxisOrientation: "maxMin" }));
bullets(s, ["Sans interactions (T = 0) : 0,240 au lieu de 0,175, et 0,50 au lieu de 0,27 pendant les événements. Environ 60 % du gain vient de là.", "Sans nœuds file WFQ : 0,428, moins bien que le Gradient Boosting. Représenter la réservation par slice est indispensable.", "Sans information d'allocation : 0,40 pour le GNN, 0,45 pour le Gradient Boosting. Le levier de l'opérateur est l'information clé."], 8.8, 1.4, 3.9, 5.3, 14);

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Résultats" });
title(s, "Décalage de distribution : vers des réseaux plus grands");
s.addChart(pres.charts.BAR, [
  { name: "test topologies aléatoires", labels: DATA.shift.map((r) => r["modèle"]), values: DATA.shift.map((r) => r["MAE_délai_topo"]) },
  { name: "test grands réseaux", labels: DATA.shift.map((r) => r["modèle"]), values: DATA.shift.map((r) => r["MAE_délai_shift"]) },
], chartBase({ x: 0.6, y: 1.3, w: 8.0, h: 5.0, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent1, HEX.accent4],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "t",
  showTitle: true, title: "MAE log10 délai (décades)", valAxisLabelFormatCode: "0.0" }));
bullets(s, ["Sur les plus grands réseaux (≥ 212 nœuds, jamais vus), tous les modèles se dégradent un peu : GNN +16 %, Gradient Boosting +6 %.", "Le GNN reste le meilleur, y compris pendant les événements de ces réseaux : 0,28 contre 0,57."], 8.8, 1.4, 3.9, 5.0, 15);

// =====================================================================
// Conclusion
// =====================================================================
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Conclusion" });
stepDot(s, 0.8, 3.05, "6", HEX.accent1, 0.9);
s.addText("Quand un GNN apporte-t-il de la valeur ?", { placeholder: "title" });
s.addText("Réponse à la question finale du guide", { placeholder: "body" });

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Conclusion" });
title(s, "Le GNN apporte de la valeur quand…");
card(s, 0.6, 1.4, 5.95, 5.3, "E8F6EF");
card(s, 6.75, 1.4, 5.95, 5.3, HEX.lt2);
s.addText("Oui, plutôt si…", { x: 0.85, y: 1.55, w: 5.5, h: 0.5, fontSize: 19, bold: true, color: HEX.accent3, isTextBox: true });
s.addText("Pas nécessairement si…", { x: 7.0, y: 1.55, w: 5.5, h: 0.5, fontSize: 19, bold: true, color: MUTED, isTextBox: true });
bullets(s, ["la QoS dépend de ressources partagées et d'effets en cascade (gain perdu sans interactions, T = 0) ;", "le graphe porte explicitement le levier de gestion : les files WFQ par slice ;", "le réseau sort de son régime habituel : erreur ×1,5 seulement, contre ×2,4 pour le Gradient Boosting ;", "la topologie change ou grandit : le même modèle s'applique, avec une dégradation modérée."], 0.85, 2.15, 5.5, 4.4, 15);
bullets(s, ["la décision est grossière et loin des seuils : sur la détection de SLA, le gain n'est que de +0,02 ;", "le coût compte : 16 min de GPU contre 2 min de CPU pour le Gradient Boosting ;", "le graphe ne représente pas l'allocation : sans files WFQ, le GNN fait moins bien que le Gradient Boosting."], 7.0, 2.15, 5.5, 4.4, 15);

s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Conclusion" });
title(s, "Limites et suites");
const lim = [
  ["Simulateur", "un seul simulateur, régimes stationnaires, pas de radio (RAN) : conclusions valables pour ce cadre."],
  ["Événements reconstruits", "pas de série temporelle : l'événement est un écart au régime habituel, pas une transition observée."],
  ["Coût", "≈ 20 s par époque sur GPU, environ 5 fois plus sur CPU ; une seule découpe aléatoire (3 initialisations)."],
  ["Suites", "boucle d'optimisation de δ guidée par le GNN, données temporelles, incertitude des prédictions."],
];
lim.forEach(([t, d], i) => {
  const y = 1.4 + i * 1.33;
  stepDot(s, 0.7, y + 0.25, String(i + 1), i === 3 ? HEX.accent3 : INK, 0.55);
  s.addText([{ text: t, options: { bold: true, breakLine: true } }, { text: d, options: { color: MUTED } }],
    { x: 1.5, y, w: 11.2, h: 1.15, fontSize: 16, color: INK, valign: "middle", isTextBox: true });
});

s = pres.addSlide({ masterName: "TITRE", sectionTitle: "Conclusion" });
s.addText("Merci", { placeholder: "title" });
s.addText("Un GNN qui voit les files WFQ et les interactions entre flux : −37 % d'erreur, et −59 % pendant les changements brusques.", { placeholder: "body" });
s.addImage({ path: LOGO, x: 10.6, y: 0.45, w: 2.1, h: 2.1 * pngSize(LOGO).h / pngSize(LOGO).w, altText: "Logo INPT" });

// ---------- contrôle du discours et version imprimable ----------
if (slideIdx !== SPEECH.length) throw new Error(`${slideIdx} diapositives mais ${SPEECH.length} entrées de discours`);
const md = [`# Discours de soutenance (${SPEECH.length} diapositives, durée totale ≈ ${mmss(elapsed)})`, "",
  "Rythme visé : ≈ 140 mots par minute. Le texte est aussi dans les notes de l'orateur de chaque diapositive.", "",
  "Répartition suggérée entre les quatre orateurs (à adapter) : diapositives 1 à 8 (introduction et état de l'art), 9 à 17 (données et méthode), 18 à 23 (GNN et comparaison), 24 à 30 (SLA, ablation, décalage, conclusion).", ""];
let cum = 0;
SPEECH.forEach((sp, i) => {
  cum += sp.sec;
  md.push(`## ${i + 1}. ${sp.titre}  ·  ≈ ${sp.sec} s  ·  cumul ${mmss(cum)}`, "", sp.texte, "");
});
fs.writeFileSync(path.join(__dirname, "discours_soutenance.md"), md.join("\n"), "utf8");
console.log(`discours : ${SPEECH.length} diapositives, ${mmss(elapsed)}`);

// ---------- écriture ----------
pres.writeFile({ fileName: OUT }).then(async () => {
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("écrit :", OUT);
});
