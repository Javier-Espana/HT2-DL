# Hoja de Trabajo #2: Transformers y Mecanismos de Atención

**Curso:** CC3092 -- Deep Learning y Sistemas Inteligentes  
**Estudiante:** Javier España (Carné: 23361)  
**Universidad del Valle de Guatemala**  
**Repositorio Git:** [https://github.com/Javier-Espana/HT2-DL.git](https://github.com/Javier-Espana/HT2-DL.git)  

---

## 📌 Descripción del Proyecto

Este repositorio contiene la resolución teórica, matemática y experimental de la **Hoja de Trabajo #2** sobre arquitecturas Transformer y mecanismos de atención.

El proyecto aborda desde los orígenes de la atención en modelos recurrentes (RNN seq2seq de Bahdanau y Luong) hasta las formulaciones modernas de auto-atención multi-cabeza (*Multi-Head Attention*), la derivación analítica de la escala $\sqrt{d_k}$, la complejidad computacional $O(n^2)$, y el análisis empírico con modelos preentrenados (**BERT** y **GPT-2**).

---

## 📂 Estructura del Repositorio

```text
HT2-DL/
├── .gitignore                      # Exclusiones de Git (venvs, checkpoints, temporales de LaTeX)
├── README.md                       # Documentación principal del proyecto
├── requirements.txt                # Dependencias reproducibles del entorno Python
├── docs/
│   ├── Hoja_de_trabajo_2_Transformers_Atencion-1.pdf # Enunciado oficial de la tarea
│   ├── Hoja_de_trabajo_2_Javier_Espana_23361.pdf     # Informe final compilado (PDF, 5 páginas)
│   └── informe/
│       ├── main.tex               # Documento fuente en LaTeX (formato estándar a 1 columna)
│       ├── main.pdf               # PDF compilado directamente desde LaTeX
│       ├── referencias.bib        # Bibliografía en formato BibTeX (IEEEtran)
│       └── figures/               # Gráficas vectoriales y de alta resolución
│           ├── fig_scaling_complexity.pdf
│           ├── fig_bert_heatmaps.pdf
│           ├── fig_winograd_resolution.pdf
│           ├── fig_gpt2_causal_sinks.pdf
│           └── fig_attention_entropy.pdf
├── notebooks/
│   └── HT2_Transformers_Atencion.ipynb # Jupyter Notebook completo, ejecutado y comentado
└── scripts/
    ├── run_experiments.py          # Script ejecutable para correr experimentos y exportar figuras
    └── build_notebook.py           # Generador programático y ejecutor del cuaderno interactivo
```

---

## 🔬 Experimentos y Hallazgos Clave

### 1. Verificación de Varianza y Escalamiento $\sqrt{d_k}$
- Se derivó formalmente que para componentes gaussianos independientes con media 0 y varianza 1, la varianza del producto punto satisface $\mathrm{Var}(q \cdot k) = d_k$.
- Se validó empíricamente con $20\,000$ pares de vectores para $d_k \in \{16, 64, 128, 256, 512, 1024\}$, obteniendo varianzas casi exactas ($16.16, 64.67, \dots, 1012.72$).
- Se demostró que omitir la división entre $\sqrt{d_k}$ provoca la saturación prematura de la función Softmax hacia valores cuasi *one-hot* ($\max P \to 0.971$), provocando la anulación de gradientes ($\frac{\partial S_i}{\partial z_j} \approx 0$).

### 2. Patrones de Atención en BERT (`bert-base-uncased`)
Se identificaron cuatro roles funcionales nítidos en las cabezas de atención:
1. **Identidad léxica / Diagonal (Capa 0, Cabeza 4):** Preserva la representación intrínseca de cada palabra.
2. **Tokens contiguos (Capa 1, Cabeza 0):** Banda subdiagonal especializada en contexto sintáctico adyacente (ej. adjetivo $\to$ sustantivo).
3. **Delimitadores especiales (Capa 7, Cabeza 2):** Concentración masiva en `[CLS]` y `[SEP]`, utilizándolos como reservas neutras de atención.
4. **Semántica contextual profunda (Capa 10, Cabeza 4):** Atención cruzada de largo alcance entre términos conceptualmente relacionados.

### 3. Resolución de Correferencias en Esquema de Winograd
Se evaluó el par clásico de desambiguación pronominal:
- $S_1$: *"The animal didn't cross the street because it was too tired."* $\to$ *"it"* debe asociarse con *"animal"*.
- $S_2$: *"The animal didn't cross the street because it was too wide."* $\to$ *"it"* debe asociarse con *"street"*.

Tras escanear las 144 cabezas de BERT, la **Capa 6, Cabeza 10** maximizó el contraste diferencial ($\Delta = 0.7954$):
- En $S_1$, la atención de *"it"* hacia *"animal"* es **$0.2843$** (frente a $0.0671$ hacia *"street"*).
- En $S_2$, la atención de *"it"* hacia *"street"* se dispara a **$0.5902$** (frente a $0.0119$ hacia *"animal"*).

### 4. Atención Causal y Attention Sinks en GPT-2
- Se comprobó la triangularidad inferior estricta de la matriz de atención autoregresiva.
- Se cuantificó empíricamente el fenómeno de **Attention Sinks** (Xiao et al., 2023): mientras que en la capa 0 el primer token recibe un $22.5\%$ de atención, a partir de la capa 4 absorbe entre el **$75\%$ y el $84\%$** de la masa de atención total, sirviendo como un anclaje o vertedero numérico para estabilizar las sumas unitarias de la softmax.

### 5. Dinámica de Entropía por Capa
- La entropía normalizada de Shannon $H/H_{\max}$ comienza elevada en capas superficiales ($0.75 - 0.76$), denotando distribuciones dispersas.
- En capas medias y profundas decae sensiblemente hasta $0.42$ en BERT y $0.30$ en GPT-2, reflejando una focalización especializada en dependencias precisas antes de la capa de salida.

---

## 🚀 Instrucciones de Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/Javier-Espana/HT2-DL.git
cd HT2-DL
```

### 2. Crear y activar entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Ejecutar los experimentos
Para regenerar todas las métricas y figuras en `docs/informe/figures/`:
```bash
python3 scripts/run_experiments.py
```

### 4. Abrir el Jupyter Notebook
```bash
jupyter notebook notebooks/HT2_Transformers_Atencion.ipynb
```

### 5. Compilar el informe en LaTeX
```bash
cd docs/informe
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📄 Formato del Informe

El informe de entrega se encuentra disponible en:
- `docs/Hoja_de_trabajo_2_Javier_Espana_23361.pdf`

Cumple con todos los lineamientos de la rúbrica:
- Documento sobrio a una sola columna en LaTeX clásico.
- Extensión exacta de 5 páginas.
- Incluye el procedimiento manual/teórico, verificaciones empíricas, gráficas de escalamiento, mapas de calor comentados, discusión crítica y referencias bibliográficas en formato IEEE.
