
# Aprendizaje por Refuerzo 1 
## GridWorld 4×4 (Q-Learning, SARSA, MC-ES, MC-IS) + Snake (DQN + CNN)

**Autora:** Noelia Melina Qualindi  
**Materia:** Aprendizaje por Refuerzo I — Maestría en IA (UBA)  
**Docente:** Miguel Augusto Azar  
**Año:** 2025

Este repositorio contiene:
- Un entorno **GridWorld 4×4** (tipo *FrozenLake*, `slip=0.2`) con **Q-Learning**, **SARSA**, **Monte Carlo ES**, y **Monte Carlo IS** (Ordinary & Weighted).  
- Un juego **Snake** con observaciones tipo imagen y un **DQN con CNN (PyTorch)**, con **app Streamlit** para jugar **manual**, hacer **inferencia** con el modelo (juega solo) y **entrenar** en vivo.  
- Notebook de **reproducibilidad** para entrenar DQN en Snake sin Streamlit.

---

## 🚀 Quick Start — GridWorld

```bash
# 1) (Opcional) crear y activar venv
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2) instalar dependencias
pip install -r requirements.txt
# o mínimas (solo GridWorld):
# pip install streamlit matplotlib numpy

# 3) correr la app GridWorld
streamlit run app_gridworld.py
# si streamlit no se reconoce:
# python -m streamlit run app_gridworld.py
```

La terminal mostrará algo como:
```
Local URL: http://localhost:8501
```
Abrí esa URL en el navegador.

---

## 🕹️ Uso de la app GridWorld

1. En la **sidebar**:
   - Ajustá `slip_prob`, `max_steps` y `seed` del entorno.
   - Elegí el **algoritmo** (Q-Learning, SARSA, MC-ES, MC-IS Weighted/Ordinary).
   - Configurá **hiperparámetros**:
     - Q-Learning / SARSA: `α`, `γ`, `ε inicial`, `ε decay`, `ε mínimo`.
     - MC-* : `γ` y `ε_b` (política de comportamiento ~ uniforme).
2. Pulsá **“Entrenar y visualizar”**.
3. Se mostrarán dos gráficos:
   - **Política greedy derivada (ε=0)** con marcadores `> v < ^` y etiquetas S/G/X.
   - **Convergencia** (reward por episodio, **media móvil=100**).
4. Debajo, se listan **métricas** (evaluación greedy, 2000 episodios):  
   **tasa de éxito**, **pasos promedio (éxitos)** y **pasos promedio (todos)**.
5. Sección **“Animación del episodio”**:
   - Elegí **FPS**, **ε de animación** y **semilla**.
   - Pulsá **“Animar episodio”** para ver el recorrido del agente con *slip*.

---

## ⚙️ Parámetros por defecto — GridWorld (demo)

- `n_episodes = 4000`, `max_steps = 100`, `slip_prob = 0.2`, `seed = 42`
- Q-Learning / SARSA: `α = 0.8`, `γ = 0.95`, `ε0 = 1.0`, `ε_decay = 0.999`, `ε_min = 0.01`
- MC-ES: *Exploring Starts*
- MC-IS (Ordinary/Weighted): \(b(a\mid s)\approx\) uniforme (`ε_b = 1.0`), con **cap** \(ρ, W \le 10^8\).

---

## 🧪 Protocolo de evaluación — GridWorld

- Evaluación **greedy (ε=0)**, **2000** episodios, `max_steps=100`.
- Métricas: **tasa de éxito**, **pasos promedio (éxitos)** y **pasos promedio (todos)**.
- Recomendado fijar `seed=42` para comparabilidad.

---

## 🐍 Snake (DQN + CNN + Streamlit)

**Qué es:** un entorno **Snake** propio (sin Gymnasium) con observación tipo imagen \((3,H,W)\): **cabeza**, **cuerpo**, **comida**.  
Implementa un **DQN** con **CNN** (PyTorch), **Replay Buffer**, **Target Network**, **ε-decay** y **pérdida Huber**.  
La app Streamlit permite **jugar manualmente**, **entrenar** y **hacer inferencia** (auto-play) con el modelo.

### Quick Start — Snake

```bash
# (con el mismo venv)
# dependencias mínimas para Snake
pip install torch matplotlib numpy streamlit

# correr la app Snake (elegí el archivo que tengas)
streamlit run app_snake_streamlit.py
# o
# streamlit run app_snake.py
```

**Device:** usa automáticamente **CUDA** (si hay GPU), sino **MPS** (Apple Silicon) y, si no, **CPU**.  
La red usa **Global Average Pool (1×1)** → compatible con MPS (evita el error “Adaptive pool MPS: input sizes must be divisible…”).

### Modos de la app Snake

- **Manual**: jugás vos con los botones (↑ ↓ ← →).  
  - Cuando el episodio termina, los botones se **deshabilitan**; presioná **Reset entorno** para reiniciar.
- **Entrenar DQN**: ajustás hiperparámetros y entrenás en vivo.  
  - Se muestra **convergencia** (Reward por episodio, **MA=50**).  
  - Al finalizar, los pesos del modelo quedan en memoria de la app.
- **Inferencia DQN**: el agente juega solo con el modelo entrenado.  
  - Podés elegir **FPS** y un **ε de inferencia** para ε-greedy.

### Hiperparámetros recomendados — Snake
- Grid \(H=W\) = 12 (slider 6–20), `max_steps=300`, `episodes=600–1000`, `seed=42`
- DQN: `γ=0.99`, `lr=1e-3` (Adam), `batch_size=64`, `buffer_size=50_000`  
  `start_learn=1_000`, `target_update_freq=1_000`, `ε0=1.0`, `ε_min=0.05`, `ε_decay=0.995`

### Tips de UI — Snake
- **Tamaño del tablero** en pantalla: control por `figsize` (XS/S/M/L) para que no se vea gigante.  
- **Persistencia**: se usa `st.session_state` para pesos, returns, `done`, etc.  
- **Botones seguros**: `do_step` ignora clicks cuando `done=True` (evita `RuntimeError`).

---

## 📓 Reproducibilidad (sin Streamlit)

Incluyo una **notebook** autocontenida para entrenar y evaluar el DQN de Snake en modo script:

- **Archivo:** `Snake_DQN_Notebook.ipynb`  
- **Instalar:** `pip install torch matplotlib numpy`  
- **Contenido:** entorno `SnakeEnv`, red `DQNCNN` con **GlobalAvgPool**, **Replay Buffer**, **Target Network**,  
  entrenamiento configurable, **gráfico de convergencia (MA=50)**, evaluación **greedy** y guardado de pesos `.pth`.

> Si preferís script CLI: `python dqn_snake.py` (incluido si trabajás con los archivos de ejemplo).

---

## 📦 Estructura sugerida

```
.
├─ app_gridworld.py                 # App Streamlit (GridWorld: entrenar, política, convergencia, animación)
├─ app_snake_streamlit.py          # App Streamlit (Snake: Manual, Entrenar DQN, Inferencia DQN)
├─ README.md                       # Este documento
├─ requirements.txt                # Dependencias mínimas (ver más abajo)
├─ rl_figs/                        # (se crea) PNGs de políticas / convergencia
└─ notebooks/                      # notebooks de apoyo
```
---

## 📋 requirements.txt (mínimo recomendado)

```txt
streamlit
matplotlib
numpy
torch
```

> Para solo GridWorld: `streamlit matplotlib numpy`. Para Snake (DQN): agregar `torch`.

---

## 🛠️ Troubleshooting (GridWorld + Snake)

- **Streamlit no abre el navegador:** copiá manualmente `http://localhost:8501`.  
- **Puerto en uso:** `streamlit run <app.py> --server.port 8502`.  
- **`streamlit` no se reconoce:** `python -m streamlit run <app.py>`.  
- **MPS (Apple Silicon) y Adaptive Pool:** usar **GlobalAvgPool 1×1** (ya incluido) para evitar el error de divisibilidad.  
- **Episodio terminado en modo manual (Snake):** los botones se deshabilitan; si insistís, `do_step` lo ignora y te pide **Reset entorno**.  
- **Dispositivo:** se elige automáticamente entre **CUDA → MPS → CPU**; si algo no está soportado en MPS, forzá `device=cpu`.

---

## ✍️ Notas finales

- En GridWorld, las políticas mostradas son **greedy (ε=0)** derivadas de \(Q\) tras el entrenamiento.  
- **SARSA** entrena con **on-policy ε-greedy**; su visualización greedy es solo para comparación.  
- En Snake, la **convergencia (MA=50)** muestra la mejora del DQN con CNN sobre observaciones pixeladas.  
- La **reproducibilidad** se garantiza con semillas y la **notebook** adjunta.
