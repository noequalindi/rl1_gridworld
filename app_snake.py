# app_snake_streamlit.py
# Streamlit: Snake con entorno propio + DQN (PyTorch, CNN)
# Modos: Manual, Inferencia DQN, Entrenar DQN (convergencia + media móvil).

import streamlit as st
import numpy as np
import random, time, math
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =============== Utils de estado de Streamlit ===============

if "snake_env" not in st.session_state: st.session_state["snake_env"] = None
if "last_obs"  not in st.session_state: st.session_state["last_obs"]  = None
if "done"      not in st.session_state: st.session_state["done"]      = True
if "score"     not in st.session_state: st.session_state["score"]     = 0
if "dqn_weights" not in st.session_state: st.session_state["dqn_weights"] = None
if "dqn_dims"    not in st.session_state: st.session_state["dqn_dims"]    = None
if "returns"     not in st.session_state: st.session_state["returns"]     = None

# =============== Entorno Snake ===============
@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict

class SnakeEnv:
    def __init__(self, width: int = 12, height: int = 12, max_steps: int = 300, seed: int = 42):
        self.W, self.H = width, height
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.actions = 4  # 0:←, 1:↓, 2:→, 3:↑
        self.reset()

    def reset(self):
        self.dir = 2  # derecha
        midr, midc = self.H // 2, self.W // 2
        self.snake: deque = deque([(midr, midc-1), (midr, midc), (midr, midc+1)])
        self._place_food()
        self.steps = 0
        self.done = False
        self.score = 0
        return self._obs()

    def _place_food(self):
        free = [(r, c) for r in range(self.H) for c in range(self.W) if (r, c) not in self.snake]
        self.food = free[self.rng.integers(len(free))] if free else None

    def _obs(self) -> np.ndarray:
        head = np.zeros((self.H, self.W), dtype=np.float32)
        body = np.zeros((self.H, self.W), dtype=np.float32)
        food = np.zeros((self.H, self.W), dtype=np.float32)
        hr, hc = self.snake[-1]
        head[hr, hc] = 1.0
        for (r, c) in list(self.snake)[:-1]:
            body[r, c] = 1.0
        if self.food is not None:
            fr, fc = self.food
            food[fr, fc] = 1.0
        return np.stack([head, body, food], axis=0)

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Episodio terminado; llamá reset().")
        self.steps += 1
        self.dir = int(action)
        drc = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}
        dr, dc = drc[self.dir]
        hr, hc = self.snake[-1]
        nr, nc = hr + dr, hc + dc

        reward = -0.01
        terminated = False
        truncated = False

        if not (0 <= nr < self.H and 0 <= nc < self.W):
            reward = -1.0
            terminated = True
        else:
            if (nr, nc) in self.snake:
                reward = -1.0
                terminated = True
            else:
                self.snake.append((nr, nc))
                if self.food is not None and (nr, nc) == self.food:
                    reward = +1.0
                    self.score += 1
                    self._place_food()
                else:
                    self.snake.popleft()

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        self.done = terminated or truncated
        return StepResult(self._obs(), reward, terminated, truncated, {"score": self.score})

# =============== Render ===============
def obs_to_rgb(obs: np.ndarray) -> np.ndarray:
    # obs: (C=3, H, W) -> RGB (H,W,3)
    head, body, food = obs
    H, W = head.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    # colores: comida=rojo, cuerpo=verde, cabeza=azul
    img[..., 0] = food
    img[..., 1] = np.maximum(body, 0.2*food) 
    img[..., 2] = head
    return img

def draw_board(obs: np.ndarray, title: str = "", grid=True, figsize=(3.0,3.0)):
    rgb = obs_to_rgb(obs)
    H, W, _ = rgb.shape
    fig, ax = plt.subplots(figsize=figsize)  # << usa figsize dinámico
    ax.imshow(rgb, interpolation="nearest")
    if grid:
        ax.set_xticks(np.arange(-.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    # << MUY IMPORTANTE: evitar que Streamlit estire la imagen
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# =============== DQN (CNN) ===============
class DQNCNN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))   # global average pooling: SIEMPRE válido (1 divide a todo)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256), nn.ReLU(),   # 64 canales * 1 * 1
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int,int,int]):
        self.capacity = capacity
        C,H,W = obs_shape
        self.s = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.ns = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False

    def push(self, s, a, r, ns, done):
        self.s[self.idx] = s
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.ns[self.idx] = ns
        self.d[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0: self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        N = len(self)
        idxs = np.random.randint(0, N, size=batch_size)
        return ( self.s[idxs], self.a[idxs], self.r[idxs], self.ns[idxs], self.d[idxs] )

def moving_average(x: List[float], w: int = 50):
    if len(x) < 2: return np.array(x, dtype=float)
    if len(x) < w: w = max(1, len(x)//2)
    return np.convolve(np.array(x, dtype=float), np.ones(w)/w, mode="valid")

# =============== Sidebar (parámetros) ===============

st.set_page_config(page_title="Snake RL — DQN + Streamlit", layout="wide")
st.title("Snake con RL (PyTorch DQN + Streamlit)")

SIZE_MAP = {
    "XS": (2.2, 2.2),
    "S":  (3.0, 3.0),
    "M":  (3.8, 3.8),
    "L":  (5.0, 5.0),
}


with st.sidebar:
    st.header("Entorno")
    grid = st.slider("Tamaño del grid", 6, 20, 12, 1)
    max_steps = st.number_input("max_steps", 50, 1000, 300, 10)
    seed = st.number_input("seed", 0, 1_000_000, 42, 1)

    board_size = st.selectbox("Tamaño del tablero en pantalla", ["XS", "S", "M", "L"], index=1)


    st.header("Modo de juego")
    mode = st.radio("Elegir modo", ["Manual", "Inferencia DQN", "Entrenar DQN"])

# =============== Inicializar / Reset ===============
cols0 = st.columns(3)
if cols0[0].button("Reset entorno"):
    st.session_state["snake_env"] = SnakeEnv(grid, grid, max_steps, seed)
    st.session_state["last_obs"]  = st.session_state["snake_env"].reset()
    st.session_state["done"]      = False         # <- re-habilita
    st.session_state["score"]     = 0


if st.session_state["snake_env"] is None:
    st.session_state["snake_env"] = SnakeEnv(grid, grid, max_steps, seed)
    st.session_state["last_obs"]  = st.session_state["snake_env"].reset()
    st.session_state["done"]      = False
    st.session_state["score"]     = 0

env = st.session_state["snake_env"]

# =============== Manual ===============
def do_step(action: int):
    # si el episodio ya terminó, no te deja seguir presionando los botones 
    if st.session_state["done"]:
        st.warning("Episodio terminado. Presioná 'Reset entorno' para reiniciar.")
        return
    try:
        res = env.step(action)
    except RuntimeError:
        st.session_state["done"] = True
        st.warning("Episodio terminado. Presioná 'Reset entorno' para reiniciar.")
        return

    st.session_state["last_obs"] = res.obs
    st.session_state["done"]     = res.terminated or res.truncated
    st.session_state["score"]    = res.info.get("score", st.session_state["score"])

def render_current():
    title = f"Score: {st.session_state['score']} — Steps: {env.steps}/{env.max_steps}" + ("  [GAME OVER]" if st.session_state["done"] else "")
    draw_board(st.session_state["last_obs"], title=title, figsize=SIZE_MAP[board_size])

if mode == "Manual":
    st.subheader("Modo Manual")
    render_current()

    disabled = st.session_state["done"]  # <- una sola bandera

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↑ Arriba", use_container_width=True, disabled=disabled):
            do_step(3)

    with c2:
        lcol, ccol, rcol = st.columns([1,1,1])
        if lcol.button("←", use_container_width=True, disabled=disabled):
            do_step(0)
        if ccol.button("●", use_container_width=True, disabled=True):
            pass
        if rcol.button("→", use_container_width=True, disabled=disabled):
            do_step(2)

    with c3:
        if st.button("↓ Abajo", use_container_width=True, disabled=disabled):
            do_step(1)

    if st.session_state["done"]:
        st.info("Episodio terminado. Presioná **Reset entorno** para reiniciar.")
        
# =============== Inferencia DQN ===============
def build_policy_from_weights(weights, in_ch, n_actions, device):
    model = DQNCNN(in_ch, n_actions).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model

import torch

def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # en Macs con Apple Silicon (PyTorch 1.12+ / 2.x)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_torch_device()
print("Using device:", device)

if mode == "Inferencia DQN":
    st.subheader("Modo Inferencia DQN")
    render_current()
    fps = st.slider("FPS (auto-play)", 1, 20, 8)
    epsilon_eval = st.slider("ε (exploración en inferencia)", 0.0, 1.0, 0.0, 0.05)
    steps_auto = st.slider("Pasos a ejecutar", 1, 400, 120, 1)

    if st.session_state["dqn_weights"] is None:
        st.warning("No hay modelo entrenado cargado. Andá a 'Entrenar DQN' y entrená un modelo.")
    else:
        in_ch, H, W, nA = st.session_state["dqn_dims"]
        policy = build_policy_from_weights(st.session_state["dqn_weights"], in_ch, nA, device)

        if st.button("Auto-play con DQN"):
            ph = st.empty()
            for _ in range(steps_auto):
                if st.session_state["done"]:
                    break
                s = st.session_state["last_obs"]
                if np.random.rand() < epsilon_eval:
                    a = np.random.randint(nA)
                else:
                    with torch.no_grad():
                        ts = torch.from_numpy(s).unsqueeze(0).to(device)
                        q = policy(ts)
                        a = int(torch.argmax(q, dim=1).item())
                do_step(a)
                # render
                rgb = obs_to_rgb(st.session_state["last_obs"])
                H, W, _ = rgb.shape
                fig, ax = plt.subplots(figsize=SIZE_MAP[board_size])
                ax.imshow(rgb, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"Score: {st.session_state['score']}  {'[GAME OVER]' if st.session_state['done'] else ''}")
                ph.pyplot(fig, use_container_width=False)
                time.sleep(1.0/max(1,fps))

        if st.button("Un paso (DQN)"):
            s = st.session_state["last_obs"]
            with torch.no_grad():
                ts = torch.from_numpy(s).unsqueeze(0).to(device)
                q = policy(ts)
                a = int(torch.argmax(q, dim=1).item())
            do_step(a)
            render_current()

# =============== Entrenar DQN ===============
def train_dqn(
    episodes=400, grid_size=12, max_steps=300,
    gamma=0.99, lr=1e-3,
    batch_size=64, buffer_size=50_000, start_learn=1000,
    target_update_freq=1000, eps_start=1.0, eps_end=0.05, eps_decay=0.995,
    seed=42, device=None, progress_cb=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv(grid_size, grid_size, max_steps, seed)
    obs = env.reset()
    C,H,W = obs.shape
    nA = env.actions

    policy_net = DQNCNN(C, nA).to(device)
    target_net = DQNCNN(C, nA).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    opt = optim.Adam(policy_net.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, (C,H,W))

    eps = eps_start
    step_count = 0
    returns = []

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for ep in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        for t in range(max_steps):
            step_count += 1
            # ε-greedy
            if np.random.rand() < eps:
                a = np.random.randint(nA)
            else:
                with torch.no_grad():
                    ts = torch.from_numpy(s).unsqueeze(0).to(device)
                    q = policy_net(ts)
                    a = int(torch.argmax(q, dim=1).item())

            res = env.step(a)
            ns, r = res.obs, res.reward
            done = res.terminated or res.truncated
            buf.push(s, a, r, ns, done)
            ep_ret += r
            s = ns

            if len(buf) >= start_learn:
                bs, ba, br, bns, bd = buf.sample(batch_size)
                bs  = torch.from_numpy(bs).to(device)
                ba  = torch.from_numpy(ba).to(device)
                br  = torch.from_numpy(br).to(device)
                bns = torch.from_numpy(bns).to(device)
                bd  = torch.from_numpy(bd.astype(np.float32)).to(device)

                qsa = policy_net(bs).gather(1, ba.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    max_next = target_net(bns).max(1)[0]
                    target = br + gamma * (1.0 - bd) * max_next
                loss = nn.SmoothL1Loss()(qsa, target)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                opt.step()

                if step_count % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done: break

        eps = max(eps_end, eps * eps_decay)
        returns.append(ep_ret)
        if progress_cb:
            ma = moving_average(returns, 50)
            progress_cb(ep+1, episodes, ep_ret, (ma[-1] if len(ma)>0 else None), eps)

    return policy_net.state_dict(), (C,H,W,nA), returns

if mode == "Entrenar DQN":
    st.subheader("Entrenamiento DQN (CNN)")
    col = st.columns(3)
    episodes = col[0].number_input("Episodios", 50, 5000, 600, 50)
    gamma    = col[1].slider("γ", 0.8, 0.999, 0.99, 0.001)
    lr       = col[2].select_slider("lr", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
    col2 = st.columns(3)
    batch_size = col2[0].select_slider("batch_size", options=[32, 64, 128], value=64)
    buf_size   = col2[1].select_slider("buffer_size", options=[20000, 50000, 100000], value=50000)
    target_upd = col2[2].select_slider("target_update_freq (steps)", options=[500,1000,2000], value=1000)

    col3 = st.columns(3)
    eps0 = col3[0].slider("ε inicial", 0.0, 1.0, 1.0, 0.01)
    eps_end = col3[1].slider("ε mínimo", 0.0, 0.2, 0.05, 0.01)
    eps_decay = col3[2].slider("ε decay", 0.90, 0.999, 0.995, 0.001)

    st.write(f"Device: **{device}**")

    prog = st.progress(0)
    status = st.empty()

    def cb(ep, total, ret, ma, eps):
        prog.progress(min(1.0, ep/total))
        if ma is None:
            status.text(f"[Ep {ep}/{total}] Return: {ret:.2f} | eps={eps:.3f}")
        else:
            status.text(f"[Ep {ep}/{total}] Return: {ret:.2f} | MA50={ma:.2f} | eps={eps:.3f}")

    if st.button("Entrenar"):
        weights, dims, returns = train_dqn(
            episodes=int(episodes), grid_size=grid, max_steps=int(max_steps),
            gamma=float(gamma), lr=float(lr),
            batch_size=int(batch_size), buffer_size=int(buf_size),
            start_learn=1000, target_update_freq=int(target_upd),
            eps_start=float(eps0), eps_end=float(eps_end), eps_decay=float(eps_decay),
            seed=int(seed), device=device, progress_cb=cb
        )
        st.session_state["dqn_weights"] = weights
        st.session_state["dqn_dims"] = dims
        st.session_state["returns"] = returns

        fig, ax = plt.subplots(figsize=(7,3.5))
        ma = moving_average(returns, 50)
        ax.plot(ma)
        ax.set_title("Convergencia DQN (Reward por episodio, MA=50)")
        ax.set_xlabel("Episodios"); ax.set_ylabel("Reward (MA=50)"); ax.grid(True)
        st.pyplot(fig); plt.close(fig)

        st.success("¡Entrenamiento finalizado! Andá a 'Inferencia DQN' para jugar con el modelo.")

st.markdown("""
**Ayuda rápida:**  
- En *Manual*, jugás con los botones de dirección.  
- En *Entrenar DQN*, aprendés una política con CNN (PyTorch) y se muestra la **convergencia (MA=50)**.  
- En *Inferencia DQN*, el agente juega solo (greedy o ε-greedy).  
""")
