import streamlit as st
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import time

# =============================
# Estado de la app (para recargas)
# =============================
if "policy" not in st.session_state: st.session_state["policy"] = None
if "rewards" not in st.session_state: st.session_state["rewards"] = None
if "algo_name" not in st.session_state: st.session_state["algo_name"] = None
if "env_params" not in st.session_state:
    st.session_state["env_params"] = {"slip": 0.2, "max_steps": 100}

# =============================
# Entorno y utilidades
# =============================

@dataclass
class StepResult:
    next_state: int; reward: float; terminated: bool; truncated: bool; info: dict

class GridWorld:
    def __init__(self, size: int = 4, slip_prob: float = 0.2, max_steps: int = 100):
        self.size = size; self.n_states = size*size; self.n_actions = 4
        self.start_state = 0; self.holes = {5,7,11,12}; self.goal = self.n_states-1
        self.slip_prob = slip_prob; self.max_steps = max_steps; self._steps = 0
        self.state = self.start_state
    def reset(self, start_state: int = None) -> int:
        self.state = self.start_state if start_state is None else start_state
        self._steps = 0; return self.state
    def _coords(self, s: int) -> Tuple[int,int]: return s//self.size, s%self.size
    def _state_from(self, r: int, c: int) -> int: return r*self.size + c
    def _move(self, s: int, a: int) -> int:
        r,c = self._coords(s)
        if a==0: c=max(0,c-1)
        elif a==1: r=min(self.size-1,r+1)
        elif a==2: c=min(self.size-1,c+1)
        elif a==3: r=max(0,r-1)
        return self._state_from(r,c)
    def step(self, action: int) -> StepResult:
        self._steps += 1
        if np.random.rand() < self.slip_prob:
            action = np.random.choice([0,1,2,3])
        ns = self._move(self.state, action)
        terminated=False; reward=0.0
        if ns in self.holes: terminated=True
        elif ns==self.goal: terminated=True; reward=1.0
        truncated = self._steps >= self.max_steps
        self.state = ns
        return StepResult(ns, reward, terminated, truncated, {})

def epsilon_greedy(q: np.ndarray, s:int, eps:float, n_actions:int) -> int:
    return np.random.randint(n_actions) if np.random.rand()<eps else int(np.argmax(q[s]))

def moving_average(x, w=100):
    x = np.array(x, dtype=float)
    if len(x) < 2:
        return x
    if len(x) < w:
        w = max(1, len(x)//2)
    return np.convolve(x, np.ones(w)/w, mode='valid')

def behavior_prob(state, action, n_actions, epsilon_b=1.0):
    # Política de comportamiento uniforme si epsilon_b=1.0
    return (1-epsilon_b)*0.0 + epsilon_b*(1.0/n_actions)

# =============================
# Algoritmos de entrenamiento
# =============================

def train_q_learning(env, n_episodes=4000, alpha=0.8, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
    q=np.zeros((env.n_states,env.n_actions)); rewards=[]
    for _ in range(n_episodes):
        s=env.reset(); tot=0.0
        while True:
            a=epsilon_greedy(q,s,epsilon,env.n_actions); res=env.step(a)
            q[s,a]+=alpha*(res.reward + gamma*np.max(q[res.next_state]) - q[s,a])
            s=res.next_state; tot+=res.reward
            if res.terminated or res.truncated: break
        epsilon=max(epsilon_min,epsilon*epsilon_decay); rewards.append(tot)
    return q,rewards

def train_sarsa(env, n_episodes=4000, alpha=0.8, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
    q=np.zeros((env.n_states,env.n_actions)); rewards=[]
    for _ in range(n_episodes):
        s=env.reset(); a=epsilon_greedy(q,s,epsilon,env.n_actions); tot=0.0
        while True:
            res=env.step(a); sn=res.next_state; an=epsilon_greedy(q,sn,epsilon,env.n_actions)
            q[s,a]+=alpha*(res.reward + gamma*q[sn,an] - q[s,a])
            s,a=sn,an; tot+=res.reward
            if res.terminated or res.truncated: break
        epsilon=max(epsilon_min,epsilon*epsilon_decay); rewards.append(tot)
    return q,rewards

def train_mc_es(env, n_episodes=4000, gamma=0.95):
    q=np.zeros((env.n_states,env.n_actions)); rewards=[]
    rs_sum=np.zeros_like(q); rs_cnt=np.zeros_like(q)+1e-8
    def greedy(s): return int(np.argmax(q[s]))
    for _ in range(n_episodes):
        s0=np.random.randint(env.n_states); a0=np.random.randint(env.n_actions)
        ep=[]; s=env.reset(start_state=s0); first=env.step(a0); ep.append((s,a0,first.reward)); s=first.next_state
        while True:
            a=greedy(s); step=env.step(a); ep.append((s,a,step.reward)); s=step.next_state
            if step.terminated or step.truncated: break
        rewards.append(sum(r for _,_,r in ep)); G=0.0; seen=set()
        for t in reversed(range(len(ep))):
            st,at,rt=ep[t]; G=gamma*G+rt
            if (st,at) not in seen:
                rs_sum[st,at]+=G; rs_cnt[st,at]+=1.0; q[st,at]=rs_sum[st,at]/rs_cnt[st,at]; seen.add((st,at))
    return q,rewards

def generate_episode_behavior(env, epsilon_b=1.0):
    s=env.reset(); ep=[]
    while True:
        a=np.random.randint(env.n_actions); res=env.step(a); ep.append((s,a,res.reward)); s=res.next_state
        if res.terminated or res.truncated: break
    return ep

def train_mc_is_weighted_control(env, n_episodes=4000, gamma=0.95, epsilon_b=1.0):
    q=np.zeros((env.n_states,env.n_actions)); rewards=[]
    C=np.zeros_like(q)+1e-8
    for _ in range(n_episodes):
        ep=generate_episode_behavior(env, epsilon_b=epsilon_b); G=0.0; W=1.0
        rewards.append(sum(r for _,_,r in ep))
        for t in reversed(range(len(ep))):
            s,a,r=ep[t]; G=gamma*G+r; C[s,a]+=W; q[s,a]+=(W/C[s,a])*(G-q[s,a])
            if a!=int(np.argmax(q[s])): break
            b=behavior_prob(s,a,env.n_actions,epsilon_b); W*=(1.0/b)
            if W>1e8: break
    return q,rewards

def train_mc_is_ordinary_control(env, n_episodes=4000, gamma=0.95, epsilon_b=1.0):
    q=np.zeros((env.n_states,env.n_actions)); rewards=[]
    cnt=np.zeros_like(q)+1e-8
    for _ in range(n_episodes):
        ep=generate_episode_behavior(env, epsilon_b=epsilon_b); G=0.0; rho=1.0
        rewards.append(sum(r for _,_,r in ep))
        for t in reversed(range(len(ep))):
            s,a,r=ep[t]; G=gamma*G+r; cnt[s,a]+=1.0; q[s,a]+=(rho*G-q[s,a])/cnt[s,a]
            if a!=int(np.argmax(q[s])): break
            b=behavior_prob(s,a,env.n_actions,epsilon_b); rho*=(1.0/b)
            if rho>1e8: break
    return q,rewards

# =============================
# Helpers de plot y animación
# =============================

def q_to_policy(q, env):
    policy = np.full(env.n_states, -1, dtype=int)
    for s in range(env.n_states):
        if s in env.holes or s == env.goal:
            continue
        policy[s] = int(np.argmax(q[s]))
    return policy

def plot_policy_triangles(policy, env, title="Política greedy derivada (ε=0)"):
    size = env.size
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.set_title(title)
    ax.set_xlim(-0.5, size-0.5); ax.set_ylim(-0.5, size-0.5)
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.invert_yaxis()  # fila 0 arriba

    # Etiquetas S, G, X (hoyos)
    sr, sc = divmod(env.start_state, size)
    gr, gc = divmod(env.goal, size)
    ax.text(sc, sr, "S", ha="center", va="center")
    ax.text(gc, gr, "G", ha="center", va="center")
    for h in env.holes:
        r, c = divmod(h, size)
        ax.text(c, r, "X", ha="center", va="center")

    # Triángulos: 0←,1↓,2→,3↑
    markers = {0:"<", 1:"v", 2:">", 3:"^"}
    for s in range(env.n_states):
        if s in env.holes or s == env.goal or s == env.start_state:
            continue
        a = policy[s]
        if a == -1: 
            continue
        r, c = divmod(s, size)
        ax.scatter([c], [r], marker=markers[a], s=350)
    st.pyplot(fig)

def plot_convergence(rewards, title="Convergencia (reward por episodio, MA=100)"):
    fig, ax = plt.subplots(figsize=(7,3.5))
    ma = moving_average(rewards, w=100)
    ax.plot(ma)
    ax.set_title(title)
    ax.set_xlabel("Episodios")
    ax.set_ylabel("Reward (MA=100)")
    ax.grid(True)
    st.pyplot(fig)

def eval_policy(env, policy, episodes=2000):
    successes=0; steps_total=0; steps_success=0; n_success=0
    for _ in range(episodes):
        s=env.reset()
        for t in range(env.max_steps):
            a = policy[s] if policy[s] != -1 else 0
            res = env.step(a)
            steps_total += 1
            if res.terminated:
                if res.reward > 0:
                    successes += 1
                    steps_success += (t+1)
                    n_success += 1
                break
            if res.truncated:
                break
            s = res.next_state
    success_rate = successes/episodes
    avg_steps_all = steps_total/max(1,episodes)
    avg_steps_success = (steps_success/max(1,n_success)) if n_success>0 else float('nan')
    return success_rate, avg_steps_success, avg_steps_all

def draw_grid(env, policy=None, agent_state=None, trail=None, title=None,
              figsize=(3.2, 3.2), tri_size=160, agent_size=340, font_size=10, line_w=1.5):
    size = env.size
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, size-0.5); ax.set_ylim(-0.5, size-0.5)
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.invert_yaxis()

    # etiquetas S/G/X
    sr, sc = divmod(env.start_state, size)
    gr, gc = divmod(env.goal, size)
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=font_size)
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=font_size)
    for h in env.holes:
        r, c = divmod(h, size)
        ax.text(c, r, "X", ha="center", va="center", fontsize=font_size)

    # política (triángulos)
    if policy is not None:
        markers = {0:"<", 1:"v", 2:">", 3:"^"}
        for s in range(env.n_states):
            if s in env.holes or s == env.goal or s == env.start_state:
                continue
            a = policy[s]
            if a == -1: 
                continue
            r, c = divmod(s, size)
            ax.scatter([c], [r], marker=markers[a], s=tri_size, alpha=0.35)

    # traza
    if trail:
        xs, ys = [], []
        for stt in trail:
            r, c = divmod(stt, size)
            xs.append(c); ys.append(r)
        ax.plot(xs, ys, linewidth=line_w, alpha=0.7)

    # agente
    if agent_state is not None:
        r, c = divmod(agent_state, size)
        ax.scatter([c], [r], s=agent_size, facecolors='none', edgecolors='black', linewidths=2)

    if title:
        ax.set_title(title, fontsize=font_size+1)
    return fig


def animate_episode(env, policy, delay=0.15, epsilon=None, seed=None,
                    container=None,
                    figcfg=dict(figsize=(3.2,3.2), tri_size=160, agent_size=340, font_size=10, line_w=1.5)):
    if seed is not None:
        np.random.seed(int(seed)); random.seed(int(seed))
    s = env.reset()
    trail = [s]
    ph = (container.empty() if container is not None else st.empty())
    while True:
        fig = draw_grid(env, policy=policy, agent_state=s, trail=trail, title="Episodio en curso", **figcfg)
        ph.pyplot(fig, use_container_width=True); plt.close(fig)

        # acción
        if epsilon is None or epsilon <= 0:
            a = 0 if policy[s] == -1 else policy[s]
        else:
            a = np.random.randint(env.n_actions) if np.random.rand() < epsilon else (0 if policy[s] == -1 else policy[s])

        res = env.step(a)
        s = res.next_state
        trail.append(s)
        time.sleep(delay)

        if res.terminated or res.truncated:
            fig = draw_grid(env, policy=policy, agent_state=s, trail=trail, title="Episodio terminado", **figcfg)
            ph.pyplot(fig, use_container_width=True); plt.close(fig)
            break

# =============================
# UI de Streamlit
# =============================

st.set_page_config(page_title="RL en vivo: GridWorld 4x4", layout="wide")
st.title("Aprendizaje por Refuerzo en vivo — GridWorld 4×4")

with st.sidebar:
    st.header("Parámetros del entorno")
    slip = st.slider("slip_prob", 0.0, 0.5, 0.2, 0.05)
    max_steps = st.number_input("max_steps", min_value=10, max_value=500, value=100, step=10)
    seed = st.number_input("seed", min_value=0, max_value=1_000_000, value=42, step=1)

    st.header("Algoritmo")
    algo = st.selectbox("Seleccionar", ["Q-Learning", "SARSA", "MC-ES", "MC-IS (Weighted)", "MC-IS (Ordinary)"])

    st.header("Hiperparámetros")
    n_episodes = st.number_input("n_episodes", min_value=100, max_value=10000, value=4000, step=100)
    gamma = st.slider("γ (discount)", 0.0, 0.999, 0.95, 0.01)
    if algo in ["Q-Learning","SARSA"]:
        alpha = st.slider("α (learning rate)", 0.0, 1.0, 0.8, 0.05)
        eps0 = st.slider("ε inicial", 0.0, 1.0, 1.0, 0.01)
        eps_decay = st.slider("ε decay", 0.900, 0.9999, 0.999, 0.0001)
        eps_min = st.slider("ε mínimo", 0.0, 0.2, 0.01, 0.01)
    else:
        alpha = None; eps0 = None; eps_decay = None; eps_min = None
        eps_b = st.slider("ε_b (behavior policy ~ uniforme)", 0.0, 1.0, 1.0, 0.05)

# Entorno "preview" para mostrar la última política entrenada, si corresponde
env_preview = GridWorld(size=4, slip_prob=float(slip), max_steps=int(max_steps))

run = st.button("Entrenar y visualizar")

if run:
    np.random.seed(int(seed)); random.seed(int(seed))
    env = GridWorld(size=4, slip_prob=float(slip), max_steps=int(max_steps))

    if algo == "Q-Learning":
        q, rewards = train_q_learning(env, n_episodes=int(n_episodes), alpha=float(alpha), gamma=float(gamma),
                                      epsilon=float(eps0), epsilon_decay=float(eps_decay), epsilon_min=float(eps_min))
    elif algo == "SARSA":
        q, rewards = train_sarsa(env, n_episodes=int(n_episodes), alpha=float(alpha), gamma=float(gamma),
                                 epsilon=float(eps0), epsilon_decay=float(eps_decay), epsilon_min=float(eps_min))
    elif algo == "MC-ES":
        q, rewards = train_mc_es(env, n_episodes=int(n_episodes), gamma=float(gamma))
    elif algo == "MC-IS (Weighted)":
        q, rewards = train_mc_is_weighted_control(env, n_episodes=int(n_episodes), gamma=float(gamma), epsilon_b=float(eps_b))
    elif algo == "MC-IS (Ordinary)":
        q, rewards = train_mc_is_ordinary_control(env, n_episodes=int(n_episodes), gamma=float(gamma), epsilon_b=float(eps_b))
    else:
        st.error("Algoritmo no soportado.")
        st.stop()

    # Política greedy y plots
    policy = q_to_policy(q, env)

    col1, col2 = st.columns(2)
    with col1:
        plot_policy_triangles(policy, env, title=f"Política greedy derivada (ε=0) — {algo}")
    with col2:
        plot_convergence(rewards, title=f"Convergencia — {algo} (MA=100)")

  
    sr, steps_suc, steps_all = eval_policy(GridWorld(size=4, slip_prob=float(slip), max_steps=int(max_steps)), policy, episodes=2000)
    st.subheader("Métricas (evaluación greedy, 2000 episodios)")
    st.write(f"**Tasa de éxito:** {sr:.3f}  |  **Pasos promedio (éxitos):** {steps_suc if steps_suc==steps_suc else float('nan'):.3f}  |  **Pasos promedio (todos):** {steps_all:.3f}")

    st.session_state["policy"] = policy
    st.session_state["rewards"] = rewards
    st.session_state["algo_name"] = algo
    st.session_state["env_params"] = {"slip": float(slip), "max_steps": int(max_steps)}

# Si hay una política guardada y no se corre el entrenamiento, mostrarla
if st.session_state["policy"] is not None and not run:
    st.info(f"Mostrando la última política entrenada: {st.session_state['algo_name']}")
    col1, col2 = st.columns(2)
    with col1:
        plot_policy_triangles(st.session_state["policy"], env_preview, 
            title=f"Política greedy derivada (ε=0) — {st.session_state['algo_name']}")
    with col2:
        plot_convergence(st.session_state["rewards"], 
            title=f"Convergencia — {st.session_state['algo_name']} (MA=100)")

# Notas
st.markdown("""
**Notas**
- Para SARSA/Q-Learning, el entrenamiento usa **ε-greedy**; los gráficos muestran la **política greedy** derivada (ε=0).
- Para MC-ES se usan *Exploring Starts*; para MC-IS, la política de comportamiento por defecto es aproximadamente uniforme (ε_b=1.0).
- La convergencia se visualiza con media móvil (ventana=100).
""")

st.subheader("Animación del episodio")
colA, colB, colC, colD = st.columns(4)
with colA:
    fps = st.slider("Velocidad (FPS)", 1, 20, 6)
with colB:
    eps_anim = st.slider("ε de animación", 0.0, 1.0, 0.0, 0.05)
with colC:
    seed_anim = st.number_input("Semilla animación", 0, 1_000_000, int(seed))
with colD:
    size_choice = st.selectbox("Tamaño", ["S", "M", "L"], index=0)

size_map = {
    "S": dict(figsize=(3.2,3.2), tri_size=140, agent_size=300, font_size=10, line_w=1.3),
    "M": dict(figsize=(4.5,4.5), tri_size=200, agent_size=420, font_size=11, line_w=1.8),
    "L": dict(figsize=(5.5,5.5), tri_size=260, agent_size=600, font_size=12, line_w=2.0),
}

btn_disabled = st.session_state["policy"] is None
col_anim, _ = st.columns([1, 2])  # 1/3 del ancho para la animación

if st.button("Animar episodio", disabled=btn_disabled):
    if st.session_state["policy"] is None:
        st.warning("Primero entrená un modelo con el botón 'Entrenar y visualizar'.")
    else:
        p = st.session_state["policy"]
        slip_saved = st.session_state["env_params"]["slip"]
        max_steps_saved = st.session_state["env_params"]["max_steps"]
        env_anim = GridWorld(size=4, slip_prob=slip_saved, max_steps=max_steps_saved)
        animate_episode(
            env_anim,
            p,
            delay=1.0/max(fps, 1),
            epsilon=(eps_anim if eps_anim > 0 else None),
            seed=seed_anim,
            container=col_anim,                
            figcfg=size_map[size_choice]       
        )
