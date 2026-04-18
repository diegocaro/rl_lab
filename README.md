# RL Lab

Un laboratorio interactivo para experimentar con algoritmos de aprendizaje por refuerzo. Actualmente incluye un agente Q-learning que aprende a equilibrar un péndulo invertido con visualización en tiempo real.

![Modo entrenamiento mostrando el péndulo y el mapa de calor de la tabla Q](assets/q-learning-pendulum.mov)

## ¿De qué se trata?

RL Lab es un entorno para experimentar con algoritmos de aprendizaje por refuerzo. El entorno actual entrena un agente Q-learning tabular para mantener un péndulo invertido en equilibrio. La simulación corre en una ventana Pygame dividida en dos paneles: la física del péndulo a la izquierda y un mapa de calor de los valores Q en tiempo real a la derecha. En cualquier momento puedes alternar entre el modo entrenamiento y el modo de observación de la política aprendida.

El proyecto está construido sobre un ciclo de simulación genérico y una clase base abstracta `Simulation`, de modo que se pueden agregar nuevos entornos sin tocar la lógica de entrenamiento.

## Características

- **Q-learning tabular** con exploración epsilon-greedy e hiperparámetros configurables
- **Visualización en tiempo real de la tabla Q** — mapa de calor 2D de valores estado × acción más un histograma de frecuencia de acciones
- **Espacio de estados flexible** — ángulo + velocidad angular, o solo ángulo con `--no-speed`
- **Dos modos de recompensa** — recompensa simple por mantenerse en pie, o recompensa moldeada que penaliza los movimientos bruscos (`--better-reward`)
- **Diseño modular** — el ciclo de juego funciona con cualquier clase que implemente el ABC `Simulation`
- **Demo de física interactiva** — controla el péndulo a mano desde el teclado

## Instalación

Requiere Python 3.13+ y [uv](https://github.com/astral-sh/uv).

```bash
git clone <url-del-repo>
cd rl_lab
uv sync
```

## Uso

### Entrenar el agente

```bash
uv run python -m rl_lab.main
```

| Opción | Descripción |
|---|---|
| `--no-speed` | Ignora la velocidad angular; el estado es solo el ángulo |
| `--better-reward` | Recompensa moldeada con bonificaciones por suavidad (menos sacudidas) |

### Controles

| Tecla | Acción |
|---|---|
| `Espacio` | Alternar entre entrenamiento y observación |
| `R` | Reiniciar el episodio actual |
| `Q` / `Esc` | Salir |

### Demo de física

Ejecuta el motor de física por separado y controla el péndulo a mano:

```bash
uv run python -m rl_lab.pendulum.physics
```

Las flechas aplican torque; `R` reinicia; `Q` sale.

## Cómo funciona

### Reinforcement Learning

El reinforcement learning consiste en aprender qué hacer — cómo mapear situaciones a acciones — de modo que se maximice una señal de recompensa numérica. A diferencia del aprendizaje supervisado, el agente no recibe instrucciones sobre qué acción tomar: debe descubrir por sí mismo qué acciones producen más recompensa, explorando el entorno. Sus decisiones pueden afectar no solo la recompensa inmediata, sino también el estado siguiente y todas las recompensas futuras.

Esto introduce el dilema fundamental: el balance entre **exploración** (probar acciones nuevas para descubrir mejores estrategias) y **explotación** (elegir las acciones que ya sabe que funcionan). Ninguna de las dos puede seguirse de forma exclusiva sin fallar en la tarea.

> *"Reinforcement learning problems involve learning what to do — how to map situations to actions — so as to maximize a numerical reward signal."* — Sutton & Barto, [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

### Q-Learning

Algoritmo *model-free* y *off-policy* que aprende por prueba y error sin necesitar un modelo del entorno. Mantiene una tabla Q con el valor esperado de cada par (estado, acción) y la actualiza con la ecuación de Bellman:

```
Q[s, a] += α · (r + γ · max(Q[s']) − Q[s, a])
```

`α` controla cuánto peso tiene la nueva información; `γ` define la importancia de las recompensas futuras. La exploración se maneja con ε-greedy: el agente elige acciones aleatorias con probabilidad ε, que decrece conforme aprende.

→ [Explicación detallada de Q-learning](docs/q-learning.md)

Hiperparámetros por defecto:

| Parámetro | Valor |
|---|---|
| Tasa de aprendizaje α | 0.15 |
| Descuento γ | 0.99 |
| Decaimiento de epsilon | 0.995 |

### Espacio de estados y acciones

El estado continuo del péndulo (ángulo, velocidad angular) se discretiza en una grilla de 32 × 16. El espacio de acciones tiene 20 torques uniformemente distribuidos en [−40, +40] N·m.

### Física del péndulo

El motor integra las ecuaciones de movimiento de cuerpo rígido con el método de Euler, incluyendo gravedad (9.81 m/s²), amortiguamiento y torque aplicado. Un episodio termina cuando la tensión en la varilla supera los 200 N, es decir, cuando el péndulo oscila con demasiada fuerza.

## Estructura del proyecto

```
src/rl_lab/
├── main.py                  # Bucle de juego genérico de Q-learning
├── agents/
│   └── q_agent.py           # Agente Q-learning tabular
├── engine/
│   ├── simulation.py        # Clase base abstracta de simulación
│   └── policy_renderer.py   # Renderizador del mapa de calor de valores Q
└── pendulum/
    ├── physics.py           # Motor de física del péndulo
    ├── env.py               # Wrapper de entorno estilo Gym
    └── sim.py               # Implementación concreta de PendulumSim
```

## Cómo extender

Para agregar un nuevo entorno, implementa el ABC `Simulation` en `src/rl_lab/engine/simulation.py` y pasa una instancia al ciclo de juego en `main.py`. El agente y el renderizador son completamente independientes del entorno.

## Desarrollo

```bash
uv sync
pre-commit install
```

La calidad del código se mantiene con Ruff (linting y formato) y Pyright (tipado estático).

## Créditos

Desarrollado por Diego Caro con asistencia de [Claude Sonnet 4.6](https://www.anthropic.com/claude) a través de [Claude Code](https://claude.ai/code).
