# Q-Learning

Q-learning es un algoritmo de aprendizaje por refuerzo *model-free* y *off-policy* que permite a los agentes aprender acciones óptimas por prueba y error, maximizando recompensas futuras sin necesitar un modelo del entorno.

## Conceptos clave

- **Agente y entorno:** el agente interactúa con el entorno eligiendo acciones y recibiendo recompensas.
- **Tabla Q:** la memoria del agente — almacena el valor esperado de cada par (estado $s$, acción $a$).
- **Función Q:** $Q(s, a)$ representa la recompensa total esperada al tomar la acción $a$ en el estado $s$.
- **Ecuación de Bellman:** la fórmula central para actualizar los valores Q a partir de la recompensa inmediata y el mejor valor futuro posible:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

donde:

- $\alpha$ — tasa de aprendizaje: cuánto peso tiene la nueva información frente a la antigua.
- $\gamma$ — factor de descuento: importancia de las recompensas futuras (0 = solo el presente, 1 = futuro igual de importante).
- $r$ — recompensa inmediata recibida al tomar la acción $a$.
- $Q(s', a')$ — fila de la tabla Q correspondiente al estado siguiente $s'$, el estado en que el agente aterriza tras ejecutar $a$. $\max_{a'} Q(s', a')$ toma la mejor acción posible desde ese estado según el conocimiento actual.

Eso último es lo que hace Q-learning *off-policy*: aunque el agente no vaya a elegir necesariamente la acción óptima en $s'$ (puede explorar), igual la usa como referencia para actualizar el valor actual. Aprende la política óptima independientemente de cómo se esté comportando en ese momento.

## Proceso de entrenamiento

1. Inicializar la tabla Q en cero.
2. Elegir una acción con estrategia $\varepsilon$-greedy (exploración aleatoria ocasional).
3. Ejecutar la acción, recibir recompensa $r$ y observar el nuevo estado $s'$.
4. Actualizar la tabla Q con la ecuación de Bellman.
5. Repetir hasta que la política converja.

## Características

- *Model-free:* aprende puramente desde la experiencia, sin modelar el entorno.
- *Off-policy:* aprende la política óptima independientemente de las acciones actuales del agente.
- *Espacio de acciones discreto:* funciona mejor con conjuntos finitos de estados y acciones.
- *Limitación:* con espacios de estado muy grandes se vuelve ineficiente; para esos casos se usan redes neuronales ([Deep Q-Network, Mnih et al., 2013](https://arxiv.org/abs/1312.5602)).

## Referencias

- Watkins, C.J.C.H. & Dayan, P. (1992). [Q-learning](https://link.springer.com/article/10.1007/BF00992698). *Machine Learning*, 8, 279–292. — paper original de Q-learning.
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press. — origen de la ecuación de Bellman.
- Sutton, R.S. & Barto, A.G. (2018). [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). MIT Press. — libro de referencia del área.
- Mnih, V. et al. (2013). [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602). DeepMind. — introducción de DQN.
