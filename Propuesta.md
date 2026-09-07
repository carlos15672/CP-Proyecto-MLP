# Entrenamiento de un MLP con paralelismo de datos (multiprocessing → DDP)

## 1. Descripción del Problema

El objetivo es resolver un problema de clasificación multiclase utilizando una Red Neuronal Artificial de tipo Perceptrón Multicapa (MLP). El modelo debe ser capaz de recibir una imagen de baja resolución de una prenda de vestir y clasificarla correctamente en una de las diez categorías posibles (camisetas, pantalones, abrigos, zapatillas, etc.). Más allá de la exactitud del modelo, el propósito central es demostrar cómo el tiempo de entrenamiento puede reducirse mediante técnicas de paralelismo de datos, transitando de una ejecución secuencial a un entorno distribuido, y medir con qué eficiencia y a qué costo de comunicación se logra.

## 2. Naturaleza de los Datos

Se utilizará el conjunto de datos **Fashion-MNIST**, el cual se ha consolidado como el reemplazo estándar del MNIST clásico para evaluar algoritmos de Machine Learning.

- **Volumen:** 60,000 imágenes para entrenamiento y 10,000 para pruebas.
- **Dimensiones:** imágenes de 28×28 píxeles en escala de grises (un solo canal).
- **Complejidad:** al representar las imágenes como vectores 1D de 784 elementos, el dataset es manejable para un MLP en CPU, pero lo suficientemente grande (cerca de mil iteraciones por época con lotes de 64) para que la distribución de la carga de trabajo muestre diferencias medibles en el tiempo de ejecución.

**Modelo:** MLP 784 → 2048 → 1024 → 10 con activación ReLU, entropía cruzada y SGD con momento. Se eligen capas ocultas amplias de forma deliberada: con un modelo pequeño el costo de sincronizar gradientes domina sobre el cómputo y el paralelismo no aporta mejora; con capas anchas el cómputo por paso es suficiente para que el reparto entre procesos se traduzca en reducción de tiempo.

## 3. Justificación del Paralelismo

El entrenamiento por descenso de gradiente con mini-lotes es paralelizable porque el gradiente de la pérdida sobre un lote es un **promedio de gradientes independientes**:

> ∇L(θ) = (1/B) · Σᵢ ∇ℓᵢ(θ)

Cada término depende únicamente de su muestra y de los mismos parámetros θ, por lo que distintos procesos pueden calcular sumas parciales sobre subconjuntos disjuntos del lote y combinarlas sin alterar el resultado. La arquitectura de paralelismo de datos aprovecha esta propiedad de la siguiente manera:

1. El modelo base se replica en múltiples unidades de procesamiento (núcleos de CPU o múltiples GPUs).
2. El lote se divide entre las réplicas: en lugar de que un solo procesador vea un lote de 256 imágenes, con 4 procesadores cada uno recibe 64 imágenes diferentes simultáneamente.
3. Cada procesador realiza el *forward pass* y el *backward pass* (cálculo de gradientes) sobre su fragmento de datos.
4. Los gradientes de todos los procesadores se promedian y sincronizan mediante una operación *all-reduce* antes de actualizar los pesos, garantizando que todas las réplicas mantengan la misma configuración en la siguiente iteración.

Lo que **no** se paraleliza es la secuencia de iteraciones (θₜ₊₁ depende de θₜ); por ello el paralelismo ocurre dentro de cada paso y no entre pasos. La sincronización de gradientes, la carga de datos y la actualización de pesos constituyen la fracción serial que acota el *speedup* alcanzable (Ley de Amdahl), y es precisamente lo que se medirá.

## 4. Versiones del Proyecto

Para evidenciar el impacto del paralelismo, el desarrollo se dividirá en tres fases:

- **V1 – Secuencial (Baseline):** entrenamiento estándar en un solo proceso. Servirá como referencia de tiempo (T₁) y de exactitud.
- **V2 – Multiprocessing local:** paralelismo de datos a nivel de CPU con `torch.multiprocessing`, implementando manualmente el promedio de gradientes entre procesos (memoria compartida), para aprovechar los múltiples núcleos de una máquina estándar y entender el mecanismo antes de usar una biblioteca.
- **V3 – Distributed Data Parallel (DDP):** migración a `torch.distributed` + `DistributedDataParallel`. Se validará primero en CPU con backend *gloo* (mismo código, varios procesos) y después en Kaggle con backend *NCCL* sobre 2 GPUs T4, el único entorno gratuito que ofrece más de una GPU.

**Métricas:** tiempo por época, *throughput* (imágenes/s), *speedup* S(p) = T₁/Tₚ, eficiencia E(p) = S(p)/p, exactitud en el conjunto de prueba (debe coincidir con V1) y descomposición del tiempo en cómputo vs. comunicación. Como extensión opcional se repetirá el estudio con CIFAR-10 (32×32 RGB, 3,072 entradas) para observar cómo cambia la escalabilidad al aumentar el cómputo por muestra.
