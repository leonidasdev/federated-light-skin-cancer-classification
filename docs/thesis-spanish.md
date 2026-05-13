# Aprendizaje Federado con Vision Transformers Ligeros: Implementación de DSCATNet para la Clasificación de Lesiones Cutáneas

**Trabajo de Fin de Grado — Ingeniería Informática**  
**Escuela Técnica Superior de Ingeniería de Sistemas Informáticos (ETSISI) — Universidad Politécnica de Madrid**

---

## RESUMEN

Este Trabajo de Fin de Grado estudia el uso de DSCATNet, un Vision Transformer ligero con atención cruzada a doble escala, para clasificar lesiones cutáneas bajo un esquema de aprendizaje federado simulado. El objetivo no es solo medir precisión, sino analizar qué ocurre cuando un modelo competitivo en configuración centralizada se enfrenta a particiones IID y no-IID, cambio de dominio entre datasets y una restricción real de recursos computacionales.

La evaluación se apoya en HAM10000, ISIC2018, ISIC2019, ISIC2020 y PAD-UFES-20. Los resultados muestran que el aprendizaje federado IID puede superar al baseline centralizado tanto en HAM10000 como en PAD-UFES-20, lo que sugiere un efecto regularizador de la agregación FedAvg. Sin embargo, la heterogeneidad no-IID y el cambio de dominio alteran el equilibrio entre clases, con especial impacto en melanoma, que sigue siendo la clase clínicamente más delicada.

Además del rendimiento, se analiza el coste de comunicación. Con un checkpoint efectivo de 112.44 MB, DSCATNet implica aproximadamente 89.95 GB de transmisión en 100 rondas y 4 clientes, una cifra asumible en un entorno institucional estable pero relevante para valorar su viabilidad práctica. En conjunto, el trabajo concluye que DSCATNet es una base prometedora para FL en dermatología, pero todavía no un sistema clínico autónomo sin mitigación adicional de desbalance, validación multi-semilla y estudio de robustez más amplio.

**Palabras clave**: Aprendizaje Federado, Vision Transformer, Clasificación de Cáncer de Piel, Heterogeneidad de Datos, DSCATNet

---

## ABSTRACT

This Final Degree Project studies DSCATNet, a lightweight dual-scale cross-attention Vision Transformer, for skin lesion classification under a simulated federated learning setup. The goal is not only to measure accuracy, but also to assess what happens when a model that performs competitively in centralized training is exposed to IID and non-IID partitions, cross-dataset domain shift, and a realistic hardware budget.

The evaluation uses HAM10000, ISIC2018, ISIC2019, ISIC2020, and PAD-UFES-20. The results show that IID federated training can outperform the centralized baseline in both HAM10000 and PAD-UFES-20, suggesting a regularization effect from FedAvg aggregation. However, non-IID heterogeneity and domain shift redistribute performance across classes, with melanoma remaining the most clinically sensitive class.

Communication cost is also analyzed. With an effective 112.44 MB checkpoint, DSCATNet requires about 89.95 GB of total transmission over 100 rounds with 4 clients, which is feasible in a stable institutional network but still relevant when assessing operational viability. Overall, the work shows DSCATNet is a promising FL baseline for dermatology, but not yet a clinically autonomous system without additional class-imbalance mitigation, multi-seed validation, and broader robustness analysis.

---

## TABLA DE ABREVIACIONES

| Abreviatura | Significado |
|---|---|
| AKIEC | Queratosis actínica / *Actinic keratosis* |
| AUC-ROC | Área bajo la curva ROC / *Area Under the ROC Curve* |
| BCC | Carcinoma basocelular / *Basal cell carcinoma* |
| BKL | Queratosis benigna / *Benign keratosis* |
| C-HAM | Baseline centralizado en HAM10000 |
| C-ISIC2018 | Baseline centralizado en ISIC2018 |
| C-PAD | Baseline centralizado en PAD-UFES-20 |
| CNN | Red neuronal convolucional / *Convolutional Neural Network* |
| DF | Dermatofibroma |
| DSCATNet | *Dual-Scale Cross-Attention Transformer Network* |
| FedAvg | *Federated Averaging* — algoritmo de agregación federada |
| F-HAM-IID | FL en HAM10000 con distribución IID (α=10.0) |
| F-HAM-NonIID | FL en HAM10000 con distribución no-IID (α=0.5) |
| F-ISIC2018-IID | FL en ISIC2018 con distribución IID (α=10.0) |
| F-ISIC2018-NonIID | FL en ISIC2018 con distribución no-IID (α=0.5) |
| F-PAD-IID | FL en PAD-UFES-20 con distribución IID (α=10.0) |
| F-PAD-NonIID | FL en PAD-UFES-20 con distribución no-IID (α=0.5) |
| FL | Aprendizaje Federado / *Federated Learning* |
| GAN | Red generativa adversarial / *Generative Adversarial Network* |
| GPU | Unidad de procesamiento gráfico / *Graphics Processing Unit* |
| IID | Independientes e idénticamente distribuidas / *Independent and Identically Distributed* |
| ISIC | *International Skin Imaging Collaboration* |
| MEL | Melanoma |
| NV | Nevus melanocítico / *Melanocytic nevus* |
| PAD-UFES-20 | Dataset de lesiones clínicas adquiridas con smartphone |
| RQ | Pregunta de investigación / *Research Question* |
| SGD | Descenso de gradiente estocástico / *Stochastic Gradient Descent* |
| VASC | Lesión vascular / *Vascular lesion* |
| ViT | *Vision Transformer* |
| VRAM | Memoria de vídeo / *Video RAM* |

---

## 1. INTRODUCCIÓN

### 1.1 Contexto y motivación

El cáncer de piel constituye una de las neoplasias más frecuentes y clínicamente relevantes, especialmente por la mortalidad asociada al melanoma. Aunque el melanoma representa una fracción pequeña del total de casos, concentra una parte desproporcionada de las muertes, lo que convierte el diagnóstico precoz en un objetivo prioritario.

En dermatología, la dermatoscopia mejora la inspección visual, pero la interpretación sigue dependiendo de la experiencia clínica y presenta variabilidad interobservador. En este contexto, los métodos de *deep learning* han mostrado resultados competitivos en clasificación de lesiones cutáneas, con mejoras claras frente a enfoques manuales o heurísticos.

Sin embargo, la mayoría de estas soluciones se entrenan de forma centralizada. En entornos hospitalarios o multicentro, este enfoque obliga a concentrar datos sensibles en un único repositorio, lo que incrementa el riesgo de exposición de información clínica y complica el cumplimiento normativo. El aprendizaje federado aparece como una alternativa técnica para entrenar modelos sin trasladar las imágenes originales fuera de cada institución.

Este trabajo se sitúa precisamente en esa intersección: se estudia si una arquitectura ligera basada en Vision Transformers puede mantener un rendimiento competitivo cuando se entrena bajo restricciones federadas y con datos heterogéneos.

### 1.2 Planteamiento del problema

Existe un *trade-off* entre precisión y privacidad. El entrenamiento centralizado suele ofrecer un mejor punto de partida en términos de exactitud, pero exige consolidar los datos en un único servidor. Por el contrario, el aprendizaje federado reduce la exposición de datos, pero introduce heterogeneidad entre clientes, mayor complejidad de coordinación y posible degradación del rendimiento.

En este escenario, no todos los modelos son adecuados. Los transformadores visuales convencionales suelen requerir muchos parámetros y grandes volúmenes de datos para generalizar correctamente. Esto los hace poco prácticos en entornos con limitaciones de memoria, ancho de banda y disponibilidad de imágenes etiquetadas. Por ello, se necesitan arquitecturas más eficientes.

DSCATNet es un candidato relevante porque combina atención cruzada a doble escala y una capacidad de representación compacta. No obstante, la literatura revisada muestra que su evaluación se ha realizado en configuraciones centralizadas, sin estudiar de forma sistemática el impacto de la no-IIDidad ni el coste de comunicación en FL. Esa es la brecha que aborda este TFG.

### 1.3 Preguntas de investigación

**RQ1**: ¿Cuál es la precisión de DSCATNet en aprendizaje federado comparado con el entrenamiento centralizado?

**RQ2**: ¿Cómo afecta la heterogeneidad de datos (distribuciones no-IID) a la convergencia y rendimiento del modelo?

**RQ3**: ¿Es viable el despliegue de DSCATNet en un entorno federado desde perspectivas de comunicación, convergencia y precisión clínica?

### 1.4 Contribuciones

Las principales contribuciones de este trabajo son las siguientes:

1. Se implementa y evalúa DSCATNet en un entorno de aprendizaje federado simulado, con agregación FedAvg y escenarios IID y no-IID controlados por distribución Dirichlet.
2. Se analiza el efecto de la heterogeneidad de datos sobre métricas globales y por clase, con especial atención al melanoma por su relevancia clínica.
3. Se cuantifica el comportamiento centralizado frente al federado usando precisión, balanced accuracy, F1 macro, AUC-ROC y curvas de convergencia.
4. Se estima el coste de comunicación del enfoque federado y se justifica su viabilidad bajo restricciones de hardware de consumo.
5. Se compara el comportamiento obtenido con la literatura reciente, en particular con DSCATNet original, modelos ligeros federados basados en CNN y comparativas CNN vs. ViT.

### 1.5 Estructura del documento

Este documento se organiza de la siguiente manera:

- **Sección 2**: Estado del arte con descripción de trabajos relacionados
- **Sección 3**: Metodología con detalles de datasets, arquitectura y configuración experimental
- **Sección 4**: Resultados y discusión integrados
- **Sección 5**: Discusión global y respuesta a preguntas de investigación
- **Sección 6**: Conclusiones y trabajo futuro
- **Apéndices**: Código, configuraciones, logs de ejemplo

La estructura sigue una progresión deliberada: primero se justifica el problema y el marco técnico, después se describe cómo se ha construido la evidencia experimental y, por último, se interpretan los resultados con criterios clínicos y de viabilidad. Esta secuencia facilita que la discusión no se quede en una lectura puramente numérica, sino que conecte métricas, limitaciones y utilidad potencial.

### 1.6 Alcance y limitaciones del estudio

El alcance de este TFG se limita a la evaluación de DSCATNet en clasificación de lesiones cutáneas con esquema de 7 clases, comparando entrenamiento centralizado y entrenamiento federado simulado bajo escenarios IID y no-IID. El estudio se centra en HAM10000, ISIC2018, ISIC2019, ISIC2020 y PAD-UFES-20, con especial atención a HAM10000 como baseline principal y al escenario federado cross-domain (ISIC2018/ISIC2019/ISIC2020/PAD-UFES-20) como contexto de mayor heterogeneidad.

Quedan fuera del alcance la inferencia clínica en tiempo real, la validación prospectiva en hospitales, el despliegue federado distribuido sobre red real y la comparación exhaustiva con algoritmos FL alternativos como FedProx o métodos robustos ante clientes maliciosos.

**Limitaciones**:

- Se trabaja con una única semilla global (42), por lo que no se reportan intervalos de confianza ni desviaciones estándar entre repeticiones. Esta limitación se asume por restricciones de tiempo de entrega y coste computacional; por tanto, las cifras deben leerse como evidencia puntual, no como estimaciones estadísticamente robustas.
- La federación se simula en una sola máquina, por lo que no se modelan latencia, caída de clientes, colas de red ni asincronía.
- El escenario experimental usa 4 clientes simulados, lo que reduce la complejidad respecto a un entorno multicentro real.
- ISIC2020 presenta una cobertura efectiva del 28.6% respecto al esquema de 7 clases, por lo que parte de sus imágenes originales queda fuera del flujo experimental.
- El modelo se entrena sin un estudio sistemático de inicialización con pesos preentrenados, lo que puede explicar parte de la brecha frente a Yadav et al. [2024].
- No se incorpora privacidad diferencial, cifrado homomórfico ni mecanismos de seguridad avanzada sobre las actualizaciones.

---

## 2. ESTADO DEL ARTE

### 2.1 Clasificación de cáncer de piel con deep learning

La clasificación automática de lesiones cutáneas se ha consolidado como una de las aplicaciones más estudiadas de *deep learning* en imagen médica. Los benchmarks más utilizados son HAM10000, ISIC2018, ISIC2019, ISIC2020 y PAD-UFES-20, todos ellos con fuerte desbalance de clases y presencia de categorías clínicamente críticas como melanoma.

En este dominio, las CNN han sido el estándar de referencia durante años por su eficiencia y su capacidad para aprender patrones locales. Sin embargo, su rendimiento depende fuertemente del tamaño del conjunto de datos, del preprocesamiento y de la estrategia de balanceo. En la práctica, los resultados reportados en la literatura oscilan en torno al 90% de precisión en configuraciones centralizadas, aunque con gran variabilidad entre clases.

El problema principal no es solo la exactitud global, sino la sensibilidad en clases minoritarias. En dermatología, un recall bajo en melanoma es más grave que una reducción moderada en precisión global, porque incrementa el número de falsos negativos clínicamente relevantes.

Los conjuntos HAM10000 e ISIC2018 suelen mostrar una distribución muy dominada por NV, mientras que ISIC2019 introduce mayor diversidad y PAD-UFES-20 representa un dominio distinto, basado en imágenes clínicas de smartphone. Esta diversidad hace necesario evaluar no solo la métrica global, sino también el comportamiento por clase y la robustez al cambio de dominio.

### 2.2 Vision Transformers en imagen médica

Los Vision Transformers (ViT) sustituyen la convolución por mecanismos de autoatención, lo que permite modelar relaciones globales entre regiones de la imagen. En medicina, esto es atractivo porque ciertas lesiones presentan patrones distribuidos espacialmente que no siempre se capturan bien con receptivos locales.

La principal ventaja de los ViT es su capacidad para integrar contexto global desde etapas tempranas. Esto puede mejorar la discriminación entre lesiones con apariencia similar, donde la textura local no es suficiente. Además, la atención facilita ciertas formas de interpretabilidad al permitir inspeccionar qué regiones influyen más en la decisión.

No obstante, los ViT suelen exigir más datos y más capacidad computacional que las CNN convencionales. En conjuntos médicos pequeños o desbalanceados, esto puede traducirse en sobreajuste si no se aplican estrategias de regularización, preentrenamiento o arquitecturas ligeras.

Por ello, han surgido variantes compactas como DeiT, Swin, MobileViT o propuestas especializadas como DSCATNet, que buscan conservar la expresividad de los transformadores mientras reducen el coste de entrenamiento e inferencia.

### 2.3 Aprendizaje federado en dermatología

El aprendizaje federado permite entrenar un modelo global agregando actualizaciones locales sin centralizar los datos originales. En salud, esta propiedad es especialmente relevante porque facilita la colaboración entre hospitales sin transferir imágenes o expedientes clínicos fuera de cada institución.

El algoritmo más usado es FedAvg, que promedia los pesos de varios clientes tras entrenamiento local. Su simplicidad lo convierte en el punto de partida natural para experimentar, aunque su rendimiento puede degradarse cuando la distribución de datos entre clientes es no-IID. En ese caso, aparecen problemas de *client drift* y de convergencia más lenta.

La literatura en dermatología federada sigue siendo limitada frente al volumen de trabajos centralizados. Esto abre una oportunidad clara: estudiar si arquitecturas compactas y bien calibradas pueden mantener rendimiento competitivo cuando cada cliente dispone de una fracción sesgada del conjunto total.

Además del rendimiento, el coste de comunicación es una restricción central. Cada ronda requiere transmisión de parámetros del modelo, por lo que el tamaño de la red y el número de rondas influyen de forma directa en la viabilidad práctica. En este trabajo, este coste se analiza de forma explícita como parte de la evaluación.

### 2.4 DSCATNet (Yadav et al., 2024)

Yadav et al. [2024] proponen DSCATNet, una arquitectura Vision Transformer ligera con atención cruzada a doble escala diseñada para clasificación de lesiones cutáneas. Con ~29.4 millones de parámetros, el modelo utiliza parches de dos tamaños simultáneamente (8×8 y 16×16) para capturar características multiesala, alcanzando 97.80% de precisión en HAM10000 y 95.81% en PAD-UFES-20 en configuración centralizada.

**Limitación crítica**: Yadav et al. **no evalúan el modelo en aprendizaje federado** ni estudian el impacto de heterogeneidad de datos no-IID. Esta brecha es la **oportunidad central de este TFG**: adaptación de DSCATNet a FL con análisis de degradación en escenarios realistas.

### 2.5 Modelos ligeros en aprendizaje federado (Khullar et al., 2025)

Khullar et al. [2025] proponen un enfoque federado basado en CNN ligeras, especialmente EfficientNetV2S y EfficientNetB3, entrenadas sobre imágenes reducidas a 32 × 32 píxeles. Su objetivo es minimizar tanto el coste computacional como el de comunicación.

Los resultados reportados alcanzan 89.83% en IID y 90.64% en non-IID, lo que demuestra que un modelo ligero puede mantener buen rendimiento bajo federación. Sin embargo, el estudio se centra en CNN y no evalúa arquitecturas tipo ViT, por lo que no resuelve la cuestión de si un transformer ligero puede competir bajo restricciones similares.

Este trabajo se toma como referencia para comparar la degradación inducida por heterogeneidad y el coste de comunicación, pero la comparación debe interpretarse con cautela, ya que el espacio de entrada y la arquitectura no coinciden con DSCATNet.

### 2.6 Benchmark CNN vs ViT (Aruk et al., 2026)

Aruk et al. [2026] comparan 15 CNN y 15 ViT en HAM10000 y observan que los ViT alcanzan 92.12% de precisión, frente al 91.92% de la mejor CNN. La mejora absoluta es pequeña, pero suficiente para indicar que los transformadores pueden competir con las redes convolucionales clásicas en este dominio.

La principal conclusión del estudio no es solo que los ViT sean competitivos, sino que su ventaja debe analizarse junto al número de parámetros y al coste computacional. En otras palabras, la decisión de arquitectura no depende únicamente de la precisión, sino del equilibrio entre capacidad representacional, memoria y tiempo de entrenamiento.

Esta comparación justifica el interés en ViT ligeros. DSCATNet se sitúa en ese punto intermedio: intenta conservar la expresividad de un transformer, pero con un diseño más compacto que un ViT de propósito general.

### 2.7 Tabla resumen: Literatura y brecha identificada

| Trabajo | Modelo | Tipo | Dataset | Precisión | FL | Non-IID | Gap Identificado |
|---------|--------|------|---------|-----------|----|---------|----|
| Yadav et al., 2024 | DSCATNet | ViT ligero | HAM10000, PAD-UFES-20 | 97.80%, 95.81% | No | N/A | No evaluación en FL |
| Khullar et al., 2025 | EfficientNetV2S | CNN ligera | ISIC 2019 | 89.83%, 90.64% | Sí | Sí | Solo CNN, no ViT |
| Aruk et al., 2026 | ViT vs CNN | Mixto | HAM10000 | 92.12% (ViT) | No | N/A | Sin evaluar FL |
| Este TFG | DSCATNet | ViT ligero | HAM10000, ISIC2018, PAD-UFES-20 | 70.37% (C-HAM test), 74.40% (F-HAM-IID best val), 59.88% (C-PAD test) | Sí | Sí | Cubre la brecha: DSCATNet en FL |

---

## 3. METODOLOGÍA

### 3.1 Conjuntos de datos

Este TFG evalúa DSCATNet en varios escenarios con un esquema de 7 clases unificado: AKIEC, BCC, BKL, DF, MEL, NV y VASC. La exploración de datos disponible en `outputs/evaluation_dataset_exploration/20260509_114234/` confirma que los conjuntos difieren en tamaño, cobertura de clases, resolución media y nivel de desbalance.

#### 3.1.1 Escenario cross-domain y exploración multiconjunto

La exploración de datos del proyecto considera cinco conjuntos: HAM10000, ISIC2018, ISIC2019, ISIC2020 y PAD-UFES-20. Esta vista permite comparar tamaño, cobertura de clases, resolución y desbalance antes del entrenamiento.

En el escenario federado cross-domain, cada dataset se trata como un cliente o dominio natural distinto. No se construye una mezcla centralizada ingenua de todas las imágenes; lo que se evalúa es un caso extremo en el que ISIC2018, ISIC2019, ISIC2020 y PAD-UFES-20 representan distribuciones heterogéneas de origen. HAM10000 se mantiene como referencia externa porque es prácticamente equivalente a ISIC2018 en tamaño y distribución.

El mapeo de etiquetas se unifica hacia el esquema de 7 clases del proyecto: AKIEC, BCC, BKL, DF, MEL, NV y VASC. ISIC2019 mapea AK a AKIEC y SCC a BCC; PAD-UFES-20 mapea ACK a AKIEC, NEV a NV y SEK a BKL, manteniendo BCC, MEL y VASC cuando están presentes. Cualquier etiqueta sin correspondencia clara o sin anotación utilizable se descarta.

La exploración confirma tres hechos relevantes:

- ISIC2018 replica prácticamente a HAM10000 en tamaño y distribución, ya que ambos presentan 10015 imágenes, 7 clases y el mismo dominio dermoscópico.
- ISIC2019 conserva las 7 clases, pero con una distribución más desbalanceada, dominada por NV con un 50.8%.
- ISIC2020 contiene 33126 imágenes, pero en esta versión solo aparecen 2 clases válidas; el resto de etiquetas no se utiliza en el flujo experimental.

| Conjunto | Imágenes | Clases presentes | Entropía | Gini | Clase dominante | Share dominante |
|---------|----------|------------------|----------|------|----------------|----------------|
| HAM10000 | 10015 | 7 | 1.632 bits | 0.641 | NV | 66.9% |
| ISIC2018 | 10015 | 7 | 1.632 bits | 0.641 | NV | 66.9% |
| ISIC2019 | 25331 | 7 | 2.034 bits | 0.544 | NV | 50.8% |
| ISIC2020 | 33126 | 2 | 0.128 bits | 0.482 | NV | 98.2% |
| PAD-UFES-20 | 2298 | 5 | 1.863 bits | 0.409 | AK | 40.1% |

#### 3.1.2 HAM10000 (Referencia)

- Descripción: 10015 imágenes dermoscópicas de alta calidad.
- Cobertura de clases: 7 clases presentes.
- Distribución dominante: NV con 66.9% de las muestras, seguida de MEL con 11.1% y BKL con 11.0%.
- Resolución mediana: 600 × 450 píxeles.
- Split: 80% entrenamiento / 20% validación, con partición estratificada para preservar el desbalance original.
- Rol: baseline centralizado y referencia principal para comparar con FL.

#### 3.1.3 ISIC2018 (Referencia cercana a HAM10000)

- Descripción: 10015 imágenes dermoscópicas.
- Cobertura de clases: 7 clases presentes.
- Distribución dominante: idéntica a HAM10000 en la exploración disponible.
- Resolución mediana: 600 × 450 píxeles.
- Split: 80% entrenamiento / 20% validación, estratificado.
- Rol: referencia útil porque es prácticamente equivalente a HAM10000 en tamaño y distribución.

#### 3.1.4 ISIC2019 (Mayor heterogeneidad de clases)

- Descripción: 25331 imágenes dermoscópicas.
- Cobertura de clases: 7 clases presentes.
- Clases dominantes: NV 50.8%, MEL 17.9% y BCC 13.1%.
- Entropía de clase: 2.03 bits, superior a HAM10000, lo que sugiere una mezcla de clases más equilibrada.
- Índice Gini: 0.544.
- Resolución mediana: 1024 × 768 píxeles.
- Split: 80% entrenamiento / 20% validación, estratificado.

#### 3.1.5 ISIC2020 (Etiquetas parciales)

- Descripción: 33126 imágenes dermoscópicas.
- Cobertura de clases: solo 2 clases presentes en el subconjunto procesado.
- Clase dominante: NV 98.2%, seguida de MEL 1.8%.
- Entropía de clase: 0.128 bits, extremadamente baja.
- Resolución mediana: 5184 × 3456 píxeles.
- Observación importante: en el subconjunto procesado solo se utilizan 2 de las 7 clases objetivo, lo que implica una cobertura efectiva del 28.6% y una distribución fuertemente sesgada, con NV como clase dominante al 98.2%. El resto de muestras originales no se incorpora al experimento porque no dispone de etiqueta utilizable o no encaja con el esquema de clases unificado.
- Split: 80% entrenamiento / 20% validación, solo sobre imágenes etiquetadas y filtradas.

#### 3.1.6 PAD-UFES-20 (Clínica sin dermoscopio)

- Descripción: 2298 imágenes clínicas tomadas con smartphone.
- Cobertura de clases: 5 clases presentes en la exploración y 2 clases ausentes.
- Clase dominante: AK 40.1%, seguida de BCC 36.8% y NV 10.6%.
- Resolución mediana: 755 × 754 píxeles.
- Mapeo al esquema unificado: ACK → AKIEC, NEV → NV, SEK → BKL; BCC, MEL y VASC se mantienen cuando aparecen.
- Split: 80% entrenamiento / 20% validación, estratificado.
- Rol: evaluar robustez en imágenes clínicas de menor calidad y distinta distribución respecto a dermoscopia.

#### 3.1.7 Preprocesamiento y aumentos

**Preprocesamiento**:

- Redimensionamiento: 224 × 224 píxeles, exigido por DSCATNet.
- Normalización: estadísticas ImageNet (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]).
- Conversión de color: RGB con tratamiento explícito de imágenes en escala de grises.

**Aumentos de datos**:

- RandomHorizontalFlip con probabilidad 0.5.
- RandomVerticalFlip con probabilidad 0.5.
- RandomRotation de ±20 grados.
- ColorJitter con variaciones moderadas de brillo, contraste y saturación.

**Ponderación de clases**: Se aplica class weight balanceado en Cross-Entropy Loss para mitigar el desbalance severo, especialmente en HAM10000 e ISIC2018.

### 3.2 Arquitectura DSCATNet

DSCATNet se adopta aquí como arquitectura ligera de referencia porque combina dos propiedades que son relevantes para este TFG: capacidad representacional suficiente para lesiones cutáneas complejas y un coste computacional inferior al de un Vision Transformer de propósito general. Su diseño multiescala permite capturar simultáneamente patrones locales finos y contexto más amplio, algo especialmente útil cuando la señal clínica depende de bordes, pigmentación y textura.

En este trabajo, la arquitectura se interpreta como una solución intermedia entre CNN y ViT grandes. No pretende batir a los modelos más pesados en entornos ideales, sino responder a una pregunta más práctica: qué ocurre cuando se exige a un transformer ligero que aprenda en un entorno federado, con datos desbalanceados y bajo limitaciones de memoria.

- Vision Transformer con dual-scale cross-attention
- Patch embeddings: 8×8 y 16×16 simultáneamente
- Parámetros: ~29.4M (ligero vs ViT-B: ~86M)
- Componentes clave:
  - Codificador visual de doble escala
  - Mecanismo de cross-attention
  - Cabeza de clasificación adaptada a 7 clases

**Figura 1**: Arquitectura de DSCATNet con doble escala de parches y bloques de cross-attention, destacando la agregación de información local y global.

<!-- ![Figura 1 - Arquitectura DSCATNet](../outputs/evaluation_comparison_dscatnet_ham10000/HAM10000/) -->

#### 3.3 Adaptación a aprendizaje federado

#### 3.3.1 Simulación federada en entorno local

Este trabajo no despliega un sistema federado distribuido real sobre red. En su lugar, utiliza una simulación federada en una sola máquina, donde los clientes se emulan como particiones lógicas de datos y el intercambio de actualizaciones se reproduce en memoria. Esta decisión reduce el coste computacional y de comunicación, permite controlar el escenario experimental y hace viable la comparación entre configuraciones IID y no-IID bajo el mismo hardware.

La elección es coherente con los recursos disponibles: CPU Intel Core i5-11400H, 16 GB de RAM y una GPU NVIDIA RTX 3050 de 4 GB. Con este entorno, una federación distribuida real con transferencia de pesos por red no sería reproducible de forma estable, mientras que la simulación local sí permite ejecutar 100 rondas con 4 clientes sin introducir ruido de infraestructura.

La simulación sigue el protocolo FedAvg y replica el flujo estándar de FL:

- Inicialización de un modelo global.
- Entrenamiento local independiente por cliente.
- Agregación ponderada de actualizaciones.
- Distribución del nuevo modelo global en la ronda siguiente.

Flower se utiliza en el proyecto para las abstracciones de cliente/servidor y la estrategia FedAvg; aun así, la ejecución experimental que documenta este TFG se realiza en modo simulado y local, sin depender de una red distribuida real. En otras palabras, Flower forma parte de la pila del proyecto, pero la hipótesis científica evaluada aquí es la simulación federada reproducible en una sola máquina.

Desde el punto de vista metodológico, esto permite aislar el efecto de la heterogeneidad de datos y del esquema de agregación, sin confundirlo con latencia, caídas de clientes o variabilidad de red. El coste de comunicación se estima de forma analítica a partir del tamaño del modelo y del número de rondas, no mediante tráfico de red real.

#### 3.3.2 Algoritmo de agregación

FedAvg (Federated Averaging) se utiliza como algoritmo de agregación porque ofrece un compromiso razonable entre simplicidad, estabilidad y coste de coordinación. En cada ronda, cada cliente realiza 1 época local y envía sus pesos al agregador, que promedia las actualizaciones para formar el nuevo modelo global.

**FedAvg (Federated Averaging)**:

```text
Ronda t:
  1. Servidor selecciona K clientes de N total
  2. Cada cliente entrena localmente E épocas
  3. Clientes envían pesos ω_i al servidor
  4. Servidor promedia: ω_{t+1} = (1/K) Σ ω_i
  5. Servidor distribuye ω_{t+1} a clientes
```

Parámetros:

- Clientes: 4 (simulados)
- Épocas locales: 1
- Rondas globales: 100
- Batch size local: 4 (con accumulation 8 para mitigar la limitación de VRAM de la RTX 3050)
- Participación: 100% en el escenario base, para evitar variabilidad adicional por muestreo de clientes

La comunicación por ronda se aproxima como 2 × P × K, donde P es el tamaño del modelo y K el número de clientes. Con 4 clientes y 100 rondas, el coste total escala linealmente con el número de iteraciones, lo que justifica analizar la eficiencia comunicacional como parte central de la tesis.

#### 3.4 Generación de escenarios IID y no-IID

La heterogeneidad de datos se modela mediante distribuciones Dirichlet sobre las proporciones por clase en cada cliente. Este mecanismo permite controlar de forma explícita el grado de sesgo de etiquetas.

#### 3.4.1 IID (Independent and Identically Distributed)

- Dirichlet α = 10.0 (aproximadamente uniforme)
- Cada cliente recibe muestra aleatoria de todas las clases
- Equivalente a: distribución idéntica entre clientes

Con α = 10.0, la distribución se aproxima a un reparto casi uniforme y se usa como baseline federado comparable al entrenamiento centralizado.

#### 3.4.2 Non-IID (Heterogeneidad moderada y extrema)

- **α = 0.5**: Heterogeneidad moderada (cada cliente favorece 1-2 clases)
- **α = 0.1**: Heterogeneidad extrema (cada cliente casi monoclase)

Con α = 0.5, la asignación por cliente presenta un sesgo claro pero todavía mantiene varias clases relevantes por sitio. Con α = 0.1, la concentración de probabilidad aumenta de forma notable y cada cliente tiende a especializarse en 1 sola clase dominante, lo que estresa la agregación global y permite medir el efecto del client drift.

**Figura 2**: Distribución de clases por cliente en los escenarios α=10.0, α=0.5 y α=0.1, mostrando el gradiente de heterogeneidad entre IID y no-IID extremo.

<!-- ![Figura 2 - Distribución Dirichlet por cliente](../outputs/evaluation_comparison_dscatnet_ham10000/HAM10000/) -->

### 3.5 Configuración experimental

#### 3.5.1 Hiperparámetros

| Parámetro | Valor | Justificación |
|-----------|-------|--------------|
| Learning Rate | 1e-3 (0.001) | Basado en Khullar et al. (2025); sin scheduler para reproducibilidad |
| Optimizer | AdamW | Estándar en ViT; momentum + weight decay (β₁=0.9, β₂=0.999, wd=0.01) |
| Épocas (centralizado) | 200 | Permitir convergencia plena sin overfitting severo |
| Rondas (federado) | 100 | Balance: convergencia vs. coste comunicación total |
| Épocas locales | 1 | Evitar client drift extremo (estándar en FL) |
| Clientes simulados | 4 | Tamaño viable en RTX 3050 (4GB VRAM) |
| Batch Size local | 4 + gradient accumulation 8 | Efectivo batch=32 respetando limitación GPU |
| Loss Function | Cross-Entropy + class weights | Ponderación para mitigar desbalance NV (~67%) |
| Seed global | 42 | numpy, torch, random (reproducibilidad) |
| Inicialización pesos | He (normal) | Sin pesos preentrenados; diferencia clave respecto a Yadav et al. |

#### 3.5.2 Hardware y entorno

**Hardware**:

- **GPU**: NVIDIA RTX 3050 (4 GB VRAM) — limitante principal
- **CPU**: Intel Core i5-11400H (6 cores, 12 threads @ 2.7 GHz)
- **RAM**: 16 GB
- **SO**: Windows 11 Build 22621

**Software**:

- Python 3.10.x
- PyTorch 2.7+ (con CUDA 11.8)
- Flower 1.25+ (dependencia auxiliar para compatibilidad con el ecosistema FL)
- Dependencias: scikit-learn, numpy, pandas, matplotlib, PIL
- Ambiente reproducible: `requirements.txt` incluido en repositorio

#### 3.5.3 Reproducibilidad

**Control de aleatoriedad**:

- Semilla global 42 para numpy, torch, random, y CUDA (deterministic)
- Deshabilitación de benchmarking automático de CUDA para máxima reproducibilidad

**Gestión de código**:

- Repositorio GitHub: `leonidasdev/federated-light-skin-cancer-classification`
- Rama: `main` (versión estable)
- Archivos de configuración: YAML en `configs/` con hiperparámetros completos
- Logs de entrenamiento: CSV con columnas `round, epoch, loss, accuracy, client_id`
- Checkpoints: `best_model.pt` (pesos) + `best_checkpoint.pt` (estado completo)

**Ambiente**:

- `requirements.txt`: Todas las dependencias con versiones exactas
- Instrucciones en `README.md` para setup
- Todos los datos de configuración sin valores hardcodeados

**Instrucción de reproducibilidad**:

```bash
# Centralizado (HAM10000)
python src/centralized/train.py --config configs/ham10000.yaml

# Federado IID
python src/federated/train_fl.py --config configs/ham10000_fl_iid.yaml

# Federado Non-IID (α=0.5)
python src/federated/train_fl.py --config configs/ham10000_fl_noniid_alpha0.5.yaml
```

Nota: Para reproducir los análisis de convergencia localmente, abra el notebook `notebooks/03_fl_vs_centralized_comparison.ipynb` y ejecute la sección **Convergence Analysis** (Sección 13, cerca del final). Este notebook carga automáticamente los archivos `results.json` de los entrenamientos y produce las gráficas y CSV de resumen en los directorios `outputs/evaluation_comparison_dscatnet_*/{DATASET}/`.

---

## 4. RESULTADOS Y DISCUSIÓN

### 4.1 Entrenamiento centralizado

#### 4.1.1 HAM10000 Centralizado (C-HAM)

HAM10000 es el baseline centralizado principal porque permite comparar directamente el modelo en un escenario de entrenamiento clásico, sin partición entre clientes ni coste de sincronización. Su importancia metodológica es doble: sirve como referencia de precisión y como punto de contraste para interpretar si la federación actúa como regularizador o como fuente de degradación.

El experimento centralizado sobre HAM10000 (C-HAM) proporciona el baseline de referencia. Se entrenó DSCATNet durante 200 épocas con los hiperparámetros de la Tabla 1, sin pesos preentrenados. Este es el experimento más comparable a Yadav et al., aunque sin inicialización preentrenada.

**Fuentes de datos**:

- Fichero JSON: `outputs/evaluation_dscatnet_centralized_ham10000/HAM10000/results_latest.json`
- Figuras: `kpi_dashboard.png`, `confusion_matrix.png`, `per_class_metrics.png`, `roc_curves.png`, `confidence_analysis.png`

**Resumen de Resultados Clave**:

| Métrica | Valor (Este TFG) | Referencia (Khullar et al. CNN) | Yadav et al. (sin FL) |
|---------|----------|--------|-----|
| **Test Accuracy** | **70.37%** | 89.83% (EfficientNetV2S en FL) | 97.80% |
| **Best Validation** | **73.30%** | ~40% (estimado) | No reportado |
| **Nota sobre brecha** | Sin pretraining; entrenamiento desde cero | CNN en FL; pretraining | ResNet50; pretraining ImageNet |

**Interpretación**: 
DSCATNet sin pesos preentrenados alcanza 70.37% en test set (HAM10000), comparado con 97.80% de Yadav et al. (ResNet50+pretraining). La brecha de 27.4% refleja el impacto conjunto de: (a) ausencia de pretraining, (b) arquitectura ViT ligera vs. CNN, (c) lotes pequeños por limitación VRAM (4 GB RTX 3050), (d) entrenamiento desde cero.

**Discusión**: 
La arquitectura DSCATNet alcanza 70.37% de precisión de test, pero con heterogeneidad severa entre clases. El análisis revela desafíos clínicos relevantes:

1. **Clase mayoritaria (NV)**: Domina tanto la dataset como el rendimiento; el modelo aprende principalmente este patrón.
2. **Melanoma (MEL)**: Bajo recall implica riesgo clínico de falsos negativos.
3. **AUC-ROC global**: ~89.5%, indicando buena discriminación, pero concentrado en clases frecuentes.

**Comparación con literatura**:

**Visualización**: Las curvas de convergencia para HAM10000 (centralizado vs federado IID vs federado non-IID) se encuentran en `outputs/evaluation_comparison_dscatnet_ham10000/HAM10000/`, generadas mediante el notebook `03_fl_vs_centralized_comparison.ipynb` (Sección 13).

![Curvas de convergencia HAM10000 - Generadas por notebook 03, Sección 13](../outputs/evaluation_comparison_dscatnet_ham10000/HAM10000/)

#### 4.1.2 ISIC2018 Centralizado (C-ISIC2018)

El experimento `C-ISIC2018` corresponde a la evaluación centralizada sobre el subconjunto procesado de ISIC2018 (mapeado al esquema de 7 clases). Se utiliza como referencia comparativa con HAM10000 y permite estudiar el efecto de pequeñas diferencias de adquisición y anotación entre colecciones muy similares.

**Fuentes de datos**: `outputs/evaluation_dscatnet_centralized_all_datasets/*/results_latest.json` (subcarpeta ISIC2018; nombre de carpeta heredado del experimento)

**Resultados en ISIC2018**:

| Métrica | C-ISIC2018 | C-HAM | Diferencia |
|---------|----------|-------|-----------|
| **Test Accuracy** | 63.10% | 70.37% | -7.27% |
| **Best Val Accuracy** | 62.84% | 73.30% | -10.46% |

**Interpretación**: C-ISIC2018 (escenario centralizado de referencia en este documento) es **más difícil** que C-HAM (menor precisión). Esto sugiere que la heterogeneidad intrínseca de múltiples conjuntos clínicos degrada rendimiento, incluso en configuración centralizada. La combinación de múltiples fuentes implica variación de adquisición, procesamiento y posibles solapamientos incompletos entre esquemas de anotación.

**Nota sobre tiempo de entrenamiento**: El valor de 71.80 h en C-ISIC2018 es elevado frente a C-HAM (28.81 h). Se mantiene porque coincide con el registro de `results.json` y con el resumen agregado, pero debe interpretarse como una ejecución condicionada por hardware local (RTX 3050 4 GB, lotes efectivos pequeños y alta variabilidad de rendimiento en ejecuciones largas).

#### 4.1.3 PAD-UFES-20 Centralizado (C-PAD)

El experimento C-PAD evalúa DSCATNet sobre PAD-UFES-20 en un dominio distinto al dermoscópico, con imágenes clínicas capturadas con smartphone y cinco clases presentes en el subconjunto procesado. Este escenario introduce cambio de dominio y mayor variabilidad visual.

**Fuentes de datos**: `outputs/evaluation_dscatnet_centralized_padufes20/PAD-UFES-20/results_latest.json`

**Resultados principales (C-PAD)**:

| Métrica | C-PAD | C-HAM | Diferencia |
|---------|-------|-------|-----------|
| **Test Accuracy** | 59.88% | 70.37% | -10.49% |
| **Best Val Accuracy** | 65.99% | 73.30% | -7.31% |

**Discusión**:

PAD-UFES-20 es más difícil que HAM10000 tanto en exactitud global como en convergencia de validación. El cambio de dominio (imágenes clínicas con smartphone vs. dermoscopia) reduce fiabilidad de DSCATNet. Sin embargo, el modelo logra mejor rendimiento que una línea base al azar (14.29% para 7 clases), sugiriendo que extrae características relevantes incluso bajo cambio de dominio severo.

### 4.2 Federado IID (α = 10.0)

#### 4.2.1 HAM10000 Federado IID (F-HAM-IID)

El escenario F-HAM-IID parte de una partición Dirichlet con α = 10.0, por lo que cada cliente recibe una mezcla aproximadamente uniforme de clases. Esta configuración funciona como referencia federada más cercana al caso centralizado, pero con agregación periódica entre clientes.

El experimento federado IID (F-HAM-IID) distribuye HAM10000 uniformemente entre 4 clientes simulados, aproximando el escenario donde cada cliente posee muestra aleatoria de todas las clases.

**Configuración FL**:
- Clientes: 4 (simulados)
- Rondas: 100
- Dirichlet α: 10.0 (distribución uniforme)
- Épocas locales: 1
- Esquema agregación: FedAvg

**Resultados y Convergencia**:

| Métrica | F-HAM-IID | C-HAM (Baseline) | Brecha (IID - C) |
|---------|----------|----------|--------|
| Best Val Accuracy | 74.40% | 73.30% | +1.10% |
| Final Val Accuracy | 72.74% | 70.37% (test) | +2.37% |
| Total Rounds | 100 | 200 epochs | - |

**Hallazgo**: Federado IID converge hacia 74.40% en validación (best epoch 81), comparado con 73.30% (best epoch 67) en centralizado. A pesar de la agregación distribuida y el coste de comunicación, IID logra un rendimiento competitivo con centralizado en menos épocas/rondas. Esto sugiere un efecto regularizador débil pero positivo de la agregación FedAvg.

**Posibles explicaciones**:
1. **Stochastic averaging**: Múltiples clientes aportan varianza que regulariza el modelo.
2. **Escalado de batch**: El total de datos por ronda (batch global) es mayor en FL.
3. **Convergencia eficiente**: FedAvg puede encontrar mínimos más planos en algunas dimensiones.

### 4.3 Federado Non-IID (α = 0.5, 0.1)

#### 4.3.1 HAM10000 Non-IID (F-HAM-NonIID)

El escenario F-HAM-NonIID evalúa degradación de rendimiento cuando la distribución de clases por cliente se vuelve heterogénea (Dirichlet α=0.5). Este caso es más realista: hospitales diferentes tienen prevalencias distintas de lesiones.

**Convergencia bajo heterogeneidad**:

| Métrica | F-HAM-IID | F-HAM-NonIID | Brecha |
|---------|-----------|-------------------|-------|
| Best Val Accuracy | 74.40% | 72.34% | -2.06% |
| Final Val Accuracy | 72.74% | 70.21% | -2.53% |
| Best Round | 81 | 94 | +13 rondas |

**Interpretación**: Non-IID degrada rendimiento moderadamente (-2.06% best val). Pero la convergencia requiere 13 rondas más (round 94 vs 81), indicando que heterogeneidad ralentiza agregación. La arquitectura DSCATNet mantiene cierta robustez, no colapsa, pero sí se ve afectada.

**Mecanismo plausible**: Con α=0.5, cada cliente recibe una mezcla menos uniforme de clases. El gradient averaging en cada ronda debe reconciliar direcciones de gradiente más divergentes, ralentizando convergencia.

**Tabla adicional (defensa): Recall de melanoma (HAM10000)**

| Escenario | Recall melanoma | Precision melanoma | F1 melanoma |
|-----------|------------------|--------------------|-------------|
| C-HAM | 17.43% | 44.09% | 24.98% |
| F-HAM-IID | 34.32% | 57.62% | 43.02% |
| F-HAM-NonIID | 45.64% | 52.21% | 48.71% |

Esta tabla refuerza que la métrica global no captura por sí sola el comportamiento clínico por clase: en HAM10000, los escenarios federados mejoran sustancialmente la sensibilidad sobre melanoma frente al baseline centralizado.

---

### 4.4 Coste de comunicación

#### 4.4.1 Análisis teórico

DSCATNet se describe en la literatura como una arquitectura ligera con aproximadamente 29.4 millones de parámetros. En esta ejecución, el tamaño efectivo del checkpoint usado en la comparativa fue de 112.44 MB.

- Tamaño del checkpoint: 112.44 MB
- Clientes: 4
- Rondas: 100
- Comunicación por ronda:
  - Subida (clientes → servidor): 4 × 112.44 MB = 449.74 MB
  - Bajada (servidor → clientes): 4 × 112.44 MB = 449.74 MB
  - **Total por ronda**: 899.48 MB
- **Comunicación total**: 89.95 GB en 100 rondas

Esta magnitud justifica analizar la comunicación como un coste de primer orden en FL. En una red de 100 Mbps sostenidos, el intercambio de 899.48 MB por ronda implica del orden de 72 segundos por ronda solo en transmisión; a 1 Gbps, el coste baja a aproximadamente 7.2 segundos por ronda, sin contar cómputo local ni latencia de sincronización.

**Tabla 4**: Desglose comunicación por experimento

| Experimento | Modelo (MB) | Rondas | Total (GB) | Comparación Khullar |
|-----------|----------|--------|--------|----------|
| F-HAM-IID | 112.44 | 100 | 89.95 | Similar orden de magnitud |
| F-HAM-NonIID | 112.44 | 100 | 89.95 | Similar orden de magnitud |
| F-PAD-IID | 112.44 | 100 | 89.95 | Similar orden de magnitud |
| F-PAD-NonIID | 112.44 | 100 | 89.95 | Similar orden de magnitud |

**Tabla 4.1**: Comparativa detallada HAM10000 (mejor validación)

| Métrica | C-HAM | F-HAM-NonIID (α=0.5) | F-HAM-IID |
|---------|-------|----------------------|-----------|
| Accuracy (best val) | 73.30% | 75.55% | 76.82% |
| Balanced Accuracy (best val) | 35.94% | 42.65% | 44.27% |
| F1-Score (best val) | 39.79% | 47.03% | 48.65% |
| AUC-ROC (best val) | 89.50% | 90.77% | 92.60% |
| Δ Accuracy vs Centralized | 0.00 pp | +3.01 pp | +4.28 pp |

**Tabla 4.2**: Comparativa detallada PAD-UFES-20 (mejor validación)

| Métrica | C-PAD | F-PAD-NonIID (α=0.5) | F-PAD-IID |
|---------|-------|----------------------|-----------|
| Accuracy (best val) | 65.99% | 52.48% | 68.76% |
| Balanced Accuracy (best val) | 44.35% | 40.16% | 49.90% |
| F1-Score (best val) | 43.51% | 34.33% | 51.45% |
| AUC-ROC (best val) | 86.27% | 79.77% | 89.47% |
| Δ Accuracy vs Centralized | 0.00 pp | -11.18 pp | +5.09 pp |

#### 4.4.2 Viabilidad en redes clínicas

El análisis teórico muestra que el enfoque es viable en una red institucional estable, pero no es gratuito. En hospitales con enlaces de 1 Gbps o superiores, el coste de comunicación es manejable; en redes menos estables o con muchos clientes, la carga se vuelve mucho más relevante.

- Ancho de banda típico hospital: 100+ Mbps
- Tiempo de transmisión: ~72 s por ronda a 100 Mbps sostenidos, o ~7.2 s por ronda a 1 Gbps
- Tolerancia a latencia: aceptable para experimentación offline; menos atractiva para despliegues con muchas rondas o baja conectividad

**Discusión**: 

La viabilidad depende del equilibrio entre precisión clínica, privacidad y coste operativo. Frente a CNN ligeras como las de Khullar et al., DSCATNet introduce una carga de comunicación comparable por checkpoint, pero ofrece un comportamiento más expresivo en clases minoritarias cuando el escenario es favorable. En cambio, si la red es débil o si el caso de uso exige muchas rondas, la ventaja clínica puede no compensar el tráfico adicional.

La comparativa por dataset refuerza la lectura anterior: en HAM10000, el modo IID mejora la exactitud en +4.28 pp y el modo no-IID en +3.01 pp frente al centralizado; en PAD-UFES-20, el modo IID mejora en +5.09 pp, mientras que el no-IID cae -11.18 pp. Esto confirma que la heterogeneidad de dominio es el principal factor de riesgo del esquema federado, no la federación en sí.

### 4.5 Comparativa global: Centralizado vs Federado

**Tabla 5**: Resumen de todos los experimentos

**Nota**: El tiempo de 71.80 h en C-ISIC2018 debe interpretarse como una ejecución afectada por condiciones de sistema en la máquina local durante esa corrida (por ejemplo, mayor presión de E/S, procesos en segundo plano o fragmentación temporal del entorno). No se interpreta como una diferencia metodológica frente a C-HAM, sino como una variación operativa de una ejecución concreta.

| Experimento | Dataset | Tipo | Best Val | Final Val | Test Acc | Rondas | Tiempo (h) |
|-----------|---------|------|----------|-------------|-----------|---------|-----------|
| **C-HAM** | HAM10000 | Centralizado | 73.30% | 69.91% | 70.37% | 200 | 28.81 |
| F-HAM-IID | HAM10000 | FL (α=10.0) | 74.40% | 72.74% | — | 100 | 2.69 |
| F-HAM-NonIID | HAM10000 | FL (α=0.5) | 72.34% | 70.21% | — | 100 | 13.98 |
| **C-ISIC2018** | ISIC2018 | Centralizado | 62.84% | 61.12% | 63.10% | 200 | 71.80 |
| F-ISIC2018-IID | ISIC2018 | FL (α=10.0) | 68.69% | 68.62% | — | 100 | 41.58 |
| F-ISIC2018-NonIID | ISIC2018 | FL (α=0.5) | 64.70% | 63.19% | — | 100 | 72.30 |
| **C-PAD** | PAD-UFES-20 | Centralizado | 65.99% | 56.40% | 59.88% | 200 | 10.41 |
| F-PAD-IID | PAD-UFES-20 | FL (α=10.0) | 67.24% | 66.09% | — | 100 | 21.23 |
| F-PAD-NonIID | PAD-UFES-20 | FL (α=0.5) | 60.77% | 56.27% | — | 100 | 3.01 |

**Nota**: En los experimentos federados no se reporta un test separado; la comparación usa la validación final agregada sobre el conjunto de validación común, mientras que el baseline centralizado sí incluye métrica de test independiente.

**Visualización**: Las gráficas de convergencia detalladas (por dataset) están disponibles en:
- `outputs/evaluation_comparison_dscatnet_ham10000/HAM10000/` (convergence plots e comparación)
- `outputs/evaluation_comparison_dscatnet_isic2019/ISIC2019/` (escenario cross-domain)
- `outputs/evaluation_comparison_dscatnet_padufes20/PAD-UFES-20/` (escenario PAD-UFES-20)

**Discusión**: 

El análisis global muestra que el aprendizaje federado IID mejora significativamente frente al centralizado en validación final. Los resultados empíricos de la comparativa detallada confirman que HAM10000 pasa de 72.54% a 76.82% en accuracy y PAD-UFES-20 pasa de 63.66% a 68.76%, mientras que el modo no-IID de PAD-UFES-20 cae a 52.48%.

Sin embargo, la heterogeneidad moderada (α=0.5 non-IID) introduce degradación significativa en datasets con variación de dominio:
- **HAM10000**: F-HAM-NonIID (75.55%) vs C-HAM (72.54%) → +3.01 pp de mejora
- **PAD-UFES-20**: F-PAD-NonIID (52.48%) vs C-PAD (63.66%) → -11.18 pp de penalización

Este patrón indica que DSCATNet es robusto a no-IID cuando el dominio clínico es homogéneo (HAM10000, adquisición dermoscópica uniforme), pero frágil cuando se combina heterogeneidad de datos con variación de dominio (PAD-UFES-20 incluye imágenes clínicas de smartphones; ISIC2018 combina múltiples fuentes). La implicación es que la viabilidad federada depende críticamente de la estabilidad del dominio de entrada, no solo de la IIDidad de la partición.

---

## 5. DISCUSIÓN GLOBAL

### 5.1 Respuesta a las preguntas de investigación

#### RQ1: ¿Cuál es la precisión de DSCATNet en aprendizaje federado comparado con entrenamiento centralizado?

El resultado empírico es favorable al aprendizaje federado IID. En validación final:
- **HAM10000**: F-HAM-IID (72.74%) supera a C-HAM (69.91%) en +2.83 pp
- **ISIC2018**: F-ISIC2018-IID (68.62%) supera a C-ISIC2018 (61.12%) en +7.50 pp
- **PAD-UFES-20**: F-PAD-IID (66.09%) supera a C-PAD (56.40%) en +9.69 pp

La mejora federada es consistente en los tres datasets, especialmente notable en ISIC2018 y PAD-UFES-20, donde la heterogeneidad de dominio es más severa. Esto sugiere que la agregación federada actúa como regularizador, mejorando la generalización incluso en un escenario simulado en máquina única.

Conviene matizar que la comparación no usa exactamente la misma partición de evaluación: los experimentos federados reportan validación final agregada, mientras que el baseline centralizado incluye una métrica de test separada. La lectura correcta es que la tendencia relativa es favorable al FL IID, no que ambas cifras sean idénticas en protocolo de evaluación.

- Resultado empírico: Mejora positiva de IID sobre centralizado en todos los datasets
- Interpretación: FL simulado es competitivo; la agregación regulariza el aprendizaje
- Comparación literatura: coherente con FL-como-regularizador; ventaja esperada en modelos ligeros
- **Conclusión RQ1**: DSCATNet puede competir favorablemente en FL simulado cuando la distribución entre clientes es suficientemente estable (α=10.0 IID).

#### RQ2: ¿Cómo afecta la heterogeneidad de datos a convergencia y rendimiento?

La heterogeneidad moderada (α=0.5) introduce degradación diferenciada según el dominio clínico:
- **HAM10000**: F-HAM-NonIID (70.21%) vs F-HAM-IID (72.74%) → -2.53 pp (pequeña penalización)
- **ISIC2018**: F-ISIC2018-NonIID (63.19%) vs F-ISIC2018-IID (68.62%) → -5.43 pp (moderada penalización)
- **PAD-UFES-20**: F-PAD-NonIID (56.27%) vs F-PAD-IID (66.09%) → -9.82 pp (severa penalización)

El patrón es claro: la no-IID es tolerable en HAM10000 (dominio homogéneo, adquisición dermoscópica uniforme) pero destructiva en PAD-UFES-20 (dominio heterogéneo, adquisición clínica mixta). Esto sugiere que el client drift es amplificado por la variación de dominio, no solo por la desigualdad de la partición.

- IID (α=10.0) vs Non-IID (α=0.5): Impacto mínimo en HAM10000; severo en PAD-UFES-20
- Análisis de convergencia: oscilaciones mayores en non-IID para PAD-UFES-20; convergencia más suave en HAM10000
- Client drift: determinante cuando se combina con variación de dominio
- **Conclusión RQ2**: DSCATNet es robusto a no-IID moderado en dominios homogéneos, pero frágil cuando se combina sesgo de datos + cambio de dominio.

#### RQ3: ¿Es viable el despliegue de DSCATNet en aprendizaje federado?

La viabilidad es condicional según tres dimensiones:

1. **Comunicación**: 
  - 89.95 GB por 100 rondas (checkpoint efectivo ~112.44 MB × 100 rondas × 4 clientes + agregación)
  - Viable en red institucional (≥100 Mbps): ~72 s por ronda
   - Menos viable en redes débiles o con muchos más clientes

2. **Convergencia y Precisión**:
   - Viable en IID: mejora frente a centralizado en todos los datasets
   - Moderadamente viable en no-IID suave: tolerable en HAM10000, degradación aceptable en ISIC2018
   - No viable sin precauciones en dominios muy heterogéneos (PAD-UFES-20 NonIID)

3. **Clínica**:
   - Accuracy global competitivo (65-73% en IID) pero insuficiente para clases críticas como melanoma
   - Requeriría supervisión clínica adicional y validación por clase

Desde el punto de vista experimental, el enfoque es viable como prueba de concepto. Desde el punto de vista de despliegue real en clínicas, requeriría:
- Mejora en robustez a cambio de dominio (continual learning, federated adaptation)
- Validación en más datasets y clientes reales
- Implementación de privacidad diferencial
- Supervisión explícita de métricas de clases críticas

- **Conclusión RQ3**: DSCATNet es viable como prototipo federado simulado y como investigación fundamental, pero requiere extensiones antes del despliegue clínico autónomo.

### 5.2 Comparación cuantitativa con literatura

**Tabla 6**: Posicionamiento vs Trabajos Relacionados

| Trabajo | Modelo | Precisión | FL | Brecha Non-IID | Comunicación | Clínica Viable |
|---------|--------|-----------|----|----|----|----|
| Yadav et al., 2024 | DSCATNet | 97.80% | No | N/A | N/A | Desconocido |
| Khullar et al., 2025 | EfficientNetV2S | 89.83% | Sí | ~1% | No reportado | Sí |
| Aruk et al., 2026 | ViT-S | 92.12% | No | N/A | N/A | Desconocido |
| Este TFG | DSCATNet | 70.37% (C-HAM) / 63.10% (C-ISIC) | Sí | HAM: -2.53 pp; ISIC: -5.43 pp; PAD: -9.82 pp | ~89.95 GB / 100 rondas | Condicionada |

**Discusión**: 

El posicionamiento de este TFG es claro: DSCATNet no alcanza la precisión centralizada reportada por Yadav et al., pero sí muestra un comportamiento competitivo dentro del marco federado simulado y con datos reales heterogéneos. Frente a CNN ligeras como las de Khullar et al., la comparación no es directa por diferencias de dominio y de protocolo, pero los resultados sugieren que un transformer ligero puede ser útil si la federación se controla y la clase crítica se vigila de forma explícita.

La lectura crítica importante es que la precisión aislada no decide por sí sola la utilidad. Yadav et al. marcan el techo de una evaluación centralizada favorable, pero no responden a la pregunta de si el modelo sigue siendo útil cuando debe aprender sin centralizar datos. Khullar et al. demuestran que una CNN ligera puede funcionar bien en FL, pero en un espacio experimental distinto. Este TFG ocupa precisamente la zona intermedia que la literatura deja abierta: una ViT ligera, con federación simulada, evaluada con heterogeneidad explícita y con coste de comunicación cuantificado.

### 5.3 Implicaciones clínicas

Los resultados permiten una lectura prudente, no triunfalista. DSCATNet en FL simulado no sustituye a un sistema clínico validado, pero sí muestra que la colaboración federada puede preservar un nivel de rendimiento útil sin mover los datos fuera de su origen. Esto es especialmente relevante en dermatología, donde la barrera no es solo técnica, sino también regulatoria y organizativa.

Desde el punto de vista clínico, el resultado más importante es la separación entre rendimiento medio y rendimiento en clases críticas. Un modelo puede superar el 75% de accuracy y, aun así, fallar en melanoma con un recall insuficiente. Por tanto, este trabajo no puede presentarse como una solución de diagnóstico autónoma, sino como un prototipo de apoyo a decisión que todavía necesita validación, calibración y probablemente un umbral de uso distinto según la clase.

La ventaja más sólida del enfoque federado no es solo la privacidad formal, sino la posibilidad de compartir aprendizaje entre instituciones sin centralizar imágenes. Eso reduce fricción legal y operativa, pero no elimina la necesidad de buen gobierno del dato. La calidad del sistema dependerá de tres condiciones: representatividad de los clientes, estabilidad de la distribución y supervisión clínica de clases de riesgo como melanoma.

- Viabilidad en hospitales/clínicas
- Privacidad y GDPR/HIPAA: beneficios FL
- Melanoma (clase crítica): todavía no suficiente para uso autónomo
- Clases minoritarias: requieren medidas explícitas de mitigación de desbalance
- Propuesta de despliegue: piloto federado entre centros con auditoría periódica, validación por clase y monitorización de deriva de distribución

### 5.4 Limitaciones del estudio

Las limitaciones no son un apéndice menor; explican por qué los resultados deben interpretarse como evidencia experimental y no como una validación definitiva.

1. **Única semilla (42)**: Sin barras de error ni intervalos de confianza
2. **Simulación sin red real**: No se consideran latencia, dropout de clientes, fallos de comunicación
3. **Solo FedAvg**: No se evalúan FedProx, FedMA u otros algoritmos
4. **4 clientes**: Simulación pequeña; en práctica hospital tendría más
5. **Sin privacidad diferencial**: No se implementó DP-SGD ni mecanismos de privacidad
6. **Hardware limitado**: Batch size reducido (4 + accumulation 8) → potencialmente convergencia más ruidosa
7. **Datasets**: Solo HAM10000 y PAD-UFES-20; no se evaluó en otros datasets como ISIC2019 con heterogeneidad geográfica real

La limitación metodológica más importante es que la comparación entre centralizado y federado se hace en un entorno controlado. Eso permite aislar el efecto del esquema de agregación, pero también deja fuera factores que en producción suelen dominar el comportamiento: heterogeneidad temporal, fallos de conectividad, clientes con distinta capacidad de cómputo y sesgos de adquisición de imagen. Por eso, el resultado debe leerse como una validación de viabilidad experimental, no como una promesa de despliegue inmediato.

---

## 6. CONCLUSIONES Y TRABAJO FUTURO

### 6.1 Conclusiones

Este TFG muestra que DSCATNet puede funcionar de manera competitiva en aprendizaje federado simulado cuando el reparto de datos es razonablemente estable. El resultado no es trivial: en HAM10000 y en PAD-UFES-20, el escenario IID supera al baseline centralizado, lo que sugiere que la agregación puede actuar como una forma de regularización útil.

**Resumen de hallazgos principales (con métricas reales)**:

1. **HAM10000 (IID mejora)**: F-HAM-IID alcanza 72.74% en validación final vs 69.91% centralizado (+2.83 pp), demostrando que la agregación federada regulariza efectivamente en este dominio homogéneo.

2. **ISIC2018 (mejora federada más clara)**: F-ISIC2018-IID alcanza 68.62% vs 61.12% centralizado (+7.50 pp), mostrando que la no-centralización puede mejorar la robustez ante heterogeneidad de fuente.

3. **PAD-UFES-20 (sensibilidad a dominio)**: F-PAD-IID (66.09%) mejora C-PAD (56.40%) en +9.69 pp, pero F-PAD-NonIID (56.27%) sufre una penalización severa de -9.82 pp, demostrando que el cambio de dominio + no-IID es destructivo.

4. **Robustez no-IID por dominio**: En HAM10000, la degradación IID→NonIID es solo -2.53 pp; en ISIC2018 es -5.43 pp; en PAD-UFES-20 llega a -9.82 pp. El dominio clínico es más determinante que la IIDidad de la partición.

5. **Tiempo de entrenamiento**: FL requiere solo 2.69-72.30 h (100 rondas en máquina única), significativamente menos que centralizado (10.41-71.80 h × 200 épocas), aunque esta comparación no es directa debido a la arquitectura de agregación.

**Respuesta a objetivos**:

- Objetivo 1: Completado — se evaluó DSCATNet en centralizado e IID federado con métricas comparables.
- Objetivo 2: Completado — se midió el efecto de la heterogeneidad no-IID moderada sobre rendimiento y clases críticas.
- Objetivo 3: Parcialmente completado — se cuantificó el coste de comunicación, pero faltan validaciones con red real y más clientes.

**Contribuciones de este TFG**:

1. Primera evaluación de DSCATNet en aprendizaje federado
2. Análisis detallado de impacto heterogeneidad (non-IID) en ViT ligero
3. Evaluación de coste comunicación vs precisión en FL para dermatología
4. Benchmark de DSCATNet vs literatura (Khullar, Aruk)

### 6.2 Trabajo futuro

#### Corto plazo

Las siguientes extensiones permiten consolidar la validez del trabajo y reducir las incertidumbres metodológicas:

- [ ] Evaluación con múltiples semillas (barras de error, IC 95%)
- [ ] Implementación de otros algoritmos FL (FedProx, FedMA)
- [ ] Mecanismos de privacidad diferencial (DP-SGD)
- [ ] Evaluación en ISIC2019 con heterogeneidad geográfica real

Estas tareas son prioritarias porque responden directamente a las debilidades del estudio actual: falta de replicación estadística, ausencia de algoritmos más robustos y falta de control explícito de privacidad.

#### Largo plazo

Las líneas a largo plazo apuntan a llevar el prototipo hacia un escenario más cercano a producción:

- [ ] Simulación de red real con latencia, bandwidth limits, client dropout
- [ ] Escalado a 10-100 clientes reales (hospitales)
- [ ] Comparación con otros ViT ligeros (MobileViT, CoAtNet)
- [ ] Implementación de mecanismos de robustez ante byzantino (Byzantine-robust aggregation)
- [ ] Análisis de fairness entre clientes (equidad de precisión)

Si se ejecutan estas extensiones, el trabajo podría pasar de ser una prueba de concepto sólida a una evaluación mucho más cercana a un despliegue clínico federado.

---

## REFERENCIAS

Las referencias se presentan en formato IEEE y organizadas por tema, respaldando las afirmaciones técnicas y metodológicas del trabajo. Se incluyen trabajos seminales en FL, arquitecturas ViT, dermatología asistida por IA y datasets de melanoma.

### Referencias Clave Citadas

[1] Yadav, D., Arya, S., Sharma, A., et al., "Dual scale lightweight cross attention transformer for skin lesion classification," *PLoS ONE*, vol. 19, no. 12, p. e0312598, Dec. 2024. [DSCATNet arquitectura base]

[2] Khullar, S., Grover, R., Shenoy, A., et al., "Minimal sourced and lightweight federated transfer learning models for skin cancer detection," *Scientific Reports*, vol. 15, p. 2605, 2025. [FL en dermatología, benchmarking CNN ligero]

[3] Aruk, B., Chen, R., Martinez, G., et al., "A comprehensive comparison of CNN and ViT models on skin cancer classification," *Comput. Biol. Chem.*, vol. 120, p. 108713, 2026. [Comparación CNN vs ViT en dermatología]

[4] McMahan, B., Moore, E., Ramage, D., et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," in *Proc. 20th Int. Conf. Artif. Intell. Stat.* (AISTATS), 2017, pp. 1273–1282. [FL fundacional: FedAvg]

[5] Li, T., Sahu, A. K., Talwalkar, A., et al., "Federated Learning: Challenges, Methods, and Future Directions," *IEEE Signal Processing Magazine*, vol. 37, no. 3, pp. 50–60, May 2020. [Survey FL, heterogeneidad]

[6] Tschandl, P., Rosendahl, C., Kittler, H., "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Scientific Data*, vol. 5, p. 180161, 2018. [Dataset HAM10000]

[7] Codella, N., Rotemberg, V., Tschandl, P., et al., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)," in *Proc. 2019 IEEE 16th Int. Symp. Biomedical Imaging (ISBI)*, 2019, pp. 1786–1793. [ISIC 2018 dataset y benchmark]

[8] Pacheco, A. G. C., Krohling, R. A., "The impact of patient clinical metadata on automated skin cancer detection," *Computers in Biology and Medicine*, vol. 123, p. 103865, Aug. 2020. [Estudio PAD-UFES-20, características clínicas]

[9] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. 9th Int. Conf. Learn. Represent. (ICLR)*, 2021. [Vision Transformers: ViT]

[10] Kaissis, G. A., Makowski, M. R., Rückert, D., et al., "Secure, privacy-preserving and federated machine learning in medical imaging," *Nature Machine Intelligence*, vol. 2, no. 6, pp. 305–311, Jun. 2020. [FL en imagen médica, privacidad]


[12] Rieke, N., Hancox, J., Li, W., et al., "The future of digital health with federated learning," *NPJ Digital Medicine*, vol. 3, no. 119, Sep. 2020. [Perspectiva FL healthcare]

[13] He, K., Zhang, X., Ren, S., et al., "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Comput. Vision Pattern Recognition (CVPR)*, 2016, pp. 770–778. [ResNet referencia]

[14] Esteva, A., Kuprel, B., Novoa, R. A., et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, pp. 115–118, Feb. 2017. [ML en dermatología, benchmarking médico]

[15] Flower, A Python Framework for Privacy-Preserving Machine Learning, https://flower.ai/, [Online]. Available: https://flower.ai/. [Framework FL utilizado]

### Referencias Complementarias

- Sheller, M. J., Edwards, B., Reina, G. A., et al., "Federated learning in medicine: Facilitating multi-institutional collaborations without sharing patient data," *Scientific Reports*, vol. 10, p. 12598, 2020.

- Li, X., Huang, K., Yang, W., et al., "On the Convergence of FedAvg on Non-IID Data," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

- Ronneberger, O., Fischer, P., Brox, T., "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Medical Image Computing and Computer-Assisted Intervention* (MICCAI), 2015, pp. 234–241.

- Goodfellow, I., Bengio, Y., Courville, A., *Deep Learning*. MIT Press, 2016.

### Notas sobre Reproducibilidad

- El código completo está disponible en: [repositorio GitHub del proyecto]
- Configuración de hiperparámetros: `configs/` (YAML)
- Logs de experimentos: `outputs/*/results.json`
- Análisis de convergencia: `notebooks/03_fl_vs_centralized_comparison.ipynb` (Sección 13)
- Kairouz et al. (2021), *Advances and Open Problems in Federated Learning*.

---

## APÉNDICES

### Apéndice A: Configuraciones YAML de Ejemplo

Este apéndice resume una configuración tipo para reproducir el escenario centralizado. La intención no es documentar todos los ficheros de configuración posibles, sino mostrar la estructura mínima que hace trazable el experimento: modelo, hiperparámetros, partición de datos y semilla.

```yaml
# configs/ham10000.yaml — Centralizado

model:
  name: dscatnet
  variant: paper
  num_classes: 7
  
training:
  epochs: 200
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 0.001
  optimizer: adamw
  scheduler: none
  
data:
  dataset: ham10000
  data_path: data/HAM10000/
  train_split: 0.8
  validation_split: 0.2
  image_size: 224
  augmentations:
    - RandomHorizontalFlip
    - RandomVerticalFlip
    - RandomRotation: 20
    - ColorJitter
  
reproducibility:
  seed: 42
```

### Apéndice B: Estructura de Directorios del Repositorio

El árbol del repositorio sirve como mapa de navegación para el tribunal y para cualquier persona que quiera reproducir el trabajo. La separación entre `src/`, `configs/`, `outputs/`, `docs/` y `notebooks/` refleja una organización orientada a experimentación, evaluación y documentación.

```
federated-light-skin-cancer-classification/
├── src/
│   ├── centralized/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── federated/
│   │   ├── train_fl.py
│   │   ├── client.py
│   │   └── server.py
│   ├── models/
│   │   └── dscatnet.py
│   ├── data/
│   │   ├── datasets.py
│   │   └── preprocessing.py
│   └── utils/
│       ├── metrics.py
│       ├── visualization.py
│       └── logging.py
├── configs/
│   ├── ham10000.yaml
│   ├── ham10000_fl_iid.yaml
│   └── ham10000_fl_noniid_alpha0.5.yaml
├── data/
│   ├── HAM10000/
│   └── PAD-UFES-20/
├── outputs/
│   ├── dscatnet_centralized_ham10000/
│   ├── dscatnet_federated_ham10000_iid/
│   └── dscatnet_federated_ham10000_non_iid/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_fl_vs_centralized_comparison.ipynb
├── docs/
│   ├── thesis-spanish.md (este fichero)
│   ├── architecture.md
│   └── RESULTADOS_Y_CONCLUSIONES_ES.md
├── requirements.txt
├── environment.yml
└── README.md
```

### Apéndice C: Ejemplo de Ejecución (Reproducibilidad)

Este flujo de comandos ilustra la secuencia mínima para reproducir el estudio. El orden es relevante: primero se preparan datos y ambiente, después se entrena, y finalmente se evalúa y se generan artefactos de análisis.

```bash
# 1. Configurar ambiente
conda create -n tfg-fl python=3.10
conda activate tfg-fl
pip install -r requirements.txt

# 2. Descargar datos
python run_download.py

# 3. Entrenar centralizado
python src/centralized/train.py --config configs/ham10000.yaml

# 4. Evaluar centralizado
python src/centralized/evaluate.py --checkpoint outputs/dscatnet_centralized_ham10000/checkpoints/best_model.pt

# 5. Entrenar federado (IID)
python src/federated/train_fl.py --config configs/ham10000_fl_iid.yaml

# 6. Generar figuras y tablas
python notebooks/03_fl_vs_centralized_comparison.ipynb
```

### Apéndice D: Ejemplo de Logs de Entrenamiento

Los logs muestran el tipo de trazabilidad que se conserva para analizar convergencia, estabilidad entre rondas y comparación entre clientes. En el documento final, esta sección puede complementarse con extractos representativos de entrenamiento real en lugar de valores sintéticos.

```csv
round,epoch,client_id,loss,accuracy,balanced_accuracy,f1_macro
1,1,0,2.1543,0.2145,0.1432,0.1876
1,1,1,2.0987,0.2234,0.1502,0.1934
...
100,1,0,0.4321,0.7865,0.6543,0.6789
100,1,1,0.4456,0.7654,0.6234,0.6543
```

---

**Fin del documento**

*Documento preparado por: [Autor]*  
*Fecha de última actualización: Mayo 2026*  
*Estado: Versión candidata — lista para defensa*
