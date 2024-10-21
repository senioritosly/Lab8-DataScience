# Informe: Reflexión sobre el Proceso de Selección y Despliegue de Modelos de Machine Learning

1. Desafíos en la Selección del Modelo y su Puesta en Producción

Durante el proceso de selección del modelo, uno de los principales desafíos fue encontrar un balance adecuado entre la precisión del modelo y la complejidad del mismo. Al evaluar modelos como la regresión lineal, Random Forest y Gradient Boosting, quedó claro que, si bien algunos modelos más complejos, como Random Forest y Gradient Boosting, eran capaces de capturar relaciones no lineales, sus errores de predicción (RMSE y MAE) resultaban ser considerablemente más altos en comparación con el modelo de regresión lineal. Esto implicaba que las técnicas más avanzadas no siempre aportan una mejora en precisión, especialmente cuando los datos presentan relaciones mayormente lineales.

Otro reto importante fue el ajuste de hiperparámetros, ya que implicaba un extenso tiempo de prueba y error para cada uno de los modelos seleccionados. En especial, los modelos de Random Forest y Gradient Boosting requerían optimización intensiva, lo que incrementaba considerablemente el tiempo de entrenamiento. Este proceso se complicó aún más por la necesidad de realizar validaciones cruzadas para garantizar la estabilidad de los resultados.

Finalmente, la puesta en producción representó un desafío adicional. Traducir el modelo entrenado a una plataforma de predicción en tiempo real exigió un enfoque cuidadoso para evitar sobrecargar el sistema. Además, se debió asegurar que las características del modelo utilizadas para el entrenamiento fueran adecuadamente mapeadas en el entorno de producción, manteniendo la consistencia entre las variables y las estructuras de datos.

2. Aprendizajes Más Significativos

Uno de los aprendizajes más valiosos fue comprender que la simplicidad del modelo no debe ser subestimada. A menudo se asume que los modelos más complejos, como Random Forest o Gradient Boosting, ofrecerán mejores resultados; sin embargo, en este caso, la regresión lineal superó a ambos en términos de precisión, lo que refuerza la importancia de probar enfoques más simples antes de optar por técnicas avanzadas.

Otro aprendizaje clave fue la importancia del preprocesamiento de datos. El manejo de valores faltantes, la normalización de las características y la transformación de variables categóricas en variables dummy fue fundamental para el éxito del modelo de regresión lineal, que es particularmente sensible a la multicolinealidad y la escala de los datos. El buen manejo de estos aspectos contribuyó directamente a la alta precisión obtenida con el modelo lineal.

Por último, la experiencia subrayó la importancia de automatizar el proceso de ajuste de hiperparámetros y validación de modelos. Utilizar herramientas como GridSearchCV o RandomizedSearchCV aceleró notablemente este proceso, mejorando la eficiencia del desarrollo.

3. Evaluación Crítica de las Fortalezas y Limitaciones del Enfoque

Una fortaleza destacable del enfoque utilizado fue la diversidad de modelos probados. Esto permitió identificar que, en este conjunto de datos particular, la regresión lineal era el modelo más adecuado, lo cual no siempre es evidente de antemano. La evaluación exhaustiva de métricas como el RMSE, MAE y R² permitió tener una visión clara de cómo cada modelo se ajustaba a los datos.

No obstante, una limitación del enfoque fue la dependencia excesiva en el ajuste manual de hiperparámetros y la falta de automatización en las primeras etapas del proceso. Aunque herramientas como GridSearchCV fueron eventualmente implementadas, una mayor automatización desde el inicio habría permitido optimizar los modelos de manera más eficiente.

Otra limitación fue el enfoque en modelos estáticos. En producción, los datos tienden a cambiar con el tiempo, lo que hace que los modelos se degraden. No se implementaron estrategias de actualización automática del modelo para adaptarse a nuevos datos (por ejemplo, técnicas de aprendizaje continuo), lo que podría ser un problema en un sistema de producción a largo plazo.

4. Sugerencias para Mejorar el Proceso de Desarrollo y Despliegue

Para mejorar el proceso de desarrollo y despliegue de modelos de Machine Learning en el futuro, se sugieren las siguientes acciones:

Automatización del Proceso de Selección y Ajuste de Modelos: Implementar herramientas que permitan realizar pruebas automatizadas de modelos y ajustes de hiperparámetros desde el inicio, para reducir la carga manual y el tiempo dedicado a este proceso. Además, sistemas de AutoML podrían ser considerados para la selección inicial de modelos.

Incorporación de Técnicas de Actualización en Producción: Para evitar la degradación de los modelos en el tiempo, sería beneficioso implementar procesos que permitan la actualización periódica de los modelos con nuevos datos. Esto podría incluir técnicas de aprendizaje en línea o pipelines automáticos de reentrenamiento basados en ciertos umbrales de rendimiento.

Monitorización del Desempeño Post-Producción: Es esencial establecer métricas de monitoreo en tiempo real que permitan detectar cuando un modelo está empezando a fallar o perder precisión. Esto permitiría intervenir rápidamente y reentrenar o ajustar el modelo antes de que afecte el servicio.

Documentación Exhaustiva y Reproducibilidad: El proceso de desarrollo debe ser completamente documentado para asegurar que cualquier modificación o reentrenamiento en el futuro se pueda realizar de manera reproducible. Esto incluye tener versiones claras de los datos, código y modelos utilizados.