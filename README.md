# ProyectoMKD
## El problema: 

En el entorno actual de los servicios financieros, las empresas enfrentan el desafío de maximizar el valor del cliente y gestionar el riesgo de manera efectiva. Uno de los problemas más comunes es identificar patrones de comportamiento en sus clientes que permitan predecir el uso de productos financieros, el riesgo de crédito y la probabilidad de default. Esta necesidad se hace evidente en el contexto de las crecientes demandas regulatorias y la competencia del mercado, donde la precisión en la segmentación y evaluación de clientes puede marcar una gran diferencia en la rentabilidad y sostenibilidad de las instituciones financieras. 

A menudo, los datos sobre el comportamiento de los clientes, como sus patrones de compra, el uso de adelantos en efectivo, los límites de crédito y su historial de pagos, se encuentran dispersos y no se analizan adecuadamente para tomar decisiones informadas. Esto crea una oportunidad para que las instituciones financieras exploten el valor de estos datos y mejoren sus procesos de gestión de riesgo, al tiempo que desarrollan estrategias personalizadas de fidelización y retención de clientes. 

## La solución: 

Una solución técnica para abordar este problema es la segmentación de clientes utilizando técnicas de machine learning, específicamente mediante algoritmos de clustering. Al analizar variables clave de comportamiento financiero, como las transacciones de compra (PURCHASES), el uso de adelantos en efectivo (CASH_ADVANCE), el límite de crédito (CREDIT_LIMIT), el saldo promedio (BALANCE), el porcentaje de pagos totales (PRC_FULL_PAYMENT), la combinación de compras y adelantos (PURCHASE_ADVANCE_COMBINED), y la relación entre el tiempo de permanencia y el saldo (TENURE_BALANCE_RATIO), podemos identificar distintos grupos de clientes con patrones de comportamiento similares. 

El objetivo de esta segmentación es permitir a las instituciones financieras desarrollar estrategias personalizadas para cada segmento, como establecer políticas de crédito adaptadas, diseñar productos financieros específicos y mejorar la retención mediante programas de fidelización más efectivos. Esta solución también puede ayudar en la gestión de riesgos, permitiendo predecir qué segmentos de clientes presentan mayor probabilidad de default o de abandonar los productos de la institución. 

Ejemplos de empresas que ya usan machine learning para esto: 

JP Morgan Chase: utiliza machine learning para clasificar a los clientes en función de su riesgo crediticio y diseñar productos financieros personalizados. 

BBVA: emplea algoritmos de segmentación para identificar segmentos de clientes que necesitan productos específicos, como préstamos o líneas de crédito adicionales. 

American Express: analiza patrones de comportamiento de gasto para identificar posibles fraudes y mejorar la experiencia del cliente mediante ofertas personalizadas. 

Planteamiento del problema: 

Con los datos disponibles, queremos responder preguntas clave como: 

¿Cuáles son los distintos patrones de comportamiento entre nuestros clientes, y cómo se relacionan con el uso de nuestros productos financieros? 

¿Qué segmentos de clientes presentan mayor riesgo de default? 

## Metodología 

### Análisis exploratorio. 

##### Histogramas
Los histogramas muestran una fuerte asimetría positiva en la mayoría de las variables, lo cual sugiere la presencia de outliers y valores concentrados en rangos bajos. 
![image](https://github.com/user-attachments/assets/1541c9cb-a540-402d-a264-22fd9628c43a)

 
BALANCE y CREDIT_LIMIT: La mayoría de los clientes tiene un saldo y límite de crédito en rangos bajos, con unos pocos casos extremos que alcanzan valores muy altos. 

PURCHASES y CASH_ADVANCE: Estas variables también están sesgadas hacia valores bajos, con clientes que realizan muy pocas compras o adelantos en efectivo. 

Frecuencias de Compra y Adelanto: Las variables como PURCHASES_FREQUENCY y CASH_ADVANCE_FREQUENCY muestran que la mayoría de los clientes tienen frecuencias altas (cerca de 1), lo cual indica un comportamiento recurrente. 

TENURE: La variable de tiempo de permanencia (TENURE) tiene valores discretos, con una mayor cantidad de clientes en el rango de 12 meses, sugiriendo una relación a largo plazo con el servicio. 

Pagos y Porcentaje de Pago Completo: PRC_FULL_PAYMENT y PAYMENTS presentan sesgos hacia valores bajos, indicando que muchos clientes no pagan el saldo completo. 

##### Diagrama de cajas.
![image](https://github.com/user-attachments/assets/60a440d5-2bd2-4c02-8619-4ababecfc6df)

observamos que varias variables presentan una gran cantidad de outliers, evidenciado por los puntos fuera de los límites de cada caja. Esto es común en datos financieros, donde ciertos clientes pueden tener comportamientos extremos, como saldos, límites de crédito y adelantos en efectivo significativamente más altos que el promedio. Las variables de frecuencia (BALANCE_FREQUENCY, CASH_ADVANCE_FREQUENCY) también muestran asimetría, con valores concentrados en un extremo. Estos outliers pueden ser indicativos de diferentes tipos de comportamiento de los clientes, lo cual será relevante en la segmentación y análisis de patrones de uso.

### Procesamiento de datos.  

##### Diagrama de correlación.
![image](https://github.com/user-attachments/assets/d4ce5f27-6e7d-46b8-a813-93b4a94b40e1)
Observamos relaciones fuertes entre algunas variables, como PURCHASES y ONEOFF_PURCHASES (0.92), o PURCHASES_FREQUENCY y PURCHASES_INSTALLMENTS_FREQUENCY (0.86), lo que indica que los clientes con compras puntuales suelen realizar compras en cuotas. La variable CREDIT_LIMIT también tiene una correlación moderada con BALANCE (0.53), sugiriendo que un límite de crédito más alto está asociado a saldos mayores. En general, estas correlaciones nos ayudan a identificar patrones de comportamiento en los clientes y a entender cómo se relacionan diferentes aspectos de su uso de crédito.

### New Features.
Frecuencia Combinada de Compras y Adelantos en Efectivo (PURCHASE_ADVANCE_COMBINED): Esta variable combina la frecuencia de compras y la frecuencia de adelantos en efectivo, proporcionando una medida general de la actividad financiera del cliente.

Ratio de Uso de Crédito en Compras (CREDIT_LIMIT_COMPRA_RATIO): Mide qué proporción del límite de crédito es utilizada en compras, lo cual ayuda a entender el grado de dependencia del cliente respecto a su crédito disponible para consumo.

Proporción de Pagos Realizados (PAYMENT_RATIO): Indica la proporción del saldo que ha sido cubierto por los pagos realizados, proporcionando una visión del comportamiento de pago del cliente.

Proporción de Adelantos de Efectivo sobre el Saldo (CASH_ADVANCE_RATIO): Calcula cuánto del saldo corresponde a adelantos en efectivo, ayudando a identificar clientes que dependen más de adelantos en efectivo que de otros tipos de transacciones.

#### Diagrama de correlacion con nuevos features: 
PURCHASE_ADVANCE_COMBINED tiene una alta correlación con PURCHASES_FREQUENCY y CASH_ADVANCE_FREQUENCY (0.87 y 0.75 respectivamente), lo que era esperado dado que esta variable fue creada a partir de ellas. Esto sugiere que resume adecuadamente la frecuencia combinada de ambos tipos de transacción.

CREDIT_LIMIT_COMPRA_RATIO muestra una correlación moderada con PURCHASES (0.55) y CREDIT_LIMIT (0.42). Esto indica que esta variable capta bien el uso del crédito en relación a los límites de crédito, proporcionando información sobre cómo los clientes usan su crédito en compras sin ser una copia directa de PURCHASES o CREDIT_LIMIT.

PAYMENT_RATIO tiene una alta correlación con BALANCE y PAYMENTS, pero relativamente baja con otras variables, lo cual es útil para entender el comportamiento de pago del cliente sin estar influenciado en gran medida por otras variables.

CASH_ADVANCE_RATIO muestra una alta correlación con CASH_ADVANCE y una correlación mínima con otras variables, lo que lo convierte en un buen indicador aislado del uso de adelantos de efectivo en comparación con el saldo.
![image](https://github.com/user-attachments/assets/9ad6eca1-2e6b-4c28-85df-0c4607a148a9)

#### Preparacion de data.
No se usó un metodo de codificacion debido a que los datos son numericos, en su lugar, se realizó una tecnica de escalado.
Como metodo de scalado se usó el Robust Scaling, puesto que a pesar de que este  método es similar al Standard Scaling, en lugar de usar la media y la desviación estándar, utiliza la mediana y el rango intercuartílico, por lo que es menos sensible a los outliers.

#### Entrenamiento y tuneo de hiperparámetros. 
Para el entrenamiento de los modelos se usó como medida .
Interpretación de los clusters: Expliquen la relación de las variables del modelo con los clusters (al menos 4) en el sentido del negocio, utilicen visualizaciones para poder ver el comportamiento de las variables en cada uno de los clusters.
### Implementación en el negocio
Integración en el Sistema de Información del Negocio
Automatización de Ofertas y Monitoreo: Integrar los clusters en el sistema de CRM o de gestión de clientes del negocio para que cada cliente reciba ofertas personalizadas de manera automática, en función de su cluster.
Dashboard de Control y Seguimiento: Crear un dashboard que permita monitorear la distribución de clientes por cluster y su evolución en tiempo real. Esto puede incluir métricas de uso de productos, riesgo de crédito y satisfacción del cliente.
### Limitaciones
Los clusters creados se han basado en variables que tiene un comportamiento transaccional, lo que no nos permite detenerminar el perfil de riesgo completo del cliente. Esto significa que los segmentos pueden no ser completamente representativos de los riesgos reales o de las necesidades de cada grupo.
Se necesitan variables relevantes como el ingreso anual, variables demograficas e informacion de retrazos en pagos.

### Conclusiones y recomendaciones
Implementar los clusters identificados proporciona una forma de personalizar la oferta de productos y servicios financieros, mejorar la gestión de riesgos, y optimizar los recursos de marketing y ventas. Al entender las características únicas de cada grupo de clientes, el negocio puede tomar decisiones más informadas que maximicen la rentabilidad y minimicen el riesgo.
### Future Work
A medida que se recopilan nuevos datos transaccionales, los clusters pueden cambiar. Implementar un sistema de monitoreo continuo que actualice los clusters de clientes periódicamente permitiría adaptar las estrategias de negocio en tiempo real y responder mejor a cambios en el comportamiento de los clientes. Esto implicaría automatizar el proceso de clustering y la integración en los sistemas CRM del negocio.
