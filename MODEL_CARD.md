# Model card — satisfacción de pasajeros

Documento descriptivo del champion que responde la API. No interviene en los pedidos ni sustituye el registro de MLflow. Las cifras provienen de las corridas del experimento `tp_airline_satisfaction`.

## Objetivo

El modelo estima si un pasajero resulta satisfecho o neutral o insatisfecho, a partir de los datos que la operación dispone al cierre del vuelo. La encuesta no forma parte de la entrada.

La clase 1 corresponde a `satisfied`. La clase 0 corresponde a `neutral or dissatisfied`. La respuesta de la API incluye la clase, la etiqueta, la probabilidad de la clase 1 y la versión del modelo.

## Identificación

| | |
|---|---|
| Registro | `airline_satisfaction_without_scores` |
| Alias | `champion` |
| Versión | 7 |
| Algoritmo | XGBoost |
| Corrida | `xgboost_without_scores` |

La API carga el modelo asociado al alias `champion`. El número de versión indica el orden de registro. El alias se reasigna cuando un entrenamiento posterior obtiene un ROC AUC de test superior.

## Variables de entrada

El modelo utiliza ocho variables:

- `Gender`
- `Customer Type`
- `Type of Travel`
- `Class`
- `Age`
- `Flight Distance`
- `Departure Delay in Minutes`
- `Arrival Delay in Minutes`

## Variables excluidas

Los catorce puntajes de la encuesta, medidos en una escala de 0 a 5, permanecen fuera del modelo. Esas respuestas describen la experiencia del vuelo y se encuentran próximas a la satisfacción que se busca estimar. El servicio debe operar con la información disponible para la operación sin recurrir a dicha encuesta.

## Métricas de entrenamiento

En la corrida `xgboost_without_scores`, correspondiente a la versión 7:

- ROC AUC en test: 0,8681
- ROC AUC de la validación cruzada: 0,8703

Ambas cifras pertenecen a esa corrida del champion.

## Auditoría

La corrida `robustness_audit` pertenece al mismo experimento, finalizó correctamente y el tag `robustness_status` tiene el valor `PASSED`. Los umbrales están definidos en el código. Los valores de la tabla son los registrados en esa corrida.

| Control | Umbral de promoción | Valor medido |
|---|---|---|
| Estabilidad: la clase se mantiene al incorporar ruido | superior a 0,90 | 0,98 |
| Paridad de género: modificar únicamente `Gender` conserva la clase | superior a 0,95 | 1,00 |
| Consistencia: un aumento de la demora de llegada no incrementa la probabilidad de satisfacción | superior a 0,90 | 0,99 |

El gráfico SHAP de esa corrida se conserva como artefacto en MLflow. Este documento no reproduce esas importancias.

## Limitaciones

El contrato de la API rechaza una edad fuera del intervalo de 0 a 120, una demora negativa y cualquier campo adicional. En el archivo crudo, la demora de llegada puede figurar vacía. En el servicio, un pedido incompleto no recibe puntuación.

La predicción constituye una estimación interna de satisfacción y no otorga un beneficio al pasajero. El conjunto de datos corresponde a una sola aerolínea. El champion fue entrenado exclusivamente con esas filas.
