## Challenge_Preauth

Flujo de ejecución que se tuvo en el proyecto:

1. crear el environment, instalar requirements.txt
2. Se realiza una exploración superficial a la data en data_exploring.ipynb, aquí se observan valores faltantes en Savings account y Checking account, ademáas posteriormente se observará el imbalanced de la data ( después de generar los targets con el modelo generativo llama 3)
3. Se ejecuta data_preparation.py que hace una pequeña transformación, los valores faltantes los coloca como 'unknown' esto posteriormente será codificado como otra categoría para las columnas Savings account y checking account.
4. Se ejecuta bedrock_description_generator.py que genera las decripciones basado en los datos, se usó el modelo meta.llama3-70b-instruct-v1:0, cabe mencionar que en el código se utiliza un 'profile' no un iam user.
5. Se ejecuta bedrock_classifier.py que clasifica en bad risk y good risk según la descripción creada en el paso anterior
6. se ejecuta sagemaker training, que utiliza XG BOOST para el entrenamiento de clasificación, se consideran las columnas iniciales y el target creado en el paso anterior, se utiliza el parámetro scale_pos_weight para manejar el imbalanced de la data, se utiliza como metrica objetivo auc, se realiza HYPERTUNING con estrategia bayesiana, para ejecutar este archivo con exito se creó previamente el bucket s3 challenge-preauth y el ROLE 'challenge-preauth-role' por simplicidad se le dio s3fullaccess y sagemakerfullaccess.
7. Se ejecuta sagemaker_deployment.py para desplegar el mejor modelo creado previamente.
8. Se ejecuta sagemaker_inference.py para realizar inferencias al modelo, este archivo acepta CSV con el mismo formato que la data original y da como resultado una carpeta 'results' con inference_resultsXXXXX.json, en los cuales muestra los resultados con el siguiente formato:
  {
    "input_data": {
      "Age": 35,
      "Sex": "male",
      "Job": 2,
      "Housing": "own",
      "Savings account": "little",
      "Checking account": "little",
      "Credit amount": 2500,
      "Duration": 12,
      "Purpose": "car"
    },
    "probability_fraud": 1.919269561767578e-05,
    "is_fraud": false,
    "target": "good risk",
    "confidence": 0.9999808073043823,
    "timestamp": "2025-07-24T22:28:47.927439"
  },

- probability_fraud es la probabilidad de que el préstamo sea de alto riesgo (bad risk)
- target nos indica la clasificación requerida

Para probar está inferencia creé un archivo create_sample.py que convierte data json en csv listo para ser consumido por sagemaker_inference.py
Tener en cuenta que en los scripts se debe reemplazar por el modelo creado de su cuenta
OBS: se observa auc en train de 1 y auc de validación 0.99147 por lo que podría indicar que no hay overfiting sin embargo, sería conveniente examinar más métricas como recall,f1, etc. por tiempo lo dejé solo en auc pero esto tiene que ser examinado más detalladamente.

Esto se puede mejorar de muchas formas, pro ejemplo realizando una arquitectura con lambda, dynamo e incluso una interfaz para consultas más interactivas.
fotos:
<img width="1601" height="259" alt="image" src="https://github.com/user-attachments/assets/8b3bc9ad-2d5e-4a52-a4a5-8964ccc4e984" />
<img width="1587" height="766" alt="image" src="https://github.com/user-attachments/assets/c2bb30e8-9e5c-46b6-b68b-4b3872b7a9ae" />
<img width="1539" height="557" alt="image" src="https://github.com/user-attachments/assets/f3deb54d-bd1c-4400-a734-b8157f9645e8" />
<img width="1322" height="427" alt="image" src="https://github.com/user-attachments/assets/5884b8da-79e2-441d-8615-e2f47ff51704" />
<img width="1230" height="644" alt="image" src="https://github.com/user-attachments/assets/7e4026a6-ccae-43d5-9df4-1a858c5e9fc0" />




