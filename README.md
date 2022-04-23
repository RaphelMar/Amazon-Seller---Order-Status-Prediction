# Amazon Seller - Order Status Prediction
 
# Sobre Dataset


## Contexto

A BL é uma pequena empresa de produtos de couro que recentemente começou a vender seus produtos na Amazon. Atualmente, possui cerca de 40 SKUs registrados no Mercado Indiano. Nos últimos meses, sofreu algumas perdas devido às ordens de devolução. Agora, BL procura ajuda para prever a probabilidade de uma nova ordem ser rejeitada. Isso os ajudaria a tomar as ações necessárias e, posteriormente, reduzir a perda.

## Objetivo

Para construir um modelo que predize o status da ordem ( Delivered to buyer ou Returned to seller)

## Dicionário de Dados

Os dados do Pedido são fornecidos em um arquivo excel. As colunas são:

    order_no - Número único da ordem da Amazon
    order_date - Data em que o pedido foi colocado
    buyer - Nome do comprador
    ship_city - Cidade do Endereço de Entrega
    ship_state - Estado do endereço de entrega
    sku - Unique identificador de um produto
    description - Descrição do produto
    quantity - Número de unidades encomendadas
    item_total - Valor total pago pelo comprador
    shipping_fee - Cargas suportadas pelo Boss Leathers para enviar o item
    cod - Modo de pagamento: Dinheiro na entrega ou não

### Label / Recurso de destino:

    order_status - Status da ordem
    
# Pipeline do Projeto

1. Importação das Bibliotecas;
0. Entendimento dos dados;
0. Tratamento dos dados (dados faltantes, coversão de valores, e criação das características de tempo)
0. Entendimento dos dados após tratamentos;
0. Visualizar insights do negócio;
0. Resumo dos insights;
0. Visualizando correlação dos dados;
0. Treinando modelos;
0. Avaliando modelos;
0. Resumo dos modelos

# 1. Importação das Bibliotecas

Manipulação de dados

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

Pré-Processamento

    from sklearn.preprocessing import LabelEncoder, StandardScaler

Modelos

    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

Seleção de modelos

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    
Avaliadores de Modelos

    from sklearn.metrics import (    
         f1_score,
         log_loss,
         recall_score,
         roc_auc_score,
         accuracy_score,
         precision_score,
         confusion_matrix,
         brier_score_loss,
         roc_curve,
         auc
    )
    
# 2. Entendimento dos dados

![image](https://user-images.githubusercontent.com/103796137/164934005-65366f82-94f0-40ac-8415-dec7d2937079.png)

![image](https://user-images.githubusercontent.com/103796137/164934024-d2ba526e-58a1-4b60-bf79-83d3d2917530.png)

![image](https://user-images.githubusercontent.com/103796137/164934031-d471f25e-b8ba-4c41-b7cb-27e548a2923b.png)

# 3. Tratamento dos dados

  Tratamento da coluna de data e criação de caractericas de tempo
  
         df['Year'] = pd.DatetimeIndex(df['order_date']).year
         df['Month'] = pd.DatetimeIndex(df['order_date']).month
         df['Date'] = pd.DatetimeIndex(df['order_date']).date
         df['Time'] = pd.DatetimeIndex(df['order_date']).time
         df['Month_Name'] = pd.DatetimeIndex(df['Date']).month_name()
         df['Day_Name'] = pd.DatetimeIndex(df['order_date']).day_name()

         df.drop(columns= 'order_date', inplace= True)

   Tratando as colunas "Ship_city" e "Ship_state"
   
         for label in ['ship_city', 'ship_state']: 
             df[label] = df[label].str.replace(',', '', regex= False)
             df[label] = df[label].str.upper()

   Tratando dados faltantes
   
         df['item_total'].fillna('0', inplace= True)
         df['shipping_fee'].fillna('0', inplace= True)
         df['cod'].fillna('Online', inplace= True)

   Removendo simbolos nas colunas "item_total", "shipping_fee" e convertendo valores
    
         for label in ['item_total', 'shipping_fee']:
             df[label] = df[label].str.replace('₹', '')
             df[label] = df[label].str.replace(',', '')
             df[label] = df[label].astype(np.float64)

   Removendo o "SKU:" da coluana "sku"
     
         df['sku'] = df['sku'].str.replace('SKU: ', '', regex= False)

# 4. Entendimento dos dados após tratamentos

![image](https://user-images.githubusercontent.com/103796137/164934196-2190b74b-3adf-4596-ac5b-74100a60e2c7.png)

![image](https://user-images.githubusercontent.com/103796137/164934215-f9be9019-db76-4d9a-9b1a-0f49d126b1d1.png)

![image](https://user-images.githubusercontent.com/103796137/164934229-36487378-c1ca-467b-82b9-3ee2cf7c12cd.png)

![image](https://user-images.githubusercontent.com/103796137/164934241-8b19bc72-c4f2-44db-a9b7-485f28e96292.png)

![image](https://user-images.githubusercontent.com/103796137/164934248-b4e138fb-a11d-4855-bf4e-ba6059bfc299.png)

Histograma da coluna "item_total"

![image](https://user-images.githubusercontent.com/103796137/164934309-ecfcebc0-4e7f-4ff6-a877-1ee22bec158f.png)

Histograma da coluna "quantity"

![image](https://user-images.githubusercontent.com/103796137/164934320-2348e1f5-74f7-4600-9a4c-b1743cf9577b.png)

Histograma da coluna "shipping_fee"

![image](https://user-images.githubusercontent.com/103796137/164934342-f178ec6a-8901-43c5-af13-eae63e3c75ea.png)

# 5. Visualizar insights do negócios
### Lucro por ano

![image](https://user-images.githubusercontent.com/103796137/164934369-0b7e35eb-5fd1-4205-87c9-883cac809000.png)

### Lucro por todo o periodo

![image](https://user-images.githubusercontent.com/103796137/164934380-d88d86ee-d814-4581-9302-fb183d8209fe.png)

### Lucro por mês

![image](https://user-images.githubusercontent.com/103796137/164934697-4d0c3261-3b98-45ff-9f03-5ad67bf4ac5c.png)

### Analise dos Meses Dezembro e Janeiro

![image](https://user-images.githubusercontent.com/103796137/164934737-02c4bc47-5039-4af7-8cca-18e86eb0e9a8.png)

### Analisando somente o mês de Dezembro

![image](https://user-images.githubusercontent.com/103796137/164934785-41701563-448e-4594-83c5-8c775f31eec6.png)

### Comparando custos de Devoluções nos meses de Dezembro e Janeiro

![image](https://user-images.githubusercontent.com/103796137/164934877-a8c6a0df-0dc1-416c-8873-4986038e6054.png)

### Analisando os lucros por dia da semana

![image](https://user-images.githubusercontent.com/103796137/164934933-b33c1da3-f080-4e36-b3b1-c34ffbc8ef2c.png)

### Lucro por Estado

![image](https://user-images.githubusercontent.com/103796137/164934996-97ac8aea-9a9a-47fd-a499-b0487a7a0ebf.png)

### Analise de cidades dos TOP 3 estados

![image](https://user-images.githubusercontent.com/103796137/164935044-2ac29674-b052-49f6-a673-afb924cf1b05.png)

### Analise das Cidades do estado "MAHARASHTRA"

![image](https://user-images.githubusercontent.com/103796137/164935101-bd60e5be-c3a3-4d1c-82ff-2b085f4fd718.png)

### Analise das Cidades do estado "WEST BENGAL"

![image](https://user-images.githubusercontent.com/103796137/164935159-cb787aae-9dd1-4f67-a1fb-cc7d35155ac3.png)

### Analise das Cidades do estado "TAMIL NADU"

![image](https://user-images.githubusercontent.com/103796137/164935227-98110a97-1165-468d-a08f-e52d68aae1b6.png)

### Top 5 Produtos

![image](https://user-images.githubusercontent.com/103796137/164935280-6fed00f2-c829-420c-8796-c34a11c23472.png)

# 6. Resumo dos insights

Limitaremos a 10 insights, mas podemos tirar muita mais que 10 insights.

1.	O lucro de 2021 foi maior que 2022, não porque 2021 foi mais performático, mas sim porque o range dos dados é de junho/2021 a fev/2022;
2.	Os compradores preferem pagar mais online do que na entrega;
3.	O mês de dezembro foi o que mais se destacou, devido ao Natal;
4.	Em dezembro, o dia da semana que teve mais venda foi a terça-feira e o dia 21/12/2021. Esse destaque ocorreu devido o natal;
5.	Observamos que em janeiro/2022 houve muita devoluções, o que demonstra o efeito do “arrependimento da compra” ;
6.	O estado que teve mais lucro foi o MAHARASHTRA seguido por WEST BENGAL e TAMIL NADU;
7.	A cidade que mais teve lucro em MAHARASHTRA foi Mumbai, com um lucro aproximado de 17,5k. Oque compõem 60% do lucro do estado;
8.	A cidade que mais teve lucro em WEST BENGAL foi Kolkata, com um lucro aproximado de 25k. Oque compõem 85% do lucro do estado;
9.	A cidade que mais teve lucro em TAMIL NADU foi Chennai, com um lucro aproximado de 20k. Oque compõem 80% do lucro do estado;
10.	O produto que teve mais performasse foi o SKU: SB-WDQN-SDN9, com um lucro aproximadamente 120k em pagamentos online e 2k em pagamentos na entrega, já o produto SKU: DN-0WDX-VYOT deve um lucro de 4k em pagamentos na entrega, comparado aos outros produtos.

# 7. Visualizando correlação dos dados

![image](https://user-images.githubusercontent.com/103796137/164936625-22505f3b-1dbb-437e-9514-25f6bf07f4b8.png)

# 8. Treinando modelos

Função para treinamento dos modelos

    def model(df, modelo, parametros):
        df= df.copy()

        # Removendo colunas desnecessarias
        df.drop(columns= ['order_no', 'Date', 'Time', 'Month_Name', 'description'], inplace= True)

        # Categorizando dados
        enconder = LabelEncoder()
        labels = ['buyer', 'ship_city', 'ship_state', 'sku', 'cod', 'order_status', 'Year', 'Day_Name']

        cat = df[labels].copy()

        for label in labels:
            cat[label] = enconder.fit_transform(cat[label])

        df.drop(columns= labels, inplace= True)
        df = pd.concat([df, cat], axis= 1)

        # Balanceando as classes
        seller = df.query('order_status == 1')
        buyer = df.query('order_status == 0').sample(n=len(seller)+10, random_state = 48)

        df_balanceado = pd.concat([seller, buyer], axis= 0, ignore_index= True)

        # Separando Dataset em X e y
        X_balanceado = df_balanceado.drop(columns= 'order_status').copy()
        y_balanceado = df_balanceado['order_status'].copy()

        X_train, X_test, y_train, y_test = train_test_split(X_balanceado, y_balanceado, test_size= .3, random_state=42)

        X_sub = df.drop(columns= 'order_status').copy()
        y_sub = df['order_status'].copy()

        # Normalizando Valores
        scaler = StandardScaler()

        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns= X_train.columns)
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns= X_test.columns)
        X_sub = pd.DataFrame(scaler.fit_transform(X_sub), columns= X_test.columns)

        # Otimizando hiperparametros
        cv = GridSearchCV(modelo, param_grid= parametros, cv=10)
        cv.fit(X_train, y_train)

        # Selecionando o melhor modelo
        modelo = cv.best_estimator_

        return modelo, X_train, y_train, X_test, y_test, X_sub, y_sub

### KNN - Mark 1

    params_knn = {
        'n_neighbors': [row for row in range(5, 50, 5)],
        'weights': ['uniform', 'distance'],
    }

    knn, X_train, y_train, X_test, y_test, X_sub, y_sub = model(df= df, modelo= KNeighborsClassifier(), parametros= params_knn)
    
### SVC - Mark 2

    params_svc = {
        'penalty': ['l1', 'l2'],
        'loss': ['hinge', 'squared_hinge'],
        'C': [N for N in range(0, 110, 5)],
    }

    svc, X_train, y_train, X_test, y_test, X_sub, y_sub = model(df= df, modelo= LinearSVC(), parametros= params_svc)

### Decision Tree Classifier - Mark 3

    params_decisiontree = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [N for N in range(0, 100, 5)],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    decisionTree, X_train, y_train, X_test, y_test, X_sub, y_sub = model(df= df, modelo= DecisionTreeClassifier(), parametros= params_decisiontree)

### Logistic Regression - Mark 4

    params_log = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [N for N in range(0, 110, 5)]
    }

    log, X_train, y_train, X_test, y_test, X_sub, y_sub = model(df= df, modelo= LogisticRegression(), parametros= params_log)
    
    
# 9. Avaliando modelos
### Funções criadas para avaliar os modelos

Resumo de Metricas

    def resume_metrics(modelo, X_train, y_train, X_test, y_test, X_sub, y_sub):

        # Prevendo dados de teste e dados de treino
        train_pred = modelo.predict(X_train)
        test_pred = modelo.predict(X_test)
        sub_pred = modelo.predict(X_sub)

        # Criando dataframe com o resumo das metricas
        metrics = {
            'Accuracy Score': [accuracy_score(y_test, test_pred), accuracy_score(y_train, train_pred), accuracy_score(y_sub, sub_pred)],
            'F1 Score': [f1_score(y_test, test_pred), f1_score(y_train, train_pred), f1_score(y_sub, sub_pred)],
            'Log Loss': [log_loss(y_test, test_pred), log_loss(y_train, train_pred), log_loss(y_sub, sub_pred)],
            'Precision Score': [precision_score(y_test, test_pred), precision_score(y_train, train_pred), precision_score(y_sub, sub_pred)],
            'Recall Score': [recall_score(y_test, test_pred), recall_score(y_train, train_pred), recall_score(y_sub, sub_pred)],
            'Brier Score Loss': [brier_score_loss(y_test, test_pred), brier_score_loss(y_train, train_pred), brier_score_loss(y_sub, sub_pred)],
            'ROC AUC Score': [roc_auc_score(y_test, test_pred), roc_auc_score(y_train, train_pred), roc_auc_score(y_sub, sub_pred)]
        }

        metrics = pd.DataFrame(metrics, index= ['Data test', 'Data train', 'Data sub'])

        return metrics


Grafico ROC

    def plot_roc_curve(modelo, X_test, y_test):

        prob = modelo.predict_proba(X_test)
        prob = prob[:, 1]

        fper, tper, thresholds = roc_curve(y_test, prob)

        plt.figure(figsize=(10, 5))
        plt.plot(fper, tper, color='#377BA6', label='ROC')
        plt.plot([0, 1], [0, 1], color='#D96B2B', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend()
        plt.show()

Comparando Modelos

    def model_metrics(dic_model, X_test, y_test):
        acc = []

        for index, modelo in dic_model.items():

            # Prevendo dados de teste e dados de treino
            test_pred = modelo.predict(X_test)

            # Criando dataframe com o resumo das metricas
            metrics = {
                'Name Model': index,
                'Accuracy Score': [accuracy_score(y_test, test_pred)],
                'F1 Score': [f1_score(y_test, test_pred)],
                'Log Loss': [log_loss(y_test, test_pred)],
                'Precision Score': [precision_score(y_test, test_pred)],
                'Recall Score': [recall_score(y_test, test_pred)],
                'Brier Score Loss': [brier_score_loss(y_test, test_pred)],
                'ROC AUC Score': [roc_auc_score(y_test, test_pred)]
            }

            metrics = pd.DataFrame(metrics)

            acc.append(metrics)

        acc = pd.concat(acc, axis= 0, ignore_index= True).sort_values(by='ROC AUC Score', ascending= False)

        return acc


Comparando ganho monetario

    def evaluating_cost(dic, X_test, y_test):
        resultado = []

        for index, modelo in dic.items():

            for i in range(10, 101, 10):

                i = i/100

                X = X_test.sample(frac= i, random_state= 2)
                y = y_test[X.index]

                cm = confusion_matrix(y, modelo.predict(X))

                FN = cm[0][1]
                FP = cm[1][0]
                VN = cm[1][1]

                proft = (-FN * 10) + (-FP * 71.86) + (VN * 61.86) 

                resultado.append({'modelo': index, 'frac': i, 'proft': proft})

        resultado = pd.DataFrame(resultado)

        plt.figure(figsize= (15, 5))
        sns.lineplot(x= resultado['frac'], y= resultado['proft'], hue= resultado['modelo'])
        plt.show

### Resumo das Metricas por modelo
#### KNN - Mark 1

Metricas

![image](https://user-images.githubusercontent.com/103796137/164940862-59544e93-c65e-4ddb-bb53-a59009a32fdb.png)

Matriz de Confusão

![image](https://user-images.githubusercontent.com/103796137/164941216-59fe974a-22bd-4e95-8a79-2cb7a462bc5f.png)

#### SVC - Mark 2

Metricas

![image](https://user-images.githubusercontent.com/103796137/164941612-78603ff1-3df3-421c-bb9f-281334a4ce72.png)

Matriz de Confusão

![image](https://user-images.githubusercontent.com/103796137/164941676-007ef0d3-00b7-4023-809e-4fbb0c9adfb0.png)


#### Decision Tree Classifier - Mark 3

Metricas

![image](https://user-images.githubusercontent.com/103796137/164941939-885d6514-d7b7-459a-898e-6d59ec650f71.png)

Matriz de Confusão

![image](https://user-images.githubusercontent.com/103796137/164942037-9082f7e5-27a6-429e-9a86-85e848255d01.png)

#### Logistic Regression - Mark 4

Metricas

![image](https://user-images.githubusercontent.com/103796137/164942226-be614e77-4809-4ac1-9ef8-6ea5d6fcd4b1.png)

Matriz de Confusão

![image](https://user-images.githubusercontent.com/103796137/164942280-03ecfff5-4a9f-41e6-9b90-919b3dc3efa4.png)

### Grafico ROC por modelo
#### KNN - Mark 1

![image](https://user-images.githubusercontent.com/103796137/164942496-be116d41-b1c0-45bf-87f1-52ec466715a1.png)

#### Decision Tree Classifier - Mark 3

![image](https://user-images.githubusercontent.com/103796137/164942543-379bdebe-363e-4700-8e4b-f9cf09018ad5.png)

#### Logistic Regression - Mark 4

![image](https://user-images.githubusercontent.com/103796137/164942555-2b2239ca-b0a5-4af4-9385-f4500ea726be.png)

### Comparando Modelos
#### Comparando metricas

![image](https://user-images.githubusercontent.com/103796137/164942562-0c0383b0-c209-4250-bada-aa66b677b911.png)

#### Comparando ganho monetario

![image](https://user-images.githubusercontent.com/103796137/164942576-0571e562-7cd2-448b-b50c-ddedc5940ab2.png)

# 10. Resumo dos modelos

1.	Foi removido as colunas 'order_no', 'Date', 'Time', 'Month_Name' e 'description' pois elas se fazem desnecessárias para o treinamento do modelo;
2.	Foi categorizado e normalizado as colunas 'buyer', 'ship_city', 'ship_state', 'sku', 'cod', 'Year', 'Day_Name' e a coluna ‘order_status' so foi categorizada;
3.	Os dados foram balanceados, pois a proporção de “Delivered to buyer” é de 94% e “Returned to seller” é de 6%;
4.	Os dados foram balanceados em 21 registros para “Delivered to buyer” e 11 registros para “Returned to seller” (os únicos registros no dataset);
5.	Foi utilizado o GridSearchCV para otimização e validação de parâmetros;
6.	Na avaliação dos modelos, foi procurado o modelo que tivesse maior “Recall Score” e “ROC AUC Score”;
7.	Modelo que saiu com melhor acurácia (93,57%) foi o Mark 3 (Decision Tree Classifier);
0. O pior modelo em “Recall Score” (27,27%) foi Mark 1;
8.	Todos os outros modelos tiveram o “Recall Score” de 100%, mas o modelo que teve o maior “ROC AUC Score” foi o Mark 3 (Decision Tree Classifier), com 96,56%;
0. Para a simulação economica, para os gastos referente a devolução foi utilizado a media de gastos de entrega, um valor de 71.86, e para fixação (incentivo, brindes ou contato) foi definido o valor de 10 pontos monetarios;
9.	O modelo que trouxe maior ganho monetário foi o Mark 3 (Decision Tree Classifier), trazendo um ganho de 600 pontos monetários.


