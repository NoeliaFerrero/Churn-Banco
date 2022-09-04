import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


class Gender (BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        mapper = {'M': 0,  'F': 1}
        col_mapped = x.replace(mapper)
        return col_mapped


class Income(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        mapper = {'Less than $40K': 1,  '$40K - $60K': 2, '$60K - $80K': 3, '$80K - $120K': 4,  '$120K +': 5}
        col_mapped = x.replace(mapper)
        return col_mapped


class Education(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        mapper = {'Uneducated': 1, 'High School': 2, 'College': 3, 'Graduate': 4,  'Post-Graduate': 5, 'Doctorate': 6}
        col_mapped = x.replace(mapper)
        return col_mapped


def drop_columns(column, df):
    df = df.drop(column, axis=1)
    return df


def eda(df, col, umbral=3):
    # nulos
    df = df.replace('Unknown', np.nan)
    df.loc[df[df[col].isnull()][col].index, col] = np.random.choice(df[df[col].notnull()][col])

    # outliers
    lista_income = ['Less than $40K', '$40K - $60K', '$60K - $80K']
    Q1 = df.Credit_Limit.quantile(0.25)
    Q3 = df.Credit_Limit.quantile(0.75)
    IQR = Q3 - Q1
    umbral_outlier = IQR * umbral
    df = df.drop(df[(df['Credit_Limit'] == 34516.000) & (df['Income_Category'].isin(lista_income))].index)
    return df


def fig_analysis(titulo, name, column, split=None):
    plt.figure(figsize=(12, 7))
    sns.displot(column)
    plt.title(titulo, fontsize=12)
    plt.savefig(name)
    if split == 'Attrition_Flag':
        plt.figure(figsize=(8, 8))
        sns.displot(data=df, x=column, hue=df.Attrition_Flag)
        plt.title(titulo, fontsize=12)
        plt.xticks(rotation=45)
        plt.savefig(name)


def dataset_split():
    df.Attrition_Flag = df.Attrition_Flag.replace({'Existing Customer': 1, 'Attrited Customer': 0})
    X = df.drop('Attrition_Flag', axis=1)
    y = df.Attrition_Flag
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


def pipeline_preprocessing(X, traintest):
    if traintest == 'train':
        X_transform = pd.DataFrame(preprocesador.fit_transform(X))
    elif traintest == 'test':
        X_transform = pd.DataFrame(preprocesador.transform(X))

    X_transform.columns = ['Marital_Status_Divorce', 'Marital_Status_Single',
                           'Marital_Status_Married', 'Card_Platinum',
                           'Card_Gold', 'Card_Silver', 'Card_Blue', 'Gender',
                           'Income_Category', 'Education_Level',
                           'Credit_Limit', 'Total_Revolving_Bal',
                           'Total_Trans_Amt', 'Customer_Age', 'Dependent_count',
                           'Months_on_book', 'Total_Relationship_Count',
                           'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
                           'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                           'Avg_Utilization_Ratio']
    return X_transform


def grid_search(pipe, params):
    grid = GridSearchCV(pipe, param_grid=params)
    grid.fit(X_train_smote, y_train_smote)
    grid.score(X_train_smote, y_train_smote)
    return grid.best_estimator_


def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(12, 7))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    for i in range(0, len(fpr)):
        plt.plot(fpr[i], tpr[i], label=str(modelos[i])[18:40])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('ROC_curve.jpg')


def cross_validation(modelos):
    fprlist = []
    tprlist = []

    for i in range(0, len(modelos)):
        modelo = modelos[i]
        modelo_pipe = make_pipeline(modelo)
        fold = KFold(n_splits=4, random_state=1, shuffle=True)
        scores = cross_validate(modelo_pipe,
                                X_train_smote,
                                y_train_smote,
                                scoring=metricas,
                                cv=fold,
                                n_jobs=-1)

        modelo_pipe.fit(X_train_smote, y_train_smote)
        probs = modelo_pipe.predict_proba(X_train_smote)
        probs = probs[:, 1]
        fpr, tpr, umbral = roc_curve(y_train_smote, probs)
        fprlist.append(fpr)
        tprlist.append(tpr)

        print('\n \033', str(modelo)[18:40], '  \033\n')
        print('Accuracy:', scores['test_accuracy'].mean())
        print('Recall:', scores['test_recall'].mean())
        print('Precision:', scores['test_precision'].mean())
        print('F1:', scores['test_f1'].mean())
        print('-----------------------------------')

    return fprlist, tprlist


if __name__ == '__main__':
    #path = sys.argv[1]  # path = C:\Users\cecim\JupyterProjects\Diplodatos\modulo2\TP\BankChurners.csv
    if len(sys.argv) < 2:
        print('Ingrese el path como un argumento')
    else:
        path = sys.argv[1]
        if isinstance(path, str):
            df = pd.read_csv(path)
            pd.set_option('display.float_format', lambda x: '%.3f' % x)
            pd.set_option('display.max.columns', 100)

            # ----------------------------- Eliminación de columnas innecesarias ------------------- #
            df = drop_columns('CLIENTNUM', df)
            df = drop_columns('Avg_Open_To_Buy', df)

            # ---------------------------------------- EDA----------------------------------------- #
            '''Se realizó en primer lugar el análisis de nulos, asignándole a las variables categóricas (que 
            son las que tienen valores Unknown) un valor en forma aleatoria, dicho valor se encuentra 
            dentro del set de datos. Luego se realizó el análisis de outliers y las distribuciones multi y 
            univariada de las variables de mayor interés.'''
            df = eda(df, 'Income_Category')
            df = eda(df, 'Marital_Status')
            df = eda(df, 'Education_Level')

            # ----------------------------- Análisis univariado y multivariado ------------------- #
            fig_analysis('Cantidad total de Transacciones', 'transacciones.jpg', df.Total_Trans_Ct)
            fig_analysis('Límite de Crédito', 'credito.jpg', df.Credit_Limit)
            fig_analysis('Cambio en la Cant de Transacciones', 'transaccionescambio.jpg', df.Total_Ct_Chng_Q4_Q1)
            fig_analysis('Monto del total de Transacciones', 'transaccionesmonto.jpg', df.Total_Trans_Amt)
            fig_analysis('Saldo remanente', 'saldo.jpg', df.Total_Revolving_Bal)
            fig_analysis('Transacciones por abandono', 'transaccionesabandono.jpg', df.Total_Trans_Ct, 'Attrition_Flag')
            fig_analysis('Ingresos por abandono', 'ingresosabandono.jpg', df.Income_Category, 'Attrition_Flag')
            fig_analysis('Límite de Crédito por abandono', 'creditoabandono.jpg', df.Credit_Limit, 'Attrition_Flag')
            fig_analysis('Género por abandono', 'generoabandono.jpg', df.Gender, 'Attrition_Flag')

            # ---------------------------- División en set de TRAIN y TEST ------------------------------------- #
            '''Como paso posterior, se realiza la división TRAIN, TEST del set de datos(75 y 25% respectivamente)
            y luego, el preprocesamiento de las variables categóricas.'''
            X_train, X_test, y_train, y_test = dataset_split()

            # ----------------------------------- Pipeline preprocesamiento--------------------------------------- #
            '''Aplicación de pipeline para tratamiento de variables categóricas (OneHoteEncoder y LabelEncoding)
             y escalamiento de variables numéricas'''
            preprocesador = make_column_transformer(
                (OneHotEncoder(), ['Marital_Status', 'Card_Category']),
                (Gender(), ['Gender']),
                (Income(), ['Income_Category']),
                (Education(), ['Education_Level']),
                (StandardScaler(), ['Credit_Limit',
                                    'Total_Revolving_Bal',
                                    'Total_Trans_Amt']),
                remainder='passthrough'
            )

            X_train_transform = pipeline_preprocessing(X_train, 'train')
            X_test_transform = pipeline_preprocessing(X_test, 'test')

            # Sampleo para balancear el target al 50%
            over_smote = SMOTE(sampling_strategy=0.5)
            X_train_smote, y_train_smote = over_smote.fit_resample(X_train_transform, y_train)

            # ---------------------------------- GridSearch ------------------------------ #
            '''GridSearch para verificar cuáles son los mejores hiperparámetros para cada modelo de interés.
            Se declara un diccionario con los hiperparámetros específicos de cada modelo.'''
            DT_pipeline = make_pipeline(DecisionTreeClassifier())
            DT_params = {'decisiontreeclassifier__criterion': ['gini', 'entropy'],
                         'decisiontreeclassifier__max_depth': [5, 7, 9, 10, 11, 13]}

            SVM_pipeline = make_pipeline(SVC())
            SVM_params = {'svc__C': [0.1, 1],
                          'svc__gamma': [1, 0.01],
                          'svc__kernel': ['rbf'],
                          'svc__probability': [True]}

            KNN_pipeline = make_pipeline(KNeighborsClassifier())
            KNN_params = {'kneighborsclassifier__n_neighbors': [2, 3, 5],
                          'kneighborsclassifier__weights': ['uniform', 'distance'],
                          'kneighborsclassifier__metric': ['euclidean', 'manhattan']}

            LR_pipeline = make_pipeline(LogisticRegression())
            LR_params = {'logisticregression__solver': ['newton-cg', 'liblinear'],  # 'lbfgs','sag','saga'
                         'logisticregression__penalty': ['l2'],  # 'l2','elastic-net'
                         'logisticregression__C': [0.1, 1]}

            RF_pipeline = make_pipeline(RandomForestClassifier())
            RF_params = {'randomforestclassifier__n_estimators': [20, 30, 50],
                         'randomforestclassifier__max_features': ['auto', 'sqrt'],
                         'randomforestclassifier__max_depth': [3, 5, 6],
                         'randomforestclassifier__min_samples_split': [2, 5],
                         'randomforestclassifier__min_samples_leaf': [1, 2]}

            modelos = [grid_search(DT_pipeline, DT_params), grid_search(LR_pipeline, LR_params),
                       grid_search(KNN_pipeline, KNN_params), grid_search(RF_pipeline, RF_params),
                       grid_search(SVM_pipeline, SVM_params)]
            print('GridSearch. Mejores parámetros de cada modelo:')
            print(modelos)
            print('------------------------------------')

            # ----------------------------------- Cross Validation --------------------------------------- #
            '''Aplicación de KFlod con 4 carpetas al TRAIN para verificar cuál modelo tiene mejores métricas
            antes de decidir entrenarlo.'''
            metricas = ['accuracy', 'recall', 'precision', 'f1']
            print('Cross Validation. Métricas de cada modelo en el TRAIN:')
            fpr_list, tpr_list = cross_validation(modelos)
            print('------------------------------------')
            # ROC curve
            '''Luego de determinar cuales son los mejores parámetros con GrisSearch y de verificar cuál 
               modelo tiene mejor desempeño con Cross Validation (K-Folds), se procede a hacer la curva ROC 
               para complemetar el análisis de kfold y poder seleccionar el mejor modelo antes de tomar la 
               decisión de entrenarlo.'''
            plot_roc_curve(fpr_list, tpr_list)

            # ------------------------------ Entrenamiento de modelo elegido ------------------------------ #
            '''Se seleccionó el modelo del árbol de decisión por ser el de mayor F1, mayor accuracy y por 
            tener un alto valor de recall. A su vez, decidimos seleccionarlo, ya que en comparación con RF 
            (el cual también presenta valores aceptables) posee menor costo computacional. Un último punto 
            que influyó fue la curva ROC, esta se acerca a 1 sin producir overfitting'''
            model_tree = grid_search(DT_pipeline, DT_params)
            model_tree.fit(X_train_smote,y_train_smote)

            # ------------------------------ Predicciones y métricas ------ ------------------------------ #
            print('Métricas del modelo Decision Tree. Set de TEST')
            y_pred = model_tree.predict(X_test_transform)
            print('Matriz de confusión:', confusion_matrix(y_test, y_pred))

            print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
            print('Precision Score : ' + str(precision_score(y_test, y_pred)))
            print('Recall Score : ' + str(recall_score(y_test, y_pred)))
            print('F1 Score : ' + str(f1_score(y_test, y_pred)))
            print('------------------------------------')
            print('termine')

        else:
            print('El argumento tiene que ser una cadena de texto')
            print('termine')
