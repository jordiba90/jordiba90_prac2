#DATA_ANALYSIS_PANDAS
import pandas as pd
import numpy as np; import matplotlib as mpl; import matplotlib.pyplot as plt
tcd_df = pd.read_csv('thecure_discography.csv', index_col=0)
"""
#
#################################################################################################
#################################################################################################
#################################################################################################
#DATA_ANALYSIS_PANDAS > Analizamos datos sobre las variables del dataset
#################################################################################################
#################################################################################################
#################################################################################################
#
#################################################################################################
#Inspect a DataFrame - Shape and Size > Obtendremos el tamaño y el número de registros/variables
#################################################################################################
print("\n\nNúmero de registros y variables del dataset:",tcd_df.shape)
print("Número de registros del dataset:",tcd_df.shape[0])
print("Número de variables del dataset:",tcd_df.shape[1])
print("Número de campos del dataset:",tcd_df.size)
#################################################################################################
#Inspect a DataFrame - Head and Tail > Buscaremos los 3 primeros y últimos registros
#################################################################################################
print("\nPrimeros 3 registros del dataset:\n",tcd_df.head(n=3))
print("\nÚltimos 3 registros del dataset:\n",tcd_df.tail(n=3)); print("\n")
#################################################################################################
#Inspect a DataFrame - Info > Obtendremos datos sobre las variables de estudio
#################################################################################################
print("Información sobre el dataset:\n")
print(tcd_df.info())
#################################################################################################
#Rows with .loc" > Buscaremos datos sobre registros filtrando según palabras
#################################################################################################
tcd_df = pd.read_csv('thecure_discography.csv', index_col=7)
print("(EJEMPLO)Datos sobre el registro cuya variable \"track_name\" sea Lovesong:\n"); print(tcd_df.loc['Lovesong'])
print("\n(EJEMPLO)Tipo de dato del registro cuya variable \"track_name\" sea Lovesong:"); print(type(tcd_df.loc['Lovesong']))
print("\n(EJEMPLO)Número de columnas del registro cuya variable \"track_name\" sea Lovesong:",tcd_df.loc['Lovesong'].shape[0])
print("\n(EJEMPLO)Extracto de los registros cuya cuya variable \"track_name\" sea \"Lovesong\" hasta aquellos cuya variable \"track_name\" sea \"Lullaby\":\n"); print(tcd_df.loc['Lovesong':'Lullaby'])
#################################################################################################
#Rows with .iloc > Buscaremos datos sobre registros filtrando según número de registro
#################################################################################################
tcd_df = pd.read_csv('thecure_discography.csv', index_col=0)
print("\nExtracto del primer registro:\n"); print(tcd_df.iloc[0])
print("\nExtracto del primer registro hasta el tercer registro:\n"); print(tcd_df.iloc[0:3])
#################################################################################################
#Columns > Obtendremos datos acerca del nombre de las variables
#################################################################################################
print("\nLista de variables del dataset:\n")
print(tcd_df.columns)
print("\nNúmero de canciones de la discografía de The Cure:",tcd_df['track_name'].shape[0],"\n")
print("Lista de todas las canciones de la discografía de The Cure:")
print(tcd_df['track_name'])
print("\nExtracto de los 3 primeros registros filtrando las variable \"track_name\" y \"track_name_popularity\":\n"); print(tcd_df[['track_name','track_popularity']].head(n=3))
#################################################################################################
#More with .loc > Obtendremos datos acerca de conjuntos de variables
#################################################################################################
print("\n(EJEMLO)Grado de liricidad, acusticidad, e instrumentalidad de las primeras 8 canciones:\n")
print(round(tcd_df.loc[:,'speechiness':'instrumentalness'].head(n=8),5))
#
####################################################################################################
#Min / Max > Cálculo del mínimo y del máximo
#####################################################################################################
print("Datos sobre la canción más antigua:\n"); print(tcd_df.min())
print("\nDatos sobre la canción más reciente:\n"); print(tcd_df.max())
####################################################################################################
#Mean > Cálculo de la media
#####################################################################################################
print("\nMedia aritmética del conjunto de variables:")
print(round(tcd_df.mean(),5))
#
print("\nMedia artimética de la variable \"energy\" de todas las canciones:",round(tcd_df['energy'].mean(),5))
#
print("\nMedia artimética de la variable \"loudness\" de todas las canciones:",round(tcd_df['loudness'].mean(),5))
#
####################################################################################################
#Median > Cálculo de la mediana
#####################################################################################################
print("\nMediana del conjunto de variables:")
print(round(tcd_df.mean(),5))
#
print("Mediana de la variable \"energy\" de todas las canciones:",round(tcd_df['energy'].median(),5))
#
print("Mediana de la variable \"loudness\" de todas las canciones:",round(tcd_df['loudness'].median(),5))
####################################################################################################
#Quantiles > Cálculo de los cuartiles
#####################################################################################################
print("\nCuartiles del conjunto de variables:")
print(tcd_df.quantile([0.25, 0.5, 0.75, 1]),"\n")
#
print("\nCuartiles de la variable \"energy\" de las canciones:\n")
print(tcd_df['energy'].quantile([0.25, 0.5, 0.75, 1]),"\n")
#
print("\nCuartiles de la variable \"loudness\" de las canciones:\n")
print(tcd_df['loudness'].quantile([0.25, 0.5, 0.75, 1]),"\n")
####################################################################################################
#Standard Deviation > Cálculo de la desviación estándar
####################################################################################################
print("\nDesviación del conjunto de variables:\n"); print(round(tcd_df.std(),5))
#
print("Desviación estándar de la variable \"energy\":",round(tcd_df['energy'].std(),5))
#
print("Desviación estándar de la variable \"loudness\":",round(tcd_df['loudness'].std(),5))
####################################################################################################
#Variance > Cálculo de la varianza
####################################################################################################
print("\nVarianza del conjunto de variables:\n"); print(round(tcd_df.var(),5))
#
print("\nVarianza de la variable \"energy\":",round(tcd_df['energy'].var(),5))
#
print("\nVarianza de la variable \"loudness\":",round(tcd_df['energy'].var(),5))
####################################################################################################
#describe() > Cálculo de datos estadísticos
####################################################################################################
print("\nDatos estadísticos del conjunto de variables:\n")
print(tcd_df.describe())
#
print("\nDatos estadísticos de la variable \"energy\":\n")
print(tcd_df['energy'].describe())
#
print("\nDatos estadísticos de la variable \"loudness\":\n")
print(tcd_df['loudness'].describe())
####################################################################################################
#Categorical Variable
####################################################################################################
print("\Frecuencia de la variable \"energy\":\n")
print(tcd_df['energy'].value_counts())
#
####################################################################################################
#Groupby
####################################################################################################
print("Agrupación de datos según la variable \"album_name\":\n")
print(tcd_df.groupby('album_name').mean())
####################################################################################################
#Aggregation
####################################################################################################
print("\nAgrupación de datos según la variable \"album_name\" filtrando la variable \"duration_ms\" para calcular el mínimo, la media, y el máximo de la duración en minutos de las canciones:\n")
print(tcd_df.groupby('album_name')['duration_ms'].agg(['min', np.mean, max])/1000/60)
#
print("\nÁlbumes según la medias de las variables \"energy\" y \"loudness\":\n")
print(tcd_df.groupby('album_name').agg({'energy':[np.mean],'loudness':[np.mean]}))
#
#################################################################################################
#################################################################################################
#################################################################################################
#DATA_VISUALIZATION_MATPLOTLIB > Visualizamos datos sobre las variables del dataset
#################################################################################################
#################################################################################################
#################################################################################################
#
####################################################################################################
#Scatterplot > Observamos la correlación entre las variables "\energy"\ y "\loudness"\
####################################################################################################
plt.scatter(tcd_df['energy'], tcd_df['loudness'],marker='+',color='b')
plt.xlabel('energy'); plt.ylabel('loudness'); plt.title('scatterplot')
plt.show()
####################################################################################################
#Histogram > Observamos la evolución de las variables
####################################################################################################
tcd_df.plot(kind='hist',title = 'dataset',bins=5)
plt.show()
#
tcd_df['energy'].plot(kind='hist',title = 'energy',bins=5)
plt.show()
#
tcd_df['loudness'].plot(kind='hist',title = 'loudness',bins=5)
plt.show()
#
#####################################################################################################
#Boxplot > Observamos la variación de las variables
#####################################################################################################
tcd_df['energy'].describe(); plt.style.use('classic'); tcd_df.boxplot(column='energy');
plt.show()
#
tcd_df['loudness'].describe(); plt.style.use('classic'); tcd_df.boxplot(column='loudness');
plt.show()
#################################################################################################
#################################################################################################
#################################################################################################
#DATA_STATISTICS_SKLEARN > Realizamos predicciones sobre las variables
#################################################################################################
#################################################################################################
#################################################################################################
from sklearn.metrics import mean_squared_error
#####################################################################################################
#Evaluating model > Realizamos un análisis de las variaciones
#####################################################################################################
X = tcd_df[['energy']]; Y = tcd_df['loudness']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)
from sklearn.linear_model import LinearRegression
model = LinearRegression(); model.fit(X_train, Y_train)
new_RM = np.array([6.5]).reshape(-1,1) # make sure it's 2d
y_test_predicted = model.predict(X_test)
plt.scatter(X_test, Y_test,label='testing data')
plt.plot(X_test, y_test_predicted,label='prediction', linewidth=3)
plt.xlabel('energy'); plt.ylabel('loudness'); plt.legend(loc='upper left')
plt.show()
#####################################################################################################
#Residuals > Analizamos los residuos
#####################################################################################################
residuals = Y_test - y_test_predicted
# plot the residuals
plt.scatter(X_test, residuals)
# plot a horizontal line at y = 0
plt.hlines(y = 0, xmin = X_test.min(), xmax=X_test.max(),linestyle='--')
# set xlim
plt.xlim((-20, 20)); plt.xlabel('energy'); plt.ylabel('residuals')
plt.show()
#####################################################################################################
#Mean Squared Error > Analizamos cuánto explicativas son las variables
#####################################################################################################
print("MSE (option1): ",(residuals**2).mean())
print("MSE (option2): ",mean_squared_error(Y_test, y_test_predicted),"\n")
#####################################################################################################
#R-squared
#####################################################################################################
print("RS (explained):",round(model.score(X_test, Y_test)*100,2),"%\n") # Model explanation variability
print("Model variation:",round(((Y_test-Y_test.mean())**2).sum(),2))
print("Total variation:",round((residuals**2).sum(),2))
"""
"""
VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV----NO FUNCIONA------VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#DATA_CLASSIFICATION_PREDICTION
import numpy as np; import pandas as pd;
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
tcd_df.drop('album_name', axis=1, inplace=True)
#####################################################################################################
#Label Prediction
#####################################################################################################

X = tcd_df[['energy', 'loudness']]; y = tcd_df['album_name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5) # instantiate
knn.fit(X_train, y_train) # fit
pred = knn.predict(X_test)

print("Neighbours first 5:")
print(pred[:5],"\n")

print("Neighbours first 5 (%):")
y_pred_prob = knn.predict_proba(X_test)
print(pred[:5],"\n")

#####################################################################################################
#Label Prediction
#####################################################################################################

y_pred_prob = knn.predict_proba(X_test)
print("Neighbours 11 to 12:")
print(y_pred_prob[10:12],"\n")

print("Neighbours 11 to 12 (%):")
y_pred_prob = knn.predict_proba(X_test)
print(pred[10:12])

"""
