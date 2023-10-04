#!/usr/bin/env python
# coding: utf-8

# ### Прогнозирование стоимости жилья

# 
# Объективная оценка стоимости недвижимого имущества необходима при:
# 
# ­ операциях купли-продажи или сдачи в аренду;
# 
# ­ акционировании предприятий и перераспределении имущественных долей;
# 
# ­ кадастровой оценке для налогообложения;
# 
# ­ страховании;
# 
# ­ кредитовании под залог объектов недвижимости;
# 
# ­ исполнении права наследования, судебного приговора;
# 
# и других операциях, связанных с реализацией имущественных прав на объекты недвижимости.
# 
# Нужно искать способы точной оценки, как со стороны продавца, так и со стороны покупателя. Поэтому крайне важно независимое, быстрое и точное знание о ценах на рынке жилой недвижимости.

# загрузим библиотеки

# In[243]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re


from scipy.stats import ttest_ind
from itertools import combinations
from collections import Counter

from datetime import datetime
import xlrd, xlwt
import re 
import warnings

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

#from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression
from sklearn import metrics #
# инструмент для разделения датасета:
from sklearn import model_selection, preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

#Готовим подвыборки
from sklearn.model_selection import train_test_split
random_seed = 42
#Подгружаем RMSLE
from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# увеличим количество отображения строк и загрузим датасет

# In[244]:


pd.set_option('display.max_rows', 80)  # показывать больше строк
pd.set_option('display.max_columns', 80)  # показывать больше колонок
#открываем файл
data = pd.read_csv('final_data.csv')


# In[4]:


#Функции
# Наличие выбросов
def outliers(column):
    outlier1 = outlier2 = 0
    """ 
    Функция проверяет наличие выбросов через межквартильный размах
    """
    q1 = data[column].quantile(q=0.25, interpolation='midpoint')
    q3 = data[column].quantile(q=0.75, interpolation='midpoint')
    MR = round(q3 - q1, 1)
    limit1 = q1 - 1.5 * MR
    limit2 = q3 + 1.5 * MR
    if data[column].min() < limit1:
        print(f'выбросы ниже {limit1}')
    elif data[column].max() > limit2:
        print(f'выбросы выше {limit2}')
    else:
        print('выбросов нет')
    return

# Визуализация признаков
def graf_cat(col):
    fig, (ax2) = plt.subplots(
        nrows=1, ncols=1,
        figsize=(12, 6)
    )

    sns.boxplot(col, "log_price", data=data, ax=ax2)

    ax2.set_title(f'Зависимость логарифма цены от {col}')
    ax2.set_xlabel(f'значения {col}')
    ax2.set_ylabel('логарифм цены')
    plt.show()
    
# Построение гистограммы
def gistogramma(col):
    fig, (ax1) = plt.subplots(
    nrows=1, ncols=1,
)

    ax1.hist(data[col], bins=77)
    ax1.set_title(f'Гистограмма {col}')
    ax1.set_xlabel(f'знечения {col}')
    ax1.set_ylabel('количество значений')


    plt.show
    
#Функция проверяет статистически значимые различия колонок
def get_stat_dif(column):

    cols = data.loc[:, column].value_counts().index[:]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(data.loc[data.loc[:, column] == comb[0], 'log_price'],
                     data.loc[data.loc[:, column] == comb[1], 'log_price']).pvalue \
                <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break
            
# Функция для определения метрик моделей           
def print_regression_metrics(Ytest, y_pred):
    mse = mean_squared_error(Ytest, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')


# Посмотрим какие данные в датасете

# In[4]:


data.info()
data.head()


# In[5]:


#Проверим на наличие дубликатов
data.duplicated().value_counts()


# In[245]:


#Удалим их и проверим данные
data.drop_duplicates(inplace=True)
display(data.shape)


# Данные содержат 18 признаков и 377135 элементов. Данные можно разделить на:

# numerical_columns = ['baths', 'sqft', 'beds', 'stories', 'mls-id', 'MlsId', 'target']
# 
# binary_columns = ['private pool', 'PrivatePool']
# 
# categorial_columns = ['status', 'propertyType', 'street', 'fireplace', 'city', 'zipcode', 'state', 'homeFacts', 'schools']

# Построим тепловую карту пропущенных значний

# In[7]:


cols = data.columns[:]
colours = ['#000099', '#ffff00'] # желтый цвет - пропущенные значения
sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours))


# Больше всего пропущенных значений в признаках: private pool, fireplace, mls-id, PrivatePool. Наличие собственного бассейна содержится в 2 признаках - можно их объединить, а в случае когда нет данных по наличию бассейна или камина -  трактовать как их отсутствие. 
# В стоимости жилья также есть пропущенные значения. удалим эти данные

# In[8]:


#Проверим наличие пропущенных значений в стоимости жилья и удалим их (данные не подходят для модели)
data.target.isna().value_counts()


# In[246]:


data.dropna(subset = ['target'], inplace=True)
display(data.shape)


# ### Предобработка данных

# #### pool&private_pool

# In[10]:


# количество значений когда есть данные 
len(data[data.PrivatePool.isna() != True])


# In[11]:


data['private pool'].value_counts()


# In[247]:


# объединим данные о наличии бассейна в признак Pool
data['private pool'] = data['private pool'].fillna('0')
data['PrivatePool'] = data['PrivatePool'].fillna('0')
data['private pool'] = data['private pool'].apply(lambda x: x.lower())
data['private pool'] = data['private pool'].apply(lambda x: x.replace('yes', '1'))
data['private pool'] = data['private pool'].astype('int')
data['PrivatePool'] = data['PrivatePool'].apply(lambda x: x.lower())
data['PrivatePool'] = data['PrivatePool'].apply(lambda x: x.replace('yes', '1'))
data['PrivatePool'] = data['PrivatePool'].astype('int')
data['Pool'] = data.PrivatePool + data['private pool']


# In[13]:


data.Pool.value_counts()


# Значит информация не дублировалась и можно удалить исходные признаки

# In[248]:


column_to_del = ['private pool', 'PrivatePool']


# #### street

# In[249]:


# Приведем к однообразию адрес
data.street = data.street.fillna('unknown address')
data.street = data.street.apply(lambda x: x.lower())
data.street = data.street.apply(lambda x: x.replace('address not disclosed', 'unknown address').replace('(undisclosed address)', 'unknown address').replace('undisclosed address', 'unknown address').replace('address not available', 'unknown address'))


# #### beds& sqft

# In[250]:


# Рассмотрим признак количество спален и заменим на наиболее частое
data['beds'] = data['beds'].fillna('3')
data.beds = data.beds.astype(str).apply(lambda x: x.replace(' Beds', '').replace(' bd', '').replace('--', '0').replace('.0', ''))
data.beds = data.beds.astype(str).apply(lambda x: x.replace('Baths', '3').replace('Bath', '3'))


# In[17]:


data.beds.value_counts().tail(10)


# Исправим ошибочно записанные данные площади в beds

# In[251]:


#отметим пропущенные значения
data['sqft'] = data['sqft'].fillna('empty')


# In[252]:


# перепишем в test2 значения площади в sqft и объединим с исходным признаком sqft(создав sqft2)
data['test2'] = data['beds'].apply(lambda x: x if 'sqft' in x else 'empty')
x = data[data.ne('empty')]
data['sqft2'] = x['test2'].combine_first(x['sqft'])


# In[253]:


# площадь в arces из признака beds запишем в test2, переведем в sqft и округлим
data['test2'] = data['beds'].apply(lambda x: x if 'acre'in x else '0')
data['test2'] = data.test2.apply(lambda x: x.replace('acres', '').replace('acre', ''))
data['test2'] = data['test2'].astype(float)
data['test2'] *= 43560
data.test2 = data.test2.apply(lambda x: round(x))


# In[254]:


#объединим с предыдущим sqft2
data['test2'] = data.test2.apply(lambda x: 'empty' if x == 0 else x)
x = data[data.ne('empty')]
data['sqft2'] = x['test2'].combine_first(x['sqft2'])


# In[255]:


#test2 нам больше не нужен
column_to_del.append('test2')


# In[23]:


#Приверим правильность действий
data[data.beds == '1 acre'].sample()


# In[256]:


# Приведем к числовому формату
data.sqft2 = data.sqft2.apply(lambda x: str(x).replace(' sqft', '').replace('--', '1200').replace(',', '').replace('Total interior livable area: ', '').replace('610-840', '725').replace('nan', '1200'))
data.sqft = data.sqft2.apply(lambda x: int(x))


# In[257]:


#Удалим ненужный признак
column_to_del.append('sqft2')


# Стоит отметить что площадь дома нулевая - значит отсутствие данных. И стоит найти способ замены на число: можно попробовать через связь с количеством спален

# In[258]:


data['beds2'] = data['beds'].apply(lambda x: 'Nan' if 'sqft' in x else('Nan' if 'acre' in x else x))
data['beds2'] = data['beds2'].apply(lambda x: x if x.isnumeric() == True else 'Nan')
data[data.beds2 != 'Nan'].groupby('beds2')['sqft'].describe().head(20)


# In[259]:


#приведем к числовому формату
data.beds2 =data.beds2.apply(lambda x: x.replace('Nan', '0'))
data.beds =data.beds2.astype(int)
data.beds2 = data.beds.apply(lambda x: 'empty'  if x == 0 else x)


# In[260]:


# перепишем в test2 значения площади в sqft и объединим с исходным признаком sqft(создав sqft2)
x = data[data.ne('empty')]
data['beds2'] = x['beds2'].combine_first(x['sqft'])


# In[261]:


data.beds2 =data.beds2.astype(int)


# In[262]:


def bath_from_sqft(x):

    if (x >0) & (x<= 144):
        a = x
    elif (x == 0)| (x >= 1500) & (x < 2100):
        a = 3
    elif (x >0) & (x < 1000):
        a = 1
    elif (x>= 1000) & (x < 1500):
        a = 2
    elif (x >= 2100 ) & (x < 3100):
        a = 4
    elif (x >= 3100) & (x < 4500):
        a = 5
    elif (x >= 4500)& (x < 5600):
        a = 6
    else:
        a = 7
    return a


# In[263]:


data['beds'] = data['beds2'].apply(bath_from_sqft)


# In[264]:


column_to_del.append('beds2')


# In[33]:


data.beds.describe()


# #### baths

# In[265]:


# Заполним пропуски и приведем к числовому формату количество ванных комнат
data.baths2 = data.baths.fillna('2')
data.baths2 = data.baths2.apply(lambda x: x.lower())
data.baths2 = data.baths2.apply(lambda x: x.replace('bathrooms: ', '').replace(' baths', '').replace(' ba', ''))
data.baths2 = data.baths2.apply(lambda x: x.replace(',', '.').replace('~', '2').replace('--', '2').replace('+', '').replace('sq. ft.', ''))
data.baths2 = data.baths2.apply(lambda x: '2' if x == ' ' else ('2' if '—' in x else x))
data.baths2 = data.baths2.apply(lambda x: x[0] if '/' in x else (x[0] if '-' in x else x))
data.baths2 = data.baths2.apply(lambda x: x if x.isnumeric() == True else '2')
data.baths2 = data.baths2.astype(str).apply(lambda x: x if int(x) < 50 else x[0])
data.baths = data.baths2.astype(int)


# #### status

# In[266]:


# Заменим пропущенные значения на "для продажи" и преобразуем текст
data['status'] = data['status'].fillna('for sale')
data['status'] = data['status'].apply(lambda x: x.lower())
data.status2 = data.status.apply(lambda x: x.replace('   showing', '').replace(' show', '').replace('a ', '').replace('active under contract', 'active'))


# In[267]:


# обработаем статус объявлений
data.status2 = data.status2.apply(lambda x: x.replace(': nov', '').replace(': dec', '').replace(': oct', '').replace('foreclosed', 'foreclosure'))
data.status2 = data.status2.str.replace('\d{1,2}', '')
data.status2 = data.status2.apply(lambda x: x.replace(' continue to', '').replace(' backups', ''))
data.status2 = data.status2.apply(lambda x: x.replace('option ', '').replace(' .', ''))
data.status2 = data.status2.apply(lambda x: x if x != 'p' else 'pre-foreclosure')
data.status2 = data.status2.apply(lambda x: x if x != ' / auction' else 'auction')
data.status2 = data.status2.apply(lambda x: x.replace('under ', '').replace('u ', '').replace(' in', '').replace(' finance andspection', '').replace(' taking', ''))
data.status = data.status2.apply(lambda x: x.replace('pre-foreclosure / ', ''))
data.status = data.status.apply(lambda x: x.replace('pending -', 'pending').replace('auction - active', 'auction').replace(' with offer', '').replace('active/contingent', 'active'))


# #### fireplace

# In[268]:


# Заменим пустые значения на наиболее частое и обработаем признак наличия камина
data['fireplace'] = data['fireplace'].fillna('empty')
data['fireplace'] = data['fireplace'].apply(lambda x: x.lower())
data['fireplace'] = data.fireplace.apply(lambda x: x.replace('not applicable', 'no').replace('0', 'no').replace('1 fireplace', '1'))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('gas logs', 'gas').replace('gas log', 'gas').replace('empty', 'yes').replace('gas/gas', 'gas'))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('fiplace', 'yes').replace('location', 'yes').replace(' burning', '').replace('fireplace yn', 'yes'))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace(', one', '').replace('1, ', '').replace('fireplace family rm', 'family room').replace(', walk-in closets', ''))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('one', '1').replace(', storage', '').replace(', utility connection', '').replace(', extra closets', ''))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('decorative', 'no').replace('gas fireplace', 'gas').replace(', in', '').replace('familyrm', 'family room'))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('# fireplaces - woodburning', 'wood').replace('# fireplaces - gas', 'gas'))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('gas living room', 'gas, living room').replace('den/', '').replace('den', 'family room'))
data.fireplace = data.fireplace.apply(lambda x: x.replace(',', '').replace(' room', ''))


# In[269]:


data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('3+', '3').replace(', redecorated', '').replace('fireplace', 'yes'))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('natural gas', 'gas').replace('family room, gas', 'gas family room').replace('gas, family room', 'gas family room').replace('other (see remarks)', 'no').replace('gas, great room', 'gas great room').replace('yess', 'yes').replace('in family room', 'family room').replace(' fuel,yes', '').replace(', fire sprinkler system', ''))
data['fireplace'] = data['fireplace'].apply(lambda x: x.replace(' frplc', ''))


# #### propertyType

# In[270]:


# заменим пропущенные значения, сделаем все буквы маленькими и обработаем признак тип собственности
data['propertyType'] = data['propertyType'].fillna('single family')
data['propertyType'] = data['propertyType'].apply(lambda x: x.lower())
data['propertyType'] = data['propertyType'].apply(lambda x: x.replace('-', ' ').replace(' / ', '/').replace('mfd/mobile', 'mobile/manufactured').replace(' home', '').replace('co op', 'coop'))
data['propertyType'] = data['propertyType'].apply(lambda x: x.replace('one', '1').replace('two', '2').replace('2 story', '2 stories').replace('singlefamilyresidence', 'single family'))
data['propertyType'] = data['propertyType'].apply(lambda x: x.replace(' (see remarks)', '').replace('/townhome', '').replace('other style', 'other').replace('lot/', '').replace(', traditional', ''))
data['propertyType'] = data['propertyType'].apply(lambda x: x.replace('/row', '').replace('/modern', '').replace('/mediterranean', '').replace('condominium', 'condo').replace('cooperative', 'coop'))
data['propertyType'] = data['propertyType'].apply(lambda x: x.replace('detached, ', ''))


# #### mls-id

# In[271]:


#заменим пропущенные значения
data['mls-id'] = data['mls-id'].fillna('No')
data['mls-id'] = data['mls-id'].apply(lambda x: x.replace(' MLS #', 'No').replace(' MLS#', '').replace('No ', 'No').replace('No', '0'))
data['mls-id'] = data['mls-id'].str.replace('\D{1,5}', '')
data['mls-id'] = data['mls-id'].str.replace('\d{2}-', '')
data['mls-id'] = data['mls-id'].apply(lambda x: 0 if x == '' else x)
data['mls-id'] = data['mls-id'].apply(lambda x: int(x))


# MlsId содержит информацию дублирующуся в других столбцах. Удалим его

# In[272]:


column_to_del.append('MlsId')


# #### homeFacts

# In[273]:


# Обработаем признак
data.homeFacts = data.homeFacts.astype(str).apply(lambda x: x.replace("'", '').replace('[', '').replace(']', '').replace('{', '').replace('}', ''))
data.homeFacts = data.homeFacts.apply(lambda x: x.replace('factValue: ', '').replace('factLabel: ', '').replace('atAGlanceFacts: ', '').replace(', ,', ', nan, '))
data.homeFacts = data.homeFacts.apply(lambda x: x.replace('Year built, ', '').replace(' Remodeled year,', '').replace('Heating, ', '').replace(' Cooling,', '').replace(' sqft', '').replace('$', '').replace(' spaces', '').replace('Parking, ', '').replace('lotsize, ', '').replace(', Price/sqft', '').replace('/sqft', ''))
data.homeFacts = data.homeFacts.apply(lambda x: x.replace(' / Sq. Ft.', ''))


# In[43]:


data.homeFacts[32]


# Создадим отдельный датафрейм где разделим все признаки

# In[274]:


data_homeFacts = data.homeFacts.str.split(', ', expand=True)
#data_homeFacts.columns = ['Year built', 'Remodeled year', 'Heating', 'Cooling', 'Parking', 'lotsize_sqft', 'Price/sqft']
data_homeFacts.head(6)


# In[275]:


#Построим тепловую карту пропущенных значний
cols = data_homeFacts.columns[:]
colours = ['#000099', '#ffff00'] # желтый цвет - пропущенные значения
sns.heatmap(data_homeFacts[cols].isnull(), cmap=sns.color_palette(colours))


# In[276]:


data_homeFacts.columns = data_homeFacts.columns.astype(str)


# In[277]:


data['Year_built'] = data.homeFacts.apply(lambda x: x[0:4])
data['Remodeled_year'] = data_homeFacts['1']
data['Heating'] = data_homeFacts['2']
data['Cooling'] = data_homeFacts['3']
data['Parking'] = data_homeFacts['4']
data['lotsize_sqft'] = data_homeFacts['5']
data['Price/sqft'] = data_homeFacts['6']


# In[278]:


data.Year_built = data.Year_built.apply(lambda x: x.replace(', na', '2019').replace(', ', '20').replace('None', '2019').replace('/,/s/d{2}', '2019').replace('No D', '2019'))


# In[279]:


data.Year_built.value_counts().sample(5)


# In[280]:


data.Year_built = data.Year_built.apply(lambda x: x.replace(',', '0').replace('120n', '2019').replace('5599', '2019'))
data.Year_built = data.Year_built.apply(lambda x: int(x))


# In[281]:


data[data.Year_built < 1500].sample(3)


# In[282]:


#удалим эти значения из датасета
data = data.loc[data['Year_built'] > 1500]


# In[283]:


data['Remodeled_year'] = data['Remodeled_year'].apply(lambda x: x.replace('nan', '0').replace('None', '0'))#['Remodeled year'] if x['Remodeled year'] != 'None' else (x['Remodeled year'] if x['Remodeled year'] != 'nan' else x.Year_built, axis=1))


# In[284]:


# заполним годом постройки если не было реконструкции здания
data['Remodeled_year'] = data['Remodeled_year'].astype(int)
data['Remodeled_year'] = data.apply(lambda x: x.Remodeled_year if x.Remodeled_year > x.Year_built else x.Year_built, axis=1)
data['Remodeled_year'] = data['Remodeled_year'].astype(int)


# In[55]:


#Создадим новый признак
#data['years_old'] = data.Remodeled_year - data.Year_built


# In[56]:


data.Remodeled_year.describe()


# In[285]:


#Обработаем признак обогрев
data['Heating'] = data['Heating'].apply(lambda x: x.lower())
data['Heating'] = data['Heating'].apply(lambda x: x.lstrip())
data['Heating'] = data['Heating'].apply(lambda x: x.replace('no data', 'forced').replace('natural ', '').replace(' heat', '').replace('none', 'forced').replace(' air', '').replace('(s)', '').replace('central gas', 'gas'))
data['Heating'] = data['Heating'].apply(lambda x: x.replace('/window unit', ''))


# In[286]:


#Аналогично обработаем признак типа охлаждения
data['Cooling'] = data['Cooling'].apply(lambda x: x.lower())
data['Cooling'] = data['Cooling'].apply(lambda x: x.lstrip())
data['Cooling'] = data['Cooling'].apply(lambda x: x.replace('a/c (electric)', 'electric').replace('a/c', 'electric').replace('no data', 'central').replace('has nan', 'nan').replace('none', 'central').replace('(s)', ''))
data['Cooling'] = data['Cooling'].apply(lambda x: x.replace(' hot air/furnace', '').replace('/window unit', '').replace('gas forced air nan', 'nan').replace('central heat', 'central').replace('window unit', 'wall').replace('central nan', 'nan'))
data['Cooling'] = data['Cooling'].apply(lambda x: x.replace('natural gas', 'gas').replace('gas nan', 'nan').replace('forced air nan', 'nan').replace('electric nan', 'nan').replace('fans', 'fan').replace('electric hot air', 'electric').replace('air conditioning-', '').replace(' air conditioning', ''))
data['Cooling'] = data['Cooling'].apply(lambda x: x.replace('wall heat', 'wall').replace('stream nan', 'nan').replace('central pump nan', 'nan').replace(' - heat', ''))
data['Cooling'] = data['Cooling'].apply(lambda x: x.replace('walls', 'wall').replace(' 1 unit', '').replace(' unit', '').replace('central electric (gas)', 'central electric, central gas').replace('gas heat', 'gas').replace('gas (hot air)', 'gas'))


# In[44]:


data['lotsize_sqft'].value_counts().head(5)


# In[287]:


#площадь помещения имеется в признаке sqft - в этом слишком много пропущенных значений.Удалим
column_to_del.append('lotsize_sqft')


# 'Price/sqft' зависит от искомой величины - стоимость жилья - удаляем признака

# In[288]:


column_to_del.append('Price/sqft')
# и удалим исходный признак
column_to_del.append('homeFacts')


# #### stories

# In[289]:


data.stories = data.stories.fillna('1.0')
data.stories = data.stories.apply(lambda x: x.lower())
data.stories = data.stories.apply(lambda x: x.replace('one', '1.0').replace('two', '2.0').replace('.00', '.0').replace(' story', '').replace(' or more', '').replace(' stories', ''))
data.stories = data.stories.apply(lambda x: x.replace('+', '').replace('three', '3').replace('lot', '3').replace('townhouse', '3'))
data.stories = data.stories.apply(lambda x: x.replace(' level', '').replace('ranch/1', '1').replace('/basement', '').replace(', site built', '').replace('condominium', '3').replace('multi/split', '3').replace('.000', ''))
data.stories = data.stories.apply(lambda x: x.replace('acreage', '3').replace('stories/levels', '1').replace('ranch', '1').replace('traditional', '1').replace(' basement', ''))
data.stories = data.stories.apply(lambda x: x if x.isnumeric() == True else x[0])


# In[290]:


data.stories = data.stories.astype(str).apply(lambda x: x if x.isnumeric() == True else '1')
data.stories = data.stories.astype(int)
#скорее всего нулевой этаж - это первый - основание - где расположена плоадь. Если понимается этажность продаваемого помещения
#data.stories = data.stories.astype(str).apply(lambda x: x.replace('0', '1'))


# Обработаем признак schools: создадим отдельный датафреймм - где разделим оценки, расстояние до , название школы и классы. Оценки и расстояния обработаем в дополнительных датафреймах. Нужные признаки запишет в исходный data

# #### schools

# In[291]:


data['schools'] = data['schools'].astype(str).apply(lambda x: None if x.strip() == '' else x)
data.schools = data.schools.astype(str).apply(lambda x: x.replace("'", '').replace('[', '').replace(']', '').replace('{', '').replace('}', ''))
data.schools = data.schools.apply(lambda x: x.replace('rating: ', '').replace('/10', '').replace(', data: Distance:', ';').replace('mi,', ',').replace(', Grades:', ';').replace(', name:', ';'))
#пропущенные значения заполним средней оценкой 3 и средним расстоянием 6
data.schools = data.schools.apply(lambda x: x.replace('; ; ; ', '0, 0, 0; 0, 0, 0; 0-0; No'))


# #расстояния 0.0 до школы - возможно стоит заменить на 0.1
# data.schools = data.schools.apply(lambda x: x.replace('0.0', '0.1'))

# In[292]:


data_schools = data['schools'].str.split(';', expand=True)
data_schools.columns = ['school_rating', 'school_distance', 'school_grades', 'school_name']
data_schools.head(6)


# In[66]:


data_schools.sample(4)


# In[293]:


data_schools.school_rating = data_schools.school_rating.apply(lambda x: x.replace('NR', '3').replace('None','3').replace('NA', '3'))


# In[294]:


def find_num(count_school):

    count = Counter(count_school) 
    return count[',']


# In[295]:


# количество школ
data['count_school'] = data_schools['school_rating'].apply(find_num) + 1


# In[296]:


# общий рейтинг школ
data['school_raiting'] = data_schools['school_rating'].apply(lambda x: x.replace(', ', ' + '))
data['school_raiting'] = data_schools['school_rating'].apply(lambda x: sum(float(y) for y in x.split(', ')))
# средний рейтинг школ
data['average_raiting_school'] = data['school_raiting'] / data.count_school


# Так как новый признак является функцией старых двух - то оставим количество школ и средний рейтинг

# In[297]:


data_schools_35 = data_schools['school_rating'].str.split(', ', expand=True)
data_schools_35.head(6)


# In[298]:


data_schools_35_max = data_schools_35.fillna(0)
data_schools_35_max = data_schools_35_max.astype(int)


# In[73]:


data_schools_35_max


# In[299]:



data['max_rating'] = data_schools_35_max.max(axis=1)


# In[300]:


data_schools_35_min = data_schools_35.fillna(11)
data_schools_35_min = data_schools_35_min.astype(int)


# In[301]:


data['min_rating'] = data_schools_35_min.min(axis=1)


# In[302]:


# Обработаем расстояние до школ
data['school_distance'] = data_schools['school_distance'].apply(lambda x: x.replace(', ', ' + '))
data['school_distance'] = data_schools['school_distance'].apply(lambda x: sum(float(y) for y in x.split(', ')))
# среднее расстояние до школ
data['average_distance_school'] = data['school_distance'] / data.count_school


# In[303]:


data_schools_35 = data_schools['school_distance'].str.split(', ', expand=True)
data_schools_35.head(6)


# In[304]:


data_schools_35_max = data_schools_35.fillna(0)
data_schools_35_max = data_schools_35_max.astype(float)
data['max_distance'] = data_schools_35_max.max(axis=1)


# In[305]:


data_schools_35_min = data_schools_35.fillna(1600)
data_schools_35_min = data_schools_35_min.astype(float)
data['min_distance'] = data_schools_35_min.min(axis=1)


# In[306]:


# Посмотрим на школы где расстояние до школы есть 0
data[data.min_distance == 0].sample(5)


# In[82]:


data.schools.iloc[31]


# In[307]:


data_schools_name35 = data_schools['school_name'].str.split(', ', expand=True)
data_schools_name35.head(6)


# In[84]:


data_schools_name35[0].value_counts().head(20)


# Обозначим отсутствие школ. Создадим признаки школ: Elementary, Middle, High, Private. И дополнительно 'Air Base', 'Hope', 'Media Arts', 'East', 'Liberty', 'Harns', 'Nyc Lab' и 'Myakka'

# In[308]:


#Обозначим отсутствие школ
data['No_school'] = data_schools_name35[0].apply(lambda x: len(x))
data['No_school'] = data.No_school.apply(lambda x: 1 if x == 3 else 0)


# In[309]:


data['Elementary'] = data.schools.apply(lambda x: 1 if 'Elementary' in x else 0)
data['Middle'] = data.schools.apply(lambda x: 1 if 'Middle' in x else 0)
data['High'] = data.schools.apply(lambda x: 1 if 'High' in x else 0)
data['Private'] = data.schools.apply(lambda x: 1 if 'Private' in x else 0)


# In[310]:


data['Air Base'] = data.schools.apply(lambda x: 1 if 'Air Base' in x else 0)
data['Hope'] = data.schools.apply(lambda x: 1 if 'Hope' in x else 0)
data['Media Arts'] = data.schools.apply(lambda x: 1 if 'Media Arts' in x else 0)
data['East'] = data.schools.apply(lambda x: 1 if 'East' in x else 0)
data['Liberty'] = data.schools.apply(lambda x: 1 if 'Liberty' in x else 0)
data['Harns'] = data.schools.apply(lambda x: 1 if 'Harns' in x else 0)
data['Nyc Lab'] = data.schools.apply(lambda x: 1 if 'Nyc Lab' in x else 0)
data['Myakka'] = data.schools.apply(lambda x: 1 if 'Myakka' in x else 0)


# In[311]:


#из признака schools все взяли - можно удалить
column_to_del.append('school_distance')
column_to_del.append('school_raiting')
column_to_del.append('schools')


# #### Parking& Heating & Cooling

# In[312]:


data.Parking = data.Parking.fillna('None')
data['Parking'] = data['Parking'].apply(lambda x: x.lower())
data['Parking'] = data['Parking'].apply(lambda x: x.lstrip())


# In[313]:


data.Parking = data.Parking.apply(lambda x: x.replace('none', 'no').replace('has nan', 'no').replace('nan', 'no').replace(' data', '').replace('1 space', '1').replace(' - side', ''))


# In[314]:


data['no_parking'] = data.Parking.apply(lambda x: 0 if x == 'no' else(0 if x == '—' else 1))
data['attached garage'] = data.Parking.apply(lambda x: 1 if x == 'attached garage' else 0)


# In[315]:


data['detached garage'] = data.Parking.apply(lambda x: 1 if x == 'detached garage' else 0)
data['3_garage'] = data.Parking.apply(lambda x: 1 if x == '3' else (1 if x == '4' else 0))
data['5_garage'] = data.Parking.apply(lambda x: 1 if x == '5' else (1 if x == '6' else 0))


# In[316]:


column_to_del.append('Parking')
column_to_del.append('Cooling')
column_to_del.append('Heating')
column_to_del.append('Remodeled_year')


# In[94]:


column_to_del


# In[317]:


# удалим ненужные признаки
data = data.drop(column_to_del, axis=1)


# In[96]:


data.info()


# In[97]:


data.sample(5).T


# ## Полученные данные можно разбить на:

# In[318]:


#
numerical_columns = ['baths', 'sqft', 'beds', 'stories', 'mls-id', 'target', 'count_school', 'average_raiting_school', 'max_rating', 'min_rating',
       'average_distance_school', 'max_distance',  'min_distance']
binary_columns = ['Pool', 'No_school', 'Elementary', 'Middle', 'High', 'Private',
       'Air Base', 'Hope', 'Media Arts', 'East', 'Liberty', 'Harns', 'Nyc Lab',
       'Myakka', 'no_parking', 'attached garage', 'detached garage', '3_garage', '5_garage']
categorial_columns = ['status', 'propertyType', 'street', 'fireplace', 'city', 'zipcode', 'state', 'homeFacts', 'schools']

data_columns = ['Year_built', 'Remodeled_year']


# 
# ### Предоработка

# In[99]:


#данные стоимости жилья с $/mo относятся к аренде жилья
data[data.target.apply(lambda x: 'mo' in x)].sample(3)


# In[319]:


#Приведем к числовому формату
data.target = data.target.astype(str).apply(lambda x: x.replace('$', '').replace('+', '').replace(',', '').replace('/mo', ''))
data.target = data.target.str.replace(' - \d{2,5}', '')
data = data.loc[data['target'] != 'nan']
data.target = data.target.apply(lambda x: int(x))


# In[101]:


# посмотрим на распределение целевой переменной
gistogramma('target')


# In[320]:


# возьмем логарифм и еще раз построим распределение
data['log_price'] = np.log(data.target)
gistogramma('log_price')


# In[103]:


# Посмотрим на наличие выбросов
outliers('log_price')


# In[104]:


#удалим данные, где логарифм цены меньше 6. сомнительная стоимость домов
data.target[data.log_price < 6].value_counts()


# In[321]:


data.drop(data.loc[np.log(data.target) < 6].index, inplace=True)


# В модель введем логарифм цены. А после предсказания вернем обратно.

# ####  Посмотрим как признаки влияют на стоимость жилья

# In[106]:


data.describe()


# In[107]:


#Наличие бассейна
graf_cat('Pool')


# Среднее значение стоимости цены на жилье выше при наличии бассейна.

# In[108]:


#уникальных городов
data.city.nunique()


# In[109]:


#Посмотрим на количество объявления по городам в %
data.city.value_counts().head(25)*100/len(data)


# Ограничимся 1% данных и оставим 22 значения

# In[322]:


# Ограничим категориалный признак город. Городов много - обрезав на 22 значений можно многие зависимости потерять. Попробуем взять 50 городов
data['city2'] = data.city
cities = data.city2.value_counts()[:22]
# Выделим 5 основных, остальные заменим общим типом "другой"
data['city2'] = data['city2'].apply(lambda x: x if x in cities else 'другой')


# In[111]:


#сгруппируем и посмотрим как меняется стоимость жилья от города
data.groupby('city2')['target'].describe().head(10)


# In[112]:


# Визуализируем
var = 'city2'
data_m = pd.concat([data['log_price'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="log_price", data=data_m)
fig.axis();
plt.xticks(rotation=90);


# In[113]:


#Обработаем категориальный признак улицы: сколько уникальный значений
data.street.nunique()


# In[114]:


data.street.value_counts()


# названия улиц даны с адресом, поэтому они не повторяются. Можно попробовать выделить тип улиц отдельно.

# In[323]:


#Введем новые бинарные признаки из наиболее частых: 'drive', 'road', 'lane', 'trail', 'ave'
data['drive'] = data.street.apply(lambda x: 1 if 'drive' in x else (1 if ' dr' in x else 0))
data['road'] = data.street.apply(lambda x: 1 if 'road' in x else (1 if ' rd' in x else 0))
data['lane'] = data.street.apply(lambda x: 1 if 'lane' in x else (1 if ' ln' in x else 0))
data['trail'] = data.street.apply(lambda x: 1 if 'trail' in x else (1 if ' trl' in x else 0))


# In[324]:


data['hwy'] = data.street.apply(lambda x: 1 if 'hwy' in x else 0)
data['ave'] = data.street.apply(lambda x: 1 if 'avenue' in x else (1 if ' ave' in x else 0))
data['street'] = data.street.apply(lambda x: 1 if 'street' in x else (1 if ' st' in x else 0))


# In[325]:


cols = ['drive', 'road', 'lane', 'trail', 'hwy', 'ave', 'street']


# In[118]:


for i in cols:
        print(i, data[i].value_counts()*100/len(data))


# hwy содержится только в 0.27 % данных. И значения 1 для признака lane только в 1.1 %. Можно удалиь эти признаки 

# In[119]:


for i in cols:
    graf_cat(i)


# drive, ave, street можно удалить сразу: разницы нет

# In[120]:


# сколько уникальных значений типа жилья
data['propertyType'].nunique()


# In[121]:


data['propertyType'].value_counts().head(30)*100/len(data)


# Ограничимся 1% и оставим 8 значений

# In[326]:


# Выделим 8 основных, остальные заменим общим типом "other"
data['propertyType2'] = data.propertyType
propertyType = data.propertyType2.value_counts()[:8]
data['propertyType2'] = data['propertyType2'].apply(lambda x: x if x in propertyType else 'other')


# In[123]:


# Визуализируем
var = 'propertyType2'
data_m = pd.concat([data['log_price'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="log_price", data=data_m)
fig.axis();
plt.xticks(rotation=90);


# In[124]:


# Рассмотрим этажность продаваемого жилья
data.stories.describe()


# In[125]:


# Посмотрим на наличие выбросов
outliers('stories')


# In[126]:


data.groupby('stories')['log_price'].mean().plot()


# In[127]:


sns.jointplot(x=data['stories'], y=data['target'], kind='reg')


# In[128]:


data[data.target > 75000000]


# In[327]:


#Удалим верхние выбросы при этажности менее 5
data = data.drop(data[data['target']>75000000].index).reset_index(drop=True)


# In[130]:


#Попробуем добавить логарифм
#data['stories_log'] = np.log(data.stories + 1)


# Чем выше этажность, тем выше цена. попробуем ограничить этажность 50 этажами

# In[131]:


outliers('beds')


# In[132]:


outliers('baths')


# In[133]:


data.beds.describe()


# In[134]:


data.baths.describe()


# In[135]:


sns.jointplot(x=data['beds'], y=data['target'], kind='reg')


# In[328]:


#Удалим выбросы - где спален больше 75, а стоимость меньше 10^7
data = data.drop(data[(data['beds']>50) 
                         & (data['target']<10000000)].index).reset_index(drop=True)


# In[137]:


data.groupby('beds')['log_price'].mean().plot()


# In[138]:


sns.jointplot(x=data['baths'], y=data['target'], kind='reg')


# In[139]:


# look at the beds & baths outlier correlation
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='beds', y='baths')
plt.xlabel(
    'Количество спален')
plt.ylabel(
    'Количество ванных комнат')
plt.title('Корреляция спальни-ванные комнаты', fontsize=12)
plt.show()


# Чем больше ванных, тем дороже жилье. Можно попробовать ограничить количество.  Или взять логарифм

# In[140]:


#Рассмотрим признак статус
data.status.nunique()


# In[141]:


data.status.value_counts().head(15)*100/len(data)


# In[329]:


# Выделим 5 основных, остальные заменим общим типом "другой"
data['status2'] = data.status
status = data.status2.value_counts()[:5]
data['status2'] = data['status2'].apply(lambda x: x if x in status else 'другой')


# In[143]:


# Визуализируем
var = 'status2'
data_m = pd.concat([data['log_price'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="log_price", data=data_m)
fig.axis();
plt.xticks(rotation=90);


# In[144]:


# Уникальных значений
data.fireplace.nunique()


# In[145]:


#Посмотрим на наиболее популярные значения в признаке
data.fireplace.value_counts().head(20)


# Введем следующие бинарные признаки:
# 
# по наличию:
# -no_fire (yes будет автоматически остальные)
# 1 не будем выделять - его можно считать yes без указания деталей
# -2_fire
# -fire_more3: если указано количество больше 3
# 
# По типу (встречающееся слово в уточнении:
# -gas
# -wood
# -air
# -electric
# 
# По расположению:
# -ceiling
# -living
# -family
# -great
# 
# Для оптимизации кода обработку этого признака можно сократить - так как мы ищем просто совпадение слов в описании признака. 

# In[330]:


data['no_fire'] = data.fireplace.apply(lambda x: 0 if 'no' in x else 1)
data['2_fire'] = data.fireplace.apply(lambda x: 1 if x == '2' else 0)
num = ['5', '3', '4', '6']
data['fire_more3'] = data.fireplace.apply(lambda x: 1 if x in num else 0)


# In[331]:


data['gas'] = data.fireplace.apply(lambda x: 1 if 'gas' in x else 0)
data['wood'] = data.fireplace.apply(lambda x: 1 if 'wood' in x else 0)
data['air'] = data.fireplace.apply(lambda x: 1 if 'air' in x else 0)
data['electric'] = data.fireplace.apply(lambda x: 1 if 'electric' in x else 0)


# In[332]:


data['ceiling'] = data.fireplace.apply(lambda x: 1 if 'ceiling' in x else 0)
data['living'] = data.fireplace.apply(lambda x: 1 if 'living' in x else 0)
data['family'] = data.fireplace.apply(lambda x: 1 if 'family' in x else 0)
data['great'] = data.fireplace.apply(lambda x: 1 if 'great' in x else 0)


# Расположение - Возможно living, great и family  описывают большую гостинную. и можно попробовать добавить признак общий, где укаазано хоть одно из трех слов

# In[333]:


data['fireplace'] = data['fireplace'].apply(lambda x: x.replace('great', 'family').replace('living', 'family'))
data['great_family'] = data.fireplace.apply(lambda x: 1 if 'family' in x else 0)
#а признаки great, living можно удалять


# In[334]:


#Итак, мы ввели следующие бинарные признаки:
new_binary = ['no_fire', '2_fire', 'fire_more3', 'gas', 'wood', 'air',
       'electric', 'ceiling', 'living', 'family', 'great', 'great_family']
        
#Исходный 'fireplace' можно удалять


# In[151]:


for i in new_binary:
    print (i, data[i].value_counts()*100/len(data))


# air встречается только в 0.07 %. electric в 0.05%. ceiling 0.35%.great - 0.23% famiy, living - 0.47, 0.43%. Удалим их

# In[152]:


for i in new_binary:
    graf_cat(i)


# Можно отметить, что количество указанных каминов (обогревателей) увеличивает стоимость жилья. что и логично - в объявлении указать максимум достоинств.
# Источники газ и электричество тоже дает цену выше, тогда как остальные источники тепла выглядят на одном уровне.
# Удалим сразу wood, air.
# 

# In[335]:


data['state2'] = data.state
state = data.state2.value_counts()[:16]
# Выделим 14 основных, остальные заменим общим типом "другой"
data['state2'] = data['state2'].apply(lambda x: x if x in state else 'другой')


# In[154]:


data.state2.nunique()


# In[155]:


# Визуализируем
var = 'state2'
data_m = pd.concat([data['log_price'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="log_price", data=data_m)
fig.axis();
plt.xticks(rotation=90);


# In[156]:


# Посмотрим на наличие выбросов
outliers('sqft')


# In[157]:


data[data.sqft > 4440]


# In[158]:


data.sqft.describe()


# In[336]:


data['log_sqft'] = np.log(data.sqft+1)


# In[161]:


gistogramma('log_sqft')


# In[162]:


# look at the beds & baths outlier correlation
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='beds', y='log_sqft')
plt.xlabel(
    'Количество спален')
plt.ylabel(
    'Площадь')
plt.title('Корреляция спальни-площадь', fontsize=12)
plt.show()


# In[337]:


#Удалим выбросы 
data = data.drop(data[data['log_sqft'] > 17].index).reset_index(drop=True)


# In[338]:


data.loc[(data.log_sqft ==0), 'log_sqft'] = 8.7


# In[339]:


cols = ['status','sqft', 'propertyType', 'fireplace', 'city', 'state', 'zipcode', 'mls-id', 'living', 'great', 'family', 'drive', 'ave', 
        'hwy', 'street', 'wood', 'air', 'electric', 'ceiling', 'lane']


# In[340]:


data = data.drop(cols, axis=1)


# In[341]:


data.loc[(data.max_distance > 130), 'max_distance'] = 130
data.loc[(data.average_distance_school > 130), 'min_distance'] = 130
data.loc[(data.average_distance_school > 130), 'average_distance_school'] = 130


# In[102]:


numeric_columns = ['baths', 'log_sqft', 'beds', 'stories', 
       'Year_built', 'count_school',
       'average_raiting_school', 'max_rating', 'min_rating', 'average_distance_school', 'max_distance',
       'min_distance', 'log_price']


# In[173]:


data[numeric_columns].corr()


# In[174]:


# Top 10 Heatmap
k = 10 #number of variables for heatmap
corrmat = data.corr()
cols = corrmat.nlargest(k, 'log_price')['log_price'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[175]:


most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr


# In[176]:


# calculate Pearson's F-value for numerical columns and plot the result
pearson = pd.Series(f_regression(data[numeric_columns], data['target'])[0], index = numeric_columns)
pearson.sort_values(inplace = True)
plt.figure(figsize=(10, 10))
pearson.plot(kind='barh')
plt.ylabel('Numeric Columns')
plt.xlabel('Pearson F-value')
plt.title("Pearson F-value between numeric columns and target")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[177]:


bynary_columns = ['Pool', 'No_school', 'Elementary', 'Middle', 'High',
       'Private', 'Air Base', 'Hope', 'Media Arts', 'East', 'Liberty', 'Harns',
       'Nyc Lab', 'Myakka', 'no_parking', 'attached garage', 'detached garage', 
       '3_garage', '5_garage', 'city2',  'road', 'trail',
       'no_fire', '2_fire', 'fire_more3', 'gas']


# In[178]:


for col in bynary_columns:
    get_stat_dif(col)


# In[343]:


categorial_columns = ['status2', 'propertyType2', 'city2', 'state2']
for col in categorial_columns:
    get_stat_dif(col)


# In[344]:


# Удалим target
data.drop(['target'], axis=1, inplace=True)


# In[180]:


data.info()


# ### ML

# In[345]:


def print_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')


# In[367]:


train_df = data.copy(deep=True)


# In[368]:



#Преобразуем не числовые данные
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))


# #### Baseline

# In[369]:


train_X = train_df.drop(["log_price"], axis=1)
train_Y = train_df["log_price"]


# In[217]:


scaler = StandardScaler()
scaler.fit(train_X)


# In[218]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_X, train_Y, test_size=0.4)


# In[223]:


print('LinearRegression')
model = LinearRegression()
model.fit(Xtrain, Ytrain)
y_pred = model.predict(Xtest)
# метрики
print_metrics(Ytest, y_pred)


# In[224]:


Xtrain


# In[220]:


print('RandomForestRegression')
model2 = RandomForestRegressor(max_depth=10, random_state=random_seed)
model2.fit(Xtrain, Ytrain)
y_pred = model2.predict(Xtest)
# метрики
print_metrics(Ytest, y_pred)


# In[221]:


#  Выведем наиболее значимые признаки
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model2.feature_importances_, index=train_X.columns)
feat_importances.nlargest(70).plot(kind='barh')


# In[351]:


cols = ['great_family', 'gas', 'Hope', '3_garage', 'attached garage', 'No_school', 'Private', 'detached garage', 
        'road', 'trail', 'Air Base', 'Middle', 'Myakka']

train_df = train_df.drop(cols, axis=1)


# In[352]:





# In[353]:



#Преобразуем не числовые данные
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))


# In[354]:


train_X = train_df.drop(["log_price"], axis=1)
train_Y = train_df["log_price"]


# In[355]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_X, train_Y, test_size=0.4)


# In[356]:


#Другая нормализация данных
scaler = MinMaxScaler()
scaler.fit(train_X)


# In[358]:


print('LinearRegression')
model = LinearRegression()
model.fit(Xtrain, Ytrain)
y_pred = model.predict(Xtest)
# метрики
print_metrics(Ytest, y_pred)


# In[359]:


print('RandomForestRegression')
model = RandomForestRegressor(max_depth=10, random_state=random_seed)
model.fit(Xtrain, Ytrain)
y_pred = model.predict(Xtest)
# метрики
print_metrics(Ytest, y_pred)


# In[360]:


#  Выведем наиболее значимые признаки
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=train_X.columns)
feat_importances.nlargest(70).plot(kind='barh')


# #### Линейная регрессия с использованием методов оптимизации

# In[ ]:


train_df = data.copy(deep=True)

#Преобразуем не числовые данные
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_X = train_df.drop(["log_price"], axis=1)
train_Y = train_df["log_price"]


# In[361]:


# функция вычисления градиента функции MSE

def calc_mse_gradient(X, y, theta):
    n = X.shape[0]
    grad = 1. / n * X.transpose().dot(X.dot(theta) - y)
    
    return grad

# функцяю, осуществляющая градиентный шаг: параметр величины шага alpha - learning rate)

def gradient_step(theta, theta_grad, alpha):
    return theta - alpha * theta_grad

# функция цикла градиентного спуска с доп. параметрами начального вектора theta и числа итераций

def optimize(X, y, grad_func, start_theta, alpha, n_iters):
    theta = start_theta.copy()
    
    for i in range(n_iters):
        theta_grad = grad_func(X, y, theta)
        theta = gradient_step(theta, theta_grad, alpha)
    
    return theta


# In[375]:


# Разобьем таблицу данных и добавим фиктивный столбец единиц (bias линейной модели)
X, y = train_X, train_Y
X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
m = X.shape[1]


# In[376]:


# Оптимизируем параметр линейной регрессии theta на всех данных
theta = optimize(X, y, calc_mse_gradient, np.ones(m), 0.001, 100)


# In[377]:


# Нормализуем даннные с помощью стандартной нормализации
X, y = train_X, train_Y
X = (X - X.mean(axis=0)) / X.std(axis=0)
# Добавим фиктивный столбец единиц (bias линейной модели)
X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])


# In[373]:


# Оптимизируем theta
#theta = optimize(X, y, calc_mse_gradient, np.ones(m), 0.01, 1000)


# In[378]:


# Разбить выборку на train/valid, оптимизировать theta,
# сделать предсказания и посчитать ошибки MSE и RMSE

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)
theta = optimize(X_train, y_train, calc_mse_gradient, np.ones(m), 0.001, 5000)
y_pred = X_valid.dot(theta)

print_metrics(y_valid, y_pred)


# #### RandomForestRegressor

# In[379]:


train_df = data.copy(deep=True)
#Преобразуем не числовые данные
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))


cols = ['great_family', 'gas', 'Hope', '3_garage', 'attached garage', 'No_school', 'Private', 'detached garage', 
        'road', 'trail', 'Air Base', 'Middle', 'Myakka']

train_df = train_df.drop(cols, axis=1)

train_X = train_df.drop(["log_price"], axis=1)
train_Y = train_df["log_price"]


# In[380]:


train_df = pd.get_dummies(train_df, columns=[ 'status2','city2','propertyType2', 'state2'], dummy_na=True)  

Xtrain,Xtest,Ytrain,Ytest = train_test_split(train_X, train_Y, test_size=0.4)


# In[382]:


#Организуем обучение в цикле для Random Forest
gr=np.arange(1,15,1)
facc=[]
acc=0
for i in gr:
    scc=0
    model = RandomForestRegressor(n_estimators=i, max_depth = 15)
    model.fit(Xtrain,Ytrain)
    y_predicted = model.predict(Xtest)
    scc=model.score(Xtest,Ytest)
    facc.append(scc)
    if scc > acc:
        acc=scc
        mf=i
        print("Random Forest: , n_estimators", i, " Точность", scc)


# In[383]:


print_metrics(y_predicted, Ytest)


# In[384]:


plt.plot(gr,facc)
plt.title("Точность модели в зависимости от числа деревьев")
plt.xlabel("Число деревьев")
plt.ylabel("Точность алгоритма")
print("best n_estimators", mf, "Наилучшая точность", acc )
scc=mean_squared_log_error(Ytest, y_predicted)
print("Error RMSLE", scc)


# In[123]:


rf = RandomForestRegressor(random_state = 42)
# Look at parameters used by our current forest
print('Параметры по умолчанию:\n')
pprint(rf.get_params())


# In[124]:



n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[125]:


rf = RandomForestRegressor(random_state=42)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10, 
                               cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(Xtrain, Ytrain)


# In[126]:


rf_random.best_params_


# {'n_estimators': 400,
#  'min_samples_split': 10,
#  'min_samples_leaf': 1,
#  'max_features': 'sqrt',
#  'max_depth': 60,
#  'bootstrap': False}

# In[391]:


y_pred = rf_random.predict(Xtest)


# In[392]:


print_metrics(y_pred, Ytest)


# GradientBoostingRegressor c заданными параметрами
# 

# In[390]:


train_df = data.copy(deep=True)
#Преобразуем не числовые данные
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))


cols = ['great_family', 'gas', 'Hope', '3_garage', 'attached garage', 'No_school', 'Private', 'detached garage', 
        'road', 'trail', 'Air Base', 'Middle', 'Myakka']

train_df = train_df.drop(cols, axis=1)

train_X = train_df.drop(["log_price"], axis=1)
train_Y = train_df["log_price"]


Xtrain,Xtest,Ytrain,Ytest = train_test_split(train_X, train_Y, test_size=0.4)


# In[188]:





# In[387]:


model1 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[388]:


model1.fit(Xtrain, Ytrain)


# In[389]:


y_pred = model1.predict(Xtest)
print_metrics(y_pred, Ytest)


# In[202]:


bagging_gbr = BaggingRegressor(GradientBoostingRegressor(random_state=random_seed                                                         , n_estimators=300                                                        , min_samples_split=5                                                        , min_samples_leaf=17                                                        , max_features='auto'                                                        , max_depth=8                                                        , learning_rate=0.13428571428571429)                                 , random_state=random_seed                                 , n_jobs=-1)


# In[ ]:


bagging_gbr.fit(Xtrain, Ytrain)

y_pred = bagging_gbr.predict(Xtest)
# # look at metrics
print_metrics(Ytest, y_pred)


# Результаты:
# 
#  нормализация: StandardScaler()
#         LinearRegression MSE = 0.84, RMSE = 0.92 
#         RandomForestRegressor MSE = 2.28, RMSE = 1.51 
# 
# нормализация MinMaxScaler()
#         LinearRegression MSE = 0.86, RMSE = 0.93
#         RandomForestRegression MSE = 0.52, RMSE = 0.72
# 
#         Линейная регрессия с использованием методов оптимизации MSE = 0.90, RMSE = 0.95
#         RandomForestRegressor c get_dummies MSE = 0.40, RMSE = 0.63
# 
#        
#         GradientBoostingRegressor MSE = 0.38, RMSE = 0.62
#  
#         Randomize(RandomForestRegressor) MSE = 0.17, RMSE = 0.41
# 
