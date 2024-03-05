import csv
import pickle
import dill
from datetime import datetime
import pandas as pd
from geopy.geocoders import Nominatim

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingClassifier





df = pd.read_csv('./data/sessions_hits_merged.csv')  #see DataPreparation.ipynb for more details
train, test = train_test_split(df, test_size=0.2, stratify=df['target'])

#load prepared geo data

with open('./data/capitals-list.csv', encoding='UTF-8') as f:
    capitals_dict = {line[0].strip():line[1].strip() for line in csv.reader(f)}

with open('./data/countries of the world.csv', encoding='UTF-8') as f:
    country_dict = {line[0].strip():line[1].strip() for line in csv.reader(f)}

cities = pd.read_csv('./data/cities_mod.csv').drop_duplicates('city')

def filter_cols(data):
    columns_to_drop = [
        'utm_keyword',
        'device_model',
        'client_id',
        'session_id',
        'geo_city',
        'geo_country',
        'visit_time',
        'visit_date',
        'device_os',
        'geo_region',
        'utm_adcontent',
        'utm_campaign',
        'utm_source',
        'utm_medium']
    data = data.drop(columns_to_drop, axis=1)
    return data


def get_coordinates(data):
    import pandas as pd
    import csv
    with open('./data/capitals-list.csv', encoding='UTF-8') as f:
        capitals_dict = {line[0].strip(): line[1].strip() for line in csv.reader(f)}
    cities = pd.read_csv('./data/cities_mod.csv').drop_duplicates('city')
    data['geo_country'] = data['geo_country'].replace('(not set)',
                                                      'Russia')  # take Russia for the default country, fill where absent

    data['geo_country'] = data['geo_country'].fillna('Russia')

    # fill missing and not valid values with capitals
    data.loc[(data['geo_city'] == '(not set)') | (
        data['geo_city'].str.contains(r'\d{4}')), 'geo_city'] = data.geo_country.apply(
        lambda x: capitals_dict[x] if x in capitals_dict.keys() else x)

    # merge the prepared list city coordinates
    data = data.merge(cities, left_on='geo_city', right_on='city', how='left', copy=False)

    # add missing values via geolocator
    lostcoord = list(data[data.lat.isna()].geo_city.value_counts().to_dict().keys())
    geolocator = Nominatim(user_agent="Geolocation", timeout=10)

    coord_dict = dict()
    for i in lostcoord:
        location = geolocator.geocode(i)
        if location:
            coord_dict[i] = (location.latitude, location.longitude)

    data.loc[data.lat.isna(), 'lat'] = data.geo_city.apply(
        lambda x: coord_dict[x][0] if x in coord_dict.keys() else None)

    data.loc[data.lng.isna(), 'lng'] = data.geo_city.apply(
        lambda x: coord_dict[x][1] if x in coord_dict.keys() else None)

    data = data.drop('city', axis=1)

    return data


def get_region(data): #add geo_region feature in place of geo_country
    import csv
    with open('./data/countries of the world.csv', encoding='UTF-8') as f:
        country_dict = {line[0].strip(): line[1].strip() for line in csv.reader(f)}
    country_dict['Russia'] = 'Russia'
    country_dict['Belarus'] = 'Belarus'
    country_dict['Czechia'] = 'EASTERN EUROPE'
    country_dict['Montenegro'] = 'EASTERN EUROPE'
    country_dict['Kosovo'] = 'EASTERN EUROPE'
    country_dict['North Macedonia'] = 'EASTERN EUROPE'
    country_dict['South Korea'] = 'ASIA (EX. NEAR EAST)'
    country_dict['North Korea'] = 'ASIA (EX. NEAR EAST)'
    country_dict['North Korea'] = 'ASIA (EX. NEAR EAST)'
    country_dict['Myanmar (Burma)'] = 'ASIA (EX. NEAR EAST)'
    country_dict['Wallis & Futuna'] = 'OCEANIA'

    data['geo_region'] = data['geo_country'].apply(lambda x: country_dict[x] if x in country_dict else x)
    return data

def process_date_time(data): #month and year not relevant (one year and not all months present )
    import pandas as pd
    data.visit_date = pd.to_datetime(data.visit_date)
    data['visit_dayofweek'] = data.visit_date.apply(lambda x: x.weekday())
    data['visit_day'] = data.visit_date.apply(lambda x: x.day)

    data.visit_time = pd.to_datetime(data.visit_time)
    data['visit_hour'] = data.visit_time.apply(lambda x: x.hour)

    return data

def add_categories(data): #group, binarize,  drop rare categories
    #здесь просится цикл, конечно,  может, позже
    brows = ['Chrome', 'Safari', 'YaBrowser', 'Safari (in-app)', 'Android Webview', 'Samsung Internet', 'Opera', 'Edge',
             'Firefox']
    data['device_browser'] = data['device_browser'].apply(lambda x: 'other' if x not in brows else x)
    data['device_browser'] = data['device_browser'].replace('Safari (in-app)', 'Safari')

    data.device_brand = data.device_brand.fillna('unknown')
    data.device_brand = data.device_brand.replace('(not set)', 'unknown')

    brands = ['Apple', 'unknown', 'Samsung', 'Xiaomi', 'Huawei', 'Realme']
    data.device_brand = data.device_brand.apply(lambda x: 'other' if x not in brands else x)

    organic = ['organic', 'referral', '(none)', '(not set)']
    data['paid_traffic'] = data.utm_medium.apply(lambda x: 0 if x in organic else 1)

    sn = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
          'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    data['social_net'] = data.utm_source.apply(lambda x: 1 if x in sn else 0)

    some_relevant_utmcampaign = ['LTuZkdKfxRGVceoWkVyg', 'LEoPHuyFvzoNfnzGgfcd', 'FTjNLDyTrXaWYgZymFkV']
    data['top_3_campaigns'] = data.utm_campaign.apply(lambda x: 1 if x in some_relevant_utmcampaign else 0)

    return data


def add_freqs(data):  # we need more numeric features

    cols = ['utm_adcontent', 'utm_source', 'utm_medium', 'geo_region', 'geo_city', 'visit_dayofweek', 'visit_day',
            'visit_hour', 'device_browser', 'device_brand', 'device_screen_resolution']
    newcols = ['utm_adcontent_freq', 'utm_source_freq', 'utm_medium_freq', 'geo_region_freq', 'geo_city_freq',
               'visit_dayofweek_freq', 'visit_day_freq', 'visit_hour_freq', 'device_browser_freq', 'device_brand_freq',
               'device_screen_resolution_freq']

    for c in range(len(cols)):
        freq = data[cols[c]].value_counts(dropna=False).to_dict()
        for i in freq:
            freq[i] = round(freq[i] / len(data.index), 4)
        data[newcols[c]] = data[cols[c]].apply(lambda x: freq[x])

    return data


def multiply_res(data):
    data.device_screen_resolution = data.device_screen_resolution.apply(lambda x: x.split('x')).apply(lambda x: int(x[0]) * int(x[1]))

    return data


def treat_outliers(data):
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = ((q25 - 1.5 * iqr), (q75 + 1.5 * iqr))
        return boundaries

    cols = ['device_screen_resolution', 'visit_number']
    for col in cols:
        boundaries = calculate_outliers(data[col])
        data.loc[data[col] < boundaries[0], col] = boundaries[0]
        data.loc[data[col] > boundaries[1], col] = boundaries[1]

    return data

constructor = Pipeline(steps=[
    ('get coordinates', FunctionTransformer(get_coordinates)),
    ('get region', FunctionTransformer(get_region)),
    ('process date', FunctionTransformer(process_date_time)),
    ('add frequencies', FunctionTransformer(add_freqs)),
    ('add categories', FunctionTransformer(add_categories)),
    ('screen resolution', FunctionTransformer(multiply_res))
     ])


cleaner = Pipeline(steps=[
    ('outliers', FunctionTransformer(treat_outliers)),
    ('filter cols', FunctionTransformer(filter_cols))
])

numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False))])

encoder = ColumnTransformer(remainder='passthrough', verbose_feature_names_out=False, transformers=[
    ('num', numerical_transformer, ['visit_number', 'device_screen_resolution', 'lat', 'lng', 'visit_dayofweek',
                                    'visit_day', 'visit_hour', 'utm_adcontent_freq', 'utm_source_freq',
                                     'utm_medium_freq', 'geo_region_freq', 'geo_city_freq', 'visit_dayofweek_freq',
                                     'visit_day_freq', 'visit_hour_freq', 'device_browser_freq', 'device_brand_freq',
                                     'device_screen_resolution_freq' ]),
    ('cat', categorical_transformer,  ['device_category', 'device_brand', 'device_browser']),
]).set_output(transform='pandas')

#let's create some more numeric features based on IsolationForest classifier.

interrim_pipe = Pipeline([
('constructor', constructor),
('encoder', encoder)])

train_iso = interrim_pipe.fit_transform(train)


def train_iso_features(data):
    train_normal = data[data['target'] == 0]
    features = ['utm_adcontent_freq', 'utm_source_freq', 'paid_traffic', 'utm_medium_freq', 'top_3_campaigns',
                'device_screen_resolution_freq', 'device_browser_freq', 'lat', 'lng', 'social_net', 'geo_city_freq',
                'visit_hour_freq', 'visit_day', 'device_screen_resolution']
    newcolslist = []
    for feat in features:
        isf = IsolationForest().fit(train_normal[[feat]])
        filename = './models/' + str(feat) + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(isf, f)


train_iso_features(train_iso)

def add_iso_features(data):
    features=['utm_adcontent_freq', 'utm_source_freq', 'paid_traffic', 'utm_medium_freq', 'top_3_campaigns',
              'device_screen_resolution_freq', 'device_browser_freq', 'lat','lng',  'social_net', 'geo_city_freq',
              'visit_hour_freq', 'visit_day', 'device_screen_resolution']

    for feat in features:
        newcol = str(feat)+'isf'
        filename = './models/'+ str(feat) + '.pkl'
        with open(filename, 'rb') as f:
            isf = pickle.load(f)
        data[newcol] =  isf.score_samples(data[[feat]])
    return data

preprocessor = Pipeline([
('constructor', constructor),
('encoder', encoder),
( 'iso', FunctionTransformer(add_iso_features)),
 ('cleaner', cleaner)
])

x_train = train.drop('target',  axis=1)
x_test = test.drop('target',  axis=1)

y_train = train['target']
y_test = test['target']

model = GradientBoostingClassifier(n_estimators=200)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)

])

model = pipe.fit(x_train, y_train)

probs = model.predict_proba(x_test)
probs = probs[:, 1]
# рассчитываем ROC AUC
clf_auc_test = roc_auc_score(y_test, probs)
print('значение метрики ROC AUC на тестовой выборке %.3f' % (clf_auc_test))

probs = model.predict_proba(x_train)
probs = probs[:, 1]
# рассчитываем ROC AUC
clf_auc = roc_auc_score(y_train, probs)
print('значение метрики ROC AUC на обучающей выборке %.3f' % (clf_auc))



train_iso = interrim_pipe.fit_transform(df)
train_iso_features(train_iso)


x = df.drop('target', axis=1)
y = df['target']


pipe.fit(x, y)

object_to_dump = {
    'model': pipe,
    'metadata': {
        'author': 'O.K.',
        'version': 1,
        'date': datetime.now(),
        'type': type(pipe.named_steps["classifier"]).__name__,
        'test ROC AUC score ': clf_auc_test

    }
}

filename = './models/pipe.pkl'
with open(filename, 'wb') as file:
    dill.dump(object_to_dump, file, recurse=True)