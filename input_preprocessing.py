import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set_style('whitegrid')

SMALL_SIZE = 10
MEDIUM_SIZE = 12

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.dpi']=150

data = pd.read_csv("/home/vignesh/PycharmProjects/SloanDigitalSkySurvey/"
                           "astronomical-observation-classification-neural-network/"
                           "Skyserver_SQL2_27_2018 6_51_39 PM.csv", skiprows=1)

# Take Action on Data - Data Filtering
data.drop(['objid', 'specobjid'], axis=1, inplace=True)

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
ax = sns.distplot(data[data['class']=='STAR'].camcol, bins = 30, ax = axes[0], kde = False)
ax.set_title('Star')
ax = sns.distplot(data[data['class']=='GALAXY'].camcol, bins = 30, ax = axes[1], kde = False)
ax.set_title('Galaxy')
ax = sns.distplot(data[data['class']=='QSO'].camcol, bins = 30, ax = axes[2], kde = False)
ax = ax.set_title('QSO')
print (data['rerun'])