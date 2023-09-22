import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from datetime import timedelta
import datetime
from statsmodels.stats.multitest import fdrcorrection

file = 'new.csv'
data = pd.read_csv(file, on_bad_lines='warn',header=None)

# idx = []; seq = []
# f = open('idx_fa.txt','r')
# g = open('seq_fa.txt','r')
# while True:

#     line = f.readline()
#     seq_line = g.readline()
#     if line:
#         line = line.split('|')
#         idx.append(line[3])
#         seq.append(seq_line.strip('\n'))
#     else:
#         break
# f.close()
# g.close()

# ii = []
# row, col = data.shape
# for i in range(row):
#     name = data.iloc[i,1]
#     if name in idx:
#         None
#     else:
#         ii.append(i)
row = data.shape[0]
print(data.columns)
data = data.drop_duplicates(subset=[11],keep='first')

# row, col = data.shape
# seq_new = []
# for i in range(row):
#     name = data.iloc[i,1]
#     index = idx.index(name)
#     seq_new.append(seq[index])

# data.insert(loc=12, column='seq', value=seq_new)

row = data.shape[0]
print(row)
delta_lineage = []; total_lineage = []
for i in range(row):
        
    lineage_bef = 0; lineage_aft = 0
    lineage = data.iloc[i,11]
    current_date = data.iloc[i,2]
    current_date = current_date.split('-')
    year = int(current_date[0]); month = int(current_date[1]); day = int(current_date[2])

    if month == 0:
        month = 1
    if day == 0:
        day = 1
    current_date = datetime.date(year,month,day)
    #before
    j = i; flag = False
    while True:
        if j < 0:
            flag = True
            break

        this_date = data.iloc[j,2]
        this_date = this_date.split('-')
        year = int(this_date[0]); month = int(this_date[1]); day = int(this_date[2])
        if month == 0:
            month = 1
        if day == 0:
            day = 1
        this_date = datetime.date(year,month,day)
        this_seq = data.iloc[j,11]
        if this_date >= current_date - datetime.timedelta(days=90):
            if this_seq == lineage:
                lineage_bef += 1
            
        else:
            break
        j -=1
    #after
    j = i
    while True:
        if j > row-1:
            flag = True
            break

        this_date = data.iloc[j,2]
        this_date = this_date.split('-')
        year = int(this_date[0]); month = int(this_date[1]); day = int(this_date[2])
        
        if month == 0:
            month = 1
        if day == 0:
            day = 1
        this_date = datetime.date(year,month,day)
        this_seq = data.iloc[j,11]
        if this_date <= current_date + datetime.timedelta(days=90):
            if this_seq == lineage:
                lineage_aft += 1
            
        else:
            break
        j +=1

    if flag:
        lineage_bef = 0; lineage_aft = 0
    delta_lineage.append(lineage_aft-lineage_bef); total_lineage.append(lineage_aft+lineage_bef)
    print(i)

data.insert(loc=12, column='delta', value=delta_lineage)
data.insert(loc=13, column='total', value=total_lineage)

data.to_csv('stat.csv', index=False)