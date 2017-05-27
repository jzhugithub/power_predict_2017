import csv
import matplotlib.pyplot as plt

def import_date(csv_name):
    data = []
    with open(csv_name, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:  # row:['record_date', 'user_id', 'power_consumption']
            if row[1] == 'predict_power_consumption':
                continue
            data.append(int(row[1]))
    return data


d26 = import_date('Tianchi_power_predict_table_5.26.csv')
d27 = import_date('Tianchi_power_predict_table_5.27.csv')
date = range(len(d26))
plt.plot(date, d26, 'k')
plt.plot(date, d27, 'r')
plt.show()
