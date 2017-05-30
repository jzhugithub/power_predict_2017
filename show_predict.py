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


d = import_date('Tianchi_power_predict_table.csv')
d26 = import_date('Tianchi_power_predict_table_5.26.csv')
d27 = import_date('Tianchi_power_predict_table_5.27.csv')
d28 = import_date('Tianchi_power_predict_table_5.28.csv')
d29 = import_date('Tianchi_power_predict_table_5.29.csv')
d30 = import_date('Tianchi_power_predict_table_5.30.csv')
date = range(len(d))
plt.plot(date, d, 'k')
plt.plot(date, d28, 'r')
plt.plot(date, d29, 'g')
plt.plot(date, d30, 'b')
plt.show()
