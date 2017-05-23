import csv
import numpy as np
import matplotlib.pyplot as plt


class User(object):
    # information
    id = 'id'
    date = []
    power = []
    power_2016_9 = []
    # model data
    average_power_train = 0.0
    average_error_vaild = 0.0

    def __init__(self, userid):
        self.id = userid
        self.date = []
        self.power = []

    def append(self, a_date, a_power):
        self.date.append(a_date)
        self.power.append(int(a_power))

    def power_without_2016_8(self):
        return self.power[0:-32]

    def power_2016_8(self):
        return self.power[-31:-1]

    def train(self):
        self.average_power_train = sum(self.power_without_2016_8()[-100:-1]) / 100.0
        # print('average_power_train ' + self.id + ': ' + str(self.average_power_train))

    def vaild(self):
        error_vaild = np.array(self.power_2016_8()) - self.average_power_train
        self.average_error_vaild = sum(abs(error_vaild)) / int(len(self.power_2016_8()))
        # print('average_error_vaild ' + self.id + ': ' + str(self.average_error_vaild))

    def predict(self):
        self.power_2016_9 = [int(self.average_power_train) for i in range(30)]
        # print('predict ' + self.id + ': ')
        # print(self.date_2016_9)
        return self.power_2016_9


def import_date(csv_name):
    print('--import date')

    user = []
    user_id = []

    with open(csv_name, 'rb') as f:
        reader = csv.reader(f)
        temp_user = User('0')
        for row in reader:  # row:['record_date', 'user_id', 'power_consumption']
            # delete first line
            if row[1] == 'user_id':
                continue
            # create list
            if (user_id == []) or (row[1] != user_id[-1]):
                user_id.append(row[1])
                temp_user = User(row[1])
                temp_user.append(row[0], row[2])
                user.append(temp_user)
            else:
                temp_user.append(row[0], row[2])
    return user, user_id


def preprocessing(user):
    print('--preprocessing--')
    print('-date count')
    date_count = {}
    for i in range(len(user)):
        date_len = len(user[i].date)
        if date_len not in date_count:
            date_count[date_len] = 1
        else:
            date_count[date_len] = date_count[date_len] + 1
    print (date_count)

    print('-lack date -add date and power')
    for i in range(len(user)):
        date_len = len(user[i].date)
        if date_len != 609:
            print(str(date_len) + ' - user_id: ' + user[i].id)
            for index, date in enumerate(user[0].date):
                if date not in user[i].date:
                    print (date)
                    # add date and power
                    user[i].date.insert(index, date)
                    add_power = (user[i].power[index - 1] + user[i].power[index]) / 2
                    user[i].power.insert(index, add_power)
    return user


def show_figure(show_index, user, sum=False):
    if sum:
        user_power_total = np.zeros(len(user[0].power))
        for temp_show_index in show_index:
            user_power_total = user_power_total + np.array(user[temp_show_index].power)
        plt.plot(range(len(user[0].date)), user_power_total, 'r')
        plt.show()
    else:
        for temp_show_index in show_index:
            plt.figure(temp_show_index)
            plt.plot(range(len(user[temp_show_index].date)), user[temp_show_index].power, 'r')
            plt.show()


def write_csv(csv_name, predict_power):
    print('--export date')
    with open(csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['predict_date', 'predict_power_consumption'])
        for date in range(30):
            writer.writerow([20160901+date, int(predict_power[date])])

if __name__ == '__main__':
    # import date
    user, user_id = import_date('Tianchi_power.csv')
    print('user number: ' + str(len(user)))
    print('date length: ' + str(len(user[0].date)))
    # preprocessing
    user = preprocessing(user)
    # show average power
    # show_figure(range(0,len(user)), user, sum = True)
    # train and predict
    power_2016_9_total = np.zeros(30)
    for i in range(len(user)):
        user[i].train()
        # user[i].vaild()
        power_2016_9_total = power_2016_9_total + np.array(user[i].predict())
    print(power_2016_9_total)
    # export
    write_csv('Tianchi_power_predict_table.csv', power_2016_9_total)




