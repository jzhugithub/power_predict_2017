import csv
import numpy as np
import matplotlib.pyplot as plt


class User(object):
    # information
    id = 'id'
    date = []
    power = []
    # model data
    average_power = 0.0
    delta_power_in_week = np.zeros(7)

    def __init__(self, userid):
        self.id = userid
        self.date = []
        self.power = []

    def append(self, a_date, a_power):
        self.date.append(a_date)
        self.power.append(int(a_power))

    def train(self, array_begin, array_end):
        self.average_power = self.get_average_power(array_end - 20, array_end)
        # print('average_power ' + self.id + ': ' + str(self.average_power))
        self.delta_power_in_week = self.get_delta_power_in_week(array_end - 395, array_end - 335)
        # print(self.delta_power_in_week)

    def predict(self, array_begin, array_end):
        power_predict = []
        for i, value in enumerate(range(array_begin, array_end)):
            # power_predict.append(self.average_power)
            power_predict.append(self.average_power + self.delta_power_in_week[value % 7])
        return power_predict

    def vaild(self, array_begin, array_end):
        power_predict = self.predict(array_begin, array_end)
        error_vaild = np.array(self.power[array_begin:array_end]) - np.array(power_predict)
        self.average_error_vaild = sum(abs(error_vaild)) / float(len(error_vaild))
        # print('average_error_vaild ' + self.id + ': ' + str(self.average_error_vaild))

    def get_average_power(self, array_begin, array_end):
        return sum(self.power[array_begin:array_end]) / float(array_end - array_begin)

    def get_delta_power_in_week(self, array_begin, array_end):
        power_in_week = np.zeros(7)
        average_power_in_week = np.zeros(7)
        date_count = np.zeros(7)
        for index in range(array_begin, array_end):
            power_in_week[index % 7] = power_in_week[index % 7] + self.power[index]
            date_count[index % 7] = date_count[index % 7] + 1
        for index in range(7):
            average_power_in_week[index] = power_in_week[index] / float(date_count[index])
        delta_power_in_week = average_power_in_week - average_power_in_week.sum() / 7.0

        return delta_power_in_week


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

def show_result(user, power_2016_9_total):
    user_power_total = np.zeros(len(user[0].power))
    for temp_show_index in range(len(user)):
        user_power_total = user_power_total + np.array(user[temp_show_index].power)
    plt.plot(range(len(user[0].date)), user_power_total, 'b')
    plt.plot(range(len(user[0].power),len(user[0].power) + 30), power_2016_9_total, 'r')
    plt.show()

def write_csv(csv_name, predict_power):
    print('--export date')
    with open(csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['predict_date', 'predict_power_consumption'])
        for date in range(30):
            writer.writerow([20160901 + date, int(predict_power[date])])


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
    average_error_vaild_total = 0
    power_2016_9_total = np.zeros(30)
    for i in range(len(user)):
        # train for vaild
        user[i].train(0, 578)
        user[i].vaild(578, 609)
        average_error_vaild_total = average_error_vaild_total + user[i].average_error_vaild
        # train for predict
        user[i].train(0, 609)
        power_2016_9_total = power_2016_9_total + np.array(user[i].predict(609, 639))
    print('average_error_vaild: ' + str(average_error_vaild_total))
    print('power_2016_9_total:')
    print(power_2016_9_total)
    # show predict result
    show_result(user, power_2016_9_total)
    # export
    write_csv('Tianchi_power_predict_table.csv', power_2016_9_total)
