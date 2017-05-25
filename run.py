import csv
import numpy as np
import matplotlib.pyplot as plt


def avg(list):
    return sum(list) / len(list)


class User(object):
    # information
    id = 'id'
    date = []
    power = []
    # model
    pre_average_length_best = 90
    delta_in_week_length_best = 0

    def __init__(self, userid):
        self.id = userid
        self.date = []
        self.power = []

    def append(self, a_date, a_power):
        self.date.append(a_date)
        self.power.append(int(a_power))

    def train(self, arrays):
        pre_average_length_min = range(len(arrays))
        delta_in_week_length = range(len(arrays))
        for i in range(len(arrays)):
            model_error_min = 1000000000
            for pal in range(10, 100, 10):
                for diwl in range(10, 150, 10):
                    model_error_temp = self.model_error(arrays[i], [pal, diwl])
                    if model_error_temp < model_error_min:
                        model_error_min = model_error_temp
                        pre_average_length_min[i] = pal
                        delta_in_week_length[i] = diwl
        self.pre_average_length_best = avg(pre_average_length_min)
        self.delta_in_week_length_best = avg(delta_in_week_length)

    def model_error(self, array, parameter):
        power_predict = self.model(array[0], array[1], parameter)
        error = np.array(self.power[array[0]:array[1]]) - np.array(power_predict)
        return avg(abs(error))

    def model(self, array_begin, array_end, parameter):
        pre_average_length = parameter[0]
        delta_in_week_length = parameter[1]
        # average: parameter: pre_average_length
        pre_average_last = self.get_average_power(array_begin - 365 - pre_average_length, array_begin - 365)
        pre_average = self.get_average_power(array_begin - pre_average_length, array_begin)
        pre_average_delta = pre_average - pre_average_last
        average = self.get_average_power(array_begin - 365, array_end) + pre_average_delta
        # delta_in_week: parameter: delta_in_week_length
        delta_in_week = self.get_delta_in_week(array_begin - delta_in_week_length, array_begin)
        # generate power_predict
        power_predict = []
        for i, value in enumerate(range(array_begin, array_end)):
            power_predict.append(average + delta_in_week[value % 7])
        return power_predict

    def evaluate(self, arrays):
        errors = range(len(arrays))
        for i in range(len(arrays)):
            errors[i] = self.model_error(arrays[i], [self.pre_average_length_best, self.delta_in_week_length_best])
        return avg(errors)

    def predict(self, array_begin, array_end):
        return self.model(array_begin, array_end, [self.pre_average_length_best, self.delta_in_week_length_best])

    def get_average_power(self, array_begin, array_end):
        return sum(self.power[array_begin:array_end]) / float(array_end - array_begin)

    def get_delta_in_week(self, array_begin, array_end):
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

    print('-add predict 2016.9 terms')
    for i in range(len(user)):
        for date in range(30):
            user[i].append('2016/9/' + str(date + 1), -1)

    return user


def merge_user(user):
    user_total = User('-1')
    user_total.date = user[0].date
    power_total = np.zeros(len(user[0].power))
    for i in range(len(user)):
        power_total = power_total + np.array(user[i].power)
    user_total.power = power_total.tolist()
    return user_total


def show_figure(show_index, user, sum_flag=False):
    if sum_flag:
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


def show_result(user_total):
    plt.plot(range(len(user_total.date) - 30), user_total.power[:-30], 'b')
    plt.plot(range(len(user_total.date) - 30, len(user_total.date)), user_total.power[-30:], 'r')
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
    user_total = merge_user(user)

    # show average power
    # show_figure(range(0,len(user)), user, sum = True)

    # train
    train_arrays = [[519, 549], [549, 579], [579, 609]]
    user_total.train(train_arrays)
    error_train = user_total.evaluate(train_arrays)
    print('error_train: ' + str(error_train))

    # vaild
    vaild_arrays = [[534, 564], [564, 594]]
    error_vaild = user_total.evaluate(vaild_arrays)
    print('error_vaild ' + str(error_vaild))

    # predict
    power_2016_9_total = user_total.predict(609, 639)
    user_total.power[-30:] = power_2016_9_total
    print('power_2016_9_total: ' + str(power_2016_9_total))

    # show predict result
    show_result(user_total)

    # export
    write_csv('Tianchi_power_predict_table.csv', power_2016_9_total)
