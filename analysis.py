import csv
import numpy as np
import matplotlib.pyplot as plt
import arma as ar
import pandas as pd


def pdvalues_to_array(pdvalue):
    return np.array([i[0] for i in pdvalue])


def avg(nparray):
    return sum(nparray) / len(nparray)


def max(nparray, num):
    return np.array([i if i > num else num for i in nparray])


def play_filter(filter, nparray, length, step):
    nparray2 = np.array(nparray)
    for i in range(int(length / 2), len(nparray) - int(length / 2) - 1):
        nparray2[i] = filter(nparray[i - int(length / 2):i + int(length / 2):step])
    return nparray2


def filter_smooth(nparray):
    nparray = play_filter(np.median, nparray, 14, 1)
    nparray = play_filter(np.mean, nparray, 14, 1)
    nparray[-37:] = nparray[-37 - 365:-365] + nparray[-37] - nparray[-37 - 365]
    nparray[-37:] = max(nparray[-37:], 0)
    return nparray


class Holiday(object):
    name_with_delta = {}
    index_with_name = {}

    def __init__(self):
        pass

    def add_name_with_delta(self, name, delta):
        self.name_with_delta[name] = delta

    def add_index_with_name(self, index, name):
        self.index_with_name[index] = name

    def has_index(self, index):
        return index in self.index_with_name

    def has_name(self, name):
        return name in self.name_with_delta

    def get_name_by_index(self, index):
        if self.has_index(index):
            return self.index_with_name[index]
        else:
            print('index is not exist')

    def get_delta_by_name(self, name):
        if self.has_name(name):
            return self.name_with_delta[name]
        else:
            print('name is not exist')

    def get_delta_by_index(self, index):
        return self.get_delta_by_name(self.get_name_by_index(index))


class User(object):
    # information
    id = 'id'
    date = []
    power = np.array([])
    power_filter = np.array([])
    power_minus_filter = np.array([])
    holiday = Holiday()
    # model
    delta_in_pd = pd.DataFrame([])
    ar_model = None

    def __init__(self, userid, a_date, a_power):
        self.id = userid
        self.date = a_date
        self.power = a_power

    def append(self, a_date, a_power):
        self.date.append(a_date)
        self.power = np.concatenate((self.power, np.array([a_power])))

    def set_holiday(self):
        self.holiday.add_index_with_name(269, 'zqj')  # zhongqiujie
        self.holiday.add_index_with_name(623, 'zqj')
        self.holiday.add_index_with_name(272, 'gq-1')  # guoqing -1
        self.holiday.add_index_with_name(638, 'gq-1')
        self.holiday.add_index_with_name(248, 'bj')  # Sunday, but have to work
        self.holiday.add_index_with_name(626, 'bj')

        self.compute_holiday_delta(269, 'zqj')
        self.compute_holiday_delta(272, 'gq-1')
        self.compute_holiday_delta(248, 'bj')

    def compute_holiday_delta(self, index, name):
        self.holiday.add_name_with_delta(name, self.power_minus_filter[int(index)])

    def train(self, arrays):
        self.set_holiday()
        self.delta_in_pd = pd.DataFrame(self.power_minus_filter, self.date)
        ar_model_best = None
        model_error_min = 1000000000
        for i in range(len(arrays)):
            self.ar_model = ar.arima_model(self.delta_in_pd[arrays[i][0] - 9:arrays[i][1]] - 1, maxLag=8)
            self.ar_model.get_proper_model()
            print 'bic:', self.ar_model.bic, 'p:', self.ar_model.p, 'q:', self.ar_model.q
            print self.ar_model.properModel.forecast()[0]
            model_error_temp = self.model_error(arrays[i])
            if model_error_temp < model_error_min:
                model_error_min = model_error_temp
                ar_model_best = self.ar_model
        self.ar_model = ar_model_best

    def model_error(self, array, parameter=None):
        power_predict = self.model(array[0], array[1], parameter)
        error = self.power[array[0]:array[1]] - power_predict
        return avg(abs(error))

    def model(self, array_begin, array_end, parameter=None):

        # predict
        ar_delta = self.ar_model.properModel.predict(self.date[array_begin], self.date[array_end - 1],
                                                     dynamic=True).values
        # generate power_predict
        power_predict = ar_delta + self.power_filter[array_begin:array_end]

        # deal with holiday
        for index in self.holiday.index_with_name:
            if index > array_begin and index < array_end:
                power_predict[index - array_begin] = self.power_filter[index] + self.holiday.get_delta_by_index(index)
        return max(power_predict, 0)

    def evaluate(self, arrays):
        errors = range(len(arrays))
        for i in range(len(arrays)):
            errors[i] = self.model_error(arrays[i])
        return avg(errors)

    def predict(self, array_begin, array_end):
        return self.model(array_begin, array_end)


def import_date(csv_name, number):
    print('--import date')
    user = []
    user_id = []
    with open(csv_name, 'rb') as f:
        reader = csv.reader(f)
        temp_date = []
        temp_power_list = []
        for row in reader:  # row:['record_date', 'user_id', 'power_consumption']
            if row[1] != 'user_id' and int(row[1]) > number:
                break
            # delete first line
            if row[1] == 'user_id':
                user_id.append('1')
                continue
            # create list
            if row[1] != user_id[-1]:
                user_id.append(row[1])  # this user id
                user.append(User(user_id[-2], temp_date, np.array(temp_power_list)))  # last user
                temp_date = [row[0]]
                temp_power_list = [int(row[2])]
            else:
                temp_date.append(row[0])
                temp_power_list.append(int(row[2]))
        user.append(User(user_id[-1], temp_date, np.array(temp_power_list)))
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
    for i in range(len(user) - 1, -1, -1):
        date_len = len(user[i].date)
        if date_len != 609:
            print(str(date_len) + ' - user_id: ' + user[i].id)
            for index, date in enumerate(user[0].date[::-1]):
                if date not in user[i].date:
                    print (date)
                    # add date and power
                    user[i].date.insert(index, date)
                    add_power = (user[i].power[index - 1] + user[i].power[index]) / 2
                    user[i].power = np.concatenate((user[i].power[:i], np.array([add_power]), user[i].power[i:]))

    print('-deal with stop users')
    stop_list = []
    for i in range(len(user) - 1, 0, -1):
        if sum(user[i].power[-30:]) <= 60:
            stop_list.append(i)
            user[i].power = user[i].power[-31] * np.ones(len(user[i].power))
    print('stop user: ' + str(stop_list))

    print('-add predict 2016.9 terms')
    for i in range(len(user)):
        for date in range(30):
            user[i].append('2016/9/' + str(date + 1), -1)

    print('-add filter for power')
    for i in range(len(user)):
        user[i].power_filter = filter_smooth(user[i].power)

    print('-add power_minus_filter')
    for i in range(len(user)):
        user[i].power_minus_filter = user[i].power - user[i].power_filter

    return user


def merge_user(user):
    user_total = User('-1', [], np.array([]))
    user_total.date = user[0].date
    power_total = np.zeros(len(user[0].power))
    for i in range(len(user)):
        power_total = power_total + np.array(user[i].power)
    user_total.power = power_total
    power_filter_total = np.zeros(len(user[0].power_filter))
    for i in range(len(user)):
        power_filter_total = power_filter_total + np.array(user[i].power_filter)
    user_total.power_filter = power_filter_total
    power_minus_filter_total = np.zeros(len(user[0].power_minus_filter))
    for i in range(len(user)):
        power_minus_filter_total = power_minus_filter_total + np.array(user[i].power_minus_filter)
    user_total.power_minus_filter = power_minus_filter_total
    return user_total


def show_figure(show_index, user, sum_flag=False):
    index1 = show_index[0]
    if sum_flag:
        user_power_total = np.zeros(len(user[index1].power))
        for temp_show_index in show_index:
            user_power_total = user_power_total + np.array(user[temp_show_index].power)
        plt.plot(range(len(user[index1].date)), user_power_total, 'k')
        plt.plot(range(len(user[index1].date))[3:len(user[index1].date):7], user_power_total[3:len(user[index1].date):7], '*r')
        plt.show()
    else:
        for temp_show_index in show_index:
            plt.figure(temp_show_index)
            plt.plot(range(len(user[temp_show_index].date)), user[temp_show_index].power, 'k')
            plt.plot(range(len(user[temp_show_index].date)), user[temp_show_index].power_filter, 'g')
            plt.plot(range(len(user[temp_show_index].date)), user[temp_show_index].power_minus_filter, 'b')
            plt.plot(range(len(user[temp_show_index].date))[3:len(user[temp_show_index].date):7],
                     user[temp_show_index].power[3:len(user[temp_show_index].date):7], '*r')

            plt.show()


def show_result(user_total, predict_range, predict_power):
    plt.plot(range(len(user_total.date)), [0 for i in range(len(user_total.date))], 'k')
    # plt.plot(range(predict_range[0], predict_range[1]), predict_power, 'r')
    plt.plot([234, 272], [user_total.power[234], user_total.power[272]], 'yo')
    plt.plot(range(len(user_total.date)), user_total.power, 'k')
    plt.plot(range(len(user_total.date)), user_total.power_filter, 'g')
    plt.plot(range(len(user_total.date) - 30), user_total.power_minus_filter[:-30], 'b')
    plt.plot(range(len(user_total.date) - 30, len(user_total.date)), user_total.power[-30:], 'b')
    plt.plot(range(len(user_total.date))[3:len(user_total.date):7], user_total.power[3:len(user[0].date):7], '*r')
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
    user, user_id = import_date('Tianchi_power.csv', 10000)  # 1454
    print('user number: ' + str(len(user)))
    print('date length: ' + str(len(user[0].date)))


    # preprocessing

    for i in range(0, 1454):
        print('i: ' + str(i))
        user[i:i+1] = preprocessing(user[i:i+1])

        show_figure([0], user[i:i+1])

