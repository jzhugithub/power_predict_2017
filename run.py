import csv
import numpy as np
import matplotlib.pyplot as plt


def avg(list):
    return sum(list) / len(list)


def max(list, num):
    return [i if i > 0 else 0 for i in list]


def filter_smooth(nparray):
    filter1 = nparray
    # filter1[-30:] = np.ones(30)*np.median(nparray[len(nparray)-60:len(nparray)-32:2])
    for i in range(7, len(nparray)-8):
        filter1[i] = np.median(nparray[i-7:i+7])
    filter2 = filter1
    for i in range(15, len(filter1)-16):
        filter2[i] = np.mean(filter1[i-15:i+15:2])
    filter2[-45:] = filter2[-45-365:-365]+filter2[-45]-filter2[-45-365]
    filter2[-45:] = max(filter2[-45:],0)
    return filter2


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
    power = []
    power_filter = np.array([])
    power_minus_filter = np.array([])
    holiday = Holiday()
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

    def set_holiday(self):
        self.holiday.add_index_with_name(269, 'zqj')# zhongqiujie
        self.holiday.add_index_with_name(623, 'zqj')
        self.holiday.add_index_with_name(272, 'gq-1')# guoqing -1
        self.holiday.add_index_with_name(638, 'gq-1')
        self.holiday.add_index_with_name(248, 'bj')# Sunday, but have to work
        self.holiday.add_index_with_name(626, 'bj')

        self.compute_holiday_delta(269, 'zqj')
        self.compute_holiday_delta(272, 'gq-1')
        self.compute_holiday_delta(248, 'bj')

    def compute_holiday_delta(self, index, name):
        pre_average = self.get_average_power(index - self.pre_average_length_best, index)
        delta = self.power[int(index)] - pre_average
        self.holiday.add_name_with_delta(name, delta)

    def train(self, arrays):
        self.set_holiday()
        pre_average_length_min = range(len(arrays))
        delta_in_week_length_min = range(len(arrays))
        for i in range(len(arrays)):
            model_error_min = 1000000000
            for pal in range(30, 100, 10):
                for diwl in range(50, 150, 10):
                    model_error_temp = self.model_error(arrays[i], [pal, diwl])
                    if model_error_temp < model_error_min:
                        model_error_min = model_error_temp
                        pre_average_length_min[i] = pal
                        delta_in_week_length_min[i] = diwl
        self.pre_average_length_best = avg(pre_average_length_min)
        self.delta_in_week_length_best = avg(delta_in_week_length_min)

    def model_error(self, array, parameter):
        power_predict = self.model(array[0], array[1], parameter)
        error = np.array(self.power[array[0]:array[1]]) - np.array(power_predict)
        return avg(abs(error))

    def model(self, array_begin, array_end, parameter):
        # pre_average_length = parameter[0]
        delta_in_week_length = parameter[1]
        # average: parameter: pre_average_length
        # pre_average_last = self.get_average_power(array_begin - 365 - pre_average_length, array_begin - 365)
        # pre_average = self.get_average_power(array_begin - pre_average_length, array_begin)
        # pre_average_delta = pre_average - pre_average_last
        # average = self.get_average_power(array_begin - 365, array_end) + pre_average_delta
        # average = self.get_average_power(array_begin - pre_average_length, array_begin) + pre_average_delta
        # average_pre = self.get_average_power(array_begin - pre_average_length, array_begin) + pre_average_delta
        # average_last = self.get_average_power(array_begin, array_end)
        # average = 0.5*average_pre + 0.5*average_last
        # delta_in_week: parameter: delta_in_week_length
        delta_in_week = self.get_delta_in_week(array_begin - delta_in_week_length, array_begin)
        # generate power_predict
        power_predict = []
        for i, value in enumerate(range(array_begin, array_end)):
            power_predict.append(self.power_filter[value] + delta_in_week[value % 7])
        # deal with holiday
        for index in self.holiday.index_with_name:
            if index>array_begin and index < array_end:
                power_predict[index - array_begin] = self.power_filter[index - array_begin] + self.holiday.get_delta_by_index(index)
                # [index - array_begin] is a important, don no forget to minus array_begin

        return max(power_predict, 0)

    def evaluate(self, arrays):
        errors = range(len(arrays))
        for i in range(len(arrays)):
            errors[i] = self.model_error(arrays[i], [self.pre_average_length_best, self.delta_in_week_length_best])
        return avg(errors)

    def predict(self, array_begin, array_end):
        print('pre_average_length_best: ' + str(self.pre_average_length_best))
        print('delta_in_week_length_best: ' + str(self.delta_in_week_length_best))
        return self.model(array_begin, array_end, [self.pre_average_length_best, self.delta_in_week_length_best])

    def get_average_power(self, array_begin, array_end):
        return sum(self.power[array_begin:array_end]) / float(array_end - array_begin)

    def get_delta_in_week(self, array_begin, array_end):
        # power_no_average = np.array(self.power[array_begin:array_end]) - self.power_minus_filter[array_begin:array_end]
        # power_no_average = np.array(self.power[array_begin:array_end]) - self.get_average_power(array_begin, array_end)
        power_in_week = np.zeros(7)
        delta_power_in_week = np.zeros(7)
        date_count = np.zeros(7)
        for index in range(array_begin, array_end):
            power_in_week[index % 7] = power_in_week[index % 7] + self.power_minus_filter[index - array_begin]
            date_count[index % 7] = date_count[index % 7] + 1
        for index in range(7):
            delta_power_in_week[index] = power_in_week[index] / float(date_count[index])
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

    print('-delete stop users')
    delete_list = []
    for i in range(len(user)-1,0,-1):
        if sum(user[i].power[-30:])<=60:
            delete_list.append(i)
            user.remove(user[i])
    print('delete user: ' + str(delete_list))

    print('-add predict 2016.9 terms')
    for i in range(len(user)):
        for date in range(30):
            user[i].append('2016/9/' + str(date + 1), -1)

    print('-add filter for power')
    for i in range(len(user)):
        user[i].power_filter = filter_smooth(np.array(user[i].power))

    print('-add power_minus_filter')
    for i in range(len(user)):
        user[i].power_minus_filter = np.array(user[i].power) - user[i].power_filter

    return user


def merge_user(user):
    user_total = User('-1')
    user_total.date = user[0].date
    power_total = np.zeros(len(user[0].power))
    for i in range(len(user)):
        power_total = power_total + np.array(user[i].power)
    user_total.power = power_total.tolist()
    power_filter_total = np.zeros(len(user[0].power_filter))
    print len(user[0].power),len(user[0].power_filter)
    for i in range(len(user)):
        power_filter_total = power_filter_total + np.array(user[i].power_filter)
    user_total.power_filter = power_filter_total
    power_minus_filter_total = np.zeros(len(user[0].power_minus_filter))
    print len(user[0].power), len(user[0].power_minus_filter)
    for i in range(len(user)):
        power_minus_filter_total = power_minus_filter_total + np.array(user[i].power_minus_filter)
    user_total.power_minus_filter = power_minus_filter_total
    return user_total


def show_figure(show_index, user, sum_flag=False):
    if sum_flag:
        user_power_total = np.zeros(len(user[0].power))
        for temp_show_index in show_index:
            user_power_total = user_power_total + np.array(user[temp_show_index].power)
        plt.plot(range(len(user[0].date)), user_power_total, 'k')
        plt.plot(range(len(user[0].date))[3:len(user[0].date):7], user_power_total[3:len(user[0].date):7], '*r')
        plt.show()
    else:
        for temp_show_index in show_index:
            plt.figure(temp_show_index)
            plt.plot(range(len(user[temp_show_index].date)), user[temp_show_index].power, 'k')
            plt.plot(range(len(user[temp_show_index].date)), user[temp_show_index].power_filter, 'g')
            plt.plot(range(len(user[temp_show_index].date))[3:len(user[temp_show_index].date):7],
                     user[temp_show_index].power[3:len(user[temp_show_index].date):7], '*r')

            plt.show()


def show_result(user_total):
    plt.plot([234, 272], [user_total.power[234], user_total.power[272]], 'yo')
    plt.plot(range(len(user_total.date) - 30), user_total.power[:-30], 'k')
    plt.plot(range(len(user_total.date)), user_total.power_filter, 'g')
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
    user, user_id = import_date('Tianchi_power.csv')
    print('user number: ' + str(len(user)))
    print('date length: ' + str(len(user[0].date)))

    # preprocessing
    user = preprocessing(user)
    user_total = merge_user(user)
    # user_total = user[0]
    # show average power
    # show_figure(range(0,len(user)), user, sum_flag = False)

    # train
    train_arrays = [[549, 579], [579, 609]]
    user_total.train(train_arrays)
    error_train = user_total.evaluate(train_arrays)
    print('error_train: ' + str(error_train))

    # vaild
    vaild_arrays = [[534, 564], [564, 594]]
    error_vaild = user_total.evaluate(vaild_arrays)
    print('error_vaild ' + str(error_vaild))

    # predict
    power_2016_9_total = user_total.predict(609, 639)
    # power_2016_9_total = user_total.predict(579, 609)
    user_total.power[-30:] = power_2016_9_total
    print('power_2016_9_total: ' + str(power_2016_9_total))

    # show predict result
    show_result(user_total)
    # plt.plot(range(len(user_total.date) - 30), user_total.power[:-30], 'k')
    # plt.plot(range(579, 609), power_2016_9_total, 'b')
    # plt.plot(range(len(user_total.date))[3:len(user_total.date):7], user_total.power[3:len(user[0].date):7], '*r')
    # plt.show()

    # export
    write_csv('Tianchi_power_predict_table.csv', power_2016_9_total)
