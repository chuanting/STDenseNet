# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by ct
** Name: milano
** Date: 1/21/18
** BSD license
********************************************/
"""
import h5py
import pickle
import numpy as np
from pandas import to_datetime
from models.MinMaxNorm import MinMaxNorm01
from dataloader.STMatrix import STMatrix


def _loader(f, nb_flow, traffic_type, height, width, crop, rows=[0, 100], cols=[0, 100]):
    if nb_flow == 1:
        if traffic_type == 'sms':
            sms_in = f['data'][:, :, :, 0]
            sms_out = f['data'][:, :, :, 1]
            data = np.sum([sms_in, sms_out], axis=0)
        elif traffic_type == 'call':
            call_in = f['data'][:, :, :, 2]
            call_out = f['data'][:, :, :, 3]
            data = np.sum([call_in, call_out], axis=0)
        elif traffic_type == 'internet':
            data = f['data'][:, :, :, 4]
        else:
            raise IOError("Unknown traffic type")
        result = data.reshape((-1, 1, height, width))

        if crop:
            result = result[:, :, rows[0]:rows[1], cols[0]:cols[1]]
        return result

    elif nb_flow == 2:
        if traffic_type == 'sms':
            data = f['data'][:, :, :, 0:2]
        elif traffic_type == 'call':
            data = f['data'][:, :, :, 2:4]
        elif traffic_type == 'internet':
            print("Internet only has one channel (please set nb_flow=1)")
            exit(0)
            # data = f['data'][:, :, 4]
        else:
            raise IOError("Unknown traffic type")

        result = []
        for i in range(len(data)):
            tmp = data[i].reshape((height*width, 2))
            result.append(tmp.transpose())
        result = np.array(result).reshape((-1, 2, height, width))

        if crop:
            result = result[:, :, rows[0]:rows[1], cols[0]:cols[1]]
        return result

    else:
        print("Wrong parameter with nb_flow")
        exit(0)


def load_data(path, traffic_type, closeness_size, period_size, trend_size, len_test, nb_flow,
              height, width, crop, rows, cols):
    f = h5py.File(path, 'r')
    data = _loader(f, nb_flow, traffic_type, height, width, crop, rows, cols)
    index = f['idx'].value.astype(str)
    index = to_datetime(index, format='%Y-%m-%d %H:%M')

    data_all = [data]
    index_all = [index]

    mmn = MinMaxNorm01()
    data_train = data[:-len_test]
    mmn.fit(data_train)

    data_all_mmn = []
    for data in data_all:
        data_all_mmn.append(mmn.transform(data))

    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    xc, xp, xt = [], [], []
    y = []
    timestamps_y = []

    for data, index in zip(data_all_mmn, index_all):
        st = STMatrix(data, index, 24)
        _xc, _xp, _xt, _y, _timestamps_y = st.create_dataset(
            len_closeness=closeness_size, len_period=period_size, len_trend=trend_size,
            PeriodInterval=1
        )

        xc.append(_xc)
        xp.append(_xp)
        xt.append(_xt)
        y.append(_y)
        timestamps_y += _timestamps_y

    xc = np.vstack(xc)
    xp = np.vstack(xp)
    xt = np.vstack(xt)
    y = np.vstack(y)

    xc_train, xp_train, xt_train, y_train = xc[:-len_test], xp[:-len_test], xt[:-len_test], y[:-len_test]
    xc_test, xp_test, xt_test, y_test = xc[-len_test:], xp[-len_test:], xt[:-len_test], y[-len_test:]
    # timestamps_train, timestamps_test = timestamps_y[:-len_test], timestamps_y[-len_test:]

    x_train = []
    x_test = []

    for l, x_ in zip([closeness_size, period_size, trend_size], [xc_train, xp_train, xt_train]):
        if l > 0:
            x_train.append(x_)

    for l, x_ in zip([closeness_size, period_size, trend_size], [xc_test, xp_test, xt_test]):
        if l > 0:
            x_test.append(x_)
    return x_train, y_train, x_test, y_test, mmn
