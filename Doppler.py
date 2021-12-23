import numpy as np
import math
import pandas as pd

class Doppler:
    def __Kinematic(x0, v0, a0=0, vm0=0, am0=0, t=0):
        """
        Вычисляет векторы координаты и скорости
        :param x0: начальное положение
        :param v0: начальная скорость
        :param a0: ускорение
        :param vm0: начальная скорость среды
        :param am0: ускорение среды
        :param t: момент времени
        :return: вектор координаты и вектор скорости
        """
        x = 0.5 * a0 * t * t + v0 * t + x0
        v = a0 * t + v0

        xm = 0.5 * am0 * t * t + vm0 * t
        vm = am0 * t + vm0

        x = x - xm
        v = v - vm
        return (x, v)

    def __ResampleValue(vo, vs, c):
        """
        Вычисляет коэффициент передискретизации
        :param vo: радиальная скорость наблюдателя
        :param vs: радиальная скорость источника
        :param c: скорость звука в среде
        :return: коэффициент передискретизации Sf
        """
        return (c + vo) / (c + vs)

    def __Approach(xo0, vo0, ao0, xs0, vs0, as0, ts, td):
        """
        Вычисляет расстояние между источником и наблюдателем
        :param xo0: начальное положение наблюдателя
        :param vo0: начальная скорость наблюдателя
        :param ao0: ускорение наблюдателя
        :param xs0: начальное положение источника
        :param vs0: начальная скорость источника
        :param as0: ускорение источника
        :param ts: приращение времени
        :param td: время наблюдения
        :return: расстояние между источником и наблюдателем
        """
        t = 0
        d0 = np.linalg.norm(xo0 - xs0)
        while (t < td):
            # print(t, td)
            x0, v0 = Doppler.__Kinematic(xo0, vo0, ao0, 0, 0, t)
            xs, vs = Doppler.__Kinematic(xs0, vs0, as0, 0, 0, t)
            los = xs - x0
            if (np.linalg.norm(los) < d0):
                d0 = np.linalg.norm(los)
            t = t + ts
        return d0

    def __Resample(b, bl, Sf, Sa):
        """
        :param b: входной буфер
        :param bl: длина входного буфера
        :param Sf: коэффициент передискретизации
        :param Sa: коэффициент усиления амплитуды
        :return: передискретизированный буфер
        """
        n = int(bl / Sf)
        bn = np.zeros(n)  # new buffer of length ln
        for i in range(n - 1):
            posb = i * Sf
            le = math.floor(posb)
            # print(le)
            bn[i] = ((b[le + 1] - b[le]) * (posb - le) + b[le]) * Sa
        return bn

    def Doppler(xo0, vo0, ao0, xs0, vs0, as0, vm0, am0, ts, c, b, lb, rb):
        """
        :param xo0: начальное положение наблюдателя
        :param vo0: начальная скорость наблюдателя
        :param ao0: ускорение наблюдателя
        :param xs0: начальное положение источника
        :param vs0: начальная скорость источника
        :param as0: ускорение источника
        :param vm0: начальная скорость среды
        :param am0: ускорение среды
        :param ts: приращение времени
        :param c: скорость звука в среде
        :param b: входной буфер
        :param lb: длина входного буфера
        :param rb: частота дискретизации входного буфера
        :return: буфер с эффектом Доплера
        """
        nb = int(lb / (rb * ts))
        rbts = int(rb * ts)
        bs = np.zeros((nb, int(rbts)))  # array of nb sample abs
        b2s = [i for i in range(nb)]

        d0 = Doppler.__Approach(xo0, vo0, ao0, xs0, vs0, as0, ts, lb / rb)
        t = 0

        for i in range(nb):
            for j in range(rbts):
                bs[i][j] = b[i * rbts + j]
            xo, vo = Doppler.__Kinematic(xo0, vo0, ao0, vm0, am0, t)
            xs, vs = Doppler.__Kinematic(xs0, vs0, ao0, vm0, am0, t)

            los = xs - xo
            # Далее нормировка в многомерном случае
            vo = np.dot(vo, los) / np.linalg.norm(los)
            vs = np.dot(vs, los) / np.linalg.norm(los)

            Sf = Doppler.__ResampleValue(vo, vs, c)
            Sa = d0 / np.linalg.norm(los)
            b2s[i] = Doppler.__Resample(bs[i], rb * ts, Sf, Sa)
            t += ts

        summ = 0
        for i in range(nb):
            summ += len(b2s[i])
        b2 = np.zeros(summ)

        k = 0
        for i in range(nb):
            l2b = len(b2s[i])
            for j in range(l2b):
                # b2.append(b2s[i][j])
                b2[k] = b2s[i][j]
                k += 1
        return b2

    def __VelFromShift(f0, f, c=343):
        """
        :param f0: исходная частота
        :param f: наблюдаемая частота
        :param c: скорость звука в среде
        :return: скорость источника
        """
        return c * (f0 - f) / f

    def GetVelocity(data: pd.DataFrame, data2: pd.DataFrame, width=1600, hight=20):
        """
        Вычисляет скорости по спектрам Фурье
        :param data: спектр испускаемых волн
        :param data2: спектр наблюдаемых волн
        :param width: пороговое значение амплитуды
        :param hight: пороговое значение частоты
        :return:
        """
        ds = data[data['F'] < width]
        first = float(ds[ds['A'] > hight][0:1]['F'])
        last = float(ds[ds['A'] > hight][-1:]['F'])

        ds2 = data2[data2['F'] < width]
        first2 = float(ds2[ds2['A'] > hight][0:1]['F'])
        last2 = float(ds2[ds2['A'] > hight][-1:]['F'])

        speed1 = Doppler.__VelFromShift(first, first2)
        speed2 = Doppler.__VelFromShift(last, last2)
        return (speed1, speed2)