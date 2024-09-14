import numpy as np
from Myalgos import *
from gurobipy import *
import xlsxwriter as xw
import time


if __name__ == '__main__':
    delta = 0.02   # confidence level
    delta_T = 100

    num_samples = 30
    # file_path = str('C:/Users/胡杨/Desktop/我的文章/A Type-free Online Reusable Service Composition/Numerical Studies/')
    file_path = str('/home/huyang/')

    '''-------------------------Test 1: Original---------------------------'''

    workbook1 = xw.Workbook(file_path + 'test1.xlsx')
    algo_names = ['Offline', 'NMWU', 'NMWU_S', 'Greedy']
    I = 10
    K = 5  # 20%, 50%, 80%
    a_bar = np.random.randint(4, 9, (I, ), dtype=int)
    w_bar = np.random.randint(2, 21, (I,), dtype=int)
    d_bar = 10

    for name in algo_names:
        exec('worksheet_' + name + ' = workbook1.add_worksheet(\'' + name + '\')')

    for counter in range(1, 101):  # num_samples is scale
        T = delta_T * counter
        for s in range(num_samples):
            [C, A, D, W, J_bar, V_bar] = CADW_generation(I, T, K, a_bar, d_bar, w_bar)

            # Reset time
            for name in algo_names:
                exec('time_' + name + ' = ' + str(0))
            # Run the algorithms
            for name in algo_names:
                if name in ['NMWU', 'NMWU_S']:
                    start = time.time()
                    exec('reward_' + name + ' = ' + name + '(delta, a_bar, d_bar, J_bar, V_bar, W, A, D, C)')
                    exec('worksheet_' + name + '.write(' + str(counter-1) + ',' + str(s) + ', reward_' + name + ')')
                    # .write(row, column, value)
                    stop = time.time()
                    exec('time_' + name + ' = time_' + name + ' + ' + str(stop - start))
                    print('Workbook 1,', name, ', T =', str(T), ', sample', str(s + 1))
                else:
                    start = time.time()
                    exec('reward_' + name + ' = ' + name + '(W, A, D, C, d_bar)')
                    exec('worksheet_' + name + '.write(' + str(counter-1) + ',' + str(s) + ', reward_' + name + ')')
                    stop = time.time()
                    exec('time_' + name + ' = time_' + name + ' + ' + str(stop - start))
                    print('Workbook 1,', name, ', T =', str(T), ', sample', str(s + 1))

        for name in algo_names:
            exec('worksheet_' + name + '.write(' + str(counter-1) + ',' + str(num_samples+1) + ', time_' + name + ')/num_samples')

    workbook1.close()

    