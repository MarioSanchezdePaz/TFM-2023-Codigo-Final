import csv
import os

from darts import TimeSeries
import glob
import datetime
import numpy as np
from ortools.linear_solver import pywraplp
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Variables globales
average_execution_time = 0
clusters = {}
virtualMachines = {}
predictionsList = []
realDataList = {}
solver = pywraplp.Solver.CreateSolver('SCIP')
solver2 = pywraplp.Solver.CreateSolver('SCIP')
work_load = []
mem_load = []
bw_load = []
longitud_lista = 150
list_for_load_overflows = []


def loop_machines_daily_cp_model(previous_day, ini, end, distance):
    x = {}
    y = {}
    z = {}
    z_abs = {}
    my_matrix = np.zeros((len(virtualMachines["all_machines"]), len(clusters["all_clusters"])))
    tmp_work_load = []
    host_load = {}
    max_load = {}
    tmp_mem = []
    tmp_bw = []
    for i in virtualMachines["all_machines"]:
        for j in clusters["all_clusters"]:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
            z[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
            z_abs[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
    for j in clusters['all_clusters']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)
    absolute_max = solver.IntVar(0, solver.infinity(), "absolute_max")
    for i in range(ini, end):
        for j in clusters["all_clusters"]:
            host_load[(i, j)] = solver.IntVar(0, solver.infinity(), "hl_%i_%i" % (j, j))

    for i in virtualMachines["all_machines"]:
        solver.Add(sum(x[i, b] for b in clusters["all_clusters"]) == 1)

    for hr in range(ini, end):
        max_load[hr] = solver.IntVar(0, solver.infinity(), "mx[%i]" % hr)
        for j in clusters["all_clusters"]:
            host_load[(hr, j)] = sum(x[i, j] * virtualMachines["cores"][i][hr]
                                     for i in virtualMachines["all_machines"])
            solver.Add(host_load[(hr, j)] <= (y[j] * clusters["cores"][j]) * 0.95)
            solver.Add(sum(x[i, j] * virtualMachines["mem"][i][hr]
                           for i in virtualMachines["all_machines"]) <= (y[j] * clusters["mem"][j]))
            input_sum = sum(x[i, j] * virtualMachines["bw_in"][i][hr] for i in virtualMachines["all_machines"])
            output_sum = sum(x[i, j] * virtualMachines["bw_out"][i][hr] for i in virtualMachines["all_machines"])
            sum_bw = input_sum + output_sum
            solver.Add(sum_bw <= (y[j] * clusters["bandwidth"][j]))

    for hr in range(ini, end):
        for j in clusters["all_clusters"]:
            solver.Add(max_load[hr] >= 0)
            solver.Add(max_load[hr] >= host_load[(hr, j)])
            solver.Add(max_load[hr] >= -(host_load[(hr, j)]))

    if distance >= 0:
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                z[i, j] = x[i, j] - previous_day[i][j]
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                solver.Add(z_abs[i, j] >= 0)
                solver.Add(z_abs[i, j] >= z[i, j])
                solver.Add(z_abs[i, j] >= -(z[i, j]))
        solver.Add(sum([z_abs[i, j] for i in virtualMachines["all_machines"] for j in
                        clusters["all_clusters"]]) / 2 <= distance)

    cores = solver.IntVar(0, solver.Infinity(), "cores")
    cores = solver.Sum([y[j] * clusters["cores"][j] for j in clusters["all_clusters"]])
    solver.Minimize(cores)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        min_cores = sum([y[j].solution_value() * clusters["cores"][j] for j in clusters["all_clusters"]])
        for hr in range(ini, end):
            solver.Add(absolute_max >= 0)
            solver.Add(absolute_max >= max_load[hr])
            solver.Add(absolute_max >= -(max_load[hr]))
        solver.Add(solver.Sum([y[j] * clusters["cores"][j] for j in clusters["all_clusters"]]) <= min_cores)
        solver.Minimize(absolute_max)
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            num_bins = 0
            for hr in range(24):
                # Calculate the number of migrations and manage it.
                print('Hour %i:00 ' % hr)
                tmp_list = []
                for j in clusters["all_clusters"]:
                    if y[j].solution_value() == 1:
                        bin_items = []
                        bin_weight = 0
                        real_weight = 0
                        current_mem = 0
                        current_bw = 0

                        real_weight = 0
                        real_mem = 0
                        real_bw = 0
                        for i in virtualMachines["all_machines"]:
                            if x[i, j].solution_value() > 0:
                                bin_items.append(i)
                                bin_weight += virtualMachines["cores"][i][hr]
                                real_weight += realDataList["cores"][i][hr]

                                current_mem += virtualMachines["mem"][i][hr]
                                current_bw += virtualMachines["bw_in"][i][hr] + virtualMachines["bw_out"][i][hr]

                                real_mem += realDataList["mem"][i][hr]
                                real_bw += realDataList["bw_in"][i][hr] + realDataList["bw_out"][i][hr]
                        if bin_items:
                            num_bins += 1
                            print('Bin number %i with weight %i' % (j, clusters["cores"][j]))
                            print('  Items packed:', bin_items)
                            print('  Total weight:', bin_weight)
                            print('  Real weight: ', real_weight)
                            pred_load_percentage = bin_weight / clusters["cores"][j] * 100
                            pred_bw_percentage = current_bw / clusters["bandwidth"][j] * 100
                            pred_mem_percentage = current_mem / clusters["mem"][j] * 100

                            real_load_percentage = real_weight / clusters["cores"][j] * 100
                            real_mem_percentage = real_mem / clusters["mem"][j] * 100
                            real_bw_percentage = real_bw / clusters["bandwidth"][j] * 100

                            print('  Predicted load %f vs Real Load %f' % (pred_load_percentage, real_load_percentage))
                            print('  Memory occupied: predicted memory: %f (%f) vs real memory: %f (%f)' %
                                  (current_mem, pred_mem_percentage, real_mem, real_mem_percentage))

                            print('  Bandwidth occupied: predicted bandwidth: %f (%f) vs real bandwidth: %f (%f)' %
                                  (current_bw, pred_bw_percentage, real_bw, real_bw_percentage))

                            if pred_load_percentage < real_load_percentage:
                                tmp_list.append(real_load_percentage - pred_load_percentage)
                            else:
                                tmp_list.append(0)
                            tmp_work_load.append((hr, round(real_load_percentage, 5)))
                            tmp_mem.append((hr, round((real_mem / clusters["mem"][j] * 100), 3)))
                            tmp_bw.append((hr, round((real_bw / clusters["bandwidth"][j] * 100), 3)))

                            print('  List size: ', len(bin_items))
                            print()
                    else:
                        tmp_work_load.append((hr, 0))
                        tmp_mem.append((hr, 0))
                        tmp_bw.append((hr, 0))
                print()
                print('Number of bins used:', num_bins / 24)
                print('Time = ', solver.WallTime(), ' milliseconds')
                work_load.append(tmp_work_load)
                mem_load.append(tmp_mem)
                bw_load.append(tmp_bw)
                list_for_load_overflows.append(tmp_list)
                tmp_work_load = []
                tmp_mem = []
                tmp_bw = []

            for i in virtualMachines["all_machines"]:
                for j in clusters["all_clusters"]:
                    my_matrix[i][j] = x[(i, j)].solution_value()
            return my_matrix
        else:
            print('The problem does not have an optimal solution.')
    else:
        print('The problem does not have an optimal solution.')


def loop_machines_each_hour_cp_model(index, previous_list, distance):
    # Each item is assigned to at most one bin.
    x = {}
    tmp_work = []
    tmp_mem = []
    tmp_bw = []
    my_matrix = np.zeros((len(virtualMachines["all_machines"]), len(clusters["all_clusters"])))
    z = {}
    z_abs = {}
    host_load = {}
    delay_for_mv = {}
    intermediate_delay = {}
    delay_abs = {}
    hot_migrations = {}
    for i in virtualMachines["all_machines"]:
        delay_for_mv[i] = solver.IntVar(-100, solver.infinity(), "delay_for_%i" % i)
        intermediate_delay[i] = solver.IntVar(0, 100, "intermediate_delay_%i" % i)
        delay_abs[i] = solver.IntVar(0, 16, "delay_abs_%i" % i)
        hot_migrations[i] = solver.BoolVar("Hot_Migration%i" % i)

    for i in virtualMachines["all_machines"]:
        for j in clusters["all_clusters"]:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
            z[(i, j)] = solver.IntVar(-1, 1, 'x_%i_%i' % (i, j))
            z_abs[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
    max_load = solver.IntVar(0, solver.infinity(), "max_load")
    y = {}
    for j in clusters['all_clusters']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)
        host_load[j] = solver.IntVar(0, solver.infinity(), "h[%i]" % j)

    for i in virtualMachines["all_machines"]:
        solver.Add(sum(x[i, b]
                       for b in clusters["all_clusters"]) == 1)

    for j in clusters["all_clusters"]:
        host_load[j] = sum(x[i, j] * virtualMachines["cores"][i][index] for i in virtualMachines["all_machines"])

    for j in clusters["all_clusters"]:
        solver.Add(sum(x[i, j] * virtualMachines["cores"][i][index]
                       for i in virtualMachines["all_machines"]) <= (y[j] * clusters["cores"][j]) * 0.95)

        solver.Add(sum(x[i, j] * virtualMachines["mem"][i][index]
                       for i in virtualMachines["all_machines"]) <= (y[j] * clusters["mem"][j]))

        input_sum = sum(x[i, j] * virtualMachines["bw_in"][i][index] for i in virtualMachines["all_machines"])
        output_sum = sum(x[i, j] * virtualMachines["bw_out"][i][index] for i in virtualMachines["all_machines"])
        sum_bw = input_sum + output_sum
        solver.Add(sum_bw <= (y[j] * clusters["bandwidth"][j]))

    if distance >= 0:
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                z[i, j] = (x[i, j] - previous_list[i][j])

        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                solver.Add(z_abs[i, j] >= 0)
                solver.Add(z_abs[i, j] >= z[i, j])
                solver.Add(z_abs[i, j] >= -(z[i, j]))

        solver.Add(sum([z_abs[i, j] for i in virtualMachines["all_machines"] for j in
                        clusters["all_clusters"]]) / 2 <= distance)

        for i in virtualMachines["all_machines"]:
            solver.Add(delay_abs[i] >= 0)
            solver.Add(delay_abs[i] >= intermediate_delay[i])
            solver.Add(delay_abs[i] >= -(intermediate_delay[i]))

        solver.Add(sum(delay_abs[i] for i in virtualMachines["all_machines"]) * 100 <= 100000)
    else:
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                z_abs[(i, j)] = x[(i, j)]
                z[(i, j)] = x[(i, j)]

        for i in virtualMachines["all_machines"]:
            delay_for_mv[i] = sum(z_abs[(i, j)] * (j + 1) for j in clusters["all_clusters"])
            delay_abs[i] = sum(x[(i, j)] * (j + 1) for j in clusters["all_clusters"])
            hot_migrations[i] = x[(i, j)]

    cores = solver.Sum([y[j] * clusters["cores"][j] for j in clusters["all_clusters"]])
    solver.Minimize(cores)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        min_cores = sum([y[j].solution_value() * clusters["cores"][j] for j in clusters["all_clusters"]])
        for j in clusters["all_clusters"]:
            solver.Add(max_load >= 0)
            solver.Add(max_load >= host_load[j])
            solver.Add(max_load >= -(host_load[j]))

        solver.Add(solver.Sum([y[j] * clusters["cores"][j] for j in clusters["all_clusters"]]) <= min_cores)
        solver.Minimize(max_load)
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            num_bins = 0
            tmp_list = []
            print('Hour %i:00 ' % index)
            for j in clusters["all_clusters"]:
                if y[j].solution_value() == 1:
                    bin_items = []
                    bin_weight = 0

                    current_mem = 0
                    current_bw = 0

                    real_weight = 0
                    real_mem = 0
                    real_bw = 0

                    for i in virtualMachines["all_machines"]:
                        if x[i, j].solution_value() > 0:
                            bin_items.append(i)
                            bin_weight += virtualMachines["cores"][i][index]
                            current_mem += virtualMachines["mem"][i][index]
                            current_bw += virtualMachines["bw_in"][i][index] + virtualMachines["bw_out"][i][index]

                            real_weight += realDataList["cores"][i][index]
                            real_mem += realDataList["mem"][i][index]
                            real_bw += realDataList["bw_in"][i][index] + realDataList["bw_out"][i][index]

                    if bin_items:
                        num_bins += 1
                        print('Bin number %i with weight %i' % (j, clusters["cores"][j]))
                        print('  Items packed:', bin_items)
                        print('  Total weight:', bin_weight)
                        print('  Real weight: ', real_weight)
                        pred_load_percentage = bin_weight / clusters["cores"][j] * 100
                        pred_bw_percentage = current_bw / clusters["bandwidth"][j] * 100
                        pred_mem_percentage = current_mem / clusters["mem"][j] * 100

                        real_load_percentage = real_weight / clusters["cores"][j] * 100
                        real_mem_percentage = real_mem / clusters["mem"][j] * 100
                        real_bw_percentage = real_bw / clusters["bandwidth"][j] * 100

                        print('  Predicted load %f vs Real Load %f' % (pred_load_percentage, real_load_percentage))
                        print('  Memory occupied: predicted memory: %f (%f) vs real memory: %f (%f)' %
                              (current_mem, pred_mem_percentage, real_mem, real_mem_percentage))

                        print('  Bandwidth occupied: predicted bandwidth: %f (%f) vs real bandwidth: %f (%f)' %
                              (current_bw, pred_bw_percentage, real_bw, real_bw_percentage))

                        if pred_load_percentage < real_load_percentage:
                            tmp_list.append(real_load_percentage - pred_load_percentage)
                        else:
                            tmp_list.append(0)
                        tmp_work.append((index, round((real_weight / clusters["cores"][j] * 100), 3)))
                        tmp_mem.append((index, round((real_mem / clusters["mem"][j] * 100), 3)))
                        tmp_bw.append((index, round((real_bw / clusters["bandwidth"][j] * 100), 3)))

                        print('  List size: ', len(bin_items))
                        print()
                else:
                    tmp_work.append((index, 0))
                    tmp_mem.append((index, 0))
                    tmp_bw.append((index, 0))
            print()
            print('Number of bins used:', num_bins)
            print('Time = ', solver.WallTime(), ' milliseconds')
            print()
            work_load.append(tmp_work)
            mem_load.append(tmp_mem)
            bw_load.append(tmp_bw)

            list_for_load_overflows.append(tmp_list)
            for i in virtualMachines["all_machines"]:
                for j in clusters["all_clusters"]:
                    my_matrix[i][j] = x[(i, j)].solution_value()
            return my_matrix
        else:
            print('The problem does not have an optimal solution.')
    else:
        print('The problem does not have an optimal solution.')


def load_data():
    clusters["cores"] = []
    clusters["mem"] = []
    clusters["bandwidth"] = []
    columns = ['courses', 'course_fee', 'course_duration', 'course_discount']
    reader = pd.read_csv("host_config_example.csv")
    with open('host_config_example.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                clusters["cores"].append(int(row["cores"]))
                clusters["mem"].append(int(row["mem"]))
                clusters["bandwidth"].append(int(row["bw"]))
                line_count += 1
        clusters["all_clusters"] = range(len(clusters["cores"]))
        print(f'Processed {line_count} lines.')

    virtualMachines["cores"] = []
    virtualMachines["mem"] = []
    virtualMachines["bw_in"] = []
    virtualMachines["bw_out"] = []

    realDataList["cores"] = []
    realDataList["mem"] = []
    realDataList["bw_in"] = []
    realDataList["bw_out"] = []

    folders = ["2013-7", "2013-8", "2013-9"]

    for f in folders:
        list_files_pred = sorted(glob.glob("predictedData/" + f + "/*.csv"), key=len)
        list_files_real = "cleanedData/" + f + "/"

        for file in list_files_pred:
            tmp_cpu, tmp_mem, tmp_bw_in, tmp_bw_out = auxiliar_load_data(file)
            virtualMachines["cores"].append(tmp_cpu)
            virtualMachines["mem"].append(tmp_mem)
            virtualMachines["bw_in"].append(tmp_bw_in)
            virtualMachines["bw_out"].append(tmp_bw_out)

            # Save only real files that are predicted...
            tmp_name = file.split("\\")[-1]
            tmp_cpu, tmp_mem, tmp_bw_in, tmp_bw_out = auxiliar_load_data(list_files_real + tmp_name)
            realDataList["cores"].append(tmp_cpu[-48:])
            realDataList["mem"].append(tmp_mem[-48:])
            realDataList["bw_in"].append(tmp_bw_in[-48:])
            realDataList["bw_out"].append(tmp_bw_out[-48:])

    realDataList["all_machines"] = range(len(realDataList["cores"]))
    virtualMachines["all_machines"] = range(len(virtualMachines["cores"]))


def auxiliar_load_data(current_file):
    tmp_cpu, tmp_mem, tmp_bw_in, tmp_bw_out = [], [], [], []

    with open(current_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            tmp_cpu.append(float(row["cpu"]))
            tmp_mem.append(float(row["mem"]))
            tmp_bw_in.append(float(row["bw_in"]))
            tmp_bw_out.append(float(row["bw_out"]))

    return tmp_cpu, tmp_mem, tmp_bw_in, tmp_bw_out


def transform_csv_files(directory, folder, ini_time_stamp):
    list_files = sorted(glob.glob(directory), key=len)
    vm_config = [["vm_id", "cores", "mem"]]
    my_data = [["time", "cpu", "mem", "bw_in", "bw_out"]]

    initial_time = ini_time_stamp
    cores, cpu_usage, mem_usage, mem_capacity, completed_set, band_width_in, band_width_out = 0, 0, 0, 0, 0, 0, 0
    line = 0
    for element in list_files:
        with open(element, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            header = next(csv_reader)
            for row in csv_reader:
                line += 1
                if completed_set == 12:
                    tmp_time = datetime.datetime.fromtimestamp(initial_time)
                    average_cpu = cpu_usage / (12 * 100) * cores
                    average_mem = (mem_usage / 12) / 1024  # Transform Kb to Mb
                    average_band_in = (band_width_in / 12) * 3600 / 1024  # Kb/s to Mb/H
                    average_band_out = (band_width_out / 12) * 3600 / 1024  # Kb/s to Mb/H
                    my_data.append([tmp_time, average_cpu, average_mem, average_band_in, average_band_out])
                    initial_time += 3600
                    cpu_usage, mem_usage, band_width_in, band_width_out, completed_set = 0, 0, 0, 0, 0

                else:
                    cores = int(row[1][:-1])
                    cpu_usage += float(row[4][:-1])
                    mem_capacity = float(row[5][:-1])
                    mem_usage += float(row[6][:-1])
                    band_width_in += float(row[9][:-1])
                    band_width_out += float(row[10][:-1])
                    completed_set += 1

        if mem_capacity != 0:
            vm_config.append([element[7:-4], cores, mem_capacity])
            file_name = element.split("\\")[-1]
            file_to_save = "cleanedData/" + folder + "/" + file_name
            with open(file_to_save, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Limit the data to 26 days
                csv_writer.writerows(my_data[:625])
        my_data.clear()
        my_data = [["time", "cpu", "mem", "bw_in", "bw_out"]]
        cpu_usage, mem_usage, completed_set = 0, 0, 0
        initial_time = ini_time_stamp

    # vms_folder = "cleanedData/rnd/vm_config_" + directory[:-2] + ".csv"
    # with open(vms_folder, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerows(vm_config)


def apply_prediction_model(directory, folder):
    list_files = sorted(glob.glob(directory + folder + "/*.csv"), key=len)
    star_time = time.time()
    clf = RandomForestRegressor(n_estimators=25, bootstrap=True, random_state=25)

    erroraverageMSE = 0
    erroraverageMAE = 0
    errorMAEcounter = 0
    id_counter = 0
    for element in list_files[:longitud_lista]:
        file_name = element.split("\\")[-1]
        train_raw = pd.read_csv(element)
        X = train_raw.drop(columns=["time"], axis=1)
        y = train_raw[["cpu", "mem", "bw_in", "bw_out"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07692, shuffle=False)

        series = TimeSeries.from_csv(element, "time", ["cpu", "mem", "bw_in", "bw_out"])
        if series.n_timesteps > 623:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracyMSE = mean_squared_error(y_test, y_pred)
            accuracyMAE = mean_absolute_error(y_test, y_pred)
            if 0.0 < accuracyMAE < 100.0:
                erroraverageMAE += accuracyMAE
                errorMAEcounter += 1
            if 0.0 < accuracyMSE < 100.0:
                erroraverageMSE += accuracyMSE
                id_counter += 1
                prediction_file = pd.DataFrame(y_pred, columns=["cpu", "mem", "bw_in", "bw_out"])
                prediction_file.to_csv("predictedData/" + folder + "/" + file_name, index=False)

    print("La media de acierto para (MSE)" + folder + " es: " + str(erroraverageMSE / id_counter))
    print("La media de acierto para (MAE)" + folder + " es: " + str(erroraverageMAE / errorMAEcounter))
    print("La ejecución dura %s seconds" % (time.time() - star_time))


def erase_faulty_models(directory):
    list_files = sorted(glob.glob(directory), key=len)
    for element in list_files:
        df = pd.read_csv(element)
        x, y = df.shape
        for i in range(x):
            if df["cpu"][i] <= 0.0 or df["mem"][i] <= 0.0:
                os.remove(element)
                break


def execute_hourly_plan(migrations):
    my_lista1 = loop_machines_each_hour_cp_model(0, [], -1)
    for hour in range(1, 24):
        my_lista1 = loop_machines_each_hour_cp_model(hour, my_lista1, migrations)
    print_final_data()


def execute_daily_plan(migrations):
    global work_load
    global mem_load
    global bw_load
    previous_day = loop_machines_daily_cp_model([], 0, 24, -1)
    print_final_data()
    work_load.clear()
    work_load = []
    host_work_message = ""
    for elem in range(longitud_lista):
        virtualMachines["cores"][elem] = virtualMachines["cores"][elem][24:]
        virtualMachines["mem"][elem] = virtualMachines["mem"][elem][24:]
        virtualMachines["bw_in"][elem] = virtualMachines["bw_in"][elem][24:]
        virtualMachines["bw_out"][elem] = virtualMachines["bw_out"][elem][24:]

        realDataList["cores"][elem] = realDataList["cores"][elem][24:]
        realDataList["mem"][elem] = realDataList["mem"][elem][24:]
        realDataList["bw_in"][elem] = realDataList["bw_in"][elem][24:]
        realDataList["bw_out"][elem] = realDataList["bw_out"][elem][24:]
    work_load = []
    mem_load = []
    bw_load = []
    loop_machines_daily_cp_model(previous_day, 0, 24, migrations)
    print_final_data()


def select_random_files(directory):
    folder_list = ["2013-7", "2013-8", "2013-9"]
    for f in folder_list:
        dir_list = sorted(glob.glob(directory + "/" + f + "/*"), key=len)
        for files in dir_list:
            tmp_name = files.split("\\")[-1]
            if os.path.isfile("predictedData/" + f + "/" + tmp_name):
                print("El fichero " + tmp_name + " existe en ambos lados.\n")
            else:
                os.remove(files)


def print_final_data():
    average_execution_time = solver.WallTime()
    print("Suma de tiempo total diario: %f\n" % average_execution_time)
    print("Media de tiempo de ejecucion: %f\n" % (average_execution_time / 24))
    for element in range(5):
        host_work_message = "Host " + str(element)
        host_mem_message = "Mem " + str(element)
        host_bw_message = "BW " + str(element)
        for index in range(24):
            host_work_message += str(work_load[index][element])
            host_mem_message += str(mem_load[index][element])
            host_bw_message += str(bw_load[index][element])
        print(host_work_message)
        print()
        print(host_mem_message)
        print()
        print(host_bw_message)
    print("\n")

    average = 0
    for element in range(5):
        tmp_sum = 0
        host_work_message = "Host average" + str(element)
        for index in range(24):
            if ((index + 1) % 6) == 0:
                average = tmp_sum / 6
                tmp_sum = 0
                host_work_message += "(" + str(average) + ")"
            else:
                tmp_sum += work_load[index][element][1]
        print(host_work_message)
    print("\n")

    for element in range(5):
        tmp_sum = 0
        host_work_message = "Host mem average" + str(element)
        for index in range(24):
            if ((index + 1) % 6) == 0:
                average = tmp_sum / 6
                tmp_sum = 0
                host_work_message += "(" + str(average) + ")"
            else:
                tmp_sum += mem_load[index][element][1]
        print(host_work_message)
    print("\n")

    for element in range(5):
        tmp_sum = 0
        host_work_message = "Host bw average" + str(element)
        for index in range(24):
            if ((index + 1) % 6) == 0:
                average = tmp_sum / 6
                tmp_sum = 0
                host_work_message += "(" + str(average) + ")"
            else:
                tmp_sum += bw_load[index][element][1]
        print(host_work_message)
    print("\n")


if __name__ == '__main__':
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)
    np.set_printoptions(linewidth=desired_width)
    apply_prediction_model("cleanedData/", "2013-7")
    apply_prediction_model("cleanedData/", "2013-8")
    apply_prediction_model("cleanedData/", "2013-9")

    load_data()

    option = int(input("Select planning model: 1.Daily    2.Hourly  ->"))
    migrations = int(input("Select migrations (between 0 and 100) ->"))
    while migrations < 0 or migrations > 100:
        migrations = int(input("Invalid value, try again. "))

    migrations = int(len(virtualMachines["all_machines"]) * (migrations / 100))

    if option == 1:
        # Daily planning code:
        execute_daily_plan(migrations)
    elif option == 2:
        # Hourly planning code:
        execute_hourly_plan(migrations)