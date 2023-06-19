import csv

from darts import TimeSeries
import glob
from darts.models import ExponentialSmoothing
from darts.models import RegressionModel
import darts.metrics as metrics
from sklearn import linear_model
import datetime
import numpy as np
from ortools.linear_solver import pywraplp
from pathlib import Path
import pandas as pd
import time

# Variables globales
average_execution_time = 0
clusters = {}
virtualMachines = {}
predictionsList = []
realDataList = []
solver = pywraplp.Solver.CreateSolver('SCIP')
solver2 = pywraplp.Solver.CreateSolver('SCIP')
work_load = []
longitud_lista = 150
list_for_load_overflows = []


def loop_machines_daily(previous_day, ini, end, distance, option):
    x = {}
    y = {}
    z = {}
    z_abs = {}
    my_matrix = np.zeros((len(virtualMachines["all_machines"]), len(clusters["all_clusters"])))
    tmp_work_load = []
    host_load = {}
    max_load = {}
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
            solver.Add(host_load[(hr, j)] <= (y[j] * clusters["cores"][j]))
    for hr in range(ini, end):
        for j in clusters["all_clusters"]:
            solver.Add(max_load[hr] >= 0)
            solver.Add(max_load[hr] >= host_load[(hr, j)])
            solver.Add(max_load[hr] >= -(host_load[(hr, j)]))

        for hr in range(ini, end):
            solver.Add(absolute_max >= 0)
            solver.Add(absolute_max >= max_load[hr])
            solver.Add(absolute_max >= -(max_load[hr]))

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
    if option == 1:
        solver.Minimize(solver.Sum([y[j] for j in clusters["all_clusters"]]))
    else:
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
                    for i in virtualMachines["all_machines"]:
                        if x[i, j].solution_value() > 0:
                            bin_items.append(i)
                            bin_weight += virtualMachines["cores"][i][hr]
                            real_weight += realDataList[i][hr]
                    if bin_items:
                        num_bins += 1
                        print('Bin number %i with weight %i' % (j, clusters["cores"][j]))
                        print('  Items packed:', bin_items)
                        print('  Total weight:', bin_weight)
                        print('  Real weight: ', real_weight)
                        pred_load_percentage = bin_weight / 64 * 100
                        real_load_percentage = real_weight / 64 * 100
                        print('  Predicted load %f vs Real Load %f' % (pred_load_percentage, real_load_percentage))
                        if pred_load_percentage < real_load_percentage:
                            tmp_list.append(real_load_percentage - pred_load_percentage)
                        else:
                            tmp_list.append(0)
                        tmp_work_load.append((hr, round(real_load_percentage, 3)))
                        print('  List size: ', len(bin_items))
                        print()
                else:
                    tmp_work_load.append((hr, 0))
            print()
            print('Number of bins used:', num_bins / 24)
            print('Time = ', solver.WallTime(), ' milliseconds')
            work_load.append(tmp_work_load)
            list_for_load_overflows.append(tmp_list)
            tmp_work_load = []
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                my_matrix[i][j] = x[(i, j)].solution_value()
        return my_matrix
    else:
        print('The problem does not have an optimal solution.')


def loop_machines_daily_cp_model(previous_day, ini, end, distance):
    x = {}
    y = {}
    z = {}
    z_abs = {}
    my_matrix = np.zeros((len(virtualMachines["all_machines"]), len(clusters["all_clusters"])))
    tmp_work_load = []
    host_load = {}
    max_load = {}
    absolute_max = 0
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
            solver.Add(host_load[(hr, j)] <= (y[j] * clusters["cores"][j]))
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
    # solver.Minimize(absolute_max)
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
                        for i in virtualMachines["all_machines"]:
                            if x[i, j].solution_value() > 0:
                                bin_items.append(i)
                                bin_weight += virtualMachines["cores"][i][hr]
                                real_weight += realDataList[i][hr]
                        if bin_items:
                            num_bins += 1
                            print('Bin number %i with weight %i' % (j, clusters["cores"][j]))
                            print('  Items packed:', bin_items)
                            print('  Total weight:', bin_weight)
                            print('  Real weight: ', real_weight)
                            pred_load_percentage = bin_weight / 64 * 100
                            real_load_percentage = real_weight / 64 * 100
                            print('  Predicted load %f vs Real Load %f' % (pred_load_percentage, real_load_percentage))
                            if pred_load_percentage < real_load_percentage:
                                tmp_list.append(real_load_percentage - pred_load_percentage)
                            else:
                                tmp_list.append(0)
                            tmp_work_load.append((hr, round(real_load_percentage, 5)))
                            print('  List size: ', len(bin_items))
                            print()
                    else:
                        tmp_work_load.append((hr, 0))
                print()
                print('Number of bins used:', num_bins / 24)
                print('Time = ', solver.WallTime(), ' milliseconds')
                work_load.append(tmp_work_load)
                list_for_load_overflows.append(tmp_list)
                tmp_work_load = []
            for i in virtualMachines["all_machines"]:
                for j in clusters["all_clusters"]:
                    my_matrix[i][j] = x[(i, j)].solution_value()
            return my_matrix
        else:
            print('The problem does not have an optimal solution.')
    else:
        print('The problem does not have an optimal solution.')


def loop_machines_each_hour(index, previous_list, distance, option):
    # Each item is assigned to at most one bin.
    x = {}
    tmp_work = []
    my_matrix = np.zeros((len(virtualMachines["all_machines"]), len(clusters["all_clusters"])))
    z = {}
    z_abs = {}
    host_load = {}
    cores_req = {}
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
                       for i in virtualMachines["all_machines"]) <= (y[j] * clusters["cores"][j]))
        solver.Add(host_load[j] <= (y[j] * clusters["cores"][j]))

    if distance >= 0:
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                z[i, j] = x[i, j] - previous_list[i][j]
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                solver.Add(z_abs[i, j] >= 0)
                solver.Add(z_abs[i, j] >= z[i, j])
                solver.Add(z_abs[i, j] >= -(z[i, j]))

        solver.Add(sum([z_abs[i, j] for i in virtualMachines["all_machines"] for j in
                        clusters["all_clusters"]]) / 2 <= distance)
    else:
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                z_abs[(i, j)] = x[(i, j)]

    for j in clusters["all_clusters"]:
        solver.Add(max_load >= 0)
        solver.Add(max_load >= host_load[j])
        solver.Add(max_load >= -(host_load[j]))

    if option == 1:
        solver.Minimize(solver.Sum([y[j] for j in clusters["all_clusters"]]))
    # status = solver.Solve()
    # min_cores = sum(y[j].solution_value() for j in clusters["all_clusters"])
    # solver.Add(sum(y[j] for j in clusters["all_clusters"]) >= min_cores)
    else:
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
                real_weight = 0
                for i in virtualMachines["all_machines"]:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        bin_weight += virtualMachines["cores"][i][index]
                        real_weight += realDataList[i][index]
                if bin_items:
                    num_bins += 1
                    print('Bin number %i with weight %i' % (j, clusters["cores"][j]))
                    print('  Items packed:', bin_items)
                    print('  Total weight:', bin_weight)
                    print('  Real weight: ', real_weight)
                    pred_load_percentage = bin_weight / 64 * 100
                    real_load_percentage = real_weight / 64 * 100
                    print('  Predicted load %f vs Real Load %f' % (pred_load_percentage, real_load_percentage))
                    if pred_load_percentage < real_load_percentage:
                        tmp_list.append(real_load_percentage - pred_load_percentage)
                    else:
                        tmp_list.append(0)
                    tmp_work.append((index, round((real_weight / 64 * 100), 3)))
                    print('  List size: ', len(bin_items))
                    print()
            else:
                tmp_work.append((index, 0))
        print()
        print('Number of bins used:', num_bins / 24)
        print('Time = ', solver.WallTime(), ' milliseconds')
        work_load.append(tmp_work)
        list_for_load_overflows.append(tmp_list)
        for i in virtualMachines["all_machines"]:
            for j in clusters["all_clusters"]:
                my_matrix[i][j] = x[(i, j)].solution_value()
        return my_matrix
    else:
        print('The problem does not have an optimal solution.')


def loop_machines_each_hour_cp_model(index, previous_list, distance):
    # Each item is assigned to at most one bin.
    x = {}
    tmp_work = []
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
        # solver.Add(host_load[j] <= (y[j] * clusters["cores"][j]))

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

        # solver.Add(solver.IfThen(delay_abs[i] <= 1, hot_migrations[i]) for i in virtualMachines["all_machines"])
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
                    real_weight = 0
                    for i in virtualMachines["all_machines"]:
                        if x[i, j].solution_value() > 0:
                            bin_items.append(i)
                            bin_weight += virtualMachines["cores"][i][index]
                            real_weight += realDataList[i][index]
                    if bin_items:
                        num_bins += 1
                        print('Bin number %i with weight %i' % (j, clusters["cores"][j]))
                        print('  Items packed:', bin_items)
                        print('  Total weight:', bin_weight)
                        print('  Real weight: ', real_weight)
                        pred_load_percentage = bin_weight / 64 * 100
                        real_load_percentage = real_weight / 64 * 100
                        print('  Predicted load %f vs Real Load %f' % (pred_load_percentage, real_load_percentage))
                        if pred_load_percentage < real_load_percentage:
                            tmp_list.append(real_load_percentage - pred_load_percentage)
                        else:
                            tmp_list.append(0)
                        tmp_work.append((index, round((real_weight / 64 * 100), 3)))
                        print('  List size: ', len(bin_items))
                        print()
                else:
                    tmp_work.append((index, 0))
            print()
            print('Number of bins used:', num_bins / 24)
            print('Time = ', solver.WallTime(), ' milliseconds')
            print()
            work_load.append(tmp_work)
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
                line_count += 1
        clusters["all_clusters"] = range(len(clusters["cores"]))
        print(f'Processed {line_count} lines.')

    virtualMachines["cores"] = []
    virtualMachines["mem"] = []
    with open('vm_config.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            virtualMachines["cores"].append(int(row["cores"]))
            virtualMachines["mem"].append(int(row["mem"]))
        virtualMachines["all_machines"] = range(len(virtualMachines["cores"]))

    list_files = sorted(glob.glob("predictions/*.csv"), key=len)

    for i in range(len(list_files[:longitud_lista])):
        tmp_prediction = []
        current_cores = virtualMachines["cores"][i]
        with open(list_files[i], mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                tmp_prediction.append(float(row["cpu"]) * current_cores)
        predictionsList.append(tmp_prediction)

    list_files = sorted(glob.glob("trace_files/*.csv"))
    for i in range(len(list_files[:longitud_lista])):
        current_cores = virtualMachines["cores"][i]
        tmp_list = []
        with open(list_files[i], mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                tmp_list.append(float(row["cpu"]) * current_cores)
        realDataList.append(tmp_list[-48:])


def transform_csv_files(directory, ini_time_stamp):
    list_files = sorted(glob.glob(directory), key=len)
    vm_config = [["vm_id", "cores", "mem"]]
    my_data = [["time", "cpu"]]

    initial_time = ini_time_stamp
    cores, cpu_usage, mem_usage, mem_capacity, completed_set, = 0, 0, 0, 0, 0
    line = 0
    for element in list_files:
        with open(element, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            header = next(csv_reader)
            for row in csv_reader:
                line += 1
                if completed_set == 12:
                    my_data.append([datetime.datetime.fromtimestamp(initial_time), cpu_usage / (12 * 100)])
                    initial_time += 3600
                    cpu_usage, mem_usage, completed_set = 0, 0, 0

                else:
                    cores = int(row[1][:-1])
                    cpu_usage += float(row[4][:-1])
                    mem_capacity = float(row[5][:-1])
                    mem_usage += float(row[6][:-1])
                    completed_set += 1
        if mem_capacity != 0:
            vm_config.append([element[7:-4], cores, mem_capacity])
            file_to_save = "bitbrainsData/" + element
            with open(file_to_save, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(my_data[:625])
        my_data.clear()
        my_data = [["time", "cpu"]]
        cpu_usage, mem_usage, completed_set = 0, 0, 0
        initial_time = ini_time_stamp

    vms_folder = "bitbrainsData/vm_config_" + directory[:-2] + ".csv"
    with open(vms_folder, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(vm_config)


def apply_prediction_model(directory):
    list_files = sorted(glob.glob(directory), key=len)

    bayesian_model = linear_model.BayesianRidge()
    gaussian = linear_model.ARDRegression()
    exp_smooth = ExponentialSmoothing()

    error_average = 0
    id_counter = 0
    ini_time = time.time()
    for element in list_files[:longitud_lista]:
        if Path(element).stat().st_size >= 2000:
            series = TimeSeries.from_csv(element, "time", "cpu")
            train, val = series[:-72], series[-72:-24]
            model_regression = RegressionModel(lags=48, output_chunk_length=48)
            model_regression.fit(train)
            prediction_linear_regression = model_regression.predict(n=48)
            final_error = metrics.rmse(val, prediction_linear_regression)
            error_average += final_error
            prediction_linear_regression.to_csv("predictions/" + str(id_counter) + ".csv")
            id_counter += 1
    print("Error accuracy: " + str(1 - error_average / len(list_files)))
    print("Execution time: " + str(time.time() - ini_time))


def execute_hourly_plan(option, migrations):
    if option == 3:
        my_lista1 = loop_machines_each_hour_cp_model(0, [], -1)
        for hour in range(1, 24):
            my_lista1 = loop_machines_each_hour_cp_model(hour, my_lista1, migrations)
    else:
        my_lista1 = loop_machines_each_hour(0, [], -1, option)
        for hour in range(1, 24):
            my_lista1 = loop_machines_each_hour(hour, my_lista1, migrations, option)
    print_final_data()


def execute_daily_plan(option, migrations):
    global work_load
    if option == 3:
        previous_day = loop_machines_daily_cp_model([], 0, 24, -1)
        print_final_data()
        work_load.clear()
        work_load = []
        host_work_message = ""
        for elem in range(longitud_lista):
            virtualMachines["cores"][elem] = virtualMachines["cores"][elem][24:]
            realDataList[elem] = realDataList[elem][24:]
        loop_machines_daily_cp_model(previous_day, 0, 24, migrations)
        print_final_data()
    else:
        previous_day = loop_machines_daily([], 0, 24, -1, option)
        print_final_data()
        work_load.clear()
        work_load = []
        host_work_message = ""
        for elem in range(longitud_lista):
            virtualMachines["cores"][elem] = virtualMachines["cores"][elem][24:]
            realDataList[elem] = realDataList[elem][24:]
        loop_machines_daily(previous_day, 0, 24, migrations, option)
    print_final_data()


def print_final_data():
    average_execution_time = solver.WallTime()
    print("Suma de tiempo total diario: %f\n" % average_execution_time)
    print("Media de tiempo de ejecucion: %f\n" % (average_execution_time / 24))
    for element in range(4):
        host_work_message = "Host " + str(element)
        for index in range(24):
            host_work_message += str(work_load[index][element])
        print(host_work_message)

    average = 0
    for element in range(4):
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


if __name__ == '__main__':

    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)
    np.set_printoptions(linewidth=desired_width)

    load_data()
    # #
    virtualMachines["cores"] = virtualMachines["cores"][:longitud_lista]
    virtualMachines["mem"] = virtualMachines["mem"][:longitud_lista]
    virtualMachines["all_machines"] = range(len(virtualMachines["cores"]))
    #
    clusters["cores"] = clusters["cores"][:4]
    clusters["mem"] = clusters["mem"][:4]
    clusters["all_clusters"] = range(len(clusters["cores"]))
    predictions_list = predictionsList[:longitud_lista]
    virtualMachines["cores"] = predictionsList
    migrations = -2
    planning = int(input("Enter planning model: \n"
                     "1.Daily   2.Hourly\n"
                     "Model -> "))
    option = int(input("Enter model option: 1.Minimize use of hosts.   2.Minimize host load.   3.Minimize use of hosts "
                      "and hosts load. ->"))
    migrations = int(input("Enter migrations percentage (between 0 and 100)"))
    while migrations > 100 or migrations < 0:
        migrations = int(input("Invalid value, please try again."))

    migrations /= 100

    if solver is None:
        print('SCIP solver unavailable.')

    if planning == 1:
        # Daily planning code:
        execute_daily_plan(option, migrations * longitud_lista)
    elif planning == 2:
        # Hourly planning code:
        execute_hourly_plan(option, migrations * longitud_lista)
