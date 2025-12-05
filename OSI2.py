import multiprocessing as mp
import random
import time


def to_matrix(shared, n):
    """
    Преобразует плоский массив SharedArray длины n*n
    в двумерный список n×n для удобного доступа как к матрице.
    """
    return [shared[i * n:(i + 1) * n] for i in range(n)]


def worker(start_row, end_row, A_shared, B_shared, C_shared, n):
    """
    Вычисляет часть строк результирующей матрицы C.
    
    Каждый процесс получает диапазон строк [start_row, end_row)
    и умножает соответствующие строки A на все столбцы B.
    Результат записывается непосредственно в общий массив C_shared.
    """
    A = to_matrix(A_shared, n)
    B = to_matrix(B_shared, n)
    C = to_matrix(C_shared, n)

    for i in range(start_row, end_row):
        for j in range(n):
            s = 0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s


def multiply_processes(n, p):
    """
    Выполняет умножение двух случайных матриц размера n×n
    с использованием p процессов.

    Создаёт общие массивы для A, B и C, равномерно делит строки
    между процессами, запускает их и замеряет время выполнения.
    Возвращает затраченное время.
    """
    A_shared = mp.Array("i", n * n)
    B_shared = mp.Array("i", n * n)
    C_shared = mp.Array("i", n * n)

    for i in range(n * n):
        A_shared[i] = random.randint(0, 9)
        B_shared[i] = random.randint(0, 9)

    rows_per_proc = n // p
    processes = []

    start_time = time.time()

    for i in range(p):
        start_row = i * rows_per_proc
        end_row = (i + 1) * rows_per_proc if i < p - 1 else n

        proc = mp.Process(
            target=worker,
            args=(start_row, end_row, A_shared, B_shared, C_shared, n)
        )
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    return time.time() - start_time


def run_tests(n, p, attempts):
    """
    Запускает функцию multiply_processes несколько раз,
    выводит время каждого прогона и вычисляет среднее.

    Используется для построения статистики и графиков.
    """
    times = []
    for i in range(attempts):
        elapsed = multiply_processes(n, p)
        print(f"Замер {i + 1}: время = {elapsed:.6f} сек")
        times.append(elapsed)

    avg = sum(times) / attempts
    print(f"\nN = {n}, процессов = {p}, среднее время по {attempts} замерам = {avg:.6f} сек\n")
    return avg
