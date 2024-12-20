import numpy as np
import argparse
import timeit
from synthetic_graph.tests.Implementations import Implementations


np.random.seed(33)

def make_test_graph(datasize=1, width=6, length=7):
    nsums = 0
    prevlayer = []

    data = np.random.uniform(-100, 100, size=datasize)
    data_node = Input(data)

    for ilayer in reversed(range(length)):
        ilayer_next = ilayer - 1
        n_groups = int(width ** ilayer_next)
        thislayer = []

        for igroup in range(n_groups):
            head = Sum()
            nsums += 1

            if prevlayer:
                for array in prevlayer:
                    array >> head
            else:
                for isource in range(width):
                    data_node >> head

            thislayer.append(head)

        prevlayer = thislayer

    return nsums, data_node, head

def run_test(head, runs):
    def test():
        head.run()

    average_time = timeit.timeit(test, number=runs) / runs
    print(f"Среднее время выполнения за {runs} прогонов: {average_time:.5e} секунд")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирование графа.")
    parser.add_argument('--width', type=int, default=6, help='Ширина графа.')
    parser.add_argument('--length', type=int, default=5, help='Длина графа.')
    parser.add_argument('--dsize', type=int, default=1, help='Размер данных.')
    parser.add_argument('--runs', type=int, default=10, help='Количество прогонов для замера времени.')
    parser.add_argument(
        '--impl',
        type=str,
        choices=[i.value for i in Implementations],
        default=Implementations.CYTHON.value,
        help='Выберите реализацию: cython, ctypes или python.'
    )
    args = parser.parse_args()

    implementation = Implementations(args.impl)
    if implementation == Implementations.CTYPES:
        from synthetic_graph.graph_ctypes import Sum, Input
    elif implementation == Implementations.PYTHON:
        from synthetic_graph.graph_python import Sum, Input
    elif implementation == Implementations.PYTHON_CTYPES:
        from synthetic_graph.graph_python_ctypes import Sum, Input
    elif implementation == Implementations.CYTHON:
        from synthetic_graph.graph_cython import Sum, Input

    nsums, data_node, head = make_test_graph(datasize=args.dsize, width=args.width, length=args.length)
    print(f"Создано узлов: {nsums}")

    if implementation != Implementations.PYTHON:
        print("Перевод классов в си структуры...")
        head.to_c_struct()

    print("Запуск...")
    run_test(head, args.runs)