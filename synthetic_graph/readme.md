Для компиляции библиотеки `graph_cython`, выполните следующие команды в терминале:
```bash
cd synthetic_graph/graph_cython
export PYTHONPATH=$PYTHONPATH:./build/lib 
python setup.py build_ext --inplace
```