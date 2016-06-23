# KCFnb
Solve the 'delay' problem of [KCFpy](https://github.com/uoip/KCFpy) by using the Ahead-of-Time compilation (AOT) facility of Numba.

### Requirements
See [KCFpy](https://github.com/uoip/KCFpy)

### Use
```shell
git clone https://github.com/uoip/KCFnb.git
cd KCFnb
python fhog_utils.py
```
then
```
python run.py
```

### Referances:
1.There is a delay when JIT-compiling a complicated function, how can I improve it? http://numba.pydata.org/numba-doc/dev/user/faq.html#there-is-a-delay-when-jit-compiling-a-complicated-function-how-can-i-improve-it<br>
2.Compiling code ahead of time http://numba.pydata.org/numba-doc/dev/user/pycc.html
