def get_problem(name, *args, **kwargs):
    name = name.lower()

    from botorch.test_functions.multi_objective import DTLZ1, DTLZ2, DTLZ3, DTLZ5, DTLZ7, VehicleSafety

    PROBLEM = {
        'vehiclesafety': VehicleSafety,
        'dtlz1': DTLZ1,
        'dtlz2': DTLZ2,
        'dtlz3': DTLZ3,
        'dtlz5': DTLZ5,
        'dtlz7': DTLZ7,
    }

    if name not in PROBLEM:
        raise Exception("Problem not found. List is still incomplete.")

    return PROBLEM[name](*args, **kwargs)
