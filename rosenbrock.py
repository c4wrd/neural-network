def rosenbrock(*args):
    x_total = len(args)
    sum = 0
    for i in range(x_total - 1):
        lhs = (1 - args[i])**2
        rhs = 100*(args[i+1] - args[i]**2)**2
        sum += lhs + rhs
    return sum