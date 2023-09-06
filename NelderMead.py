import numpy as np
from scipy.optimize import OptimizeResult

def bounded_nelder_mead(fun, x0, bounds, print_progress, maxiter=100, alpha=1.3487640718112321, gamma=2.139744408219208, rho=0.3740185211921033, sigma=0.5501390442296267, tol=1e-4):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0

    for i in range(n):
        point = np.array(x0)
        point[i] = bounds[i][0]
        simplex[i + 1] = point

        point = np.array(x0)
        point[i] = bounds[i][1]
        simplex[i + 1] = point

    fs = np.zeros(n + 1)
    for i in range(n + 1):
        fs[i] = fun(simplex[i])

    res = OptimizeResult()
    res.nfev = n + 1

    for i in range(maxiter):
        # Sort simplex according to function values
        idx = np.argsort(fs)
        simplex = simplex[idx]
        fs = fs[idx]

        # Calculate the centroid of the simplex
        xbar = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = xbar + alpha * (xbar - simplex[-1])
        xr = np.clip(xr, bounds[:, 0], bounds[:, 1])
        fxr = fun(xr)
        res.nfev += 1

        if fs[0] <= fxr < fs[-2]:
            # Successful reflection
            simplex[-1] = xr
            fs[-1] = fxr

        elif fxr < fs[0]:
            # Expansion
            xe = xbar + gamma * (xr - xbar)
            xe = np.clip(xe, bounds[:, 0], bounds[:, 1])
            fxe = fun(xe)
            res.nfev += 1

            if fxe < fxr:
                simplex[-1] = xe
                fs[-1] = fxe
            else:
                simplex[-1] = xr
                fs[-1] = fxr

        else:
            # Contraction
            xc = xbar + rho * (simplex[-1] - xbar)
            xc = np.clip(xc, bounds[:, 0], bounds[:, 1])
            fxc = fun(xc)
            res.nfev += 1

            if fxc < fs[-1]:
                simplex[-1] = xc
                fs[-1] = fxc

            else:
                # Shrink
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                for j in range(1, n + 1):
                    fs[j] = fun(simplex[j])
                    res.nfev += 1
        
        if print_progress ==1:
            print('iteration:', i+1)
            print('best value of objective function so far:', fs[0])
            print('best candidate:', simplex[0])
        # Check termination criteria
        if np.max(np.abs(fs[:-1] - fs[-1])) < tol:
            break

    res.x = simplex[0]
    res.fun = fs[0]
    res.success = True if i < maxiter - 1 else False
    res.message = 'Maximum number of iterations exceeded.' if i == maxiter - 1 else 'Maximum number of iterations not exceeded'
    res.nit = i + 1

    return res
