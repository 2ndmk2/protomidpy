import jax.numpy as jnp

RP = jnp.array([-4.79443220978201773821E9, 1.95617491946556577543E12, -
               2.49248344360967716204E14, 9.70862251047306323952E15])
RQ = jnp.array([1., 4.99563147152651017219E2, 1.73785401676374683123E5, 4.84409658339962045305E7, 1.11855537045356834862E10,
               2.11277520115489217587E12, 3.10518229857422583814E14, 3.18121955943204943306E16, 1.71086294081043136091E18])
DR1 = 5.78318596294678452118E0
DR2 = 3.04712623436620863991E1
PP = jnp.array([7.96936729297347051624E-4, 8.28352392107440799803E-2, 1.23953371646414299388E0,
               5.44725003058768775090E0, 8.74716500199817011941E0, 5.30324038235394892183E0, 9.99999999999999997821E-1])
PQ = jnp.array([9.24408810558863637013E-4, 8.56288474354474431428E-2, 1.25352743901058953537E0,
               5.47097740330417105182E0, 8.76190883237069594232E0, 5.30605288235394617618E0, 1.00000000000000000218E0])
QP = jnp.array([-1.13663838898469149931E-2, -1.28252718670509318512E0, -1.95539544257735972385E1, -9.32060152123768231369E1, -
               1.77681167980488050595E2, -1.47077505154951170175E2, -5.14105326766599330220E1, -6.05014350600728481186E0])
QQ = jnp.array([1., 6.43178256118178023184E1, 8.56430025976980587198E2, 3.88240183605401609683E3,
               7.24046774195652478189E3, 5.93072701187316984827E3, 2.06209331660327847417E3, 2.42005740240291393179E2])
PIO4 = 0.78539816339744830962
SQ2OPI = 0.79788456080286535588


def j0_jax(x):
    """Bessel function of the 1st kind, order=0.
    This is taken from https://github.com/HajimeKawahara/exojax/tree/master/src/exojax/special/j0.py
    This codes uses the implemenation in bessel.tgz at https://www.netlib.org/cephes/ for scipy
    

    Args:
       x: x

    Returns:
       J0
    """
    x = jnp.where(x > 0., x, -x)

    z = x * x
    ret = 1. - z/4.

    p = (z - DR1) * (z - DR2)
    p = p * jnp.polyval(RP, z) / jnp.polyval(RQ, z)
    ret = jnp.where(x < 1e-5, ret, p)

    # required for autograd not to fail when x includes 0
    xinv5 = jnp.where(x <= 5., 0., 1./(x+1e-10))
    w = 5.0 * xinv5
    z = w * w
    p = jnp.polyval(PP, z) / jnp.polyval(PQ, z)
    q = jnp.polyval(QP, z) / jnp.polyval(QQ, z)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    ret = jnp.where(x <= 5., ret, p * SQ2OPI * jnp.sqrt(xinv5))

    return ret