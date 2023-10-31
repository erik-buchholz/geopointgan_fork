from math import e


# Plotting
def get_sample_points(G, z, n_dim=2):
    """
      Generate 'size' data samples
      ___

      G = Generator (in G.eval() mode)
      z = Random noise tensor of shape (n, 2, 1)
      n = number of points
    """
    z_gen, _ = G(z)
    z_gen = z_gen.permute(2, 0, 1)
    n = z_gen.shape[1]
    batch_size = z_gen.shape[0]
    y_hat = z_gen.reshape(n * batch_size, n_dim)
    points = y_hat.cpu().data.numpy()
    return points


# Probability calculator with privacy budget
def dp_proba(eps, d):
    p = (e ** eps) / (e ** eps + d - 1)
    q = 1 / (e ** eps + d - 1)
    return p, q
