import jax
import jax.numpy as jnp
import jax.lax as lax

N = 64

# FIXME: I actually need to set these values
dx = 0.1
dy = 0.1
g = 1
mu = 1
A = 1

# FIXME: Z isnt needed in this function.  I need to fix how location is computed
def reisz(z, w):
    # Shape: (N, N, 2)
    c1 = lax.complex(w[:, :, 0], jnp.zeros((N, N), dtype=dtype))
    c2 = lax.complex(w[:, :, 1], jnp.zeros((N, N), dtype=dtype))

    # Shape: (N, N) --> but these are complex
    c1_fft = jnp.fft.fft2(c1)
    c2_fft = jnp.fft.fft2(c2)

    f_k = lambda i: jnp.where(
        not N % 2,
        jnp.where(i < 0, N/2 + i, i),
        jnp.where(i <= 0, (N - 1)/2 + i, i - (N - 1)/2 - 1)
    )

    # Shape: (N, N, 2)
    k = jax.vmap(lambda i: jax.vmap(lambda j: jnp.array([f_k(z[i,j,0]), f_k(z[i,j,1])]))(jnp.arange(N)))(jnp.arange(N))

    # Shape: (N, N)
    length = jnp.sqrt(k[:,:,0]*k[:,:,0] + k[:,:,1]*k[:,:,1])

    # Shape: (N, N, 2) --> have to make length (N, N, 1) for this to work
    M = jnp.divide(k, length[:,:,None])

    f_reisz_fft_real = lambda i, j: jnp.where(
        jnp.logical_not(jnp.logical_or(k[i,j,0], k[i,j,1])),
        0.0, 
        M[i,j,0]*jnp.imag(c1_fft[i,j])+M[i,j,1]*jnp.imag(c2_fft[i,j]))
    
    f_reisz_fft_imag = lambda i, j: jnp.where(
        jnp.logical_not(jnp.logical_or(k[i,j,0], k[i,j,1])),
        0.0, 
        -M[i,j,0]*jnp.real(c1_fft[i,j])-M[i,j,1]*jnp.real(c2_fft[i,j]))

    reisz_fft_real = jax.vmap(lambda i: jax.vmap(lambda j: f_reisz_fft_real(i,j))(jnp.arange(N)))(jnp.arange(N))
    reisz_fft_imag = jax.vmap(lambda i: jax.vmap(lambda j: f_reisz_fft_imag(i,j))(jnp.arange(N)))(jnp.arange(N))
    reisz_fft = lax.complex(reisz_fft_real, reisz_fft_imag)

    reisz = jnp.fft.ifft2(reisz_fft)

    return jnp.stack([jnp.real(reisz), jnp.imag(reisz)], axis=-1)


# Compute the directional derivative using 4th order central differencing
def dd(u, h, axis=0):
    return (-jnp.roll(u, -2, axis=axis) + 8*jnp.roll(u, -1, axis=axis)
            - 8*jnp.roll(u, 1, axis=axis) + jnp.roll(u, 2, axis=axis)) / (12*h)


kernel = jnp.array([
    [1, 4, 1],
    [4, -20, 4],
    [1, 4, 1]
], dtype=jnp.float32) / 6.0


# Apply 9 point stencil
def laplace(u, dx, dy):
    # Add periodic boundary padding of one element. H, W --> H+2, W+2
    padded = jnp.pad(u, ((1, 1), (1, 1)), mode="wrap")

    # batch, H+2, W+2, channels
    padded = padded[None, ..., None]

    # filter height, filter width, channel in, channel out
    k = kernel[..., None, None] / (dx*dy)

    out = lax.conv_general_dilated(
        padded, k,
        window_strides=(1,1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )

    # H, W
    return out[0, ..., 0]


def compute_haloed_derivatives(z, w, reisz):
    # FIXME: I need to think about the way this works more
    Dx_z = dd(z, dx, axis=0)
    Dy_z = dd(z, dy, axis=1)

    # Shape: (N, N)
    h11 = jnp.sum(Dx_z * Dx_z, axis=-1)
    h12 = jnp.sum(Dx_z * Dy_z, axis=-1)
    h22 = jnp.sum(Dy_z * Dy_z, axis=-1)
    deth = h11*h22 - h12*h12

    N = jnp.divide(jnp.cross(Dx_z, Dy_z), jnp.sqrt(deth)[:,:,None])

    # finalize_velocity
    zndot = jnp.divide(-0.5 * reisz[:,:,0], deth)

    zdot = zndot[:,:,None] * N

    V = (zndot * zndot 
         - 0.25*(h22*w[:,:,0]*w[:,:,0] - 2.0*h12*w[:,:,0]*w[:,:,1] + h11*w[:,:,1]*w[:,:,1]) / deth
         - 2*g*z[:,:,2])

    # Interface vorticity
    Dx_v = dd(V, dx, axis=0)
    Dy_v = dd(V, dy, axis=1)

    lap_w0 = laplace(w[...,0], dx, dy)
    lap_w1 = laplace(w[...,1], dx, dy)

    wdot0 = A * Dx_v + mu * lap_w0
    wdot1 = A * Dy_v + mu * lap_w1

    wdot = jnp.stack([wdot0, wdot1], axis=-1)

    return (zdot, wdot)

# TODO: Shape annotations + comments to aid explainability!

# https://github.com/CUP-ECS/beatnik/blob/main/src/ZModel.hpp
if __name__ == '__main__':
    dtype = jnp.float32
    init = jnp.reshape(jnp.arange(0, N*N, dtype=dtype), (N, N))

    z = jnp.stack([init, init, init], axis=-1)  # Shape: (N, N, 3)
    w = jnp.stack([init, init], axis=-1)        # Shape: (N, N, 2)

    reisz = reisz(z, w)

    print(reisz.shape)

    zdot, wdot = compute_haloed_derivatives(z, w, reisz)

    print(zdot.shape)
    print(wdot.shape)

    # FIXME: Sort out the RK3 steps
