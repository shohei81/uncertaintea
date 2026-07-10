# Device-safe, T-generic scalar log-densities used by the fused KernelAbstractions
# logjoint kernel. Every function here must be callable inside a GPU kernel:
#   * no exceptions / no error paths (out-of-support returns -Inf through ifelse),
#   * no allocations,
#   * no dependency on SpecialFunctions (a pure-Julia Lanczos loggamma is provided).
# Non-finite parameters (e.g. sigma <= 0) propagate as NaN naturally, matching the
# "garbage in -> NaN out" contract; the authoritative CPU backend still validates.

# Lanczos approximation coefficients (g = 7, n = 9). Stored as Float64 and cast to
# the working precision T on demand.
const _DEVICE_LANCZOS_G = 7.0
const _DEVICE_LANCZOS_COEFS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)

"""
    _device_loggamma(x::T) -> T

Pure-Julia log-gamma via the Lanczos approximation, valid on device. Uses the
reflection formula for `x < 0.5`. Coefficients are Float64 constants cast to `T`.
"""
@inline function _device_loggamma(x::T) where {T}
    if x < T(0.5)
        # reflection: loggamma(x) = log(pi / sin(pi x)) - loggamma(1 - x)
        return T(log(pi)) - log(abs(sin(T(pi) * x))) - _device_loggamma(one(T) - x)
    end
    xm1 = x - one(T)
    a = T(_DEVICE_LANCZOS_COEFS[1])
    t = xm1 + T(_DEVICE_LANCZOS_G) + T(0.5)
    a += T(_DEVICE_LANCZOS_COEFS[2]) / (xm1 + one(T))
    a += T(_DEVICE_LANCZOS_COEFS[3]) / (xm1 + T(2))
    a += T(_DEVICE_LANCZOS_COEFS[4]) / (xm1 + T(3))
    a += T(_DEVICE_LANCZOS_COEFS[5]) / (xm1 + T(4))
    a += T(_DEVICE_LANCZOS_COEFS[6]) / (xm1 + T(5))
    a += T(_DEVICE_LANCZOS_COEFS[7]) / (xm1 + T(6))
    a += T(_DEVICE_LANCZOS_COEFS[8]) / (xm1 + T(7))
    a += T(_DEVICE_LANCZOS_COEFS[9]) / (xm1 + T(8))
    half_log_2pi = T(0.9189385332046727) # 0.5*log(2*pi)
    return half_log_2pi + (xm1 + T(0.5)) * log(t) - t + log(a)
end

@inline _device_neginf(::Type{T}) where {T} = T(-Inf)

# log Gamma(x + 1/2) - log Gamma(x), computed WITHOUT differencing two large
# loggammas: at large x each loggamma is ~ x log x, so their O(log x) difference
# drowns in rounding (0.06 absolute at Float32 for x ~ 5e4, visible in any
# large-nu Student-t density or CDF). Below the precision-dependent crossover
# the direct difference is more accurate; above it, the Stirling-series form
#   x log1p(1/(2x)) + log(x)/2 - 1/2 + (1/(x+1/2) - 1/x)/12
# carries relative error ~1/(360 x^3).
@inline _device_loggamma_half_ratio_crossover(::Type{Float32}) = 12.0f0
@inline _device_loggamma_half_ratio_crossover(::Type{T}) where {T} = T(1200)
@inline _device_loggamma_half_ratio_crossover(::Type{DeviceDual{T}}) where {T} =
    DeviceDual{T}(_device_loggamma_half_ratio_crossover(T), zero(T))

@inline function _device_loggamma_half_ratio(x::T) where {T}
    if x > _device_loggamma_half_ratio_crossover(T)
        return x * log1p(one(T) / (T(2) * x)) + log(x) / T(2) - T(0.5) +
               (one(T) / (x + T(0.5)) - one(T) / x) / T(12)
    end
    return _device_loggamma(x + T(0.5)) - _device_loggamma(x)
end

@inline function _device_normal_logpdf(mu::T, sigma::T, x::T) where {T}
    z = (x - mu) / sigma
    return -log(sigma) - T(0.9189385332046727) - z * z / T(2)
end

@inline function _device_lognormal_logpdf(mu::T, sigma::T, x::T) where {T}
    lx = log(x)
    base = -log(sigma) - T(0.9189385332046727) - ((lx - mu) / sigma)^2 / T(2) - lx
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_exponential_logpdf(rate::T, x::T) where {T}
    return ifelse(x >= zero(T), log(rate) - rate * x, _device_neginf(T))
end

@inline function _device_gamma_logpdf(shape::T, rate::T, x::T) where {T}
    base = shape * log(rate) - _device_loggamma(shape) +
           (shape - one(T)) * log(x) - rate * x
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_laplace_logpdf(loc::T, scale::T, x::T) where {T}
    return -log(T(2) * scale) - abs(x - loc) / scale
end

@inline function _device_beta_logpdf(alpha::T, beta::T, x::T) where {T}
    logbeta = _device_loggamma(alpha) + _device_loggamma(beta) - _device_loggamma(alpha + beta)
    base = (alpha - one(T)) * log(x) + (beta - one(T)) * log1p(-x) - logbeta
    return ifelse((x > zero(T)) & (x < one(T)), base, _device_neginf(T))
end

@inline function _device_studentt_logpdf(nu::T, mu::T, sigma::T, x::T) where {T}
    z = (x - mu) / sigma
    # loggamma((nu+1)/2) - loggamma(nu/2) via the half-ratio (large-nu safe)
    return _device_loggamma_half_ratio(nu / T(2)) -
           (log(nu) + T(1.1447298858494002)) / T(2) - log(sigma) - # log(pi)
           (nu + one(T)) * log1p(z * z / nu) / T(2)
end

@inline function _device_inversegamma_logpdf(shape::T, scale::T, x::T) where {T}
    base = shape * log(scale) - _device_loggamma(shape) -
           (shape + one(T)) * log(x) - scale / x
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_weibull_logpdf(shape::T, scale::T, x::T) where {T}
    base = log(shape) + (shape - one(T)) * log(x) -
           shape * log(scale) - exp(shape * (log(x) - log(scale)))
    at_zero = ifelse(shape < one(T), T(Inf), ifelse(shape == one(T), -log(scale), _device_neginf(T)))
    return ifelse(x > zero(T), base, ifelse(x == zero(T), at_zero, _device_neginf(T)))
end

@inline function _device_bernoulli_logpdf(p::T, x::T) where {T}
    return ifelse(x > T(0.5), log(p), log1p(-p))
end

# Nonnegative exact-integer support check shared by the count families; mirrors the
# CPU `_poisson_count` acceptance (staged integer counts are exact in Float32/64).
@inline _device_count_ok(x::T, k::T) where {T} = (x >= zero(T)) & (x == k)

@inline function _device_poisson_logpdf(lambda::T, x::T) where {T}
    k = round(x)
    base = k * log(lambda) - lambda - _device_loggamma(k + one(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

@inline function _device_binomial_logpdf(trials::T, p::T, x::T) where {T}
    n = round(trials)
    k = round(x)
    in_support = _device_count_ok(x, k) & _device_count_ok(trials, n) & (k <= n)
    log_combination = _device_loggamma(n + one(T)) - _device_loggamma(k + one(T)) -
                      _device_loggamma(n - k + one(T))
    # guard the k == 0 / k == n corners so p in {0, 1} cannot produce 0 * -Inf
    base = log_combination + ifelse(k > zero(T), k * log(p), zero(T)) +
           ifelse(k < n, (n - k) * log1p(-p), zero(T))
    return ifelse(in_support, base, _device_neginf(T))
end

@inline function _device_geometric_logpdf(p::T, x::T) where {T}
    k = round(x)
    base = log(p) + ifelse(k > zero(T), k * log1p(-p), zero(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

@inline function _device_negativebinomial_logpdf(successes::T, p::T, x::T) where {T}
    k = round(x)
    base =
        _device_loggamma(k + successes) - _device_loggamma(successes) -
        _device_loggamma(k + one(T)) + successes * log(p) +
        ifelse(k > zero(T), k * log1p(-p), zero(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

# Compile-time unrolled selection of log(probabilities[k]); unmatched entries
# contribute an exact zero so heterogeneous (literal/dual) tuples promote safely.
@inline _device_categorical_pick(::Tuple{}, k, index::Int32, acc) = acc
@inline function _device_categorical_pick(probabilities::Tuple, k, index::Int32, acc)
    logp = log(first(probabilities))
    term = ifelse(k == index, logp, zero(logp))
    return _device_categorical_pick(Base.tail(probabilities), k, index + Int32(1), acc + term)
end

@inline function _device_categorical_logpdf(probabilities::Tuple, x::T) where {T}
    k = round(x)
    in_support = (x >= one(T)) & (x <= T(length(probabilities))) & (x == k)
    total = _device_categorical_pick(probabilities, k, Int32(1), zero(T))
    return ifelse(in_support, total, oftype(total, -Inf))
end

# ---- erf / erfc (W. J. Cody, SPECFUN CALERF rational approximations) ------------
# Three-interval rational forms, |relative error| ~ 1e-16 at Float64, built from
# branches and polynomials only (device-safe). The DeviceDual overloads below
# differentiate through the ANALYTIC derivative (2/sqrt(pi) * exp(-x^2)), not the
# approximation, so device gradients match the CPU analytic-gradient formulas.

const _DEVICE_ERF_A = (
    3.16112374387056560e0,
    1.13864154151050156e2,
    3.77485237685302021e2,
    3.20937758913846947e3,
    1.85777706184603153e-1,
)
const _DEVICE_ERF_B = (2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3, 2.84423683343917062e3)
const _DEVICE_ERF_C = (
    5.64188496988670089e-1,
    8.88314979438837594e0,
    6.61191906371416295e1,
    2.98635138197400131e2,
    8.81952221241769090e2,
    1.71204761263407058e3,
    2.05107837782607147e3,
    1.23033935479799725e3,
    2.15311535474403846e-8,
)
const _DEVICE_ERF_D = (
    1.57449261107098347e1,
    1.17693950891312499e2,
    5.37181101862009858e2,
    1.62138957456669019e3,
    3.29079923573345963e3,
    4.36261909014324716e3,
    3.43936767414372164e3,
    1.23033935480374942e3,
)
const _DEVICE_ERF_P = (
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
)
const _DEVICE_ERF_Q = (
    2.56852019228982242e0,
    1.87295284992346047e0,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
)

# erfc(y) * exp(y^2) for y > 0.46875 (the two outer Cody intervals), without the
# final exp(-y^2) scaling.
@inline function _device_erfcx_tail(y::T) where {T}
    if y <= T(4)
        xnum = T(_DEVICE_ERF_C[9]) * y
        xden = y
        xnum = (xnum + T(_DEVICE_ERF_C[1])) * y
        xden = (xden + T(_DEVICE_ERF_D[1])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[2])) * y
        xden = (xden + T(_DEVICE_ERF_D[2])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[3])) * y
        xden = (xden + T(_DEVICE_ERF_D[3])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[4])) * y
        xden = (xden + T(_DEVICE_ERF_D[4])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[5])) * y
        xden = (xden + T(_DEVICE_ERF_D[5])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[6])) * y
        xden = (xden + T(_DEVICE_ERF_D[6])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[7])) * y
        xden = (xden + T(_DEVICE_ERF_D[7])) * y
        return (xnum + T(_DEVICE_ERF_C[8])) / (xden + T(_DEVICE_ERF_D[8]))
    end
    inv_y2 = one(T) / (y * y)
    xnum = T(_DEVICE_ERF_P[6]) * inv_y2
    xden = inv_y2
    xnum = (xnum + T(_DEVICE_ERF_P[1])) * inv_y2
    xden = (xden + T(_DEVICE_ERF_Q[1])) * inv_y2
    xnum = (xnum + T(_DEVICE_ERF_P[2])) * inv_y2
    xden = (xden + T(_DEVICE_ERF_Q[2])) * inv_y2
    xnum = (xnum + T(_DEVICE_ERF_P[3])) * inv_y2
    xden = (xden + T(_DEVICE_ERF_Q[3])) * inv_y2
    xnum = (xnum + T(_DEVICE_ERF_P[4])) * inv_y2
    xden = (xden + T(_DEVICE_ERF_Q[4])) * inv_y2
    result = inv_y2 * (xnum + T(_DEVICE_ERF_P[5])) / (xden + T(_DEVICE_ERF_Q[5]))
    return (T(5.6418958354775628695e-1) - result) / y # 1/sqrt(pi)
end

@inline function _device_erfc_positive(y::T) where {T}
    # split exp(-y^2) as exp(-ysq^2) * exp(-del) with ysq = trunc(16y)/16 (Cody)
    ysq = trunc(y * T(16)) / T(16)
    del = (y - ysq) * (y + ysq)
    return exp(-ysq * ysq) * exp(-del) * _device_erfcx_tail(y)
end

@inline function _device_erf(x::T) where {T}
    y = abs(x)
    if y <= T(0.46875)
        ysq = ifelse(y > T(1.11e-16), y * y, zero(T))
        xnum = T(_DEVICE_ERF_A[5]) * ysq
        xden = ysq
        xnum = (xnum + T(_DEVICE_ERF_A[1])) * ysq
        xden = (xden + T(_DEVICE_ERF_B[1])) * ysq
        xnum = (xnum + T(_DEVICE_ERF_A[2])) * ysq
        xden = (xden + T(_DEVICE_ERF_B[2])) * ysq
        xnum = (xnum + T(_DEVICE_ERF_A[3])) * ysq
        xden = (xden + T(_DEVICE_ERF_B[3])) * ysq
        return x * (xnum + T(_DEVICE_ERF_A[4])) / (xden + T(_DEVICE_ERF_B[4]))
    end
    y > T(26.5) && return ifelse(x > zero(T), one(T), -one(T))
    result = one(T) - _device_erfc_positive(y)
    return ifelse(x < zero(T), -result, result)
end

@inline function _device_erfc(x::T) where {T}
    y = abs(x)
    y <= T(0.46875) && return one(T) - _device_erf(x)
    # no explicit big-argument cutoff: exp(-ysq^2) underflows naturally, so
    # representable subnormal tails (e.g. erfc(26.51) ~ 1.3e-307 at Float64)
    # survive instead of collapsing to zero before the type does
    isinf(y) && return ifelse(x > zero(T), zero(T), T(2))
    result = _device_erfc_positive(y)
    return ifelse(x < zero(T), T(2) - result, result)
end

# Analytic-derivative duals: d/dx erf(x) = 2/sqrt(pi) exp(-x^2). An infinite
# value (an unbounded truncation side) is pinned to a zero derivative -- its
# incoming `deriv` can itself be infinite (Inf flowed through the standardizing
# division), and 0 * Inf would poison the gradient with NaN; this mirrors the
# CPU analytic gradient's isinf guards.
@inline function _device_erf(a::DeviceDual{T}) where {T}
    v = a.value
    deriv = ifelse(isinf(v), zero(T), T(1.1283791670955126) * exp(-v * v) * a.deriv)
    return DeviceDual{T}(_device_erf(v), deriv)
end
@inline function _device_erfc(a::DeviceDual{T}) where {T}
    v = a.value
    deriv = ifelse(isinf(v), zero(T), -T(1.1283791670955126) * exp(-v * v) * a.deriv)
    return DeviceDual{T}(_device_erfc(v), deriv)
end

# ---- regularized incomplete beta / Student-t CDF ---------------------------------
# Continued fraction (Lentz) for I_x(a, b), mirroring the Numerical Recipes form.
# Called with plain reals only: the DeviceDual overload of `_device_std_t_cdf`
# differentiates via the analytic d/dz = t-pdf(z) instead.

@inline function _device_betacf(a::T, b::T, x::T) where {T}
    qab = a + b
    qap = a + one(T)
    qam = a - one(T)
    fpmin = T(1e-30)
    c = one(T)
    d = one(T) - qab * x / qap
    abs(d) < fpmin && (d = fpmin)
    d = one(T) / d
    h = d
    for m = 1:200
        mf = T(m)
        m2 = 2 * mf
        aa = mf * (b - mf) * x / ((qam + m2) * (a + m2))
        d = one(T) + aa * d
        abs(d) < fpmin && (d = fpmin)
        c = one(T) + aa / c
        abs(c) < fpmin && (c = fpmin)
        d = one(T) / d
        h *= d * c
        aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
        d = one(T) + aa * d
        abs(d) < fpmin && (d = fpmin)
        c = one(T) + aa / c
        abs(c) < fpmin && (c = fpmin)
        d = one(T) / d
        delta = d * c
        h *= delta
        abs(delta - one(T)) <= T(4) * eps(T) && break
    end
    return h
end

# `x` and `y` are the caller-computed pair with x + y == 1 in exact arithmetic;
# passing both lets the caller keep each accurate near its own zero (recomputing
# `1 - x` at x ~ 1 destroys the low bits Float32 needs, e.g. the Student-t CDF
# at small |z|).
@inline function _device_beta_inc_reg_parts(a::T, b::T, x::T, y::T) where {T}
    x <= zero(T) && return zero(T)
    y <= zero(T) && return one(T)
    logx = ifelse(x < T(0.5), log(x), log1p(-y))
    logy = ifelse(y < T(0.5), log(y), log1p(-x))
    # the Student-t CDF calls this with b == 1/2, where loggamma(a+b) -
    # loggamma(a) is exactly the half-ratio; computing it as a ratio keeps
    # large-a (large-nu) accuracy that the difference of two huge loggammas
    # loses. Other b fall back to the direct difference.
    log_gamma_ratio = ifelse(
        b == T(0.5),
        _device_loggamma_half_ratio(a),
        _device_loggamma(a + b) - _device_loggamma(a),
    )
    front = exp(a * logx + b * logy + log_gamma_ratio - _device_loggamma(b))
    if x < (a + one(T)) / (a + b + T(2))
        return front * _device_betacf(a, b, x) / a
    end
    return one(T) - front * _device_betacf(b, a, y) / b
end

@inline _device_beta_inc_reg(a::T, b::T, x::T) where {T} = _device_beta_inc_reg_parts(a, b, x, one(T) - x)

# log I_x(a, b) computed IN LOG SPACE on the direct branch (x small), so a tail
# probability that underflows as a plain value (light Student-t tails at large
# nu, Float32) still yields a finite log. The symmetric branch is the
# complement of a small quantity and is safe as log1p.
@inline function _device_log_beta_inc_reg_parts(a::T, b::T, x::T, y::T) where {T}
    x <= zero(T) && return _device_neginf(T)
    y <= zero(T) && return zero(T)
    logx = ifelse(x < T(0.5), log(x), log1p(-y))
    logy = ifelse(y < T(0.5), log(y), log1p(-x))
    log_gamma_ratio = ifelse(
        b == T(0.5),
        _device_loggamma_half_ratio(a),
        _device_loggamma(a + b) - _device_loggamma(a),
    )
    log_front = a * logx + b * logy + log_gamma_ratio - _device_loggamma(b)
    if x < (a + one(T)) / (a + b + T(2))
        return log_front + log(_device_betacf(a, b, x)) - log(a)
    end
    return log1p(-exp(log_front) * _device_betacf(b, a, y) / b)
end

# log(T_cdf(z; nu)), staying in log space so light tails survive underflow.
@inline function _device_std_t_log_cdf(z::T, nu::T) where {T}
    isinf(z) && return ifelse(z > zero(T), zero(T), _device_neginf(T))
    z == zero(T) && return -log(T(2))
    denominator = nu + z * z
    x = nu / denominator
    y = z * z / denominator
    if z < zero(T)
        return _device_log_beta_inc_reg_parts(nu / T(2), T(0.5), x, y) - log(T(2))
    end
    regularized = _device_beta_inc_reg_parts(nu / T(2), T(0.5), x, y)
    return log1p(-regularized / T(2))
end

# d/dz log cdf = exp(log pdf - log cdf), computed in log space so a tail whose
# plain cdf underflows still gets a finite derivative ratio.
@inline function _device_std_t_log_cdf(z::DeviceDual{T}, nu::Real) where {T}
    nu_value = convert(T, nu)
    v = z.value
    log_cdf = _device_std_t_log_cdf(v, nu_value)
    log_pdf = _device_studentt_logpdf(nu_value, zero(T), one(T), v)
    deriv = ifelse(isinf(v), zero(T), exp(log_pdf - log_cdf) * z.deriv)
    return DeviceDual{T}(log_cdf, deriv)
end
@inline _device_std_t_log_cdf(z::DeviceDual{T}, nu::DeviceDual{T}) where {T} = _device_std_t_log_cdf(z, nu.value)

# Standard Student-t CDF for a lowering-guaranteed constant nu; mirrors the CPU
# `_std_t_cdf` branch structure exactly.
@inline function _device_std_t_cdf(z::T, nu::T) where {T}
    isinf(z) && return ifelse(z > zero(T), one(T), zero(T))
    z == zero(T) && return T(0.5)
    denominator = nu + z * z
    x = nu / denominator
    y = z * z / denominator # accurate complement of x (see _device_beta_inc_reg_parts)
    regularized = _device_beta_inc_reg_parts(nu / T(2), T(0.5), x, y)
    return ifelse(z > zero(T), one(T) - regularized / T(2), regularized / T(2))
end

# Standard Student-t pdf with the infinite-argument guard used by the CPU
# analytic gradient (an unbounded truncation side contributes zero density).
@inline function _device_std_t_pdf(z::T, nu::T) where {T}
    isinf(z) && return zero(T)
    return exp(_device_studentt_logpdf(nu, zero(T), one(T), z))
end

# d/dz T_cdf(z; nu) = t-pdf(z; nu); nu is a literal (derivative-free), so the nu
# channel is deliberately ignored, matching the CPU analytic gradient's omitted
# (genuinely zero) d/dnu term.
@inline function _device_std_t_cdf(z::DeviceDual{T}, nu::Real) where {T}
    nu_value = convert(T, nu)
    v = z.value
    # pin the derivative to zero on an infinite argument (see the erf duals)
    deriv = ifelse(isinf(v), zero(T), _device_std_t_pdf(v, nu_value) * z.deriv)
    return DeviceDual{T}(_device_std_t_cdf(v, nu_value), deriv)
end
# nu is a lowering-guaranteed literal, so a promoted dual nu carries derivative
# zero; only its value participates (disambiguates against the plain method).
@inline _device_std_t_cdf(z::DeviceDual{T}, nu::DeviceDual{T}) where {T} = _device_std_t_cdf(z, nu.value)

# ---- truncated families -----------------------------------------------------------

# Standard normal pdf with the infinite-bound guard (matches `_std_normal_pdf`).
@inline function _device_std_normal_pdf(z::T) where {T}
    isinf(z) && return zero(T)
    return exp(-z * z / T(2)) * T(0.3989422804014327) # 1/sqrt(2*pi)
end

# log(erfc(x)) with an asymptotic fallback once erfc underflows (Float32 loses
# erfc around x ~ 9.3, far before the CPU Float64 reference):
# erfc(x) ~ exp(-x^2)/(x sqrt(pi)) * (1 - u/2 + 3u^2/4 - 15u^3/8), u = 1/x^2.
# At the smallest engaging x the truncated series is accurate to ~1e-7, well
# inside the Float32 tolerance contract; the derivative matches to the same
# order.
@inline function _device_log_erfc(x::T) where {T}
    e = _device_erfc(x)
    u = one(T) / (x * x)
    # clamp: the series is only SELECTED for large x (u <= ~0.012, argument
    # ~ -0.006), but ifelse evaluates it everywhere and small x would push the
    # log1p argument below -1 (a DomainError on CPU and GPU alike)
    series = log1p(max(-u / T(2) + T(0.75) * u * u - T(1.875) * u * u * u, -T(0.5)))
    # log(abs(x)): the asymptote is only SELECTED for large positive x (erfc
    # underflow), but ifelse evaluates both branches and a GPU log() throws a
    # DomainError on negative arguments instead of returning NaN
    asymptotic = -x * x - log(abs(x)) - T(0.5723649429247001) + series # log(sqrt(pi))
    return ifelse(e > zero(T), log(e), asymptotic)
end

# Scaled complementary error function erfcx(y) = erfc(y) exp(y^2) for y >= 0:
# the Cody tail intervals compute it natively; the central interval scales the
# plain erfc (both factors representable there). The unselected branch is
# domain-safe: exp(y^2) can only overflow where erfc is exactly 0.
@inline _device_erfcx(y::T) where {T} =
    ifelse(y > T(0.46875), _device_erfcx_tail(y), _device_erfc(y) * exp(y * y))

# log(erfc(a) - erfc(b)) for 0 < a < b, cancellation- and underflow-free:
# erfc(a) - erfc(b) = e^{-a^2} erfcx(a) (1 - e^{a^2-b^2} erfcx(b)/erfcx(a)),
# so the tail magnitude lives in the -a^2 log term and the parenthesis is
# -expm1(s) with s < 0, exact for narrow AND wide intervals alike.
@inline function _device_log_erfc_diff(a::T, b::T) where {T}
    ea = _device_erfcx(a)
    eb = _device_erfcx(b)
    s = (a - b) * (a + b) + log(eb / ea)
    return -a * a + log(ea) + log(-expm1(min(s, -_device_tiny(T))))
end

# smallest useful positive magnitude for clamping (keeps -expm1(s) > 0 when
# rounding lands s exactly on zero)
@inline _device_tiny(::Type{T}) where {T} = T(1e-30)
@inline _device_tiny(::Type{DeviceDual{T}}) where {T} = DeviceDual{T}(T(1e-30), zero(T))

# log of a two-term difference with a cancellation guard: when the difference
# has lost (nearly) all significance relative to its larger term -- Float32
# rounding on a narrow interval, or an upper-tail CDF difference -- fall back to
# the midpoint approximation log(pdf(mid)) + log(width), exact as width -> 0.
@inline function _device_log_diff_or_midpoint(big::T, small::T, log_offset::T, log_fallback::T) where {T}
    diff = big - small
    # abs: a rounded-negative diff never SELECTS the log branch, but ifelse
    # evaluates it, and a GPU log() throws on negative arguments
    return ifelse(diff > T(8) * eps(T) * big, log(abs(diff)) - log_offset, log_fallback)
end

# log(Phi(zb) - Phi(za)), same branch structure as the CPU
# `_log_normal_cdf_diff` but with each branch computed in log space: one-sided
# tails via log(erfc) (asymptotic past underflow), same-sign two-sided
# intervals via the erfcx difference identity (exact for narrow and far-tail
# intervals where plain erfc differences cancel or underflow), and the
# straddling branch directly (an addition -- it cannot cancel).
@inline function _device_log_normal_cdf_diff(za::T, zb::T) where {T}
    root2 = sqrt(T(2))
    log2 = log(T(2))
    lower_infinite = isinf(za) & (za < zero(T))
    upper_infinite = isinf(zb) & (zb > zero(T))
    if lower_infinite && upper_infinite
        return zero(T)
    elseif lower_infinite
        return _device_log_erfc(-zb / root2) - log2
    elseif upper_infinite
        return _device_log_erfc(za / root2) - log2
    elseif za > zero(T)
        return _device_log_erfc_diff(za / root2, zb / root2) - log2
    elseif zb < zero(T)
        return _device_log_erfc_diff(-zb / root2, -za / root2) - log2
    end
    return log(_device_erf(zb / root2) - _device_erf(za / root2)) - log2
end

@inline function _device_truncatednormal_logpdf(mu::T, sigma::T, lower::T, upper::T, x::T) where {T}
    base = _device_normal_logpdf(mu, sigma, x)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    # collapsed/cancelled differences fall back to the log-space midpoint
    # approximation inside `_device_log_normal_cdf_diff` itself
    log_z = _device_log_normal_cdf_diff(za, zb)
    # degrade the degenerate lower >= upper (zero-width) case to -Inf; the
    # exception-free device path must never accept it as an infinite density
    in_support = (x >= lower) & (x <= upper) & (lower < upper)
    return ifelse(in_support, base - log_z, _device_neginf(T))
end

# log(T_cdf(zb) - T_cdf(za)) for a constant nu. One-sided normalizers use the
# symmetry S(z) = cdf(-z) so a tail probability is computed DIRECTLY (the
# regularized incomplete beta is small there and keeps relative accuracy) rather
# than as 1 - cdf, which cancels at Float32. Two-sided same-sign intervals use
# the tail-side representation for the same reason, with the midpoint fallback
# guarding what cancellation remains; the straddling branch spans ~0.5 of mass
# on each side, so only a narrow straddle can cancel (also caught by the guard,
# and its midpoint is finite by construction).
@inline function _device_t_log_normalizer(nu::T, za::T, zb::T) where {T}
    lower_infinite = isinf(za) & (za < zero(T))
    upper_infinite = isinf(zb) & (zb > zero(T))
    if lower_infinite && upper_infinite
        return zero(T)
    elseif lower_infinite
        return _device_std_t_log_cdf(zb, nu)
    elseif upper_infinite
        return _device_std_t_log_cdf(-za, nu)
    end
    if za > zero(T) # right tail: S(za) - S(zb) = cdf(-za) - cdf(-zb)
        big = _device_std_t_cdf(-za, nu)
        small = _device_std_t_cdf(-zb, nu)
    else # left tail and straddling: cdf(zb) - cdf(za)
        big = _device_std_t_cdf(zb, nu)
        small = _device_std_t_cdf(za, nu)
    end
    # log-space midpoint density (t tails only underflow at astronomical |z|,
    # but the log form is exact, cheaper, and symmetric with the normal path)
    log_fallback = _device_studentt_logpdf(nu, zero(T), one(T), (za + zb) / T(2)) + log(abs(zb - za))
    return _device_log_diff_or_midpoint(big, small, zero(T), log_fallback)
end

@inline function _device_truncatedstudentt_logpdf(nu::T, mu::T, sigma::T, lower::T, upper::T, x::T) where {T}
    base = _device_studentt_logpdf(nu, mu, sigma, x)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    log_z = _device_t_log_normalizer(nu, za, zb)
    # the CPU reference REJECTS lower >= upper (ArgumentError); the exception-free
    # device contract degrades that to -Inf instead of an infinite density
    in_support = (x >= lower) & (x <= upper) & (lower < upper)
    return ifelse(in_support, base - log_z, _device_neginf(T))
end

# Parameter transform codes shared by lowering, kernel, and staging.
const DEVICE_TRANSFORM_IDENTITY = Int32(0)
const DEVICE_TRANSFORM_LOG = Int32(1)
const DEVICE_TRANSFORM_LOGIT = Int32(2)

# Applies an unconstrained -> constrained transform, returning
# (constrained_value, logabsdet_jacobian). Mirrors src/parameters.jl and the
# batched transform branch in src/batched/workspace.jl for the supported set.
@inline function _device_transform(tcode::Int32, u::T) where {T}
    if tcode == DEVICE_TRANSFORM_LOG
        return (exp(u), u)
    elseif tcode == DEVICE_TRANSFORM_LOGIT
        c = inv(one(T) + exp(-u))
        return (c, log(c) + log1p(-c))
    else # DEVICE_TRANSFORM_IDENTITY
        return (u, zero(T))
    end
end
