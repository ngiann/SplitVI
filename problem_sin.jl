function problem_sin()


    w(x)    = sin(2π*x/4)

    U(z)    = 0.5*((z[2]-w(z[1]))/0.4)^2

    loglikel(z) = -U(z)

    logprior(z) = Distributions.logpdf(MvNormal(zeros(2), 1.9), z)

    logp(z) = loglikel(z) + logprior(z)

end
