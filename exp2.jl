# define problem
using ApproximateVI, Distributions, PyPlot, Optim, MiscUtil, LinearAlgebra, StatsFuns, Suppressor

include("problem_3normals.jl")
include("problem_sin.jl")
include("RegionalSoftmax.jl")

function runme(logp; D=D, K=K, iterations=1)


    function plotdensity(p)
        xgrid = -6:0.03:6
        aux1 = zeros(length(xgrid), length(xgrid))
        aux2 = zeros(length(xgrid), length(xgrid))
        for i in 1:length(xgrid)
            for j in 1:length(xgrid)
                aux1[i,j] = xgrid[i]
                aux2[i,j] = xgrid[j]
            end
        end
        figure()
        pcolor(aux1, aux2, [p([x;y]) for x in xgrid, y in xgrid])
    end


    post0, logev0 = VI(logp, randn(D), S=750, iterations = 1000, show_every=10, optimiser = NelderMead())


    plotdensity(x -> exp(logp(x)))

    regionalsoftmax = RegionalSoftmax(D=D, K=K)

    function splitvi(param)

        @assert(length(param) == 1 + D*K)

        local initpost = MvNormal(zeros(D), Matrix(diagm(ones(D))))

        local regionprior = regionalsoftmax(param)

        local logtarget = [x -> logp(x) + log(rp(x)) for rp in regionprior]


        local output = map(logtarget) do logt
            @suppress VI(logt, initpost, S=250, iterations = 1000, show_every=10, optimiser = NelderMead())
        end


        local ℓ = [o[2] for o in output]

        local ω = exp.(ℓ .- logsumexp(ℓ))

        sum(ω.*(ℓ .- log.(ω))), [o[1] for o in output], ω

    end



    result = optimize(x->-splitvi(x)[1], [0.0; randn(D*K)], NelderMead(), Optim.Options(show_every=1, show_trace=true, iterations=iterations))

    logevcombined, post, ω = splitvi(result.minimizer)


    for (i,p) in enumerate(post)
        display(p)
        plot_ellipse(p, "r")
    end

    plot_ellipse(post0, "k")

    logev0, logevcombined, ω
end
