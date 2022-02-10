# define problem
using ApproximateVI, Distributions, PyPlot, Optim, MiscUtil, LinearAlgebra, StatsFuns, Suppressor

include("problem_2normals.jl")
include("problem_sin.jl")

function runme(iterations=1)

    logtarget = problem_2normals()

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


    post, logev0 = VI(logtarget, randn(2), S=1010, iterations = 1000, show_every=10, optimiser = NelderMead())


    plotdensity(x -> exp(logtarget(x)))
    plot_ellipse(post)


    function splitvi(param)

        post1, post2 = MvNormal(zeros(2), Matrix(diagm(ones(2)))), MvNormal(zeros(2), Matrix(diagm(ones(2))))

        @assert(length(param) == 5)

        logr, m1, m2 = param[1], param[2:3], param[4:5]

        r = exp(logr) + 0.1

        pr1(x) = exp(-r*sum((x-m1).^2))

        pr2(x) = exp(-r*sum((x-m2).^2))

        logtarget1(x) = log(exp(logtarget(x)) * pr1(x)/(pr1(x) + pr2(x)))

        logtarget2(x) = log(exp(logtarget(x)) * pr2(x)/(pr1(x) + pr2(x)))


        post1, logev1 = @suppress VI(logtarget1, post1, S=500, iterations = 1000, show_every=10, optimiser = NelderMead())

        post2, logev2 = @suppress VI(logtarget2, post2, S=500, iterations = 1000, show_every=10, optimiser = NelderMead())


        ℓ = [logev1; logev2]

        ω = exp.(ℓ .- logsumexp(ℓ))

        sum(ω.*(ℓ .- log.(ω))), post1, post2

    end



    result = optimize(x->-splitvi(x)[1], 3*randn(5), NelderMead(), Optim.Options(show_every=1, show_trace=true, iterations=iterations))

    logevcombined, post1, post2 = splitvi(result.minimizer)

    plot_ellipse(post1, "r"); plot_ellipse(post2, "c")

    logev0, logevcombined#, post1, post2
end
