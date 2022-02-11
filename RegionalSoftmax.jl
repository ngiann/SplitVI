struct RegionalSoftmax
    D::Int
    K::Int
end


function RegionalSoftmax(;D::Int = D, K::Int = K)

    RegionalSoftmax(D, K)

end


function unnormalisedregion(r, centre)

    x -> exp.(-r * sum((x - centre).^2))

end


function (a::RegionalSoftmax)(params)

    logr, centres = unpack(D = a.D, K = a.K, params = params)

    a(exp(logr), centres)

end


function (a::RegionalSoftmax)(r, centres)

    aux = [unnormalisedregion(r, c) for c in centres]

    function producefunction(index)

        function f(x)

            values = map(a -> a(x), aux)

            values[index] / sum(values)

        end

    end

    map(producefunction, 1:a.K)

end


function unpack(; D = D, K = K, params = params)

    r = params[1]

    @assert(length(params[2:end]) == D * K)

    p = reshape(params[2:end], D, K)

    centres = [p[:,k] for k in 1:K]

    (r = r, centres = centres)

end
