function problem_2normals()

    function logdensity(x)

        d1, d2 = MvNormal([-3;1], 1.0), MvNormal([3;-1], 1.0)

        log(pdf(d1, x)*0.5 + pdf(d2, x)*0.5)

    end

end
