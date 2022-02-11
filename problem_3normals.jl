function problem_3normals()

    function logdensity(x)

        d1, d2, d3 = MvNormal([-3;1], 1.0), MvNormal([3;-1], 1.0), MvNormal([0;0], 1.0)

        log(pdf(d1, x)/3 + pdf(d2, x)/3 + pdf(d3, x)/3)

    end

end
