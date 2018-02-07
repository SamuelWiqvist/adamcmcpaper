include("posteriorquantileinterval.jl")

data = rand(1,100)

calcquantileint(data, 5, 95)

data = rand(100,4)

calcquantileint(data)

data = rand(100)

calcquantileint(data)
