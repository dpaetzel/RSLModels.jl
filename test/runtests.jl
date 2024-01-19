using Test
using TestSetExtensions

@testset ExtendedTestSet "All tests" begin
    # This allows me to run tests selectively (by their filename without
    # extension). If the user does not specify anything, all `.jl` files
    # in the `/test` directory are run.
    @includetests ARGS
end
