using RSLModels.Intervals

# intersection

i1 = Intervals.draw_interval([0.5]; spread_min=0.2)
i2 = Intervals.draw_interval([0.5]; spread_min=0.2)
@test intersection(i1, i2) != nothing

i3 = Intervals.Interval(i1.ubound, i1.ubound .+ 0.2)
inters13 = intersection(i1, i3)
@test inters13.lbound == inters13.ubound

i4 = Intervals.Interval(i1.ubound, i1.ubound .+ 0.2; lopen=BitVector((true,)))
@test intersection(i1, i4) == nothing

i4 = Intervals.Interval(i1.ubound, i1.ubound .+ 0.2; uopen=BitVector((true,)))
@test intersection(i1, i4) != nothing

i4 = Intervals.Interval(
    i1.ubound,
    i1.ubound .+ 0.2;
    lopen=BitVector((true,)),
    uopen=BitVector((true,)),
)
@test intersection(i1, i4) == nothing

i1 = Intervals.draw_interval([0.5, 0.5, 0.5]; spread_min=0.2)
i2 = Intervals.draw_interval([0.5, 0.5, 0.5]; spread_min=0.2)
@test intersection(i1, i2) != nothing

i3 = Intervals.Interval(i1.ubound, i1.ubound .+ 0.2)
inters13 = intersection(i1, i3)
@test inters13.lbound == inters13.ubound

i4 = Intervals.Interval(
    i1.ubound,
    i1.ubound .+ 0.2;
    lopen=BitVector((true, false, false)),
)
@test intersection(i1, i4) == nothing

i4 = Intervals.Interval(
    i1.ubound,
    i1.ubound .+ 0.2;
    uopen=BitVector((true, false, false)),
)
@test intersection(i1, i4) != nothing

i4 = Intervals.Interval(
    i1.ubound,
    i1.ubound .+ 0.2;
    lopen=BitVector((false, true, false)),
    uopen=BitVector((false, false, true)),
)
@test intersection(i1, i4) == nothing

# draw

dim = 3
spread_min = 0.1
spread_max = 0.3
x_min = 0.0
x_max = 1.0

spread = Intervals.draw_spread(
    dim;
    spread_min=spread_min,
    spread_max=spread_max,
    x_min=x_min,
    x_max=x_max,
)

@test all(spread_min .<= spread)
@test all(spread .<= spread_max)
@test all(x_min .<= spread)
@test all(spread .<= x_max)

x = [0.4]

interval = draw_interval(
    x;
    spread_min=spread_min,
    spread_max=spread_max,
    x_min=x_min,
    x_max=x_max,
)

@test elemof(x, interval)
@test all(x_min .<= interval.lbound)
@test all(interval.ubound .<= x_max)

# Check that remove_full_overlaps does not change coverage.
rate_coverage, intervals =
    draw_intervals(dim; remove_final_fully_overlapped=false)
X = rand(10000, dim)
count_match = count(any(elemof(X, intervals); dims=2))
count_match2 =
    count(any(elemof(X, remove_fully_overlapped(intervals, X)); dims=2))
@test count_match == count_match2

# Ensure that usemmap does not change the result.
rate_coverage, intervals1 = draw_intervals(Random.Xoshiro(1), dim)
rate_coverage, intervals2 =
    draw_intervals(Random.Xoshiro(1), dim; usemmap=true)
@test all(intervals1 .== intervals2)
