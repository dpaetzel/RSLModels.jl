module AbstractModels

export output, output_dist, output_mean, output_variance

inputs_required = """
Requires (an) input(s), if the model's output distribution depends on an input
(this is the case for most predictive models).
"""

"""
Return an output of the given model.

If the model is probabilistic, draw a random output from the output
distribution.

$inputs_required
"""
function output end
# TODO Use Random.rand combined with the output distribution instead probably

"""
Return the model's output distribution.

$inputs_required
"""
function output_dist end

"""
Return the model's output distribution's mean.

$inputs_required
"""
function output_mean end

"""
Return the model's output distribution's variance.

$inputs_required
"""
function output_variance end

end
