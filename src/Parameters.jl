module Parameters

"""
Number of training examples to generate for the given dimension.
"""
function n(dims)
    return Int(round(200 * 10^(dims / 5)))
end

end
