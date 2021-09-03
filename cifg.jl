# Coupled Input and Forget Gate
using Flux

CIFGate(x) = sigmoid(sigmoid(x) * tanh(x))
OutGate(x, Ct) = sigmoid(x) * Ct

function CIFGCell(prev_ct, prev_ht, input)
	combine = prev_ht + input
	Ct = prev_ct

	Ct *= CIFGate(combine)

	return Ct, OutGate(combine, Ct)
end

x, y = CIFGCell(0.221, 0.114, 0.525)
println("Ct: ", x)
println("Ht: ", y)
