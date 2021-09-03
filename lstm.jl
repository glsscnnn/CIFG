using Flux

# 1 forget 0 remember
function forget_gate(x)
	return sigmoid(x)
end

# update our cell state
function input_gate(x)
	a = sigmoid(x)
	b = tanh(x)
	return a * b
end

# output hidden state
function output_gate(x, Ct)
	return sigmoid(x) * Ct
end

# LSTM cell arch
function LSTMCell(prev_ct, prev_ht, input)
	combine = prev_ht + input
	Ct = prev_ct

	Ct *= forget_gate(combine)
	Ct += input_gate(combine)

	Ht = output_gate(combine, Ct)
	return Ct, Ht
end

x, y = LSTMCell(0.221, 0.114, 0.525)

println("Ct: ", x)
println("Ht: ", y)
