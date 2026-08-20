local learningRate = 0.0003

local function reLU(x)
	return math.max(0, x)
end

local function reLUDeriv(y)
	return y > 0 and 1 or 0
end

local module = {}

function module.getNetwork(sizes, numParents, parentSizes, includeParentOutputs, rng)
	numParents = numParents or 0
	includeParentOutputs = includeParentOutputs == true
	rng = rng or Random.new()

	local hasParents = numParents > 0 and type(parentSizes) == "table"
	local network = {}

	for i = 2, #sizes do
		local layerSize = sizes[i]
		local ownPrevSize = sizes[i - 1]
		local lateralPerParent = 0
		local parentOutputPerParent = 0

		if hasParents and i > 2 then
			lateralPerParent = parentSizes[i - 1] or 0

			if includeParentOutputs and i == #sizes then
				parentOutputPerParent = parentSizes[#parentSizes] or 0
			end
		end

		local totalLateral = numParents * (lateralPerParent + parentOutputPerParent)
		local totalPrevSize = ownPrevSize + totalLateral
		local heBound = math.sqrt(6 / totalPrevSize)

		local layer = {}

		for j = 1, layerSize do
			local neuron = {}
			neuron.bias = 0
			neuron.weights = {}
			neuron.ownInputCount = ownPrevSize
			neuron.lateralInputCount = totalLateral
			neuron.parentCount = numParents
			neuron.parentHiddenInputCount = lateralPerParent
			neuron.parentOutputInputCount = parentOutputPerParent

			for k = 1, totalPrevSize do
				local isLateral = k > ownPrevSize
				local scale = isLateral and 0.1 or 1.0
				neuron.weights[k] = (rng:NextNumber() * 2 - 1) * heBound * scale
			end

			layer[j] = neuron
		end

		network[i - 1] = layer
	end

	return network
end

local function buildLayerInputs(network, activations, layerIndex, parentActivationsList, lateralScale)
	local currentInputs = activations[layerIndex]
	local neuron = network[layerIndex][1]
	local lateralCount = neuron.lateralInputCount or 0

	if lateralCount == 0 then
		return currentInputs
	end

	local layerInputs = table.clone(currentInputs)
	local parentCount = neuron.parentCount or 0
	local parentHiddenInputCount = neuron.parentHiddenInputCount or 0
	local parentOutputInputCount = neuron.parentOutputInputCount or 0

	for parentIndex = 1, parentCount do
		local parentActivations = parentActivationsList and parentActivationsList[parentIndex] or nil
		local parentLayerInputs = parentActivations and parentActivations[layerIndex] or nil

		for inputIndex = 1, parentHiddenInputCount do
			local value = parentLayerInputs and parentLayerInputs[inputIndex] or 0
			table.insert(layerInputs, value * lateralScale)
		end

		local parentOutputs = parentActivations and parentActivations[#parentActivations] or nil

		for outputIndex = 1, parentOutputInputCount do
			local value = parentOutputs and parentOutputs[outputIndex] or 0
			table.insert(layerInputs, value * lateralScale)
		end
	end

	return layerInputs
end

function module.forwardPass(network, inputs, parentActivationsList, lateralScale)
	lateralScale = lateralScale or 1

	local activations = {inputs}

	for i = 1, #network do
		local currentInputs = buildLayerInputs(network, activations, i, parentActivationsList, lateralScale)
		local layerActivations = {}

		for j = 1, #network[i] do
			local neuron = network[i][j]
			local activation = neuron.bias

			for k = 1, #neuron.weights do
				activation += currentInputs[k] * neuron.weights[k]
			end

			if i ~= #network then
				activation = reLU(activation)
			end

			layerActivations[j] = activation
		end

		activations[i + 1] = layerActivations
	end

	return activations
end

function module.backpropagate(network, activations, outputErrors, parentActivationsList, lateralScale)
	lateralScale = lateralScale or 1

	local deltas = {}
	deltas[#network] = outputErrors

	for i = #network - 1, 1, -1 do
		local layer = network[i]
		local nextLayer = network[i + 1]
		local layerDelta = {}

		for j = 1, #layer do
			local delta = 0

			for k = 1, #nextLayer do
				delta += deltas[i + 1][k] * nextLayer[k].weights[j]
			end

			delta *= reLUDeriv(activations[i + 1][j])
			layerDelta[j] = delta
		end

		deltas[i] = layerDelta
	end

	local gradients = {}

	for i = 1, #network do
		local layerGradients = {}
		local currentInputs = buildLayerInputs(network, activations, i, parentActivationsList, lateralScale)

		for j = 1, #network[i] do
			local neuronGradients = {}
			neuronGradients.bias = deltas[i][j]
			neuronGradients.weights = {}

			for k = 1, #network[i][j].weights do
				neuronGradients.weights[k] = currentInputs[k] * deltas[i][j]
			end

			layerGradients[j] = neuronGradients
		end

		gradients[i] = layerGradients
	end

	return gradients
end

function module.updateWeights(network, gradients)
	for i = 1, #network do
		for j = 1, #network[i] do
			local neuron = network[i][j]
			local gradient = gradients[i][j]

			neuron.bias += learningRate * gradient.bias

			for k = 1, #neuron.weights do
				neuron.weights[k] += learningRate * gradient.weights[k]
			end
		end
	end
end

function module.getBlankGradients(network)
	local blankGradients = {}

	for i = 1, #network do
		local layerGradients = {}

		for j = 1, #network[i] do
			local neuronGradients = {}
			neuronGradients.bias = 0
			neuronGradients.weights = {}

			for k = 1, #network[i][j].weights do
				neuronGradients.weights[k] = 0
			end

			layerGradients[j] = neuronGradients
		end

		blankGradients[i] = layerGradients
	end

	return blankGradients
end

function module.addGradients(accumulatedGradients, newGradients)
	for i = 1, #accumulatedGradients do
		for j = 1, #accumulatedGradients[i] do
			accumulatedGradients[i][j].bias += newGradients[i][j].bias

			for k = 1, #accumulatedGradients[i][j].weights do
				accumulatedGradients[i][j].weights[k] += newGradients[i][j].weights[k]
			end
		end
	end
end

function module.compressNetwork(network, sizes)
	local compressedNetwork = module.getNetwork(sizes)

	for i = 1, #compressedNetwork do
		for j = 1, #compressedNetwork[i] do
			compressedNetwork[i][j].bias = network[i][j].bias

			for k = 1, #compressedNetwork[i][j].weights do
				compressedNetwork[i][j].weights[k] = network[i][j].weights[k]
			end
		end
	end

	return compressedNetwork
end

function module.forwardArchivedColumns(parentNetworks, inputs, lateralScale)
	local activationsList = {}

	for _, parentNetwork in ipairs(parentNetworks) do
		local activations = module.forwardPass(
			parentNetwork,
			inputs,
			activationsList,
			lateralScale or 1
		)
		table.insert(activationsList, activations)
	end

	return activationsList
end

function module.maxAbsOutputDifference(leftActivations, rightActivations)
	local left = leftActivations[#leftActivations]
	local right = rightActivations[#rightActivations]
	assert(#left == #right, "cannot compare output layers of different sizes")

	local maximum = 0
	for index = 1, #left do
		maximum = math.max(maximum, math.abs(left[index] - right[index]))
	end
	return maximum
end

return module
