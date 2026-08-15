local learningRate = 0.0003

local function reLU(x)
	return math.max(0, x)
end

local function reLUDeriv(y)
	return y > 0 and 1 or 0
end

local module = {}

function module.getNetwork(sizes, parentSizes, outputExtraInputCount, rng)
	outputExtraInputCount = outputExtraInputCount or 0
	rng = rng or Random.new()

	local hasParentSizes = type(parentSizes) == "table"
	local network = {}

	for i = 2, #sizes do
		local layerSize = sizes[i]
		local ownPrevSize = sizes[i - 1]
		local lateralSize = 0
		local extraSize = 0

		if hasParentSizes and i > 2 then
			lateralSize = parentSizes[i - 1] or 0
		end

		if i == #sizes then
			extraSize = outputExtraInputCount
		end

		local totalPrevSize = ownPrevSize + lateralSize + extraSize
		local heBound = math.sqrt(6 / totalPrevSize)

		local layer = {}

		for j = 1, layerSize do
			local neuron = {}
			neuron.bias = 0
			neuron.weights = {}
			neuron.ownInputCount = ownPrevSize
			neuron.lateralInputCount = lateralSize
			neuron.outputExtraInputCount = extraSize

			for k = 1, totalPrevSize do
				local isExtraInput = k > ownPrevSize
				local scale = isExtraInput and 0.1 or 1.0
				neuron.weights[k] = (rng:NextNumber() * 2 - 1) * heBound * scale
			end

			layer[j] = neuron
		end

		network[i - 1] = layer
	end

	return network
end

local function buildLayerInputs(network, activations, layerIndex, parentActivations, lateralScale, outputExtraInputs)
	local currentInputs = activations[layerIndex]
	local neuron = network[layerIndex][1]
	local lateralCount = neuron.lateralInputCount or 0
	local extraCount = neuron.outputExtraInputCount or 0

	if lateralCount == 0 and extraCount == 0 then
		return currentInputs
	end

	local layerInputs = table.clone(currentInputs)

	if lateralCount > 0 then
		local parentLayerInputs = parentActivations and parentActivations[layerIndex] or nil

		for i = 1, lateralCount do
			local value = parentLayerInputs and parentLayerInputs[i] or 0
			table.insert(layerInputs, value * lateralScale)
		end
	end

	if layerIndex == #network and extraCount > 0 then
		for i = 1, extraCount do
			local value = outputExtraInputs and outputExtraInputs[i] or 0
			table.insert(layerInputs, value * lateralScale)
		end
	end

	return layerInputs
end

function module.forwardPass(network, inputs, parentActivations, lateralScale, outputExtraInputs)
	lateralScale = lateralScale or 1

	local activations = {inputs}

	for i = 1, #network do
		local currentInputs = buildLayerInputs(network, activations, i, parentActivations, lateralScale, outputExtraInputs)
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

function module.backpropagate(network, activations, outputErrors, parentActivations, lateralScale, outputExtraInputs)
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
		local currentInputs = buildLayerInputs(network, activations, i, parentActivations, lateralScale, outputExtraInputs)

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

function module.updateWeights(network, gradients, learningRateOverride, firstTrainableLayer)
	local effectiveLearningRate = learningRateOverride or learningRate
	local firstLayer = firstTrainableLayer or 1

	assert(firstLayer >= 1 and firstLayer <= #network, "firstTrainableLayer is outside the network")
	assert(type(effectiveLearningRate) == "number" and effectiveLearningRate > 0, "learning rate must be positive")

	for i = firstLayer, #network do
		for j = 1, #network[i] do
			local neuron = network[i][j]
			local gradient = gradients[i][j]

			neuron.bias += effectiveLearningRate * gradient.bias

			for k = 1, #neuron.weights do
				neuron.weights[k] += effectiveLearningRate * gradient.weights[k]
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

return module
