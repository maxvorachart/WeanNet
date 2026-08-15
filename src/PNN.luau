local DataStoreService = game:GetService("DataStoreService")
local HttpService = game:GetService("HttpService")
local ServerStorage = game:GetService("ServerStorage")

local aiDataStore = DataStoreService:GetDataStore("GenerationalPNNSpeedStageTest1")

local oldWaypointsFolder = workspace:WaitForChild("WaypointsFolder")
local newWaypointsFolder = workspace:WaitForChild("TrueWaypointsFolder")
local divider = workspace:WaitForChild("Divider")
local spawner = workspace:WaitForChild("Spawner")

local pnnModule = require(ServerStorage:WaitForChild("PNNModule"))

local agent = workspace:WaitForChild("Agent6")
local agentColor = agent:WaitForChild("Highlight").FillColor
local defaultColor = Color3.fromRGB(255, 255, 0)
local raycastsFolder = agent:WaitForChild("RaycastsFolder")
local humanoid = agent:WaitForChild("Humanoid")
local humanoidRootPart = agent:WaitForChild("HumanoidRootPart")

local save = true
local key = 73

local gamma = 0.95
local temperature = 1.0
local batchSize = 512
local stageLengthBatches = 1000
local maxBatches = stageLengthBatches * 2
local networkShape = {39, 64, 64, 4}
local newHeadSeeds = {1, 2, 3, 4}
local legacySeed = 5
local parentOutputLateralConnectionOptions = {false, true}
local checkpointNamespace = "MultiSeedExperimentV3"
local checkpointVersion = 3
local runStorageVersion = 4
local maxEncodedCheckpointBytes = 3500000
local expectedBatches = maxBatches
local configFingerprint = "reviewer-v3|architecture=PNN|key=73|seeds=1,2,3,4,5|tasks=A-B|stageLength=1000|maxBatches=2000|batchSize=512|shape=39-64-64-4|gamma=0.95|temperature=1.0|rewardMultiplier=1.0|maskedInactiveCoordinates=true|raycastExclusions=agents+waypoints|waypointDisplay=regularOwnerActiveOnly|fullDependencyArchive=true|lateralScale=1.0|parentOutputOptions=false,true|evaluation=greedy|evaluationEpisodes=5|evaluationStepsPerEpisode=256|seenTasksOnly=true|evaluationSuccess=anyWaypointCompletion|evaluationResetActionState=true"
local checkpointEveryBatches = 5
local evaluationEpisodes = 5
local evaluationStepsPerEpisode = 256
local rngDrawCount = 0
local rng

local function createTrackedRandom(seed, drawsToReplay)
	local innerRandom = Random.new(seed)
	local replayCount = drawsToReplay or 0

	for _ = 1, replayCount do
		innerRandom:NextNumber()
	end

	rngDrawCount = replayCount

	local trackedRandom = {}

	function trackedRandom:NextNumber()
		rngDrawCount += 1
		return innerRandom:NextNumber()
	end

	return trackedRandom
end

local normalWalkSpeed = 16
local chaseReachDistance = 3
local chaseProgressRewardScale = 1.0
local chaseReachReward = 8
local livingPenalty = 0.01
local newSideTerminalMultiplier = 1.0

local lateralScale = 1.0

local oldTargetWaypoint = 1
local newTargetWaypoint = 1
local previousOldDistance
local previousNewDistance
local batch = {}
local averageRewardsHistory = {}
local evaluationByGeneration = {}
local evaluationTaskOverride = nil
local evaluationWaypointCompletions = 0
local batchesCompleted = 0

local lastRebirthStageNumber = 1
local generation = 1
local network
local parentNetworks = {}

local aiType = "PNN"

local function getStageNumber()
	return math.floor(batchesCompleted / stageLengthBatches) + 1
end

local function isStageTwo()
	if evaluationTaskOverride ~= nil then
		return evaluationTaskOverride
	end
	return getStageNumber() % 2 == 0
end

local function getStageName()
	return isStageTwo() and "MirrorAlternate" or "NormalParkour"
end

local function getWalkSpeedForStage()
	return normalWalkSpeed
end

local function getPlanarDistanceToPosition(position)
	local yDifference = humanoidRootPart.Position.Y - position.Y
	local zDifference = humanoidRootPart.Position.Z - position.Z
	return math.sqrt(yDifference * yDifference + zDifference * zDifference)
end

local function getOldWaypoint(index)
	return oldWaypointsFolder:FindFirstChild(tostring(index))
end

local function getNewWaypoint(index)
	return newWaypointsFolder:FindFirstChild(tostring(index))
end

local function isOnNewSide()
	return humanoidRootPart.Position.Z > divider.Position.Z
end

local function updateGoals()
end

local raycastExclusions = {oldWaypointsFolder, newWaypointsFolder}
for _, candidate in ipairs(workspace:GetChildren()) do
	if string.match(candidate.Name, "^Agent") then
		table.insert(raycastExclusions, candidate)
	end
end

local function updateRayPart(name, origin, direction, distance)
	local part = raycastsFolder:FindFirstChild(name)

	if not part then
		part = Instance.new("Part")
		part.Name = name
		part.Anchored = true
		part.CanCollide = false
		part.CanTouch = false
		part.CastShadow = false
		part.CanQuery = false
		part.Material = Enum.Material.Neon
		part.Color = Color3.new(1, 0, 0)
		part.Parent = raycastsFolder
	end

	part.Size = Vector3.new(0.1, 0.1, distance)
	part.CFrame = CFrame.lookAt(origin + direction.Unit * (distance / 2), origin + direction)
end

local function newExperiment()
	return {
		version = checkpointVersion,
		configFingerprint = configFingerprint,
		expectedBatches = expectedBatches,
		runs = {},
	}
end

local function assertCompatibleExperiment(experiment)
	assert(experiment.version == checkpointVersion, "checkpoint version mismatch")
	assert(experiment.configFingerprint == configFingerprint, "checkpoint configFingerprint mismatch")
	assert(experiment.expectedBatches == expectedBatches, "checkpoint expectedBatches mismatch")
end

local function loadLegacyData(keyValue)
	for attempt = 1, 5 do
		local success, storedRoot = pcall(function()
			return aiDataStore:GetAsync(keyValue)
		end)

		if success then
			if type(storedRoot) == "table" and type(storedRoot[checkpointNamespace]) == "table" then
				local experiment = storedRoot[checkpointNamespace]
				assertCompatibleExperiment(experiment)
				experiment.runs = type(experiment.runs) == "table" and experiment.runs or {}
				return experiment
			end

			return newExperiment()
		end

		warn(string.format("Legacy checkpoint load failed (attempt %d): %s", attempt, tostring(storedRoot)))
		task.wait(1)
	end

	error("Legacy checkpoint load failed after 5 attempts; refusing to start with empty data")
end

local function getRunStorageKey(keyValue, runId)
	local storageKey = string.format("MSE4:%s:%s", tostring(keyValue), runId)
	assert(#storageKey <= 50, "run checkpoint key exceeds DataStore's 50-character limit")
	return storageKey
end

local function isDeterministicSizeError(errorMessage)
	local message = string.lower(tostring(errorMessage))
	return string.find(message, "valuetoolarge", 1, true) ~= nil
		or string.find(message, "serialized value exceeds", 1, true) ~= nil
		or string.find(message, "105:", 1, true) ~= nil
end

local function loadRunData(keyValue, runId, legacyExperiment)
	local storageKey = getRunStorageKey(keyValue, runId)

	for attempt = 1, 5 do
		local success, storedEnvelope = pcall(function()
			return aiDataStore:GetAsync(storageKey)
		end)

		if success then
			if storedEnvelope == nil then
				local legacyRun = legacyExperiment.runs[runId]
				return legacyRun, legacyRun and "legacy" or "new"
			end

			assert(type(storedEnvelope) == "table", "run checkpoint envelope is not a table")
			assert(storedEnvelope.storageVersion == runStorageVersion, "run checkpoint storage version mismatch")
			assert(storedEnvelope.namespace == checkpointNamespace, "run checkpoint namespace mismatch")
			assert(tostring(storedEnvelope.baseKey) == tostring(keyValue), "run checkpoint base key mismatch")
			assert(storedEnvelope.runId == runId, "run checkpoint runId mismatch")
			assert(
				storedEnvelope.seedsPerHead == #newHeadSeeds
					or storedEnvelope.seedsPerHead == legacySeed,
				"run checkpoint seedsPerHead mismatch"
			)
			assert(type(storedEnvelope.run) == "table", "run checkpoint payload is missing")
			return storedEnvelope.run, "shard"
		end

		warn(string.format(
			"Run checkpoint load failed for %s (attempt %d): %s",
			runId,
			attempt,
			tostring(storedEnvelope)
		))
		task.wait(1)
	end

	error(string.format(
		"Run checkpoint load failed for %s after 5 attempts; refusing to overwrite it",
		runId
	))
end

local function saveData(keyValue, runId, runData, seedsPerHead)
	local storageKey = getRunStorageKey(keyValue, runId)
	local envelope = {
		storageVersion = runStorageVersion,
		namespace = checkpointNamespace,
		baseKey = keyValue,
		runId = runId,
		seedsPerHead = seedsPerHead,
		run = runData,
	}

	local encodeSuccess, encodedOrError = pcall(function()
		return HttpService:JSONEncode(envelope)
	end)
	if not encodeSuccess then
		warn(string.format("Checkpoint serialization failed for %s: %s", runId, tostring(encodedOrError)))
		return false
	end
	if #encodedOrError > maxEncodedCheckpointBytes then
		warn(string.format(
			"Checkpoint for %s is %d bytes; refusing write above %d-byte safety limit",
			runId,
			#encodedOrError,
			maxEncodedCheckpointBytes
		))
		return false
	end

	for attempt = 1, 5 do
		local success, errorMessage = pcall(function()
			aiDataStore:UpdateAsync(storageKey, function(existing)
				if type(existing) == "table" then
					assert(existing.storageVersion == runStorageVersion, "existing run storage version mismatch")
					assert(existing.namespace == checkpointNamespace, "existing run namespace mismatch")
					assert(tostring(existing.baseKey) == tostring(keyValue), "existing run base key mismatch")
					assert(existing.runId == runId, "existing runId mismatch")
				end
				return envelope
			end)
		end)

		if success then
			return true
		end

		warn(string.format("Checkpoint save failed (attempt %d): %s", attempt, tostring(errorMessage)))
		if isDeterministicSizeError(errorMessage) then
			return false
		end
		task.wait(1)
	end

	return false
end

local function softmax(outputs)
	local maxOutput = -math.huge

	for _, value in ipairs(outputs) do
		if value > maxOutput then
			maxOutput = value
		end
	end

	local expSum = 0
	local exps = {}

	for i, value in ipairs(outputs) do
		exps[i] = math.exp((value - maxOutput) / temperature)
		expSum += exps[i]
	end

	for i, value in ipairs(exps) do
		exps[i] = value / expSum
	end

	return exps
end

local function getDiscountedRewards(rewards)
	local discountedRewards = {}
	local runningAdd = 0

	for i = #rewards, 1, -1 do
		runningAdd = runningAdd * gamma + rewards[i]
		discountedRewards[i] = runningAdd
	end

	return discountedRewards
end

local function shuffleList(targetList)
	for i = #targetList, 2, -1 do
		local j = math.floor(rng:NextNumber() * i) + 1
		targetList[i], targetList[j] = targetList[j], targetList[i]
	end

	return targetList
end

local function doAction(rawOutputs)
	local probabilities = softmax(rawOutputs)
	local roll = rng:NextNumber()
	local cumulative = 0

	for i, probability in ipairs(probabilities) do
		cumulative += probability

		if roll < cumulative then
			if i == 1 then
				humanoid.Jump = true
			elseif i == 2 then
				humanoid:Move(Vector3.new(0, 0, 1))
			elseif i == 3 then
				humanoid:Move(Vector3.new(0, 0, -1))
			elseif i == 4 then
				humanoid:Move(Vector3.new(0, 0, 0))
			end

			return i
		end
	end

	humanoid:Move(Vector3.new(0, 0, 0))
	return 4
end

local function advanceOldWaypoint()
	local nextWaypoint = oldWaypointsFolder:FindFirstChild(tostring(oldTargetWaypoint + 1))

	if nextWaypoint then
		oldTargetWaypoint += 1
	else
		oldTargetWaypoint = 1
	end

	previousOldDistance = nil
	updateGoals()
end

local function advanceNewWaypoint()
	local nextWaypoint = newWaypointsFolder:FindFirstChild(tostring(newTargetWaypoint + 1))

	if nextWaypoint then
		newTargetWaypoint += 1
	else
		newTargetWaypoint = 1
	end

	previousNewDistance = nil
	updateGoals()
end

local function getReward()
	local stageTwo = isStageTwo()
	local activeWaypoint = stageTwo
		and getNewWaypoint(newTargetWaypoint)
		or getOldWaypoint(oldTargetWaypoint)
	local activeDistance = activeWaypoint
		and getPlanarDistanceToPosition(activeWaypoint.Position)
		or nil
	local previousDistance = stageTwo and previousNewDistance or previousOldDistance
	local reward = -livingPenalty

	if activeDistance and previousDistance then
		reward += (previousDistance - activeDistance) * chaseProgressRewardScale
	end

	if stageTwo then
		previousOldDistance = nil
		previousNewDistance = activeDistance
	else
		previousOldDistance = activeDistance
		previousNewDistance = nil
	end

	if activeDistance and activeDistance <= chaseReachDistance then
		reward += chaseReachReward * newSideTerminalMultiplier
		if evaluationTaskOverride ~= nil then
			evaluationWaypointCompletions += 1
		end
		if stageTwo then
			advanceNewWaypoint()
		else
			advanceOldWaypoint()
		end
	end

	return reward
end

local function getInputs()
	local inputs = {}
	local rayCount = 16
	local rayLength = 20
	local spread = 15
	local halfSpread = spread / 2
	local velocitySensitivity = 0.1
	local waypointSensitivity = 0.1

	local params = RaycastParams.new()
	params.FilterType = Enum.RaycastFilterType.Exclude
	params.CollisionGroup = humanoidRootPart.CollisionGroup
	params.FilterDescendantsInstances = raycastExclusions

	local stageTwo = isStageTwo()
	local oldWaypoint = getOldWaypoint(oldTargetWaypoint)
	local newWaypoint = getNewWaypoint(newTargetWaypoint)

	local oldGoalPosition = oldWaypoint and oldWaypoint.Position or humanoidRootPart.Position
	local newGoalPosition = newWaypoint and newWaypoint.Position or humanoidRootPart.Position

	local relativeOld = stageTwo
		and Vector3.zero
		or humanoidRootPart.CFrame:PointToObjectSpace(oldGoalPosition)
	local relativeNew = stageTwo
		and humanoidRootPart.CFrame:PointToObjectSpace(newGoalPosition)
		or Vector3.zero

	table.insert(inputs, math.tanh(relativeOld.Y * waypointSensitivity))
	table.insert(inputs, math.tanh(relativeOld.Z * waypointSensitivity))
	table.insert(inputs, math.tanh(relativeNew.Y * waypointSensitivity))
	table.insert(inputs, math.tanh(relativeNew.Z * waypointSensitivity))

	local currentVelocity = humanoidRootPart.AssemblyLinearVelocity

	table.insert(inputs, math.tanh(currentVelocity.Y * velocitySensitivity))
	table.insert(inputs, math.tanh(currentVelocity.Z * velocitySensitivity))
	table.insert(inputs, humanoidRootPart.Orientation.Y / 180)

	for i = 1, rayCount do
		local verticalOffset = ((i - 1) / (rayCount - 1) * spread) - halfSpread
		local origin = humanoidRootPart.Position + Vector3.new(0, verticalOffset, 0)
		local direction = Vector3.new(0, 0, rayLength)
		local result = workspace:Raycast(origin, direction, params)
		local distance = result and result.Distance or rayLength

		updateRayPart("ForwardRay" .. i, origin, direction, distance)
		table.insert(inputs, distance / rayLength)
	end

	for i = 1, rayCount do
		local verticalOffset = ((i - 1) / (rayCount - 1) * spread) - halfSpread
		local origin = humanoidRootPart.Position + Vector3.new(0, verticalOffset, 0)
		local direction = Vector3.new(0, 0, -rayLength)
		local result = workspace:Raycast(origin, direction, params)
		local distance = result and result.Distance or rayLength

		updateRayPart("BackwardRay" .. i, origin, direction, distance)
		table.insert(inputs, distance / rayLength)
	end

	return inputs
end

local function reset()
	humanoid.WalkSpeed = getWalkSpeedForStage()
	humanoidRootPart.CFrame = CFrame.new(spawner.Position + Vector3.new(0, 4, 0))
	humanoidRootPart.AssemblyLinearVelocity = Vector3.new(0, 0, 0)
	humanoidRootPart.AssemblyAngularVelocity = Vector3.new(0, 0, 0)
	oldTargetWaypoint = 1
	newTargetWaypoint = 1
	previousOldDistance = nil
	previousNewDistance = nil
	updateGoals()
end

local function getParentActivationsList(inputs)
	return pnnModule.forwardArchivedColumns(parentNetworks, inputs, lateralScale)
end

local function getEvaluationOutputs(inputs)
	local parentActivationsList = pnnModule.forwardArchivedColumns(
		parentNetworks,
		inputs,
		lateralScale
	)
	local activations = pnnModule.forwardPass(
		network,
		inputs,
		parentActivationsList,
		lateralScale
	)
	return activations[#activations]
end

local function doGreedyAction(rawOutputs)
	local bestIndex = 1
	local bestValue = rawOutputs[1]
	for index = 2, #rawOutputs do
		if rawOutputs[index] > bestValue then
			bestIndex = index
			bestValue = rawOutputs[index]
		end
	end

	if bestIndex == 1 then
		humanoid.Jump = true
	elseif bestIndex == 2 then
		humanoid:Move(Vector3.new(0, 0, 1))
	elseif bestIndex == 3 then
		humanoid:Move(Vector3.new(0, 0, -1))
	else
		humanoid:Move(Vector3.new(0, 0, 0))
	end
end

local function runGreedyEvaluation(taskIsB)
	local savedOverride = evaluationTaskOverride
	local savedWaypointCompletions = evaluationWaypointCompletions
	local savedCFrame = humanoidRootPart.CFrame
	local savedLinearVelocity = humanoidRootPart.AssemblyLinearVelocity
	local savedAngularVelocity = humanoidRootPart.AssemblyAngularVelocity
	local savedWalkSpeed = humanoid.WalkSpeed
	local savedJump = humanoid.Jump
	local savedMoveDirection = humanoid.MoveDirection
	local savedOldTargetWaypoint = oldTargetWaypoint
	local savedNewTargetWaypoint = newTargetWaypoint
	local savedPreviousOldDistance = previousOldDistance
	local savedPreviousNewDistance = previousNewDistance

	evaluationTaskOverride = taskIsB
	evaluationWaypointCompletions = 0
	local rewardSum = 0
	local successfulEpisodes = 0

	for _ = 1, evaluationEpisodes do
		humanoid.Jump = false
		humanoid:Move(Vector3.new(0, 0, 0))
		reset()
		local completionsAtEpisodeStart = evaluationWaypointCompletions

		for _ = 1, evaluationStepsPerEpisode do
			task.wait()
			local inputs = getInputs()
			local rawOutputs = getEvaluationOutputs(inputs)
			doGreedyAction(rawOutputs)
			rewardSum += getReward()
		end

		if evaluationWaypointCompletions > completionsAtEpisodeStart then
			successfulEpisodes += 1
		end
	end

	local result = {
		episodes = evaluationEpisodes,
		stepsPerEpisode = evaluationStepsPerEpisode,
		totalSteps = evaluationEpisodes * evaluationStepsPerEpisode,
		meanRewardPerStep = rewardSum / (evaluationEpisodes * evaluationStepsPerEpisode),
		waypointCompletions = evaluationWaypointCompletions,
		successfulEpisodes = successfulEpisodes,
		successRate = successfulEpisodes / evaluationEpisodes,
	}

	evaluationTaskOverride = savedOverride
	evaluationWaypointCompletions = savedWaypointCompletions
	oldTargetWaypoint = savedOldTargetWaypoint
	newTargetWaypoint = savedNewTargetWaypoint
	previousOldDistance = savedPreviousOldDistance
	previousNewDistance = savedPreviousNewDistance
	humanoid.WalkSpeed = savedWalkSpeed
	humanoidRootPart.CFrame = savedCFrame
	humanoidRootPart.AssemblyLinearVelocity = savedLinearVelocity
	humanoidRootPart.AssemblyAngularVelocity = savedAngularVelocity
	humanoid.Jump = savedJump
	humanoid:Move(savedMoveDirection)

	return result
end

local function ensureEvaluationForCompletedGeneration()
	if batchesCompleted <= 0 or batchesCompleted % stageLengthBatches ~= 0 then
		return
	end

	local completedGeneration = math.floor(batchesCompleted / stageLengthBatches)
	if evaluationByGeneration[completedGeneration] then
		return
	end

	local tasks = {
		A = runGreedyEvaluation(false),
	}
	if completedGeneration >= 2 then
		tasks.B = runGreedyEvaluation(true)
	end

	evaluationByGeneration[completedGeneration] = {
		generation = completedGeneration,
		afterBatches = batchesCompleted,
		tasks = tasks,
	}
end

local legacyExperiment = save and loadLegacyData(key) or newExperiment()
local currentRunId
local currentSeed
local currentRunNumber = 0
local currentSeedsPerHead = #newHeadSeeds
local currentParentOutputLateralConnection = false

local function getRunId(seed, useParentOutput)
	return string.format("parent_%s_seed_%d", tostring(useParentOutput), seed)
end

local seedsByParentOutput = {}
local totalRuns = 0
for _, useParentOutput in ipairs(parentOutputLateralConnectionOptions) do
	local headSeeds = table.clone(newHeadSeeds)
	if type(legacyExperiment.runs[getRunId(legacySeed, useParentOutput)]) == "table" then
		table.insert(headSeeds, legacySeed)
	end
	seedsByParentOutput[useParentOutput] = headSeeds
	totalRuns += #headSeeds
end

local function printProgress(label)
	local remaining = math.max(0, maxBatches - batchesCompleted)

	print(string.format(
		"%s | %s | run %d/%d | parent output %s | seed %d | batch %d/%d | %d left",
		label,
		aiType,
		currentRunNumber,
		totalRuns,
		tostring(currentParentOutputLateralConnection),
		currentSeed or 0,
		batchesCompleted,
		maxBatches,
		remaining
	))
end

local function saveCurrentRun(completed)
	if not save or not currentRunId then
		return true
	end

	local runData = {
		version = checkpointVersion,
		configFingerprint = configFingerprint,
		expectedBatches = expectedBatches,
		completed = completed == true,
		seed = currentSeed,
		usedParentOutputLateralConnection = currentParentOutputLateralConnection,
		rngDrawCount = rngDrawCount,
		network = network,
		parentNetworks = parentNetworks,
		batchesCompleted = batchesCompleted,
		lastRebirthStageNumber = lastRebirthStageNumber,
		generation = generation,
		averageRewardsHistory = averageRewardsHistory,
		evaluationByGeneration = evaluationByGeneration,
	}

	return saveData(key, currentRunId, runData, currentSeedsPerHead)
end

game:BindToClose(function()
	if currentRunId and batchesCompleted < maxBatches then
		if saveCurrentRun(false) then
			printProgress("Saved on shutdown")
		else
			printProgress("SAVE FAILED on shutdown")
		end
	end
end)

local completedRunOffset = 0
for _, useParentOutputLateralConnection in ipairs(parentOutputLateralConnectionOptions) do
	local seedResults = {}
	local headSeeds = seedsByParentOutput[useParentOutputLateralConnection]
	currentSeedsPerHead = #headSeeds

	for seedIndex, seed in ipairs(headSeeds) do
		currentSeed = seed
		currentRunNumber = completedRunOffset + seedIndex
		currentParentOutputLateralConnection = useParentOutputLateralConnection
		currentRunId = getRunId(seed, useParentOutputLateralConnection)

		local data = loadRunData(key, currentRunId, legacyExperiment)

		if type(data) ~= "table"
			or not data.network
			or data.version ~= checkpointVersion
			or data.configFingerprint ~= configFingerprint
			or data.expectedBatches ~= expectedBatches
		then
			data = nil
		end

		rng = createTrackedRandom(seed, data and data.rngDrawCount or 0)
		batchesCompleted = data and data.batchesCompleted or 0
		averageRewardsHistory = data and data.averageRewardsHistory or {}
		evaluationByGeneration = data and data.evaluationByGeneration or {}
		table.clear(batch)

		if data then
			network = data.network
			parentNetworks = data.parentNetworks or {}
			lastRebirthStageNumber = data.lastRebirthStageNumber or getStageNumber()
			generation = data.generation or 1
		else
			network = pnnModule.getNetwork(networkShape, 0, nil, false, rng)
			parentNetworks = {}
			lastRebirthStageNumber = getStageNumber()
			generation = 1
		end

		local progressLabel = data
			and ((data.completed or batchesCompleted >= maxBatches) and "Loaded complete" or "Resumed")
			or "Started"
		printProgress(progressLabel)
		ensureEvaluationForCompletedGeneration()

while batchesCompleted < maxBatches do
	local stageNumberForBatch = getStageNumber()

	if stageNumberForBatch ~= lastRebirthStageNumber then
		local verificationInputs = getInputs()
		local previousParentActivations = getParentActivationsList(verificationInputs)
		local outputsBeforeArchive = pnnModule.forwardPass(
			network,
			verificationInputs,
			previousParentActivations,
			lateralScale
		)

		table.insert(parentNetworks, network)

		local archivedActivations = getParentActivationsList(verificationInputs)
		local outputsAfterArchive = archivedActivations[#archivedActivations]
		local archiveError = pnnModule.maxAbsOutputDifference(
			outputsBeforeArchive,
			outputsAfterArchive
		)
		assert(
			archiveError <= 1e-12,
			string.format("PNN archive changed frozen-column logits (max error %.17g)", archiveError)
		)

		network = pnnModule.getNetwork(
			networkShape,
			#parentNetworks,
			networkShape,
			useParentOutputLateralConnection,
			rng
		)
		lastRebirthStageNumber = stageNumberForBatch
		generation += 1
	end

	reset()

	for i = 1, batchSize do
		task.wait()

		local inputs = getInputs()
		local parentActivationsList = getParentActivationsList(inputs)

		local activations = pnnModule.forwardPass(network, inputs, parentActivationsList, lateralScale)
		local rawOutputs = activations[#activations]
		local actionIndex = doAction(rawOutputs)
		local reward = getReward()

		batch[i] = {
			activations = activations,
			reward = reward,
			actionIndex = actionIndex,
			rawOutputs = rawOutputs,
			parentActivationsList = parentActivationsList,
			lateralScale = lateralScale
		}
	end

	local rewards = {}
	local rawRewardSum = 0

	for i = 1, #batch do
		rewards[i] = batch[i].reward
		rawRewardSum += batch[i].reward
	end

	local averageReward = rawRewardSum / #batch

	averageRewardsHistory[batchesCompleted + 1] = averageReward

	local newRewards = getDiscountedRewards(rewards)

	local rewardSum = 0

	for i = 1, #newRewards do
		rewardSum += newRewards[i]
	end

	local rewardMean = rewardSum / #newRewards
	local varianceSum = 0

	for i = 1, #newRewards do
		varianceSum += math.pow(newRewards[i] - rewardMean, 2)
	end

	local rewardStdDev = math.sqrt(varianceSum / #newRewards)

	if rewardStdDev == 0 then
		rewardStdDev = 1e-8
	end

	for i = 1, #newRewards do
		batch[i].reward = (newRewards[i] - rewardMean) / rewardStdDev
	end

	local shuffledBatch = shuffleList(batch)
	local accumulatedGradients = pnnModule.getBlankGradients(network)

	for i = 1, #shuffledBatch do
		local memory = shuffledBatch[i]
		local outputErrors = {}
		local probabilities = softmax(memory.rawOutputs)

		for j = 1, 4 do
			if j == memory.actionIndex then
				outputErrors[j] = memory.reward * (1 - probabilities[j])
			else
				outputErrors[j] = memory.reward * (0 - probabilities[j])
			end
		end

		local stepGradients = pnnModule.backpropagate(
			network,
			memory.activations,
			outputErrors,
			memory.parentActivationsList,
			memory.lateralScale
		)
		pnnModule.addGradients(accumulatedGradients, stepGradients)
	end

	pnnModule.updateWeights(network, accumulatedGradients)

	batchesCompleted += 1
	ensureEvaluationForCompletedGeneration()

	if save and batchesCompleted < maxBatches and batchesCompleted % checkpointEveryBatches == 0 then
		if saveCurrentRun(false) then
			printProgress("Checkpoint saved")
		else
			printProgress("CHECKPOINT SAVE FAILED")
		end
	end

	table.clear(batch)
end

ensureEvaluationForCompletedGeneration()

if saveCurrentRun(true) then
	printProgress("Completed")
else
	printProgress("FINAL SAVE FAILED")
end

table.insert(seedResults, {
	seed = seed,
	averageRewardPerBatch = table.clone(averageRewardsHistory),
	evaluationByGeneration = evaluationByGeneration
})
	end

local groupedResultTable = {
	version = checkpointVersion,
	configFingerprint = configFingerprint,
	expectedBatches = expectedBatches,
	stageSequence = {"A", "B"},
	stageLengthBatches = stageLengthBatches,
	architecture = aiType,
	usedParentOutputLateralConnection = useParentOutputLateralConnection,
	seedsPerHead = #headSeeds,
	storageVersion = runStorageVersion,
	seeds = seedResults
}
print("WEANNET_GROUPED_RESULT", groupedResultTable)
completedRunOffset += #headSeeds
end
