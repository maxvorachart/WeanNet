local DataStoreService = game:GetService("DataStoreService")
local HttpService = game:GetService("HttpService")
local ServerStorage = game:GetService("ServerStorage")

local aiDataStore = DataStoreService:GetDataStore("GenerationalWeanNetSpeedStageTest1")

local oldWaypointsFolder = workspace:WaitForChild("WaypointsFolder")
local newWaypointsFolder = workspace:WaitForChild("TrueWaypointsFolder")
local divider = workspace:WaitForChild("Divider")
local spawner = workspace:WaitForChild("Spawner")

local aiModule = require(ServerStorage:WaitForChild("AIModule"))

local agent = workspace:WaitForChild("Agent1")
local agentColor = agent:WaitForChild("Highlight").FillColor
local defaultColor = Color3.fromRGB(255, 255, 0)
local raycastsFolder = agent:WaitForChild("RaycastsFolder")
local humanoid = agent:WaitForChild("Humanoid")
local humanoidRootPart = agent:WaitForChild("HumanoidRootPart")

local save = true
local key = 74

local gamma = 0.95
local temperature = 1.0
local batchSize = 512
local stageLengthBatches = 1000
local maxBatches = stageLengthBatches * 2
local networkShape = {39, 64, 64, 4}
local seeds = {1, 2, 3, 4, 5}
local parentOutputLateralConnectionOptions = {false, true}
local checkpointNamespace = "MultiSeedExperimentV4"
local checkpointVersion = 4
local runStorageVersion = 5
local maxEncodedCheckpointBytes = 3500000
local checkpointMaxAttempts = 5
local expectedBatches = maxBatches
local configFingerprint = "reviewer-v4|architecture=WeanNet|key=74|seeds=1,2,3,4,5|tasks=A-B-persistent-left-higher-right|stageLength=1000|maxBatches=2000|batchSize=512|shape=39-64-64-4|gamma=0.95|temperature=1.0|oldTerminalReward=8|newTerminalMultiplier=2.0|stage2BothTargets=true|stage2BothGoalCoordinates=true|raycastExclusions=agents+waypoints|waypointDisplay=stage1Old-stage2Both|lateralSchedule=linear1to0over250|parentRemovedAtZero=true|pairedInit=PNNReset|parentOutputOptions=false,true|evaluation=greedy|evaluationEpisodes=5|evaluationStepsPerEpisode=256|seenTasksOnly=true|evaluationSuccess=anyWaypointCompletion|evaluationSideCompletionCounts=true|evaluationResetActionState=true"
local checkpointEveryBatches = 5
local evaluationEpisodes = 5
local evaluationStepsPerEpisode = 256
local rngDrawCount = 0
local rng
local checkpointPersistenceEnabled = save
local checkpointPersistenceFailure = nil

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
local newSideTerminalMultiplier = 2.0

local oldTargetWaypoint = 1
local newTargetWaypoint = 1
local previousOldDistance
local previousNewDistance
local batch = {}
local averageRewardsHistory = {}
local evaluationByGeneration = {}
local evaluationTaskOverride = nil
local evaluationWaypointCompletions = 0
local evaluationOldWaypointCompletions = 0
local evaluationNewWaypointCompletions = 0
local parentRemovalBatches = {}
local batchesCompleted = 0

local initialLateralScale = 1.0
local lateralDecayBatches = 250
local useLateralScaleDecay = true

local batchesSinceRebirth = 0
local lastRebirthStageNumber = 1
local generation = 1
local network
local parentNetwork

local aiType = "WeanNet"

local function getCurrentLateralScale()
	if not parentNetwork then
		return 0
	end
	if not useLateralScaleDecay then
		return initialLateralScale
	end
	local progress = math.clamp(batchesSinceRebirth / lateralDecayBatches, 0, 1)
	return initialLateralScale * (1 - progress)
end

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
	return isStageTwo() and "PersistentChoiceParkour" or "NormalParkour"
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
	local stageTwo = isStageTwo()

	for _, waypoint in ipairs(oldWaypointsFolder:GetDescendants()) do
		if waypoint:IsA("BasePart") then
			waypoint.Transparency = 0
			waypoint.CanCollide = false
			waypoint.CanTouch = false
			waypoint.CanQuery = false
		end
	end

	for _, waypoint in ipairs(newWaypointsFolder:GetDescendants()) do
		if waypoint:IsA("BasePart") then
			waypoint.Transparency = stageTwo and 0 or 1
			waypoint.CanCollide = false
			waypoint.CanTouch = false
			waypoint.CanQuery = false
		end
	end
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

local function disableCheckpointPersistence(reason)
	if not checkpointPersistenceEnabled then
		return
	end

	checkpointPersistenceEnabled = false
	checkpointPersistenceFailure = tostring(reason)
	warn(string.format(
		"Checkpoint persistence disabled: %s. Training will continue in memory; copy the final grouped result from the server output.",
		checkpointPersistenceFailure
		))
end

local function getRunStorageKey(keyValue, runId)
	local storageKey = string.format("MSE5:%s:%s", tostring(keyValue), runId)
	assert(#storageKey <= 50, "run checkpoint key exceeds DataStore's 50-character limit")
	return storageKey
end

local function isNonRetryableDataStoreError(errorMessage)
	local message = string.lower(tostring(errorMessage))
	return string.find(message, "valuetoolarge", 1, true) ~= nil
		or string.find(message, "serialized value exceeds", 1, true) ~= nil
		or string.find(message, "cantstorevalue", 1, true) ~= nil
		or string.find(message, "can't store", 1, true) ~= nil
		or string.find(message, "storage quota", 1, true) ~= nil
		or string.find(message, "quota exceeded", 1, true) ~= nil
		or string.find(message, "storage limit", 1, true) ~= nil
		or string.find(message, "incompatible existing run checkpoint", 1, true) ~= nil
end

local function waitBeforeDataStoreRetry(attempt)
	task.wait(math.min(8, 2 ^ (attempt - 1)))
end

local function loadRunData(keyValue, runId)
	if not checkpointPersistenceEnabled then
		return nil
	end

	local storageKey = getRunStorageKey(keyValue, runId)
	for attempt = 1, checkpointMaxAttempts do
		local success, storedEnvelope = pcall(function()
			return aiDataStore:GetAsync(storageKey)
		end)

		if success then
			if storedEnvelope == nil then
				return nil
			end

			if type(storedEnvelope) ~= "table"
				or storedEnvelope.storageVersion ~= runStorageVersion
				or storedEnvelope.namespace ~= checkpointNamespace
				or tostring(storedEnvelope.baseKey) ~= tostring(keyValue)
				or storedEnvelope.runId ~= runId
				or type(storedEnvelope.run) ~= "table"
			then
				disableCheckpointPersistence("existing run checkpoint is incompatible")
				return nil
			end

			return storedEnvelope.run
		end

		warn(string.format(
			"Run checkpoint load failed for %s (attempt %d): %s",
			runId,
			attempt,
			tostring(storedEnvelope)
			))
		if isNonRetryableDataStoreError(storedEnvelope) then
			disableCheckpointPersistence(storedEnvelope)
			return nil
		end
		if attempt < checkpointMaxAttempts then
			waitBeforeDataStoreRetry(attempt)
		end
	end

	disableCheckpointPersistence("run checkpoint load failed after retry limit")
	return nil
end

local function saveData(keyValue, runId, runData)
	if not checkpointPersistenceEnabled then
		return false
	end

	local storageKey = getRunStorageKey(keyValue, runId)
	local envelope = {
		storageVersion = runStorageVersion,
		namespace = checkpointNamespace,
		baseKey = keyValue,
		runId = runId,
		run = runData,
	}

	local encodeSuccess, encodedOrError = pcall(function()
		return HttpService:JSONEncode(envelope)
	end)
	if not encodeSuccess then
		disableCheckpointPersistence("checkpoint serialization failed: " .. tostring(encodedOrError))
		return false
	end
	if #encodedOrError > maxEncodedCheckpointBytes then
		disableCheckpointPersistence(string.format(
			"checkpoint for %s is %d bytes, above the %d-byte safety limit",
			runId,
			#encodedOrError,
			maxEncodedCheckpointBytes
			))
		return false
	end

	for attempt = 1, checkpointMaxAttempts do
		local success, errorMessage = pcall(function()
			aiDataStore:UpdateAsync(storageKey, function(existing)
				if existing ~= nil then
					if type(existing) ~= "table"
						or existing.storageVersion ~= runStorageVersion
						or existing.namespace ~= checkpointNamespace
						or tostring(existing.baseKey) ~= tostring(keyValue)
						or existing.runId ~= runId
					then
						error("incompatible existing run checkpoint")
					end
				end
				return envelope
			end)
		end)

		if success then
			return true
		end

		warn(string.format("Checkpoint save failed (attempt %d): %s", attempt, tostring(errorMessage)))
		if isNonRetryableDataStoreError(errorMessage) then
			disableCheckpointPersistence(errorMessage)
			return false
		end
		if attempt < checkpointMaxAttempts then
			waitBeforeDataStoreRetry(attempt)
		end
	end

	disableCheckpointPersistence("checkpoint save failed after retry limit")
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
	local oldWaypoint = getOldWaypoint(oldTargetWaypoint)
	local newWaypoint = stageTwo and getNewWaypoint(newTargetWaypoint) or nil
	local oldDistance = oldWaypoint
		and getPlanarDistanceToPosition(oldWaypoint.Position)
		or nil
	local newDistance = newWaypoint
		and getPlanarDistanceToPosition(newWaypoint.Position)
		or nil
	local reward = -livingPenalty

	if stageTwo then
		-- Task B keeps the old route as a local optimum while adding a more
		-- valuable new route. Reward progress toward either route, not the
		-- sum, so moving right is not penalized for leaving the old route.
		local bestProgress = 0
		if oldDistance and previousOldDistance then
			bestProgress = math.max(bestProgress, previousOldDistance - oldDistance)
		end
		if newDistance and previousNewDistance then
			bestProgress = math.max(bestProgress, previousNewDistance - newDistance)
		end
		reward += bestProgress * chaseProgressRewardScale
	elseif oldDistance and previousOldDistance then
		reward += (previousOldDistance - oldDistance) * chaseProgressRewardScale
	end

	previousOldDistance = oldDistance
	previousNewDistance = newDistance

	-- Only the new/right route receives the terminal multiplier in Task B.
	if stageTwo and newDistance and newDistance <= chaseReachDistance then
		reward += chaseReachReward * newSideTerminalMultiplier
		if evaluationTaskOverride ~= nil then
			evaluationWaypointCompletions += 1
			evaluationNewWaypointCompletions += 1
		end
		advanceNewWaypoint()
	elseif oldDistance and oldDistance <= chaseReachDistance then
		reward += chaseReachReward
		if evaluationTaskOverride ~= nil then
			evaluationWaypointCompletions += 1
			evaluationOldWaypointCompletions += 1
		end
		advanceOldWaypoint()
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

	local relativeOld = humanoidRootPart.CFrame:PointToObjectSpace(oldGoalPosition)
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

local function getEvaluationOutputs(inputs)
	local currentLateralScale = getCurrentLateralScale()
	local parentActivations = nil
	local parentOutputs = nil
	if parentNetwork then
		parentActivations = aiModule.forwardPass(parentNetwork, inputs)
		parentOutputs = parentActivations[#parentActivations]
	end
	local activations = aiModule.forwardPass(
		network,
		inputs,
		parentActivations,
		currentLateralScale,
		parentOutputs
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
	local savedOldWaypointCompletions = evaluationOldWaypointCompletions
	local savedNewWaypointCompletions = evaluationNewWaypointCompletions
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
	evaluationOldWaypointCompletions = 0
	evaluationNewWaypointCompletions = 0
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
		oldWaypointCompletions = evaluationOldWaypointCompletions,
		newWaypointCompletions = evaluationNewWaypointCompletions,
		successfulEpisodes = successfulEpisodes,
		successRate = successfulEpisodes / evaluationEpisodes,
	}

	evaluationTaskOverride = savedOverride
	evaluationWaypointCompletions = savedWaypointCompletions
	evaluationOldWaypointCompletions = savedOldWaypointCompletions
	evaluationNewWaypointCompletions = savedNewWaypointCompletions
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

local savedExperiment = newExperiment()
local currentRunId
local currentSeed
local currentRunNumber = 0
local currentParentOutputLateralConnection = false
local totalRuns = #seeds * #parentOutputLateralConnectionOptions

local function getRunId(seed, useParentOutput)
	return string.format("parent_%s_seed_%d", tostring(useParentOutput), seed)
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
	if not checkpointPersistenceEnabled then
		return false
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
		parentNetwork = parentNetwork,
		batchesCompleted = batchesCompleted,
		batchesSinceRebirth = batchesSinceRebirth,
		lastRebirthStageNumber = lastRebirthStageNumber,
		generation = generation,
		averageRewardsHistory = averageRewardsHistory,
		evaluationByGeneration = evaluationByGeneration,
		parentRemovalBatches = parentRemovalBatches,
	}

	savedExperiment.runs[currentRunId] = runData
	local success = saveData(key, currentRunId, runData)

	if success then
		savedExperiment.runs[currentRunId] = runData
	end

	return success
end

game:BindToClose(function()
	if currentRunId and batchesCompleted < maxBatches then
		if checkpointPersistenceEnabled and saveCurrentRun(false) then
			printProgress("Saved on shutdown")
		elseif not checkpointPersistenceEnabled then
			printProgress("Shutdown without checkpoint persistence")
		else
			printProgress("SAVE FAILED on shutdown")
		end
	end
end)

for parentOptionIndex, useParentOutputLateralConnection in ipairs(parentOutputLateralConnectionOptions) do
	local seedResults = {}

	for seedIndex, seed in ipairs(seeds) do
		currentSeed = seed
		currentRunNumber = (parentOptionIndex - 1) * #seeds + seedIndex
		currentParentOutputLateralConnection = useParentOutputLateralConnection
		currentRunId = getRunId(seed, useParentOutputLateralConnection)

		local data = savedExperiment.runs[currentRunId]
		if data == nil and checkpointPersistenceEnabled then
			data = loadRunData(key, currentRunId)
		end

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
		parentRemovalBatches = data and data.parentRemovalBatches or {}
		table.clear(batch)

		if data then
			network = data.network
			parentNetwork = data.parentNetwork
			batchesSinceRebirth = data.batchesSinceRebirth or 0
			lastRebirthStageNumber = data.lastRebirthStageNumber or getStageNumber()
			generation = data.generation or 1
		else
			network = aiModule.getNetwork(networkShape, nil, 0, rng)
			parentNetwork = nil
			batchesSinceRebirth = 0
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
				parentNetwork = aiModule.compressNetwork(network, networkShape)
				network = aiModule.getNetwork(networkShape, networkShape, useParentOutputLateralConnection and networkShape[#networkShape] or 0, rng)
				batchesSinceRebirth = 0
				lastRebirthStageNumber = stageNumberForBatch
				generation += 1
			end

			reset()

			local currentLateralScale = getCurrentLateralScale()

			if parentNetwork and currentLateralScale <= 0 then
				parentNetwork = nil
				if parentRemovalBatches[#parentRemovalBatches] ~= batchesCompleted then
					table.insert(parentRemovalBatches, batchesCompleted)
				end
			end

			for i = 1, batchSize do
				task.wait()

				local inputs = getInputs()
				local parentActivations = nil
				local parentOutputs = nil

				if parentNetwork then
					parentActivations = aiModule.forwardPass(parentNetwork, inputs)
					parentOutputs = parentActivations[#parentActivations]
				end

				local activations = aiModule.forwardPass(network, inputs, parentActivations, currentLateralScale, parentOutputs)
				local rawOutputs = activations[#activations]
				local actionIndex = doAction(rawOutputs)
				local reward = getReward()

				batch[i] = {
					activations = activations,
					reward = reward,
					actionIndex = actionIndex,
					rawOutputs = rawOutputs,
					parentActivations = parentActivations,
					parentOutputs = parentOutputs,
					lateralScale = currentLateralScale
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
			local accumulatedGradients = aiModule.getBlankGradients(network)

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

				local stepGradients = aiModule.backpropagate(
					network,
					memory.activations,
					outputErrors,
					memory.parentActivations,
					memory.lateralScale,
					memory.parentOutputs
				)
				aiModule.addGradients(accumulatedGradients, stepGradients)
			end

			aiModule.updateWeights(network, accumulatedGradients)

			batchesCompleted += 1
			ensureEvaluationForCompletedGeneration()

			if parentNetwork then
				batchesSinceRebirth += 1
			end

			if checkpointPersistenceEnabled and batchesCompleted < maxBatches and batchesCompleted % checkpointEveryBatches == 0 then
				if saveCurrentRun(false) then
					printProgress("Checkpoint saved")
				else
					printProgress("CHECKPOINT SAVE FAILED")
				end
			end

			table.clear(batch)
		end

		ensureEvaluationForCompletedGeneration()

		if checkpointPersistenceEnabled and saveCurrentRun(true) then
			printProgress("Completed")
		elseif not checkpointPersistenceEnabled then
			printProgress("Completed without checkpoint persistence")
		else
			printProgress("FINAL SAVE FAILED")
		end

		table.insert(seedResults, {
			seed = seed,
			averageRewardPerBatch = table.clone(averageRewardsHistory),
			evaluationByGeneration = evaluationByGeneration,
			parentRemovalBatches = table.clone(parentRemovalBatches),
			parentPhysicallyRemovedAfterDecay = true
		})
	end

	local groupedResultTable = {
		version = checkpointVersion,
		configFingerprint = configFingerprint,
		expectedBatches = expectedBatches,
		stageSequence = {"A", "B"},
		stageLengthBatches = stageLengthBatches,
		architecture = aiType,
		checkpointStorage = "per-run MSE5 shards",
		checkpointPersistenceEnabled = checkpointPersistenceEnabled,
		checkpointPersistenceFailure = checkpointPersistenceFailure,
		usedParentOutputLateralConnection = useParentOutputLateralConnection,
		seeds = seedResults
	}
	print("WEANNET_GROUPED_RESULT", groupedResultTable)
end
