local DataStoreService = game:GetService("DataStoreService")
local ServerStorage = game:GetService("ServerStorage")

local aiDataStore = DataStoreService:GetDataStore("ResetReinforceSpeedStageTest1")

local oldWaypointsFolder = workspace:WaitForChild("WaypointsFolder")
local newWaypointsFolder = workspace:WaitForChild("TrueWaypointsFolder")
local divider = workspace:WaitForChild("Divider")
local spawner = workspace:WaitForChild("Spawner")

local aiModule = require(ServerStorage:WaitForChild("AIModule"))

local agent = workspace:WaitForChild("Agent3")
local agentColor = agent:WaitForChild("Highlight").FillColor
local defaultColor = Color3.fromRGB(255, 255, 0)
local raycastsFolder = agent:WaitForChild("RaycastsFolder")
local humanoid = agent:WaitForChild("Humanoid")
local humanoidRootPart = agent:WaitForChild("HumanoidRootPart")

local save = true
local key = 66

local gamma = 0.95
local temperature = 1.0
local batchSize = 512
local stageLengthBatches = 1000
local maxBatches = stageLengthBatches * 2
local networkShape = {39, 64, 64, 4}

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
local batchesCompleted = 0
local network
local networkStageNumber

local aiType = "Reset"

local function getStageNumber()
	return math.floor(batchesCompleted / stageLengthBatches) + 1
end

local function isStageTwo()
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
	local stageTwo = isStageTwo()

	for _, waypoint in ipairs(oldWaypointsFolder:GetChildren()) do
		waypoint.Transparency = 0
		waypoint.CanTouch = true
		waypoint.CanCollide = false
		waypoint.CanQuery = true

		if waypoint.Name == tostring(oldTargetWaypoint) then
			waypoint.Color = agentColor
		else
			waypoint.Color = defaultColor
		end
	end

	for _, waypoint in ipairs(newWaypointsFolder:GetChildren()) do
		if stageTwo then
			waypoint.Transparency = 0
			waypoint.CanTouch = true
			waypoint.CanCollide = false
			waypoint.CanQuery = true

			if waypoint.Name == tostring(newTargetWaypoint) then
				waypoint.Color = agentColor
			else
				waypoint.Color = defaultColor
			end
		else
			waypoint.Transparency = 1
			waypoint.CanTouch = false
			waypoint.CanCollide = false
			waypoint.CanQuery = false
		end
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

local function loadData(keyValue)
	for i = 1, 5 do
		local success, data = pcall(function()
			return aiDataStore:GetAsync(keyValue)
		end)

		if success then
			return data
		end

		task.wait(1)
	end
end

local function saveData(keyValue, data)
	for i = 1, 5 do
		local success = pcall(function()
			aiDataStore:SetAsync(keyValue, data)
		end)

		if success then
			return
		end

		task.wait(1)
	end
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
		local j = math.random(1, i)
		targetList[i], targetList[j] = targetList[j], targetList[i]
	end

	return targetList
end

local function doAction(rawOutputs)
	local probabilities = softmax(rawOutputs)
	local roll = math.random()
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
	local oldDistance = oldWaypoint and getPlanarDistanceToPosition(oldWaypoint.Position) or nil
	local newDistance = newWaypoint and getPlanarDistanceToPosition(newWaypoint.Position) or nil
	local onNewSide = stageTwo and isOnNewSide()

	local reward = -livingPenalty

	if onNewSide then
		if newDistance and previousNewDistance then
			reward += (previousNewDistance - newDistance) * chaseProgressRewardScale
		end
	else
		if oldDistance and previousOldDistance then
			reward += (previousOldDistance - oldDistance) * chaseProgressRewardScale
		end
	end

	previousOldDistance = oldDistance
	previousNewDistance = newDistance

	if onNewSide and newDistance and newDistance <= chaseReachDistance then
		reward += chaseReachReward * newSideTerminalMultiplier
		advanceNewWaypoint()
	elseif (not onNewSide) and oldDistance and oldDistance <= chaseReachDistance then
		reward += chaseReachReward
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
	params.FilterDescendantsInstances = {agent}

	local oldWaypoint = getOldWaypoint(oldTargetWaypoint)
	local newWaypoint = getNewWaypoint(newTargetWaypoint)

	local oldGoalPosition = oldWaypoint and oldWaypoint.Position or humanoidRootPart.Position
	local newGoalPosition = newWaypoint and newWaypoint.Position or humanoidRootPart.Position

	local relativeOld = humanoidRootPart.CFrame:PointToObjectSpace(oldGoalPosition)
	local relativeNew = humanoidRootPart.CFrame:PointToObjectSpace(newGoalPosition)

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

if save then
	local data = loadData(key)
	network = data and data.network or aiModule.getNetwork(networkShape)
	batchesCompleted = data and data.batchesCompleted or 0
	networkStageNumber = data and data.networkStageNumber or getStageNumber()
	averageRewardsHistory = data and data.averageRewardsHistory or {}

	game:BindToClose(function()
		saveData(key, {
			network = network,
			batchesCompleted = batchesCompleted,
			networkStageNumber = networkStageNumber,
			averageRewardsHistory = averageRewardsHistory
		})
	end)
else
	network = aiModule.getNetwork(networkShape)
	networkStageNumber = getStageNumber()
	averageRewardsHistory = {}
end

while batchesCompleted < maxBatches do
	local stageNumberForBatch = getStageNumber()

	if stageNumberForBatch ~= networkStageNumber then
		network = aiModule.getNetwork(networkShape)
		networkStageNumber = stageNumberForBatch
	end

	reset()

	for i = 1, batchSize do
		task.wait()

		local inputs = getInputs()
		local activations = aiModule.forwardPass(network, inputs)
		local rawOutputs = activations[#activations]
		local actionIndex = doAction(rawOutputs)
		local reward = getReward()

		batch[i] = {
			activations = activations,
			reward = reward,
			actionIndex = actionIndex,
			rawOutputs = rawOutputs
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

		local stepGradients = aiModule.backpropagate(network, memory.activations, outputErrors)
		aiModule.addGradients(accumulatedGradients, stepGradients)
	end

	aiModule.updateWeights(network, accumulatedGradients)

	batchesCompleted += 1

	table.clear(batch)
end

if save then
	saveData(key, {
		network = network,
		batchesCompleted = batchesCompleted,
		networkStageNumber = networkStageNumber,
		averageRewardsHistory = averageRewardsHistory
	})
end

print(aiType .. " Rewards:", averageRewardsHistory)
