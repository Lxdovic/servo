#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/OBSBuilders/DefaultObs.h>
#include <RLGymCPP/OBSBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>

using namespace GGL;
using namespace RLGC;

EnvCreateResult EnvCreateFunc(int index) {
	std::vector<WeightedReward> rewards = {

		// Movement
		{ new AirReward(), 0.1f },

		// Player-ball
		{ new FaceBallReward(), 1.f },
		{ new VelocityPlayerToBallReward(), 8.f },
		{ new TouchBallReward(), 0.1f },
		{ new StrongTouchReward(), 40.f },

		// Ball-goal
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 4.0f },

		// Boost
		// { new PickupBoostReward(), 10.f },
		// { new SaveBoostReward(), 0.2f },

		// Game events
		// { new ZeroSumReward(new BumpReward(), 0.5f), 20.f },
		// { new ZeroSumReward(new DemoReward(), 0.5f), 80.f },
		{ new GoalReward(), 150.f }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(30),
		new GoalScoreCondition()
	};

	int playersPerTeam = 1;
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	FuzzedKickoffState* kickoffStateSetter = new FuzzedKickoffState();
	RandomState* randState = new RandomState(true, true, true);

	CombinedState* combined = new CombinedState({
		{ kickoffStateSetter, 0.6f },
		{ randState, 0.4f }
	});

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new AdvancedObs();
	result.stateSetter = combined;
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;

	result.arena = arena;

	combined->ResetArena(arena);

	return result;
}

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	bool doExpensiveMetrics = (rand() % 4) == 0;

	for (auto& state : states) {
		if (doExpensiveMetrics) {
			for (auto& player : state.players) {
				report.AddAvg("Player/In Air Ratio", !player.isOnGround);
				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
				report.AddAvg("Player/Demoed Ratio", player.isDemoed);

				report.AddAvg("Player/Speed", player.vel.Length());
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));

				report.AddAvg("Player/Boost", player.boost);

				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

int main(int argc, char* argv[]) {
	RocketSim::Init("C:\\Users\\ludov\\Documents\\collision_meshes");

	LearnerConfig cfg = {};

	cfg.deviceType = LearnerDeviceType::GPU_CUDA;

	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

	cfg.numGames = 512;

	cfg.randomSeed = 123;

	int tsPerItr = 50'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 50'000; // Lower this if too much VRAM is being allocated

	// 1, 2 or 3
	cfg.ppo.epochs = 1;

	cfg.ppo.entropyScale = 0.035f;

	// Rate of reward decay
	// Starting low tends to work out
	cfg.ppo.gaeGamma = 0.99;

	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;

	cfg.ppo.sharedHead.layerSizes = { 1024, 1024 };
	cfg.ppo.policy.layerSizes = { 1024, 512, 512 };
	cfg.ppo.critic.layerSizes = { 1024, 512, 512 };

	auto optim = ModelOptimType::ADAM;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;

	auto activation = ModelActivationType::RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;

	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;

	cfg.metricsProjectName = "servo";
	cfg.metricsGroupName = "servo-runs";
	cfg.metricsRunName = "servo-run";

	cfg.sendMetrics = true;
	cfg.renderMode = false;

	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	learner->Start();

	return EXIT_SUCCESS;
}
