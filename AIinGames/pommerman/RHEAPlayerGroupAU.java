package players.groupAU;

import core.GameState;
import players.Player;
import players.groupAU.utils.RHEAParamsGroupAU;
import players.optimisers.ParameterizedPlayer;
import utils.ElapsedCpuTimer;
import utils.Types;

import java.util.Random;

import static players.rhea.utils.Constants.TIME_BUDGET;

public class RHEAPlayerGroupAU extends ParameterizedPlayer {
    private RollingHorizonPlayerGroupAU player;
    private GameInterface gInterface;
    private RHEAParamsGroupAU params;

    private Types.TILETYPE[][] memoryBoard;
    private int[][] memoryboardIDs;

    public RHEAPlayerGroupAU(long seed, int playerID) {
        this(seed, playerID, new RHEAParamsGroupAU());
    }

    public RHEAPlayerGroupAU(long seed, int playerID, RHEAParamsGroupAU params) {
        super(seed, playerID, params);
        reset(seed, playerID);
    }

    @Override
    public void reset(long seed, int playerID) {
        super.reset(seed, playerID);

        // Make sure we have parameters
        this.params = (RHEAParamsGroupAU) getParameters();
        if (this.params == null) {
            this.params = new RHEAParamsGroupAU();
            super.setParameters(this.params);
        }

        // Set up random generator
        Random randomGenerator = new Random(seed);

        // Create interface with game
        gInterface = new GameInterface(this.params, randomGenerator, playerID - Types.TILETYPE.AGENT0.getKey());

        // Set up player
        player = new RollingHorizonPlayerGroupAU(randomGenerator, this.params, gInterface);
    }

    @Override
    public Types.ACTIONS act(GameState gs) {

        ElapsedCpuTimer elapsedTimer = null;
        if (params.budget_type == TIME_BUDGET) {
            elapsedTimer = new ElapsedCpuTimer();
            elapsedTimer.setMaxTimeMillis(params.time_budget);
        }
        setup(gs, elapsedTimer);
        return gInterface.translate(player.getAction(elapsedTimer, gs.nActions()));
    }


    @Override
    public int[] getMessage() {
        // default message
        return new int[Types.MESSAGE_LENGTH];
    }

    private void setup(GameState rootState, ElapsedCpuTimer elapsedTimer) {
        gInterface.initTick(rootState, elapsedTimer);
    }

    @Override
    public Player copy() {
        return new RHEAPlayerGroupAU(seed, playerID, params);
    }
}
