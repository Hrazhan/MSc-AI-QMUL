How to run the Agent
-------------------

```java

// imports
import players.groupAU.RHEAPlayerGroupAU;
import players.groupAU.utils.ConstantsGroupAU;
import players.groupAU.utils.RHEAParamsGroupAU;

        RHEAParamsGroupAU rheaParamsGroupAU = new RHEAParamsGroupAU();
        rheaParamsGroupAU.modeling = true;

        rheaParamsGroupAU.budget_type = ConstantsGroupAU.ITERATION_BUDGET;
        rheaParamsGroupAU.iteration_budget = 200;
        rheaParamsGroupAU.population_size = 5;
        rheaParamsGroupAU.individual_length = 12;

        rheaParamsGroupAU.init_type = ConstantsGroupAU.INIT_MCTS;

        rheaParamsGroupAU.genetic_operator = ConstantsGroupAU.MUTATION_AND_CROSSOVER;
        rheaParamsGroupAU.selection_type = ConstantsGroupAU.SELECT_RANK;
        rheaParamsGroupAU.crossover_type = ConstantsGroupAU.CROSS_UNIFORM;
        rheaParamsGroupAU.mutation_type = ConstantsGroupAU.MUTATION_UNIFORM;
        rheaParamsGroupAU.mutation_rate = 0.5;

        rheaParamsGroupAU.heurisic_type = ConstantsGroupAU.CUSTOM_HEURISTIC;

        rheaParamsGroupAU.shift_buffer = true;
        rheaParamsGroupAU.evaluate_act = ConstantsGroupAU.EVALUATE_ACT_DISCOUNT;


        p = new RHEAPlayerGroupAU(seed, playerID++, rheaParamsGroupAU);
        playerStr[i-4] = "Group AU Agent";
    
```

All our implementations is in ```GameInterface.java``` file

P.S. we have also included a data folder tht contains all the data we collected.

